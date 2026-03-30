#!/usr/bin/env python3
"""Quantize Qwen3-TTS model weights to int4 (symmetric, group_size=64).

Produces a new safetensors file with:
- Linear weights packed as int4 in u32 (8 values per u32, LSB-first)
- Per-group f32 scales
- Non-quantized tensors (norms, embeddings) passed through as BF16

Usage:
    nix-shell -p python3Packages.numpy python3Packages.safetensors --run \
        "python3 scripts/quantize_int4.py \
            --input test_data/models/0.6B-CustomVoice/model.safetensors \
            --output test_data/models/0.6B-CustomVoice/model_int4.safetensors"
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────

GROUP_SIZE = 64
BITS = 4
# Asymmetric unsigned int4 range: [0, 15]
QMIN = 0
QMAX = (1 << BITS) - 1  # 15

# Keys to quantize: Linear weights with 2D shape, excluding norms/embeddings
SKIP_PATTERNS = ["norm", "embed", "codec_head"]


# ── Safetensors I/O ───────────────────────────────────────────────────────

def read_safetensors(path: Path):
    """Read safetensors file, return (header_dict, raw_bytes)."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data = f.read()
    return header, data


def extract_tensor(header, data, key):
    """Extract a tensor as numpy array from safetensors data."""
    info = header[key]
    dtype_str = info["dtype"]
    shape = info["shape"]
    start, end = info["data_offsets"]
    raw = data[start:end]

    if dtype_str == "BF16":
        # Read as uint16, convert to float32 via bit manipulation
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        return bf16_to_f32(u16)
    elif dtype_str == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
    elif dtype_str == "F16":
        return np.frombuffer(raw, dtype=np.float16).copy().reshape(shape).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype {dtype_str} for key {key}")


def bf16_to_f32(u16_array):
    """Convert BF16 (as uint16) to float32."""
    # BF16 is the upper 16 bits of float32
    f32_bits = u16_array.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def f32_to_bf16(f32_array):
    """Convert float32 to BF16 (as uint16), round-to-nearest-even."""
    f32_bits = f32_array.view(np.uint32)
    # Round to nearest even: add 0x7FFF + bit 16 (for round-to-even)
    rounding = (f32_bits >> 16) & 1
    f32_bits = f32_bits + 0x7FFF + rounding
    return (f32_bits >> 16).astype(np.uint16)


# ── Quantization ───────────────────────────────────────────────────────────

def quantize_asymmetric_int4_hqq(weight_f32, group_size=GROUP_SIZE):
    """Quantize a 2D weight matrix to asymmetric unsigned int4 using HQQ.

    Uses Half-Quadratic Quantization (HQQ) proximal optimization to find
    optimal scale and zero-point that minimize reconstruction error.
    Critical for TTS: naive round-to-nearest (RTN) produces garbage audio
    because it doesn't preserve token decision boundaries.

    Asymmetric quantization maps [min, max] → [0, 15]:
      q = clamp(round(x / scale + zero), 0, 15)
      x_hat = (q - zero) * scale

    HQQ iteratively optimizes zero-point via Lp-norm proximal shrinkage.

    Reference: https://github.com/mobiusml/hqq

    Args:
        weight_f32: numpy array of shape [N, K], float32
        group_size: number of elements per quantization group

    Returns:
        packed_u32: numpy array of shape [N, K // 8], uint32
            8 unsigned int4 values packed per u32, LSB-first
        scales: numpy array of shape [N, K // group_size], float32
        zeros: numpy array of shape [N, K // group_size], float32
    """
    N, K = weight_f32.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert group_size % 8 == 0, f"group_size={group_size} must be divisible by 8"

    num_groups = K // group_size
    # Reshape to [N, num_groups, group_size]
    w = weight_f32.reshape(N, num_groups, group_size)

    # --- Step 1: Initial scale and zero from min/max (RTN baseline) ---
    w_min = w.min(axis=2)  # [N, num_groups]
    w_max = w.max(axis=2)  # [N, num_groups]

    scale = (w_max - w_min) / QMAX  # [N, num_groups]
    scale = np.maximum(scale, 1e-10)
    zero = np.round(-w_min / scale)  # [N, num_groups]
    zero = np.clip(zero, 0, QMAX)

    # --- Step 2: HQQ proximal optimization of zero-point ---
    # Iteratively refine zero to minimize reconstruction error
    lp_norm = 0.7
    beta = 10.0
    kappa = 1.01
    iters = 20

    scale_exp = scale[:, :, np.newaxis]  # [N, num_groups, 1]
    zero_exp = zero[:, :, np.newaxis]    # [N, num_groups, 1]

    best_err = np.full((N, num_groups), np.inf)
    best_zero = zero.copy()

    for _it in range(iters):
        # Quantize
        w_q = np.round(w / scale_exp + zero_exp)
        w_q = np.clip(w_q, QMIN, QMAX)

        # Dequantize
        w_r = (w_q - zero_exp) * scale_exp

        # Per-group mean absolute error
        err = np.abs(w - w_r).mean(axis=2)  # [N, num_groups]

        # Track best zero per group
        improved = err < best_err
        best_err = np.where(improved, err, best_err)
        best_zero = np.where(improved, zero, best_zero)

        # Residual for shrinkage
        residual = w - w_r

        # Lp proximal shrinkage operator
        abs_res = np.abs(residual) + 1e-10
        shrink = abs_res - (1.0 / beta) * np.power(abs_res, lp_norm - 1.0)
        shrink = np.maximum(shrink, 0.0) * np.sign(residual)

        # Update zero using shrinkage-adjusted weights
        w_adj = w - shrink
        # Recompute zero from adjusted weights (keep scale fixed)
        zero = np.round(-w_adj.min(axis=2) / scale)
        zero = np.clip(zero, 0, QMAX)
        zero_exp = zero[:, :, np.newaxis]

        beta *= kappa

    # Use best zero found during optimization
    zero = best_zero
    zero_exp = zero[:, :, np.newaxis]

    # --- Final quantization with optimized zero ---
    w_q = np.round(w / scale_exp + zero_exp)
    w_q = np.clip(w_q, QMIN, QMAX).astype(np.int32)

    # Pack 8 unsigned int4 values into each u32, LSB-first
    w_q_flat = w_q.reshape(N, K)
    w_q_groups = w_q_flat.reshape(N, K // 8, 8)
    w_q_u4 = w_q_groups.astype(np.uint32) & 0xF

    packed = np.zeros((N, K // 8), dtype=np.uint32)
    for i in range(8):
        packed |= w_q_u4[:, :, i] << (i * 4)

    return packed, scale.astype(np.float32), zero.astype(np.float32)


def dequantize_int4(packed_u32, scales, zeros, K, group_size=GROUP_SIZE):
    """Dequantize packed asymmetric int4 weights back to float32."""
    N = packed_u32.shape[0]
    num_groups = K // group_size

    # Unpack u32 → 8 unsigned int4 values
    w_q = np.zeros((N, K), dtype=np.int32)
    for i in range(8):
        vals = (packed_u32 >> (i * 4)) & 0xF
        w_q[:, i::8] = vals.astype(np.int32)

    # Dequantize: x = (q - zero) * scale
    w_q = w_q.reshape(N, num_groups, group_size).astype(np.float32)
    scales_exp = scales[:, :, np.newaxis]
    zeros_exp = zeros[:, :, np.newaxis]
    return ((w_q - zeros_exp) * scales_exp).reshape(N, K)


def should_quantize(key, shape, group_size=GROUP_SIZE):
    """Determine if a tensor should be quantized to int4."""
    if "weight" not in key:
        return False
    if len(shape) != 2:
        return False
    for pat in SKIP_PATTERNS:
        if pat in key:
            return False
    # Check dimensions are compatible
    N, K = shape
    if K % group_size != 0:
        return False
    return True


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quantize model weights to int4")
    parser.add_argument("--input", required=True, help="Input safetensors file")
    parser.add_argument("--output", required=True, help="Output safetensors file")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE, help="Quantization group size")
    parser.add_argument("--validate", action="store_true", help="Validate dequantization error")
    args = parser.parse_args()

    group_size = args.group_size

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Reading {input_path}...")
    header, data = read_safetensors(input_path)

    # Collect all tensors
    metadata = header.get("__metadata__", {})
    tensor_keys = sorted([k for k in header if k != "__metadata__"])

    # Process tensors
    output_tensors = {}  # key -> (dtype_str, shape, bytes)
    quantized_count = 0
    passthrough_count = 0
    total_original_bytes = 0
    total_quantized_bytes = 0

    for key in tensor_keys:
        info = header[key]
        shape = info["shape"]
        dtype_str = info["dtype"]
        start, end = info["data_offsets"]
        original_bytes = end - start

        if should_quantize(key, shape, group_size):
            # Quantize this weight
            w_f32 = extract_tensor(header, data, key)
            N, K = shape

            packed, scales, zeros = quantize_asymmetric_int4_hqq(w_f32, group_size)

            if args.validate:
                w_recon = dequantize_int4(packed, scales, zeros, K, group_size)
                rel_err = np.abs(w_f32 - w_recon) / (np.abs(w_f32) + 1e-10)
                max_abs_err = np.abs(w_f32 - w_recon).max()
                mean_rel_err = rel_err.mean()
                print(f"  {key}: shape={shape} max_abs_err={max_abs_err:.4e} mean_rel_err={mean_rel_err:.4e}")

            packed_bytes = packed.tobytes()
            scales_bytes = scales.tobytes()
            zeros_bytes = zeros.tobytes()

            output_tensors[f"{key}_int4"] = ("U32", [N, K // 8], packed_bytes)
            output_tensors[f"{key}_scales"] = ("F32", [N, K // group_size], scales_bytes)
            output_tensors[f"{key}_zeros"] = ("F32", [N, K // group_size], zeros_bytes)

            quantized_bytes = len(packed_bytes) + len(scales_bytes) + len(zeros_bytes)
            total_original_bytes += original_bytes
            total_quantized_bytes += quantized_bytes
            quantized_count += 1
        else:
            # Pass through unchanged
            raw = data[start:end]
            output_tensors[key] = (dtype_str, shape, raw)
            passthrough_count += 1

    print(f"\nQuantized: {quantized_count} tensors")
    print(f"Passthrough: {passthrough_count} tensors")
    if total_original_bytes > 0:
        ratio = total_quantized_bytes / total_original_bytes
        print(f"Quantized weight size: {total_original_bytes / 1e6:.1f} MB → {total_quantized_bytes / 1e6:.1f} MB ({ratio:.1%})")

    # Write output safetensors
    print(f"\nWriting {output_path}...")
    write_safetensors(output_path, output_tensors, {
        **metadata,
        "quantization": f"int4_symmetric_g{group_size}",
    })
    print("Done.")


def write_safetensors(path, tensors, metadata):
    """Write tensors to a safetensors file.

    tensors: dict of key -> (dtype_str, shape, bytes)
    metadata: dict of string -> string
    """
    # Build header
    header = {}
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}

    offset = 0
    for key in sorted(tensors.keys()):
        dtype_str, shape, raw = tensors[key]
        header[key] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_size = len(header_json)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_json)
        for key in sorted(tensors.keys()):
            _, _, raw = tensors[key]
            f.write(raw)


if __name__ == "__main__":
    main()
