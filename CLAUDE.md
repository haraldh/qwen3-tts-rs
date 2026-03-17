# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
cargo build                                    # CPU debug
cargo build --release --features rocm,cli      # ROCm release with CLI
cargo build --release --features cuda,cli      # CUDA release with CLI
cargo test --lib                               # Unit tests (no model weights needed)
cargo test --test integration                  # Integration tests (no weights)
cargo test --lib -- generation::sampling       # Single test module
cargo clippy --features cli,hub -- -D warnings  # Lint (matches CI)
cargo fmt -- --check                           # Format check
cargo bench                                    # Criterion micro-benchmarks (no weights)
```

Python scripts in `scripts/` are linted with:

```bash
uvx ruff format --check scripts/
uvx ruff check scripts/
```

Pre-commit (runs both Rust and Python checks):

```bash
make pre-commit
```

## Profiling & Benchmarks

Model weights required:

```bash
make profile-chrome MODEL_DIR=test_data/models/1.7B-CustomVoice
make profile-flamegraph MODEL_DIR=test_data/models/1.7B-CustomVoice
make audit-gpu-syncs
```

E2E benchmarks:

```bash
cargo run --release --features rocm,cli --bin e2e_bench -- \
  --model-dir test_data/models/0.6B-CustomVoice --iterations 3 --warmup 2 --streaming
```

## Architecture

Two parallel model stacks exist in `src/`:

- **`burn_models/`** — Active model stack using the [Burn](https://burn.dev) framework. Multi-backend: CPU (NdArray), CUDA, ROCm (HIP), WGPU (Vulkan/SPIR-V). This is where new development happens.
- **`models/`** — Legacy Candle-based models, gated behind `_candle_legacy` feature flag. Kept for reference during migration.

### Three-stage TTS pipeline

1. **TalkerModel** (`burn_models/talker.rs`) — 28-layer transformer generating semantic tokens from text. Uses MRoPE, KV caching. 0.6B: hidden=1024, 1.7B: hidden=2048.

2. **CodePredictor** (`burn_models/code_predictor.rs`) — 5-layer transformer generating 15 acoustic codes per semantic token. Always hidden=1024; 1.7B models use `small_to_mtp_projection` to bridge from talker's 2048-dim space. Called every frame during generation.

3. **Decoder12Hz** (`burn_models/codec/decoder_12hz.rs`) — ConvNeXt + transposed convolution decoder converting 16-codebook codes to 24kHz audio. Always F32.

### Generation loop

`burn_models/facade.rs` contains `Qwen3TTS<B: Backend>`, the main facade that ties the pipeline together:

```
For each frame:
  1. CodePredictor generates 15 acoustic codes from last_hidden + semantic embedding
  2. All 16 codes (semantic + 15 acoustic) are embedded and summed (residual VQ)
  3. Trailing text embedding fused in (or tts_pad after text exhausted)
  4. Talker forward step → next hidden state + logits
  5. Sample next semantic token from logits
```

### Weight loading

`burn_models/weight_loader.rs` loads safetensors files directly into Burn modules (not using Burn's record format). All weights are converted through f32 on load, then cast to the backend's compute dtype.

## Key Types

- `Qwen3TTS<B: Backend>` (`burn_models/facade.rs`) — main facade, generic over Burn backend
- `VoiceClonePrompt<B>` (`burn_models/facade.rs`) — speaker embedding + optional ICL data
- `SynthesisOptions` / `GenerationConfig` — hyperparameters
- `AudioBuffer` (`audio/io.rs`) — PCM samples + sample_rate
- `ModelType` (`models/config.rs`) — enum: Base, CustomVoice, VoiceDesign

## Binaries

- `generate_audio` — CLI tool for batch synthesis (requires `cli` feature)
- `serve` — OpenAI-compatible TTS HTTP server at `POST /v1/audio/speech` (requires `serve` feature, which implies `cli`)
- `e2e_bench` — End-to-end benchmark (requires `cli` feature)

Backend selection in binaries uses compile-time feature flags with priority: cuda > rocm > wgpu > cpu.

## Model Variants

Five variants, auto-detected from `config.json`:

- **Base** (0.6B, 1.7B): Voice cloning via ECAPA-TDNN speaker encoder. ICL mode uses speech encoder + reference text.
- **CustomVoice** (0.6B, 1.7B): 9 preset speakers (Ryan, Serena, etc.) via discrete speaker token IDs.
- **VoiceDesign** (1.7B only): Text-described voices via instruct prompt with ChatML framing.

## Feature Flags

| Feature | Effect |
|---------|--------|
| `cpu` (default) | CPU inference via NdArray |
| `cuda` | NVIDIA GPU via Burn CUDA backend |
| `rocm` | AMD GPU via Burn ROCm/HIP backend (BF16) |
| `wgpu` | GPU via WGPU (WebGPU/Vulkan) |
| `vulkan` | Alias: enables `wgpu` + Burn Vulkan/SPIR-V |
| `cli` | CLI binaries (generate_audio, e2e_bench) |
| `serve` | OpenAI-compatible HTTP server (implies cli) |
| `hub` | HuggingFace Hub downloads |
| `profiling` | tracing-chrome spans (zero overhead when disabled) |
| `all-portable` | cpu + cli + hub (safe for `cargo check/test`) |

Platform-specific features (`cuda`, `rocm`, `wgpu`) conflict — enable only one.

## Codec Token IDs

Generation uses codec vocabulary (0–3071), not text vocabulary:

- EOS: 2150 (generation stops here)
- BOS: 2149, PAD: 2148
- Speakers: Ryan=3061, Serena=3066, etc.
- Languages: English=2050, Chinese=2055, etc.

## Conventions

- All models are generic over `B: Backend` — new code must work across all backends
- Profiling spans: `#[cfg(feature = "profiling")]` gated, `info_span!("snake_case")`
- GPU sync points: `tracing::trace!(target: "gpu_sync", ...)` markers
- Decoder and speaker encoder always run in F32 regardless of backend compute dtype
- Tests don't require model weights — use synthetic tensors
- Local path dependencies: `burn` and `cubecl` are patched to local builds (see `[patch.crates-io]` in Cargo.toml)
