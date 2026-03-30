//! Raw HIP code predictor — bypasses CubeCL for the autoregressive loop.
//!
//! Extracts weight data from Burn tensors at init time, manages its own
//! GPU buffers, and launches custom HIP kernels directly. This reduces
//! per-frame kernel launches from ~7071 to ~1200, cutting code predictor
//! time from ~107ms to ~15-20ms.

use std::cell::Cell;
use std::ffi::c_void;

use burn::prelude::*;
use burn::tensor::TensorData;
use cubecl_hip_sys::{
    hipFunction_t, hipMemcpyKind_hipMemcpyDeviceToDevice, hipMemcpyKind_hipMemcpyDeviceToHost,
    hipMemcpyKind_hipMemcpyHostToDevice, hipStream_t, HIP_SUCCESS,
};

use super::kernels::HipKernels;
use crate::burn_models::code_predictor::{CodePredictor, CodePredictorConfig};

/// GPU weight pointer: either BF16 or packed int4 with per-group scales.
#[derive(Clone, Copy)]
pub(crate) enum WeightPtr {
    /// Raw BF16 weight buffer: [N, K] in row-major order.
    Bf16(*mut c_void),
    /// Packed asymmetric int4 weights + per-group f32 scales and zeros.
    /// packed: [N, K/8] as u32 (8 unsigned int4 [0..15] per u32, LSB-first)
    /// scales: [N, K/64] as f32
    /// zeros: [N, K/64] as f32
    Int4 {
        packed: *mut c_void,
        scales: *mut c_void,
        zeros: *mut c_void,
    },
}

/// Per-layer weight pointers (on GPU).
struct LayerWeights {
    q_weight: WeightPtr,
    k_weight: WeightPtr,
    v_weight: WeightPtr,
    o_weight: WeightPtr,
    gate_weight: WeightPtr,
    up_weight: WeightPtr,
    down_weight: WeightPtr,
    input_ln: *mut c_void,
    post_ln: *mut c_void,
    q_norm: *mut c_void,
    k_norm: *mut c_void,
}

/// Raw HIP code predictor.
///
/// Holds all weight data in its own GPU buffers and drives the autoregressive
/// loop with direct HIP kernel launches (no CubeCL overhead).
pub struct HipCodePredictor {
    kernels: HipKernels,
    own_stream: hipStream_t,
    active_stream: Cell<hipStream_t>,

    layers: Vec<LayerWeights>,
    final_norm: *mut c_void,
    codec_embeds: Vec<*mut c_void>,
    lm_heads: Vec<*mut c_void>,
    mtp_weight: Option<*mut c_void>,
    mtp_bias: Option<*mut c_void>,
    cos_table: *mut c_void,
    sin_table: *mut c_void,

    // KV caches (per-layer)
    k_caches: Vec<*mut c_void>,
    v_caches: Vec<*mut c_void>,

    // Scratch buffers
    input_buf: *mut c_void,
    normed_buf: *mut c_void,
    q_buf: *mut c_void,
    k_buf: *mut c_void,
    v_buf: *mut c_void,
    attn_out_buf: *mut c_void,
    projected_buf: *mut c_void,
    gate_buf: *mut c_void,
    up_buf: *mut c_void,
    mlp_buf: *mut c_void,
    logits_buf: *mut c_void,
    code_idx_buf: *mut c_void,
    embed_sum_buf: *mut c_void,
    codes_out_buf: *mut c_void,

    all_allocs: Vec<*mut c_void>,

    // Dimensions
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_layers: usize,
    num_acoustic: usize,
    max_seq: usize,
    codec_embed_dim: usize,
    rms_norm_eps: f32,
}

unsafe impl Send for HipCodePredictor {}
unsafe impl Sync for HipCodePredictor {}

impl HipCodePredictor {
    /// Build from a Burn CodePredictor by extracting all weight data.
    ///
    /// One-time GPU→CPU→GPU copy (~260MB for 0.6B model).
    pub fn from_burn<B: Backend>(
        cp: &CodePredictor<B>,
        config: &CodePredictorConfig,
        cos_data: &TensorData,
        sin_data: &TensorData,
    ) -> Result<Self, String> {
        tracing::info!("Initializing HIP code predictor (bypassing CubeCL)...");
        let t0 = std::time::Instant::now();

        let kernels = HipKernels::compile()?;

        let mut stream: hipStream_t = std::ptr::null_mut();
        hip_check(
            unsafe { cubecl_hip_sys::hipStreamCreate(&mut stream) },
            "hipStreamCreate",
        )?;

        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let vocab_size = config.vocab_size;
        let num_layers = config.num_hidden_layers;
        let num_acoustic = config.num_code_groups - 1;
        let codec_embed_dim = config.codec_embed_dim();
        let max_seq = 20; // prefill 2 + up to 14 decode + margin

        let mut ga = GpuAlloc::new();

        // Per-layer weights (Linear weights transposed from [in, out] to [out, in])
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let la = &cp.layers[i];
            let a = &la.self_attn;
            let m = &la.mlp;
            layers.push(LayerWeights {
                q_weight: WeightPtr::Bf16(ga.upload_transposed(
                    &a.q_proj.weight.val().into_data(),
                    hidden_size,
                    q_dim,
                )),
                k_weight: WeightPtr::Bf16(ga.upload_transposed(
                    &a.k_proj.weight.val().into_data(),
                    hidden_size,
                    kv_dim,
                )),
                v_weight: WeightPtr::Bf16(ga.upload_transposed(
                    &a.v_proj.weight.val().into_data(),
                    hidden_size,
                    kv_dim,
                )),
                o_weight: WeightPtr::Bf16(ga.upload_transposed(
                    &a.o_proj.weight.val().into_data(),
                    q_dim,
                    hidden_size,
                )),
                gate_weight: WeightPtr::Bf16(ga.upload_transposed(
                    &m.gate_proj.weight.val().into_data(),
                    hidden_size,
                    intermediate_size,
                )),
                up_weight: WeightPtr::Bf16(ga.upload_transposed(
                    &m.up_proj.weight.val().into_data(),
                    hidden_size,
                    intermediate_size,
                )),
                down_weight: WeightPtr::Bf16(ga.upload_transposed(
                    &m.down_proj.weight.val().into_data(),
                    intermediate_size,
                    hidden_size,
                )),
                input_ln: ga.upload(&la.input_layernorm.gamma.val().into_data()),
                post_ln: ga.upload(&la.post_attention_layernorm.gamma.val().into_data()),
                q_norm: ga.upload(&a.q_norm.gamma.val().into_data()),
                k_norm: ga.upload(&a.k_norm.gamma.val().into_data()),
            });
        }

        let final_norm = ga.upload(&cp.norm.gamma.val().into_data());

        let codec_embeds: Vec<_> = (0..num_acoustic)
            .map(|i| ga.upload(&cp.codec_embeddings[i].weight.val().into_data()))
            .collect();
        // LM heads: Linear [hidden_size, vocab_size] → transposed to [vocab_size, hidden_size]
        let lm_heads: Vec<_> = (0..num_acoustic)
            .map(|i| {
                ga.upload_transposed(
                    &cp.lm_heads[i].weight.val().into_data(),
                    hidden_size,
                    vocab_size,
                )
            })
            .collect();

        // MTP projection: Linear [codec_embed_dim, hidden_size] → transposed
        let mtp_weight = cp.small_to_mtp_projection.as_ref().map(|p| {
            ga.upload_transposed(&p.weight.val().into_data(), codec_embed_dim, hidden_size)
        });
        let mtp_bias = cp
            .small_to_mtp_projection
            .as_ref()
            .and_then(|p| p.bias.as_ref().map(|b| ga.upload(&b.val().into_data())));

        let cos_table = ga.upload(cos_data);
        let sin_table = ga.upload(sin_data);

        // KV caches
        let cache_size = num_kv_heads * max_seq * head_dim * 2;
        let k_caches: Vec<_> = (0..num_layers).map(|_| ga.alloc(cache_size)).collect();
        let v_caches: Vec<_> = (0..num_layers).map(|_| ga.alloc(cache_size)).collect();

        // Scratch buffers
        let input_buf = ga.alloc(hidden_size * 2);
        let normed_buf = ga.alloc(hidden_size * 2);
        let q_buf = ga.alloc(q_dim * 2);
        let k_buf = ga.alloc(kv_dim * 2);
        let v_buf = ga.alloc(kv_dim * 2);
        let attn_out_buf = ga.alloc(q_dim * 2);
        let projected_buf = ga.alloc(usize::max(hidden_size, codec_embed_dim) * 2);
        let gate_buf = ga.alloc(intermediate_size * 2);
        let up_buf = ga.alloc(intermediate_size * 2);
        let mlp_buf = ga.alloc(intermediate_size * 2);
        let logits_buf = ga.alloc(vocab_size * 2);
        let code_idx_buf = ga.alloc(4);
        let embed_sum_buf = ga.alloc(codec_embed_dim * 2);
        let codes_out_buf = ga.alloc(num_acoustic * 4);

        tracing::info!(
            "HIP code predictor ready in {:.0}ms ({:.1}MB weights, {} buffers)",
            t0.elapsed().as_secs_f64() * 1000.0,
            ga.total_bytes as f64 / 1_048_576.0,
            ga.ptrs.len(),
        );
        let all_allocs = ga.ptrs;

        Ok(Self {
            kernels,
            own_stream: stream,
            active_stream: Cell::new(stream),
            layers,
            final_norm,
            codec_embeds,
            lm_heads,
            mtp_weight,
            mtp_bias,
            cos_table,
            sin_table,
            k_caches,
            v_caches,
            input_buf,
            normed_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            projected_buf,
            gate_buf,
            up_buf,
            mlp_buf,
            logits_buf,
            code_idx_buf,
            embed_sum_buf,
            codes_out_buf,
            all_allocs,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            num_layers,
            num_acoustic,
            max_seq,
            codec_embed_dim,
            rms_norm_eps: config.rms_norm_eps as f32,
        })
    }

    /// Generate all 15 acoustic codes.
    ///
    /// Inputs are BF16 bytes of shape [1, 1, codec_embed_dim].
    /// Replace BF16 layer weights with int4 packed weights from a quantized file.
    pub fn load_int4_weights(
        &mut self,
        int4_file: &crate::burn_models::weight_loader::Int4SafeTensors,
    ) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        let mut ga = GpuAlloc::new();
        let mut replaced = 0usize;

        for (l, lw) in self.layers.iter_mut().enumerate() {
            let prefix = format!("talker.code_predictor.model.layers.{l}");
            let projs = [
                (&mut lw.q_weight, "self_attn.q_proj.weight"),
                (&mut lw.k_weight, "self_attn.k_proj.weight"),
                (&mut lw.v_weight, "self_attn.v_proj.weight"),
                (&mut lw.o_weight, "self_attn.o_proj.weight"),
                (&mut lw.gate_weight, "mlp.gate_proj.weight"),
                (&mut lw.up_weight, "mlp.up_proj.weight"),
                (&mut lw.down_weight, "mlp.down_proj.weight"),
            ];
            for (weight_ptr, suffix) in projs {
                let int4_key = format!("{prefix}.{suffix}_int4");
                let scales_key = format!("{prefix}.{suffix}_scales");
                let zeros_key = format!("{prefix}.{suffix}_zeros");
                if int4_file.has_key(&int4_key) {
                    let packed_bytes =
                        int4_file.raw_bytes(&int4_key).map_err(|e| e.to_string())?;
                    let scale_bytes =
                        int4_file.raw_bytes(&scales_key).map_err(|e| e.to_string())?;
                    let zeros_bytes =
                        int4_file.raw_bytes(&zeros_key).map_err(|e| e.to_string())?;
                    if let WeightPtr::Bf16(old) = *weight_ptr {
                        unsafe { cubecl_hip_sys::hipFree(old) };
                    }
                    *weight_ptr = ga.upload_int4_weight(packed_bytes, scale_bytes, zeros_bytes);
                    replaced += 1;
                }
            }
        }

        self.all_allocs.extend(ga.ptrs);

        tracing::info!(
            "Loaded int4 weights for HIP code predictor: {} projections replaced ({:.1}MB) in {:.0}ms",
            replaced,
            ga.total_bytes as f64 / 1_048_576.0,
            t0.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(())
    }

    /// Returns (codes, embed_sum_bf16_bytes).
    ///
    /// Takes `&self` because all mutations happen through raw GPU pointers,
    /// not Rust-owned data. Single-threaded use assumed.
    pub fn generate(
        &self,
        talker_hidden_bytes: &[u8],
        semantic_embed_bytes: &[u8],
    ) -> (Vec<u32>, Vec<u8>) {
        // Zero the embed_sum accumulator
        let zeros = vec![0u8; self.codec_embed_dim * 2];
        hip_h2d(self.embed_sum_buf, zeros.as_ptr() as *const _, zeros.len());

        // Prefill: process talker_hidden (offset=0) and semantic_embed (offset=1)
        // as two sequential single-token steps (equivalent to causal attention on seq=2)
        self.upload_input(talker_hidden_bytes);
        self.forward_one_token(0);

        self.upload_input(semantic_embed_bytes);
        self.forward_one_token(1);

        // First acoustic code (group 0)
        self.run_lm_head(0);

        // Autoregressive decode: groups 1..14
        for group_idx in 1..self.num_acoustic {
            // Embed previous code → accumulate into embed_sum, write to projected_buf
            self.launch_embedding_gather_add(
                self.codec_embeds[group_idx - 1],
                self.code_idx_buf,
                self.embed_sum_buf,
                self.projected_buf,
                self.codec_embed_dim,
            );

            // Project or copy embedding to input_buf
            if let (Some(w), Some(b)) = (self.mtp_weight, self.mtp_bias) {
                self.launch_gemv(
                    WeightPtr::Bf16(w),
                    self.projected_buf,
                    self.input_buf,
                    self.hidden_size,
                    self.codec_embed_dim,
                    b,
                );
            } else {
                self.d2d_copy(self.projected_buf, self.input_buf, self.hidden_size * 2);
            }

            let offset = 2 + group_idx - 1;
            self.forward_one_token(offset);
            self.run_lm_head(group_idx);
        }

        // Embed final code for embed_sum
        self.launch_embedding_gather_add(
            self.codec_embeds[self.num_acoustic - 1],
            self.code_idx_buf,
            self.embed_sum_buf,
            self.gate_buf, // dummy output
            self.codec_embed_dim,
        );

        // Sync and read results
        unsafe { cubecl_hip_sys::hipStreamSynchronize(self.active_stream.get()) };

        let mut codes_i32 = vec![0i32; self.num_acoustic];
        hip_d2h(
            codes_i32.as_mut_ptr() as *mut c_void,
            self.codes_out_buf,
            self.num_acoustic * 4,
        );

        let mut embed_bytes = vec![0u8; self.codec_embed_dim * 2];
        hip_d2h(
            embed_bytes.as_mut_ptr() as *mut c_void,
            self.embed_sum_buf,
            self.codec_embed_dim * 2,
        );

        let codes = codes_i32.iter().map(|&c| c as u32).collect();
        (codes, embed_bytes)
    }

    /// Upload input data and apply optional MTP projection.
    fn upload_input(&self, data: &[u8]) {
        if let (Some(w), Some(b)) = (self.mtp_weight, self.mtp_bias) {
            hip_h2d(self.projected_buf, data.as_ptr() as *const _, data.len());
            self.launch_gemv(
                WeightPtr::Bf16(w),
                self.projected_buf,
                self.input_buf,
                self.hidden_size,
                self.codec_embed_dim,
                b,
            );
        } else {
            hip_h2d(self.input_buf, data.as_ptr() as *const _, data.len());
        }
    }

    /// Run LM head + argmax + copy code index to output.
    fn run_lm_head(&self, group_idx: usize) {
        self.launch_gemv(
            WeightPtr::Bf16(self.lm_heads[group_idx]),
            self.normed_buf,
            self.logits_buf,
            self.vocab_size,
            self.hidden_size,
            std::ptr::null_mut(),
        );
        self.launch_argmax(self.logits_buf, self.code_idx_buf, self.vocab_size);
        // Copy code_idx (4 bytes) to codes_out[group_idx]
        let dst = unsafe { (self.codes_out_buf as *mut u8).add(group_idx * 4) as *mut c_void };
        self.d2d_copy(self.code_idx_buf, dst, 4);
    }

    /// Process input_buf through all layers + final norm → normed_buf.
    fn forward_one_token(&self, offset: usize) {
        let hs = self.hidden_size;
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let inter = self.intermediate_size;
        let eps = self.rms_norm_eps;

        for l in 0..self.num_layers {
            let lw = &self.layers[l];

            // Input layernorm
            self.launch_rmsnorm(self.input_buf, lw.input_ln, self.normed_buf, hs, eps);

            // QKV projections
            self.launch_gemv(
                lw.q_weight,
                self.normed_buf,
                self.q_buf,
                q_dim,
                hs,
                std::ptr::null_mut(),
            );
            self.launch_gemv(
                lw.k_weight,
                self.normed_buf,
                self.k_buf,
                kv_dim,
                hs,
                std::ptr::null_mut(),
            );
            self.launch_gemv(
                lw.v_weight,
                self.normed_buf,
                self.v_buf,
                kv_dim,
                hs,
                std::ptr::null_mut(),
            );

            // QK norm + RoPE
            self.launch_qk_norm_rope(lw.q_norm, lw.k_norm, offset);

            // KV cache update
            self.launch_kv_cache_append(l, offset);

            // Attention
            self.launch_attention_decode(l, offset + 1);

            // O projection
            self.launch_gemv(
                lw.o_weight,
                self.attn_out_buf,
                self.projected_buf,
                hs,
                q_dim,
                std::ptr::null_mut(),
            );

            // Residual: input_buf += projected_buf
            self.launch_add_inplace(self.input_buf, self.projected_buf, hs);

            // Post-attention layernorm
            self.launch_rmsnorm(self.input_buf, lw.post_ln, self.normed_buf, hs, eps);

            // MLP
            self.launch_gemv(
                lw.gate_weight,
                self.normed_buf,
                self.gate_buf,
                inter,
                hs,
                std::ptr::null_mut(),
            );
            self.launch_gemv(
                lw.up_weight,
                self.normed_buf,
                self.up_buf,
                inter,
                hs,
                std::ptr::null_mut(),
            );
            self.launch_silu_mul(inter);
            self.launch_gemv(
                lw.down_weight,
                self.mlp_buf,
                self.projected_buf,
                hs,
                inter,
                std::ptr::null_mut(),
            );

            // Residual: input_buf += projected_buf
            self.launch_add_inplace(self.input_buf, self.projected_buf, hs);
        }

        // Final norm
        self.launch_rmsnorm(self.input_buf, self.final_norm, self.normed_buf, hs, eps);
    }

    // ==================== Kernel launch wrappers ====================
    //
    // Each wrapper stores argument values in local variables to ensure they
    // live until hipModuleLaunchKernel copies the params synchronously.

    fn launch_gemv(
        &self,
        weight: WeightPtr,
        input: *mut c_void,
        output: *mut c_void,
        n: usize,
        k: usize,
        bias: *mut c_void,
    ) {
        match weight {
            WeightPtr::Bf16(w) => {
                let mut p_input = input;
                let mut p_weight = w;
                let mut p_output = output;
                let mut p_bias = bias;
                let mut k_i32 = k as i32;
                let mut n_i32 = n as i32;
                let mut args: [*mut c_void; 6] = [
                    ptr_of(&mut p_input),
                    ptr_of(&mut p_weight),
                    ptr_of(&mut p_output),
                    ptr_of(&mut p_bias),
                    ptr_of(&mut k_i32),
                    ptr_of(&mut n_i32),
                ];
                self.launch_kernel(
                    self.kernels.gemv,
                    n as u32,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    &mut args,
                );
            }
            WeightPtr::Int4 { packed, scales, zeros } => {
                let mut p_input = input;
                let mut p_packed = packed;
                let mut p_output = output;
                let mut p_bias = bias;
                let mut p_scales = scales;
                let mut p_zeros = zeros;
                let mut k_i32 = k as i32;
                let mut n_i32 = n as i32;
                let mut args: [*mut c_void; 8] = [
                    ptr_of(&mut p_input),
                    ptr_of(&mut p_packed),
                    ptr_of(&mut p_output),
                    ptr_of(&mut p_bias),
                    ptr_of(&mut p_scales),
                    ptr_of(&mut p_zeros),
                    ptr_of(&mut k_i32),
                    ptr_of(&mut n_i32),
                ];
                self.launch_kernel(
                    self.kernels.gemv_int4,
                    n as u32,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    &mut args,
                );
            }
        }
    }

    fn launch_rmsnorm(
        &self,
        input: *mut c_void,
        weight: *mut c_void,
        output: *mut c_void,
        d: usize,
        eps: f32,
    ) {
        let mut p_in = input;
        let mut p_w = weight;
        let mut p_out = output;
        let mut d_i32 = d as i32;
        let mut eps_f32 = eps;
        let mut args: [*mut c_void; 5] = [
            ptr_of(&mut p_in),
            ptr_of(&mut p_w),
            ptr_of(&mut p_out),
            ptr_of(&mut d_i32),
            ptr_of(&mut eps_f32),
        ];
        self.launch_kernel(self.kernels.rmsnorm, 1, 1, 1, 256, 1, 1, 0, &mut args);
    }

    fn launch_qk_norm_rope(&self, q_norm_w: *mut c_void, k_norm_w: *mut c_void, position: usize) {
        let total_blocks = (self.num_heads + self.num_kv_heads) as u32;
        let smem = (128 + self.head_dim) * 4; // reduce + normed floats

        let mut p_q = self.q_buf;
        let mut p_k = self.k_buf;
        let mut p_qn = q_norm_w;
        let mut p_kn = k_norm_w;
        let mut p_cos = self.cos_table;
        let mut p_sin = self.sin_table;
        let mut num_q = self.num_heads as i32;
        let mut num_kv = self.num_kv_heads as i32;
        let mut hd = self.head_dim as i32;
        let mut pos = position as i32;
        let mut eps = self.rms_norm_eps;

        let mut args: [*mut c_void; 11] = [
            ptr_of(&mut p_q),
            ptr_of(&mut p_k),
            ptr_of(&mut p_qn),
            ptr_of(&mut p_kn),
            ptr_of(&mut p_cos),
            ptr_of(&mut p_sin),
            ptr_of(&mut num_q),
            ptr_of(&mut num_kv),
            ptr_of(&mut hd),
            ptr_of(&mut pos),
            ptr_of(&mut eps),
        ];
        self.launch_kernel(
            self.kernels.qk_norm_rope,
            total_blocks,
            1,
            1,
            128,
            1,
            1,
            smem as u32,
            &mut args,
        );
    }

    fn launch_kv_cache_append(&self, layer: usize, offset: usize) {
        let total = self.num_kv_heads * self.head_dim;
        let blocks = div_ceil(total, 256) as u32;

        let mut p_k = self.k_buf;
        let mut p_v = self.v_buf;
        let mut p_kc = self.k_caches[layer];
        let mut p_vc = self.v_caches[layer];
        let mut n_kv = self.num_kv_heads as i32;
        let mut ms = self.max_seq as i32;
        let mut hd = self.head_dim as i32;
        let mut off = offset as i32;

        let mut args: [*mut c_void; 8] = [
            ptr_of(&mut p_k),
            ptr_of(&mut p_v),
            ptr_of(&mut p_kc),
            ptr_of(&mut p_vc),
            ptr_of(&mut n_kv),
            ptr_of(&mut ms),
            ptr_of(&mut hd),
            ptr_of(&mut off),
        ];
        self.launch_kernel(
            self.kernels.kv_cache_append,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            &mut args,
        );
    }

    fn launch_attention_decode(&self, layer: usize, seq_kv: usize) {
        let mut p_q = self.q_buf;
        let mut p_kc = self.k_caches[layer];
        let mut p_vc = self.v_caches[layer];
        let mut p_out = self.attn_out_buf;
        let mut n_q = self.num_heads as i32;
        let mut n_kv = self.num_kv_heads as i32;
        let mut hd = self.head_dim as i32;
        let mut ms = self.max_seq as i32;
        let mut skv = seq_kv as i32;
        let mut scale = (self.head_dim as f32).powf(-0.5);

        let mut args: [*mut c_void; 10] = [
            ptr_of(&mut p_q),
            ptr_of(&mut p_kc),
            ptr_of(&mut p_vc),
            ptr_of(&mut p_out),
            ptr_of(&mut n_q),
            ptr_of(&mut n_kv),
            ptr_of(&mut hd),
            ptr_of(&mut ms),
            ptr_of(&mut skv),
            ptr_of(&mut scale),
        ];
        self.launch_kernel(
            self.kernels.attention_decode,
            self.num_heads as u32,
            1,
            1,
            128,
            1,
            1,
            0,
            &mut args,
        );
    }

    fn launch_silu_mul(&self, n: usize) {
        let blocks = div_ceil(n, 256) as u32;
        let mut p_gate = self.gate_buf;
        let mut p_up = self.up_buf;
        let mut p_out = self.mlp_buf;
        let mut n_i32 = n as i32;
        let mut args: [*mut c_void; 4] = [
            ptr_of(&mut p_gate),
            ptr_of(&mut p_up),
            ptr_of(&mut p_out),
            ptr_of(&mut n_i32),
        ];
        self.launch_kernel(self.kernels.silu_mul, blocks, 1, 1, 256, 1, 1, 0, &mut args);
    }

    fn launch_add_inplace(&self, y: *mut c_void, x: *mut c_void, n: usize) {
        let blocks = div_ceil(n, 256) as u32;
        let mut p_y = y;
        let mut p_x = x;
        let mut n_i32 = n as i32;
        let mut args: [*mut c_void; 3] = [ptr_of(&mut p_y), ptr_of(&mut p_x), ptr_of(&mut n_i32)];
        self.launch_kernel(
            self.kernels.add_inplace,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            &mut args,
        );
    }

    fn launch_argmax(&self, input: *mut c_void, result: *mut c_void, n: usize) {
        let mut p_in = input;
        let mut p_res = result;
        let mut n_i32 = n as i32;
        let mut args: [*mut c_void; 3] =
            [ptr_of(&mut p_in), ptr_of(&mut p_res), ptr_of(&mut n_i32)];
        self.launch_kernel(self.kernels.argmax, 1, 1, 1, 256, 1, 1, 0, &mut args);
    }

    fn launch_embedding_gather_add(
        &self,
        table: *mut c_void,
        index: *mut c_void,
        acc: *mut c_void,
        output: *mut c_void,
        dim: usize,
    ) {
        let blocks = div_ceil(dim, 256) as u32;
        let mut p_t = table;
        let mut p_i = index;
        let mut p_a = acc;
        let mut p_o = output;
        let mut d_i32 = dim as i32;
        let mut args: [*mut c_void; 5] = [
            ptr_of(&mut p_t),
            ptr_of(&mut p_i),
            ptr_of(&mut p_a),
            ptr_of(&mut p_o),
            ptr_of(&mut d_i32),
        ];
        self.launch_kernel(
            self.kernels.embedding_gather_add,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            &mut args,
        );
    }

    fn d2d_copy(&self, src: *mut c_void, dst: *mut c_void, size: usize) {
        unsafe {
            cubecl_hip_sys::hipMemcpyAsync(
                dst,
                src,
                size,
                hipMemcpyKind_hipMemcpyDeviceToDevice,
                self.active_stream.get(),
            );
        }
    }

    fn launch_kernel(
        &self,
        func: hipFunction_t,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        smem: u32,
        args: &mut [*mut c_void],
    ) {
        let status = unsafe {
            cubecl_hip_sys::hipModuleLaunchKernel(
                func,
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
                smem,
                self.active_stream.get(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        debug_assert_eq!(status, HIP_SUCCESS, "Kernel launch failed: {status}");
    }

    // ==================== GPU-to-GPU interface ====================

    /// Generate acoustic codes from GPU-resident inputs.
    ///
    /// Uses the provided stream for all kernel launches. Does NOT synchronize.
    /// After return, codes are in `codes_out_buf` and embed sum in `embed_sum_buf`.
    pub fn generate_gpu_to_gpu(
        &self,
        stream: hipStream_t,
        hidden_ptr: *mut c_void,
        semantic_ptr: *mut c_void,
    ) {
        let _guard = StreamGuard::new(&self.active_stream, stream);

        // Zero embed_sum (async on active stream)
        unsafe {
            cubecl_hip_sys::hipMemsetAsync(
                self.embed_sum_buf,
                0,
                self.codec_embed_dim * 2,
                self.active_stream.get(),
            );
        }

        // Prefill: hidden (offset=0), semantic (offset=1)
        self.upload_input_from_gpu(hidden_ptr);
        self.forward_one_token(0);

        self.upload_input_from_gpu(semantic_ptr);
        self.forward_one_token(1);

        // First acoustic code
        self.run_lm_head(0);

        // Autoregressive decode: groups 1..14
        for group_idx in 1..self.num_acoustic {
            self.launch_embedding_gather_add(
                self.codec_embeds[group_idx - 1],
                self.code_idx_buf,
                self.embed_sum_buf,
                self.projected_buf,
                self.codec_embed_dim,
            );

            if let (Some(w), Some(b)) = (self.mtp_weight, self.mtp_bias) {
                self.launch_gemv(
                    WeightPtr::Bf16(w),
                    self.projected_buf,
                    self.input_buf,
                    self.hidden_size,
                    self.codec_embed_dim,
                    b,
                );
            } else {
                self.d2d_copy(self.projected_buf, self.input_buf, self.hidden_size * 2);
            }

            let offset = 2 + group_idx - 1;
            self.forward_one_token(offset);
            self.run_lm_head(group_idx);
        }

        // Embed final code for embed_sum
        self.launch_embedding_gather_add(
            self.codec_embeds[self.num_acoustic - 1],
            self.code_idx_buf,
            self.embed_sum_buf,
            self.gate_buf, // dummy output
            self.codec_embed_dim,
        );
        // No sync — caller is responsible
    }

    /// Upload input from a GPU pointer (d2d copy with optional MTP projection).
    fn upload_input_from_gpu(&self, src_ptr: *mut c_void) {
        if let (Some(w), Some(b)) = (self.mtp_weight, self.mtp_bias) {
            self.d2d_copy(src_ptr, self.projected_buf, self.codec_embed_dim * 2);
            self.launch_gemv(
                WeightPtr::Bf16(w),
                self.projected_buf,
                self.input_buf,
                self.hidden_size,
                self.codec_embed_dim,
                b,
            );
        } else {
            self.d2d_copy(src_ptr, self.input_buf, self.hidden_size * 2);
        }
    }

    // ==================== Accessors for frame loop ====================

    pub(crate) fn embed_sum_ptr(&self) -> *mut c_void {
        self.embed_sum_buf
    }
    pub(crate) fn codes_out_ptr(&self) -> *mut c_void {
        self.codes_out_buf
    }
    pub(crate) fn num_acoustic(&self) -> usize {
        self.num_acoustic
    }
    pub(crate) fn kernel_add_inplace(&self) -> hipFunction_t {
        self.kernels.add_inplace
    }
    pub(crate) fn kernel_embedding_gather_add(&self) -> hipFunction_t {
        self.kernels.embedding_gather_add
    }
}

impl Drop for HipCodePredictor {
    fn drop(&mut self) {
        unsafe { cubecl_hip_sys::hipStreamSynchronize(self.own_stream) };
        for ptr in &self.all_allocs {
            unsafe { cubecl_hip_sys::hipFree(*ptr) };
        }
        unsafe { cubecl_hip_sys::hipStreamDestroy(self.own_stream) };
    }
}

/// RAII guard that temporarily sets a Cell<hipStream_t> to a new value
/// and restores the original on drop (even on panic).
pub(crate) struct StreamGuard<'a> {
    cell: &'a Cell<hipStream_t>,
    original: hipStream_t,
}

impl<'a> StreamGuard<'a> {
    pub(crate) fn new(cell: &'a Cell<hipStream_t>, new_stream: hipStream_t) -> Self {
        let original = cell.get();
        cell.set(new_stream);
        Self { cell, original }
    }
}

impl Drop for StreamGuard<'_> {
    fn drop(&mut self) {
        self.cell.set(self.original);
    }
}

// ==================== Helpers ====================

#[inline]
pub(crate) fn ptr_of<T>(val: &mut T) -> *mut c_void {
    val as *mut T as *mut c_void
}

pub(crate) fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

pub(crate) fn hip_check(status: u32, op: &str) -> Result<(), String> {
    if status != HIP_SUCCESS {
        Err(format!("{op} failed: {status}"))
    } else {
        Ok(())
    }
}

pub(crate) fn hip_malloc(size: usize) -> *mut c_void {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let s = unsafe { cubecl_hip_sys::hipMalloc(&mut ptr, size) };
    assert_eq!(s, HIP_SUCCESS, "hipMalloc({size}) failed: {s}");
    ptr
}

pub(crate) fn hip_h2d(dst: *mut c_void, src: *const c_void, size: usize) {
    let s =
        unsafe { cubecl_hip_sys::hipMemcpy(dst, src, size, hipMemcpyKind_hipMemcpyHostToDevice) };
    assert_eq!(s, HIP_SUCCESS, "hipMemcpy H2D failed: {s}");
}

pub(crate) fn hip_d2h(dst: *mut c_void, src: *mut c_void, size: usize) {
    let s = unsafe {
        cubecl_hip_sys::hipMemcpy(
            dst,
            src as *const _,
            size,
            hipMemcpyKind_hipMemcpyDeviceToHost,
        )
    };
    assert_eq!(s, HIP_SUCCESS, "hipMemcpy D2H failed: {s}");
}

/// GPU memory allocator that tracks all allocations for cleanup.
pub(crate) struct GpuAlloc {
    pub(crate) ptrs: Vec<*mut c_void>,
    pub(crate) total_bytes: usize,
}

impl GpuAlloc {
    pub(crate) fn new() -> Self {
        Self {
            ptrs: Vec::new(),
            total_bytes: 0,
        }
    }

    pub(crate) fn upload(&mut self, data: &TensorData) -> *mut c_void {
        let bytes = data.as_bytes();
        self.total_bytes += bytes.len();
        let ptr = hip_malloc(bytes.len());
        hip_h2d(ptr, bytes.as_ptr() as *const c_void, bytes.len());
        self.ptrs.push(ptr);
        ptr
    }

    /// Upload a 2D weight matrix, transposing from [rows, cols] to [cols, rows].
    /// Burn stores Linear weights as [d_input, d_output] but our gemv kernel
    /// expects [d_output, d_input] for coalesced reads.
    pub(crate) fn upload_transposed(
        &mut self,
        data: &TensorData,
        rows: usize,
        cols: usize,
    ) -> *mut c_void {
        let bytes = data.as_bytes();
        assert_eq!(bytes.len(), rows * cols * 2, "Weight size mismatch");
        self.total_bytes += bytes.len();

        // Transpose BF16 elements: src[i * cols + j] → dst[j * rows + i]
        let mut transposed = vec![0u8; bytes.len()];
        for i in 0..rows {
            for j in 0..cols {
                let src_off = (i * cols + j) * 2;
                let dst_off = (j * rows + i) * 2;
                transposed[dst_off] = bytes[src_off];
                transposed[dst_off + 1] = bytes[src_off + 1];
            }
        }

        let ptr = hip_malloc(bytes.len());
        hip_h2d(ptr, transposed.as_ptr() as *const c_void, transposed.len());
        self.ptrs.push(ptr);
        ptr
    }

    /// Upload raw bytes to GPU. Used for pre-packed int4 weights and scales.
    pub(crate) fn upload_raw(&mut self, bytes: &[u8]) -> *mut c_void {
        self.total_bytes += bytes.len();
        let ptr = hip_malloc(bytes.len());
        hip_h2d(ptr, bytes.as_ptr() as *const c_void, bytes.len());
        self.ptrs.push(ptr);
        ptr
    }

    /// Upload asymmetric int4 packed weights + scales + zeros, returning a WeightPtr::Int4.
    pub(crate) fn upload_int4_weight(
        &mut self,
        packed_bytes: &[u8],
        scale_bytes: &[u8],
        zeros_bytes: &[u8],
    ) -> WeightPtr {
        let packed = self.upload_raw(packed_bytes);
        let scales = self.upload_raw(scale_bytes);
        let zeros = self.upload_raw(zeros_bytes);
        WeightPtr::Int4 { packed, scales, zeros }
    }

    pub(crate) fn alloc(&mut self, size: usize) -> *mut c_void {
        let ptr = hip_malloc(size);
        self.ptrs.push(ptr);
        ptr
    }
}
