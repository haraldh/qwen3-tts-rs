//! Raw HIP talker — bypasses CubeCL for the autoregressive decode step.
//!
//! Extracts weight data from Burn TalkerModel at init time, manages its own
//! GPU buffers and KV caches, and launches custom HIP kernels directly.
//! Prefill still runs through Burn; only the per-frame decode step uses HIP.

use std::cell::Cell;
use std::ffi::c_void;

use burn::prelude::*;
use cubecl_hip_sys::{hipMemcpyKind_hipMemcpyDeviceToDevice, hipStream_t, HIP_SUCCESS};

use super::kernels::HipKernels;
use crate::burn_models::kv_cache::KVCache;
use crate::burn_models::talker::TalkerModel;
use crate::burn_models::transformer::RoPEType;

// Reuse helpers from code_predictor module
use super::code_predictor::{div_ceil, hip_check, hip_d2h, hip_h2d, ptr_of, GpuAlloc, StreamGuard};

/// Per-layer weight pointers (all on GPU, BF16).
struct LayerWeights {
    q_weight: *mut c_void,
    k_weight: *mut c_void,
    v_weight: *mut c_void,
    o_weight: *mut c_void,
    gate_weight: *mut c_void,
    up_weight: *mut c_void,
    down_weight: *mut c_void,
    input_ln: *mut c_void,
    post_ln: *mut c_void,
    q_norm: *mut c_void,
    k_norm: *mut c_void,
}

/// Raw HIP talker for autoregressive decode steps.
///
/// Prefill runs through Burn, then KV cache data is transferred to HIP buffers.
/// Subsequent decode steps (one per frame) run entirely through HIP kernels.
pub struct HipTalker {
    kernels: HipKernels,
    own_stream: hipStream_t,
    active_stream: Cell<hipStream_t>,

    layers: Vec<LayerWeights>,
    final_norm: *mut c_void,
    codec_head_weight: *mut c_void,
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

    all_allocs: Vec<*mut c_void>,

    // Dimensions
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    codec_vocab_size: usize,
    num_layers: usize,
    max_seq: usize,
    rms_norm_eps: f32,
}

unsafe impl Send for HipTalker {}
unsafe impl Sync for HipTalker {}

impl HipTalker {
    /// Build from a Burn TalkerModel by extracting all weight data.
    pub fn from_burn<B: Backend>(
        talker: &TalkerModel<B>,
        rope: &RoPEType<B>,
        max_seq: usize,
    ) -> Result<Self, String> {
        tracing::info!("Initializing HIP talker (bypassing CubeCL for decode)...");
        let t0 = std::time::Instant::now();

        let kernels = HipKernels::compile()?;

        let mut stream: hipStream_t = std::ptr::null_mut();
        hip_check(
            unsafe { cubecl_hip_sys::hipStreamCreate(&mut stream) },
            "hipStreamCreate",
        )?;

        let config = talker.config();
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let codec_vocab_size = config.codec_vocab_size;
        let num_layers = config.num_hidden_layers;

        let mut ga = GpuAlloc::new();

        // Per-layer weights (Linear weights transposed from [in, out] to [out, in])
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let la = &talker.layers[i];
            let a = &la.self_attn;
            let m = &la.mlp;
            layers.push(LayerWeights {
                q_weight: ga.upload_transposed(
                    &a.q_proj.weight.val().into_data(),
                    hidden_size,
                    q_dim,
                ),
                k_weight: ga.upload_transposed(
                    &a.k_proj.weight.val().into_data(),
                    hidden_size,
                    kv_dim,
                ),
                v_weight: ga.upload_transposed(
                    &a.v_proj.weight.val().into_data(),
                    hidden_size,
                    kv_dim,
                ),
                o_weight: ga.upload_transposed(
                    &a.o_proj.weight.val().into_data(),
                    q_dim,
                    hidden_size,
                ),
                gate_weight: ga.upload_transposed(
                    &m.gate_proj.weight.val().into_data(),
                    hidden_size,
                    intermediate_size,
                ),
                up_weight: ga.upload_transposed(
                    &m.up_proj.weight.val().into_data(),
                    hidden_size,
                    intermediate_size,
                ),
                down_weight: ga.upload_transposed(
                    &m.down_proj.weight.val().into_data(),
                    intermediate_size,
                    hidden_size,
                ),
                input_ln: ga.upload(&la.input_layernorm.gamma.val().into_data()),
                post_ln: ga.upload(&la.post_attention_layernorm.gamma.val().into_data()),
                q_norm: ga.upload(&a.q_norm.gamma.val().into_data()),
                k_norm: ga.upload(&a.k_norm.gamma.val().into_data()),
            });
        }

        let final_norm = ga.upload(&talker.norm.gamma.val().into_data());

        // codec_head: Linear [hidden_size, codec_vocab_size] → transposed
        let codec_head_weight = ga.upload_transposed(
            &talker.codec_head.weight.val().into_data(),
            hidden_size,
            codec_vocab_size,
        );

        // RoPE cos/sin tables
        let (cos_data, sin_data) = rope.cos_sin_data();
        let cos_table = ga.upload(&cos_data);
        let sin_table = ga.upload(&sin_data);

        // KV caches
        let cache_size = num_kv_heads * max_seq * head_dim * 2; // BF16
        let k_caches: Vec<_> = (0..num_layers).map(|_| ga.alloc(cache_size)).collect();
        let v_caches: Vec<_> = (0..num_layers).map(|_| ga.alloc(cache_size)).collect();

        // Scratch buffers
        let input_buf = ga.alloc(hidden_size * 2);
        let normed_buf = ga.alloc(hidden_size * 2);
        let q_buf = ga.alloc(q_dim * 2);
        let k_buf = ga.alloc(kv_dim * 2);
        let v_buf = ga.alloc(kv_dim * 2);
        let attn_out_buf = ga.alloc(q_dim * 2);
        let projected_buf = ga.alloc(hidden_size * 2);
        let gate_buf = ga.alloc(intermediate_size * 2);
        let up_buf = ga.alloc(intermediate_size * 2);
        let mlp_buf = ga.alloc(intermediate_size * 2);
        let logits_buf = ga.alloc(codec_vocab_size * 2);

        tracing::info!(
            "HIP talker ready in {:.0}ms ({:.1}MB GPU, {} buffers, max_seq={})",
            t0.elapsed().as_secs_f64() * 1000.0,
            ga.total_bytes as f64 / 1_048_576.0,
            ga.ptrs.len(),
            max_seq,
        );
        let all_allocs = ga.ptrs;

        Ok(Self {
            kernels,
            own_stream: stream,
            active_stream: Cell::new(stream),
            layers,
            final_norm,
            codec_head_weight,
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
            all_allocs,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            codec_vocab_size,
            num_layers,
            max_seq,
            rms_norm_eps: config.rms_norm_eps as f32,
        })
    }

    /// Load prefill KV cache data from Burn into HIP GPU buffers.
    ///
    /// Must be called once after Burn prefill, before the first decode step.
    /// The Burn KV cache layout is [1, num_kv_heads, max_seq, head_dim] (BF16).
    /// The HIP layout is [num_kv_heads, max_seq, head_dim] (same, minus batch dim).
    pub fn load_prefill_cache<B: Backend>(&self, burn_caches: &[KVCache<B>]) {
        for (layer, cache) in burn_caches.iter().enumerate() {
            let (k_tensor, v_tensor, prefill_len) = cache.raw_kv();
            if prefill_len == 0 {
                continue;
            }

            // Extract the valid portion: narrow to [1, num_kv_heads, prefill_len, head_dim]
            let k_valid = k_tensor.clone().narrow(2, 0, prefill_len);
            let v_valid = v_tensor.clone().narrow(2, 0, prefill_len);

            let k_data = k_valid.into_data();
            let v_data = v_valid.into_data();
            let k_bytes = k_data.as_bytes();
            let v_bytes = v_data.as_bytes();

            // We need to copy into HIP cache which has layout
            // [num_kv_heads, self.max_seq, head_dim]. The Burn data is
            // [1, num_kv_heads, prefill_len, head_dim] (contiguous).
            // We must scatter each head's contiguous prefill_len*head_dim block
            // into the correct position in the HIP cache (which has stride max_seq*head_dim per head).
            let head_stride_src = prefill_len * self.head_dim * 2; // bytes
            let head_stride_dst = self.max_seq * self.head_dim * 2; // bytes

            for h in 0..self.num_kv_heads {
                let src_off = h * head_stride_src;
                let dst_off = h * head_stride_dst;
                let copy_size = head_stride_src;

                let k_dst =
                    unsafe { (self.k_caches[layer] as *mut u8).add(dst_off) as *mut c_void };
                let v_dst =
                    unsafe { (self.v_caches[layer] as *mut u8).add(dst_off) as *mut c_void };

                hip_h2d(
                    k_dst,
                    unsafe { k_bytes.as_ptr().add(src_off) as *const c_void },
                    copy_size,
                );
                hip_h2d(
                    v_dst,
                    unsafe { v_bytes.as_ptr().add(src_off) as *const c_void },
                    copy_size,
                );
            }
        }
    }

    /// Run one decode step: input → 28 layers → norm → codec_head → logits.
    ///
    /// Takes BF16 bytes of shape [hidden_size], returns (last_hidden_bytes, logits_bytes).
    pub fn forward_decode(&self, input_bytes: &[u8], offset: usize) -> (Vec<u8>, Vec<u8>) {
        // Upload input
        hip_h2d(
            self.input_buf,
            input_bytes.as_ptr() as *const c_void,
            input_bytes.len(),
        );

        // Run all layers
        self.forward_one_token(offset);

        // codec_head: normed_buf → logits_buf
        self.launch_gemv(
            self.codec_head_weight,
            self.normed_buf,
            self.logits_buf,
            self.codec_vocab_size,
            self.hidden_size,
            std::ptr::null_mut(),
        );

        // Sync and read results
        unsafe { cubecl_hip_sys::hipStreamSynchronize(self.active_stream.get()) };

        // Read last_hidden from normed_buf (after final RMSNorm, matching Burn's
        // generate_step_with_embed which returns hidden AFTER norm.forward())
        let mut hidden_bytes = vec![0u8; self.hidden_size * 2];
        hip_d2h(
            hidden_bytes.as_mut_ptr() as *mut c_void,
            self.normed_buf,
            self.hidden_size * 2,
        );

        let mut logits_bytes = vec![0u8; self.codec_vocab_size * 2];
        hip_d2h(
            logits_bytes.as_mut_ptr() as *mut c_void,
            self.logits_buf,
            self.codec_vocab_size * 2,
        );

        (hidden_bytes, logits_bytes)
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

            // Attention (long-sequence variant with dynamic shared memory)
            self.launch_attention_decode_long(l, offset + 1);

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

        // Final norm: input_buf → normed_buf
        self.launch_rmsnorm(self.input_buf, self.final_norm, self.normed_buf, hs, eps);
    }

    // ==================== Kernel launch wrappers ====================

    fn launch_gemv(
        &self,
        weight: *mut c_void,
        input: *mut c_void,
        output: *mut c_void,
        n: usize,
        k: usize,
        bias: *mut c_void,
    ) {
        let mut p_input = input;
        let mut p_weight = weight;
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
        self.launch_kernel(self.kernels.gemv, n as u32, 1, 1, 256, 1, 1, 0, &mut args);
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

    fn launch_attention_decode_long(&self, layer: usize, seq_kv: usize) {
        // Dynamic shared memory: seq_kv floats for scores + 128 floats for reduction
        let smem_bytes = (seq_kv + 128) * 4;

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
            self.kernels.attention_decode_long,
            self.num_heads as u32,
            1,
            1,
            128,
            1,
            1,
            smem_bytes as u32,
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
        func: cubecl_hip_sys::hipFunction_t,
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

    /// Run one decode step from GPU-resident input.
    ///
    /// Copies input from `input_ptr` to internal buffer, runs all layers + codec_head.
    /// After return, last hidden is in `input_buf`, logits in `logits_buf`.
    /// Does NOT synchronize.
    pub fn forward_decode_gpu_to_gpu(
        &self,
        stream: hipStream_t,
        input_ptr: *mut c_void,
        offset: usize,
    ) {
        let _guard = StreamGuard::new(&self.active_stream, stream);

        // Copy input to input_buf (async d2d)
        self.d2d_copy(input_ptr, self.input_buf, self.hidden_size * 2);

        // Run all layers
        self.forward_one_token(offset);

        // codec_head: normed_buf → logits_buf
        self.launch_gemv(
            self.codec_head_weight,
            self.normed_buf,
            self.logits_buf,
            self.codec_vocab_size,
            self.hidden_size,
            std::ptr::null_mut(),
        );
        // No sync — caller is responsible
    }

    // ==================== Accessors for frame loop ====================

    #[allow(dead_code)]
    pub(crate) fn input_ptr(&self) -> *mut c_void {
        self.input_buf
    }
    /// Pointer to the normed hidden state (after final RMSNorm).
    ///
    /// This is the correct hidden state to pass to the code predictor,
    /// matching Burn's `generate_step_with_embed` which returns hidden
    /// AFTER `norm.forward()`.
    pub(crate) fn normed_ptr(&self) -> *mut c_void {
        self.normed_buf
    }
    pub(crate) fn logits_ptr(&self) -> *mut c_void {
        self.logits_buf
    }
    #[allow(dead_code)]
    pub(crate) fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    #[allow(dead_code)]
    pub(crate) fn codec_vocab_size(&self) -> usize {
        self.codec_vocab_size
    }
}

impl Drop for HipTalker {
    fn drop(&mut self) {
        unsafe { cubecl_hip_sys::hipStreamSynchronize(self.own_stream) };
        for ptr in &self.all_allocs {
            unsafe { cubecl_hip_sys::hipFree(*ptr) };
        }
        unsafe { cubecl_hip_sys::hipStreamDestroy(self.own_stream) };
    }
}
