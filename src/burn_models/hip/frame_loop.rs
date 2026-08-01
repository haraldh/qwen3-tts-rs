//! Zero-copy HIP frame loop — keeps all intermediate data on GPU.
//!
//! Eliminates CPU↔GPU round-trips between the code predictor, talker,
//! and sampling stages by orchestrating the entire per-frame generation
//! loop with direct HIP kernel launches. Only one GPU sync per frame
//! (for logits + codes readback).

use std::collections::HashSet;
use std::ffi::c_void;

use burn::prelude::*;
use cubecl_hip_sys::{hipFunction_t, hipStream_t, HIP_SUCCESS};

use super::code_predictor::{
    div_ceil, hip_check, hip_d2h, hip_h2d, ptr_of, GpuAlloc, HipCodePredictor,
};
use super::talker::HipTalker;
use crate::burn_models::sampling::{self, GenerationConfig, SamplingContext};
use crate::burn_models::talker::{codec_tokens, TalkerModel};
use crate::FrameCodes;

/// GPU-resident frame loop that keeps all intermediate data on HIP.
///
/// Eliminates 3 GPU sync barriers and ~14 CubeCL kernel launches per frame
/// by running embedding lookup, code predictor, embed+fuse, talker decode,
/// and logits readback all on a single HIP stream.
pub struct HipFrameLoop {
    stream: hipStream_t,

    // GPU buffers owned by this struct
    semantic_embed_buf: *mut c_void, // [hidden_size] BF16
    text_embeds_buf: *mut c_void,    // [max_text_len * hidden_size] BF16
    tts_pad_buf: *mut c_void,        // [hidden_size] BF16
    token_idx_buf: *mut c_void,      // [1] i32 (embedding gather index)
    codec_embed_table: *mut c_void,  // [codec_vocab_size, hidden_size] BF16
    scratch_buf: *mut c_void,        // [hidden_size] BF16 (dummy accumulator)

    // Kernel function handles (borrowed from CP's compiled module)
    add_inplace_fn: hipFunction_t,
    embedding_gather_add_fn: hipFunction_t,

    // CPU-side suppression mask
    suppression_mask: Vec<f32>, // [codec_vocab_size], 0.0 or -inf

    // Dimensions
    hidden_size: usize,
    codec_vocab_size: usize,
    max_text_len: usize,

    all_allocs: Vec<*mut c_void>,
}

unsafe impl Send for HipFrameLoop {}
unsafe impl Sync for HipFrameLoop {}

impl HipFrameLoop {
    /// Build from a Burn TalkerModel and HIP code predictor.
    ///
    /// Extracts the codec embedding table and tts_pad embedding, uploads them
    /// to GPU once. Borrows kernel function handles from the code predictor's
    /// compiled module (valid for the lifetime of the code predictor).
    pub fn from_burn<B: Backend>(
        talker: &TalkerModel<B>,
        hip_cp: &HipCodePredictor,
        device: &B::Device,
    ) -> Result<Self, String> {
        tracing::info!("Initializing HIP frame loop (zero-copy)...");
        let t0 = std::time::Instant::now();

        let mut stream: hipStream_t = std::ptr::null_mut();
        hip_check(
            unsafe { cubecl_hip_sys::hipStreamCreate(&mut stream) },
            "hipStreamCreate",
        )?;

        let hidden_size = talker.config().hidden_size;
        let codec_vocab_size = talker.config().codec_vocab_size;
        let max_text_len = 4096; // generous upper bound

        let mut ga = GpuAlloc::new();

        // Upload codec embedding table [codec_vocab_size, hidden_size] BF16
        let codec_embed_data = talker.codec_embedding.weight.val().into_data();
        let codec_embed_table = ga.upload(&codec_embed_data);

        // Upload tts_pad embedding [1, 1, hidden_size] → flatten to [hidden_size]
        let tts_pad = talker.get_tts_pad_embed(device);
        let tts_pad_data = tts_pad.into_data();
        let tts_pad_buf = ga.upload(&tts_pad_data);

        // Scratch buffers
        let semantic_embed_buf = ga.alloc(hidden_size * 2);
        let text_embeds_buf = ga.alloc(max_text_len * hidden_size * 2);
        let token_idx_buf = ga.alloc(4);
        let scratch_buf = ga.alloc(hidden_size * 2);

        // Get kernel handles from CP (valid as long as CP lives)
        let add_inplace_fn = hip_cp.kernel_add_inplace();
        let embedding_gather_add_fn = hip_cp.kernel_embedding_gather_add();

        // Build CPU suppression mask (same logic as tts::build_suppression_mask)
        let suppress_start = codec_vocab_size - 1024;
        let eos_id = codec_tokens::CODEC_EOS;
        let mut suppression_mask = vec![0.0f32; codec_vocab_size];
        for i in suppress_start..codec_vocab_size {
            if i as u32 != eos_id {
                suppression_mask[i] = f32::NEG_INFINITY;
            }
        }

        tracing::info!(
            "HIP frame loop ready in {:.0}ms ({:.1}MB GPU)",
            t0.elapsed().as_secs_f64() * 1000.0,
            ga.total_bytes as f64 / 1_048_576.0,
        );
        let all_allocs = ga.ptrs;

        Ok(Self {
            stream,
            semantic_embed_buf,
            text_embeds_buf,
            tts_pad_buf,
            token_idx_buf,
            codec_embed_table,
            scratch_buf,
            add_inplace_fn,
            embedding_gather_add_fn,
            suppression_mask,
            hidden_size,
            codec_vocab_size,
            max_text_len,
            all_allocs,
        })
    }

    /// Upload per-call data to GPU buffers.
    ///
    /// Must be called before `run_frame()` for each synthesis call.
    /// Uploads initial hidden state to the talker's input buffer and
    /// trailing text embeddings to the frame loop's text buffer.
    pub fn init_run(
        &self,
        initial_hidden_bytes: &[u8],
        trailing_text_bytes: &[u8],
        trailing_text_len: usize,
        hip_talker: &HipTalker,
    ) {
        assert!(
            trailing_text_len <= self.max_text_len,
            "trailing_text_len ({trailing_text_len}) exceeds max ({max})",
            max = self.max_text_len,
        );

        // Upload initial hidden (post-norm from Burn prefill) to talker's normed_buf.
        // The code predictor reads from normed_ptr() (matching Burn's post-norm convention).
        // After each talker step, forward_one_token writes normed_buf via final RMSNorm.
        hip_h2d(
            hip_talker.normed_ptr(),
            initial_hidden_bytes.as_ptr() as *const c_void,
            initial_hidden_bytes.len(),
        );

        // Upload trailing text embeddings
        if !trailing_text_bytes.is_empty() {
            hip_h2d(
                self.text_embeds_buf,
                trailing_text_bytes.as_ptr() as *const c_void,
                trailing_text_bytes.len(),
            );
        }
    }

    /// Run one frame of the generation loop entirely on HIP.
    ///
    /// Returns `(frame_16_codes, next_semantic_token)`.
    /// All GPU work runs on the frame loop's single stream, with one sync
    /// for logits + codes readback at the end.
    #[allow(clippy::too_many_arguments)]
    pub fn run_frame(
        &self,
        hip_cp: &HipCodePredictor,
        hip_talker: &HipTalker,
        semantic_token: u32,
        frame_idx: usize,
        offset: usize,
        trailing_text_len: usize,
        penalty_set: &HashSet<u32>,
        gen_config: &GenerationConfig,
        sampling_ctx: &mut SamplingContext,
        token_count: usize,
    ) -> (Vec<u32>, u32) {
        // 1. Codec embedding lookup: semantic_token → semantic_embed_buf
        //    Upload token ID (4 bytes, synchronous — sub-microsecond)
        let token_id_i32 = semantic_token as i32;
        hip_h2d(
            self.token_idx_buf,
            &token_id_i32 as *const i32 as *const c_void,
            4,
        );
        self.launch_embedding_gather_add(
            self.codec_embed_table,
            self.token_idx_buf,
            self.scratch_buf, // dummy accumulator
            self.semantic_embed_buf,
            self.hidden_size,
        );

        // 2. Code predictor: reads from talker.normed_buf (post-norm hidden) + semantic_embed_buf
        //    Uses normed_ptr (after final RMSNorm) to match Burn's generate_step_with_embed
        //    which returns hidden AFTER norm.forward().
        hip_cp.generate_gpu_to_gpu(
            self.stream,
            hip_talker.normed_ptr(),
            self.semantic_embed_buf,
        );

        // 3. Embed+fuse: semantic_embed_buf += acoustic_embed_sum + text
        self.launch_add_inplace(
            self.semantic_embed_buf,
            hip_cp.embed_sum_ptr(),
            self.hidden_size,
        );

        let text_ptr = if frame_idx < trailing_text_len {
            unsafe {
                (self.text_embeds_buf as *mut u8).add(frame_idx * self.hidden_size * 2)
                    as *mut c_void
            }
        } else {
            self.tts_pad_buf
        };
        self.launch_add_inplace(self.semantic_embed_buf, text_ptr, self.hidden_size);

        // 4. Talker step: semantic_embed_buf → talker.input_buf (hidden) + talker.logits_buf
        hip_talker.forward_decode_gpu_to_gpu(self.stream, self.semantic_embed_buf, offset);

        // 5. ONE sync — then read logits + codes
        unsafe { cubecl_hip_sys::hipStreamSynchronize(self.stream) };

        // Read logits (6KB BF16)
        let logits_size = self.codec_vocab_size * 2;
        let mut logits_bytes = vec![0u8; logits_size];
        hip_d2h(
            logits_bytes.as_mut_ptr() as *mut c_void,
            hip_talker.logits_ptr(),
            logits_size,
        );

        // Read acoustic codes (60 bytes)
        let num_acoustic = hip_cp.num_acoustic();
        let mut codes_i32 = vec![0i32; num_acoustic];
        hip_d2h(
            codes_i32.as_mut_ptr() as *mut c_void,
            hip_cp.codes_out_ptr(),
            num_acoustic * 4,
        );

        // Build frame: [semantic, acoustic_0..14]
        let mut frame = Vec::with_capacity(16);
        frame.push(semantic_token);
        for &c in &codes_i32 {
            frame.push(c as u32);
        }

        // 6. CPU sampling from BF16 logits
        let next_token = sampling::sample_from_bf16_logits(
            &logits_bytes,
            self.codec_vocab_size,
            &self.suppression_mask,
            penalty_set,
            gen_config,
            sampling_ctx,
            token_count,
        );

        (frame, next_token)
    }

    /// Run the complete frame generation loop.
    ///
    /// Convenience wrapper around `init_run()` + `run_frame()` loop.
    #[allow(clippy::too_many_arguments)]
    pub fn run_loop(
        &self,
        hip_cp: &HipCodePredictor,
        hip_talker: &HipTalker,
        initial_hidden_bytes: &[u8],
        trailing_text_bytes: &[u8],
        trailing_text_len: usize,
        mut semantic_token: u32,
        start_offset: usize,
        gen_config: &GenerationConfig,
        sampling_ctx: &mut SamplingContext,
    ) -> FrameCodes {
        self.init_run(
            initial_hidden_bytes,
            trailing_text_bytes,
            trailing_text_len,
            hip_talker,
        );

        let vocab_size = self.codec_vocab_size;
        let mut all_codes: FrameCodes = Vec::new();
        let mut token_count: usize = 1;
        let mut penalty_set = HashSet::new();
        if (semantic_token as usize) < vocab_size {
            penalty_set.insert(semantic_token);
        }

        #[cfg(feature = "profiling")]
        let loop_start = std::time::Instant::now();

        for frame_idx in 0..gen_config.max_new_tokens {
            if gen_config.eos_token_id == Some(semantic_token) {
                break;
            }

            let (frame, next_token) = self.run_frame(
                hip_cp,
                hip_talker,
                semantic_token,
                frame_idx,
                start_offset + frame_idx,
                trailing_text_len,
                &penalty_set,
                gen_config,
                sampling_ctx,
                token_count,
            );
            all_codes.push(frame);

            semantic_token = next_token;
            if (semantic_token as usize) < vocab_size {
                penalty_set.insert(semantic_token);
            }
            token_count += 1;
        }

        #[cfg(feature = "profiling")]
        {
            let loop_elapsed = loop_start.elapsed();
            let frames = all_codes.len();
            eprintln!(
                "HIP frame loop: {} frames in {:.1?} ({:.1}ms/frame)",
                frames,
                loop_elapsed,
                loop_elapsed.as_secs_f64() * 1000.0 / frames.max(1) as f64
            );
        }

        all_codes
    }

    // ==================== Kernel launch wrappers ====================

    fn launch_add_inplace(&self, y: *mut c_void, x: *mut c_void, n: usize) {
        let blocks = div_ceil(n, 256) as u32;
        let mut p_y = y;
        let mut p_x = x;
        let mut n_i32 = n as i32;
        let mut args: [*mut c_void; 3] = [ptr_of(&mut p_y), ptr_of(&mut p_x), ptr_of(&mut n_i32)];
        self.launch_kernel(self.add_inplace_fn, blocks, 1, 1, 256, 1, 1, 0, &mut args);
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
            self.embedding_gather_add_fn,
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
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        debug_assert_eq!(status, HIP_SUCCESS, "Kernel launch failed: {status}");
    }
}

impl Drop for HipFrameLoop {
    fn drop(&mut self) {
        unsafe { cubecl_hip_sys::hipStreamSynchronize(self.stream) };
        for ptr in &self.all_allocs {
            unsafe { cubecl_hip_sys::hipFree(*ptr) };
        }
        unsafe { cubecl_hip_sys::hipStreamDestroy(self.stream) };
    }
}
