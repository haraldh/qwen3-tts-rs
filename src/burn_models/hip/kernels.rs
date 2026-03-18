//! HIP kernel source code and runtime compilation for the raw code predictor.

use cubecl_hip_sys::{
    self, hipFunction_t, hipModule_t, hiprtcProgram, hiprtcResult_HIPRTC_SUCCESS, HIP_SUCCESS,
};
use std::ffi::{c_char, CString};

/// All compiled kernel function handles.
pub(crate) struct HipKernels {
    pub module: hipModule_t,
    pub gemv: hipFunction_t,
    pub rmsnorm: hipFunction_t,
    pub qk_norm_rope: hipFunction_t,
    pub kv_cache_append: hipFunction_t,
    pub attention_decode: hipFunction_t,
    pub attention_decode_long: hipFunction_t,
    pub silu_mul: hipFunction_t,
    pub add_inplace: hipFunction_t,
    pub argmax: hipFunction_t,
    pub embedding_gather_add: hipFunction_t,
}

/// HIP C++ kernel source code for all code predictor operations.
/// Compiled at runtime via hiprtc.
const KERNEL_SOURCE: &str = r#"
typedef unsigned short bf16;

__device__ __forceinline__ float bf16_to_f32(bf16 x) {
    unsigned int bits = ((unsigned int)x) << 16;
    return __uint_as_float(bits);
}

__device__ __forceinline__ bf16 f32_to_bf16(float x) {
    unsigned int bits = __float_as_uint(x);
    // Round to nearest even
    unsigned int lsb = (bits >> 16) & 1;
    unsigned int rounding_bias = 0x7fff + lsb;
    bits += rounding_bias;
    return (bf16)(bits >> 16);
}

// ============ GEMV ============
// y[j] = sum_i x[i] * W[j * K + i]  (+bias if non-null)
// Grid: N blocks, Block: 256 threads
// Each block cooperatively reduces one output element.
#define GEMV_BLOCK 256

extern "C" __global__ void gemv_bf16(
    const bf16* __restrict__ x,
    const bf16* __restrict__ W,
    bf16* __restrict__ y,
    const bf16* __restrict__ bias,
    int K, int N)
{
    __shared__ float shared[GEMV_BLOCK];
    int j = blockIdx.x;
    int tid = threadIdx.x;

    if (j >= N) return;

    float sum = 0.0f;
    const bf16* row = W + (long long)j * K;
    for (int i = tid; i < K; i += GEMV_BLOCK) {
        sum += bf16_to_f32(x[i]) * bf16_to_f32(row[i]);
    }
    shared[tid] = sum;
    __syncthreads();

    for (int s = GEMV_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float result = shared[0];
        if (bias != nullptr) {
            result += bf16_to_f32(bias[j]);
        }
        y[j] = f32_to_bf16(result);
    }
}

// ============ RMSNorm ============
// y[i] = x[i] * weight[i] * rsqrt(mean(x^2) + eps)
#define NORM_BLOCK 256

extern "C" __global__ void rmsnorm_bf16(
    const bf16* __restrict__ x,
    const bf16* __restrict__ weight,
    bf16* __restrict__ y,
    int D, float eps)
{
    __shared__ float shared[NORM_BLOCK];
    int tid = threadIdx.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += NORM_BLOCK) {
        float val = bf16_to_f32(x[i]);
        sum_sq += val * val;
    }
    shared[tid] = sum_sq;
    __syncthreads();

    for (int s = NORM_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(shared[0] / (float)D + eps);

    for (int i = tid; i < D; i += NORM_BLOCK) {
        y[i] = f32_to_bf16(bf16_to_f32(x[i]) * bf16_to_f32(weight[i]) * scale);
    }
}

// ============ QK Norm + RoPE (fused) ============
// For each head: per-head RMSNorm then RoPE rotation.
// Blocks 0..num_q_heads-1 process Q, num_q_heads..total process K.
// Block size = 128 (= head_dim).

extern "C" __global__ void qk_norm_rope_bf16(
    bf16* __restrict__ Q,
    bf16* __restrict__ K,
    const bf16* __restrict__ q_norm_weight,
    const bf16* __restrict__ k_norm_weight,
    const bf16* __restrict__ cos_table,
    const bf16* __restrict__ sin_table,
    int num_q_heads, int num_kv_heads, int head_dim, int position,
    float eps)
{
    int block_id = blockIdx.x;
    bool is_q = (block_id < num_q_heads);
    int head_idx = is_q ? block_id : (block_id - num_q_heads);

    bf16* data = is_q ? (Q + head_idx * head_dim) : (K + head_idx * head_dim);
    const bf16* norm_weight = is_q ? q_norm_weight : k_norm_weight;

    int tid = threadIdx.x;
    int half_dim = head_dim / 2;

    // Step 1: RMSNorm (reduction in shared mem)
    extern __shared__ float smem[];
    // smem layout: [0..blockDim.x) for reduction, [blockDim.x..blockDim.x+head_dim) for normed
    float* s_reduce = smem;
    float* s_normed = smem + blockDim.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = bf16_to_f32(data[i]);
        sum_sq += val * val;
    }
    s_reduce[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(s_reduce[0] / (float)head_dim + eps);

    // Write normalized values to shared memory
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_normed[i] = bf16_to_f32(data[i]) * bf16_to_f32(norm_weight[i]) * scale;
    }
    __syncthreads();

    // Step 2: RoPE rotation
    const bf16* cos_row = cos_table + position * half_dim;
    const bf16* sin_row = sin_table + position * half_dim;

    for (int i = tid; i < half_dim; i += blockDim.x) {
        float x1 = s_normed[i];
        float x2 = s_normed[i + half_dim];
        float c = bf16_to_f32(cos_row[i]);
        float s = bf16_to_f32(sin_row[i]);

        data[i] = f32_to_bf16(x1 * c - x2 * s);
        data[i + half_dim] = f32_to_bf16(x2 * c + x1 * s);
    }
}

// ============ KV Cache Append ============
// Copy K,V vectors to cache at position offset.
extern "C" __global__ void kv_cache_append_bf16(
    const bf16* __restrict__ K_new,
    const bf16* __restrict__ V_new,
    bf16* __restrict__ K_cache,
    bf16* __restrict__ V_cache,
    int num_kv_heads, int max_seq, int head_dim, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_kv_heads * head_dim;
    if (idx >= total) return;

    int h = idx / head_dim;
    int d = idx % head_dim;

    int cache_idx = h * max_seq * head_dim + offset * head_dim + d;
    K_cache[cache_idx] = K_new[idx];
    V_cache[cache_idx] = V_new[idx];
}

// ============ Attention Decode (seq_q=1) ============
// One block per query head. Handles GQA mapping internally.
// Shared memory: 256 floats for reduction + 64 floats for scores.
#define ATTN_BLOCK 128
#define ATTN_MAX_SEQ 64

extern "C" __global__ void attention_decode_bf16(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K_cache,
    const bf16* __restrict__ V_cache,
    bf16* __restrict__ output,
    int num_q_heads, int num_kv_heads, int head_dim,
    int max_seq, int seq_kv, float scale)
{
    int h = blockIdx.x;
    int kv_h = h * num_kv_heads / num_q_heads;
    int tid = threadIdx.x;

    const bf16* q = Q + h * head_dim;
    const bf16* k_base = K_cache + kv_h * max_seq * head_dim;
    const bf16* v_base = V_cache + kv_h * max_seq * head_dim;
    bf16* out = output + h * head_dim;

    __shared__ float s_scores[ATTN_MAX_SEQ];
    __shared__ float s_reduce[ATTN_BLOCK];

    // Step 1: Compute attention scores via dot products
    for (int s = 0; s < seq_kv; s++) {
        const bf16* k = k_base + s * head_dim;
        float sum = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            sum += bf16_to_f32(q[d]) * bf16_to_f32(k[d]);
        }
        s_reduce[tid] = sum;
        __syncthreads();

        for (int r = blockDim.x / 2; r > 0; r >>= 1) {
            if (tid < r) s_reduce[tid] += s_reduce[tid + r];
            __syncthreads();
        }

        if (tid == 0) s_scores[s] = s_reduce[0] * scale;
        __syncthreads();
    }

    // Step 2: Softmax (single-thread for small seq_kv)
    if (tid == 0) {
        float max_val = s_scores[0];
        for (int s = 1; s < seq_kv; s++) max_val = fmaxf(max_val, s_scores[s]);
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_kv; s++) {
            s_scores[s] = expf(s_scores[s] - max_val);
            sum_exp += s_scores[s];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int s = 0; s < seq_kv; s++) s_scores[s] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum of V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int s = 0; s < seq_kv; s++) {
            val += s_scores[s] * bf16_to_f32(v_base[s * head_dim + d]);
        }
        out[d] = f32_to_bf16(val);
    }
}

// ============ Attention Decode Long (seq_q=1, flat parallel) ============
// Optimized for long sequences (up to 4096+). Each thread independently computes
// full dot products for its assigned KV positions — no __syncthreads in the score loop.
// This eliminates ~7*seq_kv barrier syncs compared to the cooperative reduction approach.
// One block per query head. Dynamic shared memory for scores.
#define ATTN_LONG_BLOCK 128

extern "C" __global__ void attention_decode_long_bf16(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K_cache,
    const bf16* __restrict__ V_cache,
    bf16* __restrict__ output,
    int num_q_heads, int num_kv_heads, int head_dim,
    int max_seq, int seq_kv, float scale)
{
    int h = blockIdx.x;
    int kv_h = h * num_kv_heads / num_q_heads;
    int tid = threadIdx.x;

    const bf16* q = Q + h * head_dim;
    const bf16* k_base = K_cache + kv_h * max_seq * head_dim;
    const bf16* v_base = V_cache + kv_h * max_seq * head_dim;
    bf16* out = output + h * head_dim;

    // Dynamic shared memory: [0..seq_kv) = scores, [seq_kv..seq_kv+BLOCK) = reduce
    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_reduce = smem + seq_kv;

    // Step 1: Each thread computes full dot products for its assigned positions.
    // No sync needed — each thread works independently.
    for (int s = tid; s < seq_kv; s += blockDim.x) {
        const bf16* k = k_base + s * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += bf16_to_f32(q[d]) * bf16_to_f32(k[d]);
        }
        s_scores[s] = dot * scale;
    }
    __syncthreads();

    // Step 2: Softmax (thread-parallel)
    float local_max = -1e30f;
    for (int s = tid; s < seq_kv; s += blockDim.x) {
        local_max = fmaxf(local_max, s_scores[s]);
    }
    s_reduce[tid] = local_max;
    __syncthreads();
    for (int r = blockDim.x / 2; r > 0; r >>= 1) {
        if (tid < r) s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + r]);
        __syncthreads();
    }
    float max_val = s_reduce[0];

    float local_sum = 0.0f;
    for (int s = tid; s < seq_kv; s += blockDim.x) {
        float e = expf(s_scores[s] - max_val);
        s_scores[s] = e;
        local_sum += e;
    }
    s_reduce[tid] = local_sum;
    __syncthreads();
    for (int r = blockDim.x / 2; r > 0; r >>= 1) {
        if (tid < r) s_reduce[tid] += s_reduce[tid + r];
        __syncthreads();
    }
    float inv_sum = 1.0f / s_reduce[0];

    for (int s = tid; s < seq_kv; s += blockDim.x) {
        s_scores[s] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum of V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int s = 0; s < seq_kv; s++) {
            val += s_scores[s] * bf16_to_f32(v_base[s * head_dim + d]);
        }
        out[d] = f32_to_bf16(val);
    }
}

// ============ SiLU * Mul (fused) ============
extern "C" __global__ void silu_mul_bf16(
    const bf16* __restrict__ gate,
    const bf16* __restrict__ up,
    bf16* __restrict__ y,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float g = bf16_to_f32(gate[i]);
    float u = bf16_to_f32(up[i]);
    float silu_g = g / (1.0f + expf(-g));
    y[i] = f32_to_bf16(silu_g * u);
}

// ============ Add (in-place) ============
extern "C" __global__ void add_inplace_bf16(
    bf16* __restrict__ y,
    const bf16* __restrict__ x,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    y[i] = f32_to_bf16(bf16_to_f32(y[i]) + bf16_to_f32(x[i]));
}

// ============ Argmax ============
// Single block, finds index of maximum value.
extern "C" __global__ void argmax_bf16(
    const bf16* __restrict__ x,
    int* __restrict__ result,
    int N)
{
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    int tid = threadIdx.x;

    float max_val = -1e30f;
    int max_idx = 0;

    for (int i = tid; i < N; i += blockDim.x) {
        float val = bf16_to_f32(x[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    s_val[tid] = max_val;
    s_idx[tid] = max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) result[0] = s_idx[0];
}

// ============ Embedding Gather + Accumulate ============
// output[i] = table[index * dim + i]
// acc[i] += table[index * dim + i]
extern "C" __global__ void embedding_gather_add_bf16(
    const bf16* __restrict__ table,
    const int* __restrict__ index,
    bf16* __restrict__ acc,
    bf16* __restrict__ output,
    int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    int idx = index[0];
    bf16 val = table[idx * dim + i];
    output[i] = val;
    acc[i] = f32_to_bf16(bf16_to_f32(acc[i]) + bf16_to_f32(val));
}
"#;

impl HipKernels {
    /// Compile all kernels and extract function handles.
    pub(crate) fn compile() -> Result<Self, String> {
        let source = CString::new(KERNEL_SOURCE).unwrap();
        let name = CString::new("hip_code_predictor.hip").unwrap();

        // Create program
        let mut program: hiprtcProgram = std::ptr::null_mut();
        let status = unsafe {
            cubecl_hip_sys::hiprtcCreateProgram(
                &mut program,
                source.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if status != hiprtcResult_HIPRTC_SUCCESS {
            return Err(format!("hiprtcCreateProgram failed: {status}"));
        }

        // Compile
        let include_path = cubecl_hip_sys::get_hip_include_path()
            .map_err(|e| format!("No HIP include path: {e}"))?;
        let include_opt = CString::new(format!("-I{include_path}")).unwrap();
        let std_opt = CString::new("--std=c++17").unwrap();
        let opt_level = CString::new("-O3").unwrap();
        let mut options = vec![std_opt.as_ptr(), include_opt.as_ptr(), opt_level.as_ptr()];

        let status = unsafe {
            cubecl_hip_sys::hiprtcCompileProgram(
                program,
                options.len() as i32,
                options.as_mut_ptr(),
            )
        };
        if status != hiprtcResult_HIPRTC_SUCCESS {
            let log = get_compile_log(program);
            unsafe { cubecl_hip_sys::hiprtcDestroyProgram(&mut program) };
            return Err(format!("hiprtcCompileProgram failed ({status}):\n{log}"));
        }

        // Get compiled code
        let mut code_size: usize = 0;
        unsafe { cubecl_hip_sys::hiprtcGetCodeSize(program, &mut code_size) };
        let mut code = vec![0i8; code_size];
        unsafe { cubecl_hip_sys::hiprtcGetCode(program, code.as_mut_ptr() as *mut c_char) };
        unsafe { cubecl_hip_sys::hiprtcDestroyProgram(&mut program) };

        // Load module
        let mut module: hipModule_t = std::ptr::null_mut();
        let status =
            unsafe { cubecl_hip_sys::hipModuleLoadData(&mut module, code.as_ptr() as *const _) };
        if status != HIP_SUCCESS {
            return Err(format!("hipModuleLoadData failed: {status}"));
        }

        // Extract function handles
        let get_fn = |name: &str| -> Result<hipFunction_t, String> {
            let cname = CString::new(name).unwrap();
            let mut func: hipFunction_t = std::ptr::null_mut();
            let s =
                unsafe { cubecl_hip_sys::hipModuleGetFunction(&mut func, module, cname.as_ptr()) };
            if s != HIP_SUCCESS {
                return Err(format!("hipModuleGetFunction({name}) failed: {s}"));
            }
            Ok(func)
        };

        Ok(Self {
            module,
            gemv: get_fn("gemv_bf16")?,
            rmsnorm: get_fn("rmsnorm_bf16")?,
            qk_norm_rope: get_fn("qk_norm_rope_bf16")?,
            kv_cache_append: get_fn("kv_cache_append_bf16")?,
            attention_decode: get_fn("attention_decode_bf16")?,
            attention_decode_long: get_fn("attention_decode_long_bf16")?,
            silu_mul: get_fn("silu_mul_bf16")?,
            add_inplace: get_fn("add_inplace_bf16")?,
            argmax: get_fn("argmax_bf16")?,
            embedding_gather_add: get_fn("embedding_gather_add_bf16")?,
        })
    }
}

impl Drop for HipKernels {
    fn drop(&mut self) {
        unsafe {
            cubecl_hip_sys::hipModuleUnload(self.module);
        }
    }
}

fn get_compile_log(program: hiprtcProgram) -> String {
    let mut log_size: usize = 0;
    let s = unsafe { cubecl_hip_sys::hiprtcGetProgramLogSize(program, &mut log_size) };
    if s != hiprtcResult_HIPRTC_SUCCESS || log_size == 0 {
        return String::from("<no log>");
    }
    let mut log = vec![0u8; log_size];
    let s =
        unsafe { cubecl_hip_sys::hiprtcGetProgramLog(program, log.as_mut_ptr() as *mut c_char) };
    if s != hiprtcResult_HIPRTC_SUCCESS {
        return String::from("<log fetch failed>");
    }
    String::from_utf8_lossy(&log).to_string()
}
