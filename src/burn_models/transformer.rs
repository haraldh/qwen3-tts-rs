//! Shared transformer building blocks for Qwen3-TTS (Burn)
//!
//! Contains `RotaryEmbedding`, `MRoPE`, `Attention`, `MLP`, and `DecoderLayer`
//! — used by both `TalkerModel` and `CodePredictor`.

use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::DType;

use super::kv_cache::KVCache;

/// Create a causal attention mask as a boolean tensor.
///
/// Returns a `[1, 1, seq_len, offset + seq_len]` Bool tensor where position `(i, j)`
/// is `true` if `j > offset + i` (masked) and `false` (allowed).
///
/// `true` = masked (set to -inf before softmax).
pub fn create_causal_mask<B: Backend>(
    seq_len: usize,
    offset: usize,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    let total_len = offset + seq_len;

    // Row indices [offset, offset+1, ..., offset+seq_len-1] as [1,1,seq_len,1]
    let rows: Vec<f32> = (0..seq_len).map(|i| (i + offset) as f32).collect();
    let rows = Tensor::<B, 1>::from_floats(rows.as_slice(), device).reshape([1, 1, seq_len, 1]);

    // Col indices [0, 1, ..., total_len-1] as [1,1,1,total_len]
    let cols: Vec<f32> = (0..total_len).map(|j| j as f32).collect();
    let cols = Tensor::<B, 1>::from_floats(cols.as_slice(), device).reshape([1, 1, 1, total_len]);

    // mask[i][j] = col > row  =>  true means "masked"
    cols.greater(rows)
}

/// Apply RoPE rotation to a tensor.
///
/// `x` has shape `[batch, heads, seq_len, head_dim]`.
/// `cos` and `sin` have shape `[seq_len, head_dim/2]`.
fn apply_rope_rotation<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [_b, _h, _seq, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.narrow(3, half, half);

    // Broadcast cos/sin from [seq_len, half_dim] to [1, 1, seq_len, half_dim]
    let cos = cos.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
    let sin = sin.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

    // Standard RoPE: [x1*cos - x2*sin, x2*cos + x1*sin]
    let part1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let part2 = x2 * cos + x1 * sin;
    Tensor::cat(vec![part1, part2], 3)
}

/// Rotary position embedding (standard RoPE).
///
/// Pre-computes cos/sin tables up to `max_seq_len`.
pub struct RotaryEmbedding<B: Backend> {
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn new(dim: usize, max_seq_len: usize, theta: f64, device: &B::Device) -> Self {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let half_dim = inv_freq.len();

        // Compute cos/sin in F32, then cast to backend dtype for storage.
        // Matches PyTorch: Qwen2RotaryEmbedding computes in F32, returns .to(dtype=x.dtype).
        let native_dtype = Tensor::<B, 1>::zeros([1], device).dtype();
        let inv_freq = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device).cast(DType::F32);
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::<B, 1>::from_floats(positions.as_slice(), device)
            .cast(DType::F32)
            .unsqueeze_dim::<2>(1);

        // [max_seq_len, 1] @ [1, half_dim] -> [max_seq_len, half_dim]
        let freqs = positions.matmul(inv_freq.unsqueeze_dim::<2>(0));
        let cos = freqs.clone().cos().cast(native_dtype);
        let sin = freqs.sin().cast(native_dtype);

        // Verify shapes
        debug_assert_eq!(cos.dims(), [max_seq_len, half_dim]);

        Self { cos, sin }
    }

    pub fn apply(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let seq_len = q.dims()[2];
        let cos = self.cos.clone().narrow(0, offset, seq_len);
        let sin = self.sin.clone().narrow(0, offset, seq_len);

        let q_rot = apply_rope_rotation(q, cos.clone(), sin.clone());
        let k_rot = apply_rope_rotation(k, cos, sin);
        (q_rot, k_rot)
    }
}

/// Multimodal Rotary Embedding (MRoPE) for 3D positions.
///
/// For TTS, all 3 position dimensions use the same value, so this is
/// equivalent to standard RoPE but preserves the frequency interleaving.
pub struct MRoPE<B: Backend> {
    inv_freq: Tensor<B, 1>,
    device: B::Device,
}

impl<B: Backend> MRoPE<B> {
    pub fn new(dim: usize, theta: f64, _mrope_section: [usize; 3], device: &B::Device) -> Self {
        // Store inv_freq in F32 to preserve precision for on-the-fly cos/sin computation.
        // Matches PyTorch's Qwen2RotaryEmbedding which keeps inv_freq in F32.
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device).cast(DType::F32);

        Self {
            inv_freq,
            device: device.clone(),
        }
    }

    pub fn apply(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        offset: usize,
        seq_len: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let positions: Vec<f32> = (offset..offset + seq_len).map(|i| i as f32).collect();
        let pos = Tensor::<B, 1>::from_floats(positions.as_slice(), &self.device).cast(DType::F32);

        let pos_col = pos.unsqueeze_dim::<2>(1); // [seq_len, 1]
        let inv_freq_row = self.inv_freq.clone().unsqueeze_dim::<2>(0); // [1, half_dim]
        let freqs = pos_col.matmul(inv_freq_row); // [seq_len, half_dim]

        // Compute cos/sin in F32, cast to Q/K dtype (matching PyTorch)
        let q_dtype = q.dtype();
        let cos = freqs.clone().cos().cast(q_dtype);
        let sin = freqs.sin().cast(q_dtype);

        let q_rot = apply_rope_rotation(q, cos.clone(), sin.clone());
        let k_rot = apply_rope_rotation(k, cos, sin);
        (q_rot, k_rot)
    }
}

/// Either standard RoPE or MRoPE (multimodal).
pub enum RoPEType<B: Backend> {
    Standard(RotaryEmbedding<B>),
    Multimodal(MRoPE<B>),
}

impl<B: Backend> RoPEType<B> {
    pub fn apply(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        match self {
            RoPEType::Standard(rope) => rope.apply(q, k, offset),
            RoPEType::Multimodal(mrope) => {
                let seq_len = q.dims()[2];
                mrope.apply(q, k, offset, seq_len)
            }
        }
    }
}

/// Attention configuration (extracted from model config).
#[derive(Config, Debug)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
}

/// Multi-head attention with grouped-query attention and QK normalization.
///
/// Uses manual matmul-based SDPA (flash attention has BF16 bugs on CubeCL).
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub(crate) q_proj: Linear<B>,
    pub(crate) k_proj: Linear<B>,
    pub(crate) v_proj: Linear<B>,
    pub(crate) o_proj: Linear<B>,
    pub(crate) q_norm: RmsNorm<B>,
    pub(crate) k_norm: RmsNorm<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        let q_proj = LinearConfig::new(self.hidden_size, self.num_heads * self.head_dim)
            .with_bias(false)
            .init(device);
        let k_proj = LinearConfig::new(self.hidden_size, self.num_kv_heads * self.head_dim)
            .with_bias(false)
            .init(device);
        let v_proj = LinearConfig::new(self.hidden_size, self.num_kv_heads * self.head_dim)
            .with_bias(false)
            .init(device);
        let o_proj = LinearConfig::new(self.num_heads * self.head_dim, self.hidden_size)
            .with_bias(false)
            .init(device);

        let q_norm = RmsNormConfig::new(self.head_dim)
            .with_epsilon(self.rms_norm_eps)
            .init(device);
        let k_norm = RmsNormConfig::new(self.head_dim)
            .with_epsilon(self.rms_norm_eps)
            .init(device);

        Attention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
        }
    }
}

impl<B: Backend> Attention<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        rope: &RoPEType<B>,
        is_causal: bool,
        kv_cache: Option<&mut KVCache<B>>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = hidden_states.dims();

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        // Reshape to [batch, seq, heads, head_dim] for QK norm
        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);

        // Apply QK normalization (per-head RMSNorm)
        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Apply rotary embeddings
        let (q, k) = rope.apply(q, k, offset);

        // Update KV cache
        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(k, v)
        } else {
            (k, v)
        };

        // Repeat KV heads for GQA
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        // Use burn's attention() which dispatches to flash attention on CubeCL backends.
        let options = burn::tensor::ops::AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal,
        };
        let attn_output = burn::tensor::module::attention(q, k, v, None, None, options);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output =
            attn_output
                .swap_dims(1, 2)
                .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn_output)
    }

    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return x;
        }

        let [batch, num_kv_heads, seq_len, head_dim] = x.dims();
        x.unsqueeze_dim::<5>(2)
            .expand([batch, num_kv_heads, n_rep, seq_len, head_dim])
            .reshape([batch, num_kv_heads * n_rep, seq_len, head_dim])
    }
}

/// MLP configuration.
#[derive(Config, Debug)]
pub struct MLPConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

/// MLP block with SwiGLU activation.
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub(crate) gate_proj: Linear<B>,
    pub(crate) up_proj: Linear<B>,
    pub(crate) down_proj: Linear<B>,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        MLP {
            gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = burn::tensor::activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Decoder layer configuration.
#[derive(Config, Debug)]
pub struct DecoderLayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
}

/// Transformer decoder layer.
#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    pub(crate) self_attn: Attention<B>,
    pub(crate) mlp: MLP<B>,
    pub(crate) input_layernorm: RmsNorm<B>,
    pub(crate) post_attention_layernorm: RmsNorm<B>,
}

impl DecoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderLayer<B> {
        let attn_config = AttentionConfig::new(
            self.hidden_size,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rms_norm_eps,
        );
        let mlp_config = MLPConfig::new(self.hidden_size, self.intermediate_size);

        DecoderLayer {
            self_attn: attn_config.init(device),
            mlp: mlp_config.init(device),
            input_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }
}

impl<B: Backend> DecoderLayer<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        rope: &RoPEType<B>,
        is_causal: bool,
        kv_cache: Option<&mut KVCache<B>>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let normed = self.input_layernorm.forward(hidden_states);
        let attn_output = self
            .self_attn
            .forward(normed, rope, is_causal, kv_cache, offset);
        let hidden_states = residual + attn_output;

        let residual = hidden_states.clone();
        let normed = self.post_attention_layernorm.forward(hidden_states);
        let mlp_output = self.mlp.forward(normed);
        residual + mlp_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_rotary_embedding_creation() {
        let device = Default::default();
        let rope = RotaryEmbedding::<B>::new(64, 512, 10000.0, &device);
        assert_eq!(rope.cos.dims(), [512, 32]); // [max_seq, dim/2]
        assert_eq!(rope.sin.dims(), [512, 32]);
    }

    #[test]
    fn test_rotary_embedding_apply() {
        let device = Default::default();
        let rope = RotaryEmbedding::<B>::new(16, 512, 10000.0, &device);

        let q = Tensor::<B, 4>::zeros([2, 4, 10, 16], &device);
        let k = Tensor::<B, 4>::zeros([2, 4, 10, 16], &device);

        let (q_rot, k_rot) = rope.apply(q.clone(), k.clone(), 0);
        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());
    }

    #[test]
    fn test_rotary_embedding_with_offset() {
        let device = Default::default();
        let rope = RotaryEmbedding::<B>::new(16, 512, 10000.0, &device);

        let q = Tensor::<B, 4>::zeros([1, 2, 5, 16], &device);
        let k = Tensor::<B, 4>::zeros([1, 2, 5, 16], &device);

        let (q_rot, k_rot) = rope.apply(q, k, 100);
        assert_eq!(q_rot.dims(), [1, 2, 5, 16]);
        assert_eq!(k_rot.dims(), [1, 2, 5, 16]);
    }

    #[test]
    fn test_causal_mask() {
        let device = Default::default();
        let mask = create_causal_mask::<B>(3, 0, &device);
        assert_eq!(mask.dims(), [1, 1, 3, 3]);
        // Verify mask values: true = masked (future positions)
        let data: Vec<bool> = mask.reshape([9]).into_data().to_vec().unwrap();
        // Row 0: [false, true, true]   (pos 0 can only see pos 0)
        // Row 1: [false, false, true]  (pos 1 can see 0,1)
        // Row 2: [false, false, false] (pos 2 can see all)
        assert_eq!(
            data,
            vec![false, true, true, false, false, true, false, false, false]
        );
    }

    #[test]
    fn test_mlp() {
        let device = Default::default();
        let mlp = MLPConfig::new(64, 128).init::<B>(&device);
        let input = Tensor::<B, 3>::zeros([2, 10, 64], &device);
        let output = mlp.forward(input);
        assert_eq!(output.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_attention_forward() {
        let device = Default::default();
        let attn = AttentionConfig::new(64, 4, 2, 16, 1e-6).init::<B>(&device);
        let rope = RoPEType::Standard(RotaryEmbedding::new(16, 512, 10000.0, &device));

        let input = Tensor::<B, 3>::zeros([1, 10, 64], &device);
        let output = attn.forward(input, &rope, true, None, 0);
        assert_eq!(output.dims(), [1, 10, 64]);
    }

    #[test]
    fn test_attention_with_cache() {
        let device = Default::default();
        let attn = AttentionConfig::new(64, 4, 2, 16, 1e-6).init::<B>(&device);
        let rope = RoPEType::Standard(RotaryEmbedding::new(16, 512, 10000.0, &device));
        let mut cache = KVCache::new(1, 2, 64, 16, &device);

        let input1 = Tensor::<B, 3>::zeros([1, 5, 64], &device);
        let _out1 = attn.forward(input1, &rope, true, Some(&mut cache), 0);

        let input2 = Tensor::<B, 3>::zeros([1, 3, 64], &device);
        let out2 = attn.forward(input2, &rope, false, Some(&mut cache), 5);
        assert_eq!(out2.dims(), [1, 3, 64]);
    }

    #[test]
    fn test_decoder_layer() {
        let device = Default::default();
        let layer = DecoderLayerConfig::new(64, 128, 4, 2, 16, 1e-6).init::<B>(&device);
        let rope = RoPEType::Standard(RotaryEmbedding::new(16, 512, 10000.0, &device));
        let mut cache = KVCache::new(1, 2, 64, 16, &device);

        let input = Tensor::<B, 3>::zeros([1, 8, 64], &device);
        let output = layer.forward(input, &rope, true, Some(&mut cache), 0);
        assert_eq!(output.dims(), [1, 8, 64]);
    }

    #[test]
    fn test_repeat_kv_no_repeat() {
        let device = Default::default();
        let attn = AttentionConfig::new(64, 4, 4, 16, 1e-6).init::<B>(&device);

        let x = Tensor::<B, 4>::zeros([1, 4, 10, 16], &device);
        let repeated = attn.repeat_kv(x.clone());
        assert_eq!(repeated.dims(), x.dims());
    }

    #[test]
    fn test_repeat_kv_with_repeat() {
        let device = Default::default();
        let attn = AttentionConfig::new(128, 8, 2, 16, 1e-6).init::<B>(&device);

        let x = Tensor::<B, 4>::zeros([1, 2, 10, 16], &device);
        let repeated = attn.repeat_kv(x);
        assert_eq!(repeated.dims(), [1, 8, 10, 16]);
    }
}
