//! 12Hz Audio Decoder for Qwen3-TTS (Burn)
//!
//! Full decoder that converts discrete codec tokens to audio waveforms.
//! Uses the Mimi-based architecture with BigVGAN-style upsampling.
//!
//! Components:
//! - CausalConv1d: Conv1d with left-only zero-padding
//! - CausalTransConv1d: ConvTranspose1d with right trimming
//! - SnakeBeta: x + (1/β) * sin²(α * x) activation
//! - ConvNeXtBlock: Depthwise conv + LayerNorm + pointwise convs + GELU
//! - ResidualUnit: SnakeBeta + dilated conv + SnakeBeta + 1×1 conv + residual
//! - DecoderBlock: BigVGAN-style upsample + 3 residual units
//! - Decoder12Hz: Full decoder pipeline

use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, RmsNorm,
    RmsNormConfig,
};
use burn::prelude::*;

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for the 12Hz decoder
#[derive(Debug, Clone)]
pub struct Decoder12HzConfig {
    pub codebook_dim: usize,
    pub latent_dim: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_quantizers: usize,
    pub codebook_size: usize,
    pub upsampling_ratios: Vec<usize>,
    pub decoder_dim: usize,
    pub upsample_rates: Vec<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub layer_scale: f64,
}

impl Default for Decoder12HzConfig {
    fn default() -> Self {
        Self {
            codebook_dim: 512,
            latent_dim: 1024,
            hidden_size: 512,
            num_layers: 8,
            num_heads: 16,
            head_dim: 64,
            intermediate_size: 1024,
            num_quantizers: 16,
            codebook_size: 2048,
            upsampling_ratios: vec![2, 2],
            decoder_dim: 1536,
            upsample_rates: vec![8, 5, 4, 3],
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            layer_scale: 0.01,
        }
    }
}

// ── Causal Conv1d ───────────────────────────────────────────────────────

/// Causal 1D convolution with left-only zero-padding.
#[derive(Module, Debug)]
pub struct CausalConv1d<B: Backend> {
    pub(crate) conv: Conv1d<B>,
    #[module(skip)]
    causal_padding: usize,
}

impl<B: Backend> CausalConv1d<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        groups: usize,
        device: &B::Device,
    ) -> Self {
        let conv = Conv1dConfig::new(in_channels, out_channels, kernel_size)
            .with_dilation(dilation)
            .with_groups(groups)
            .with_bias(true)
            .with_padding(burn::nn::PaddingConfig1d::Explicit(0, 0))
            .init(device);
        let causal_padding = dilation * (kernel_size - 1);
        Self {
            conv,
            causal_padding,
        }
    }

    /// Create a causal conv with stride (for encoder downsampling).
    pub fn new_strided(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        device: &B::Device,
    ) -> Self {
        let conv = Conv1dConfig::new(in_channels, out_channels, kernel_size)
            .with_stride(stride)
            .with_dilation(dilation)
            .with_groups(groups)
            .with_bias(true)
            .with_padding(burn::nn::PaddingConfig1d::Explicit(0, 0))
            .init(device);
        let causal_padding = dilation * (kernel_size - 1);
        Self {
            conv,
            causal_padding,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x_padded = if self.causal_padding > 0 {
            let [batch, channels, _seq_len] = x.dims();
            let pad = Tensor::<B, 3>::zeros([batch, channels, self.causal_padding], &x.device());
            Tensor::cat(vec![pad, x], 2)
        } else {
            x
        };
        self.conv.forward(x_padded)
    }
}

// ── Causal Transposed Conv1d ────────────────────────────────────────────

/// Causal transposed 1D convolution with right trimming.
#[derive(Module, Debug)]
pub struct CausalTransConv1d<B: Backend> {
    pub(crate) conv: ConvTranspose1d<B>,
    #[module(skip)]
    right_trim: usize,
}

impl<B: Backend> CausalTransConv1d<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        device: &B::Device,
    ) -> Self {
        let conv = ConvTranspose1dConfig::new([in_channels, out_channels], kernel_size)
            .with_stride(stride)
            .with_bias(true)
            .init(device);
        let right_trim = kernel_size.saturating_sub(stride);
        Self { conv, right_trim }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = self.conv.forward(x);
        if self.right_trim > 0 {
            let out_len = out.dims()[2];
            let end = out_len.saturating_sub(self.right_trim);
            out.narrow(2, 0, end)
        } else {
            out
        }
    }
}

// ── SnakeBeta Activation ────────────────────────────────────────────────

/// SnakeBeta activation: x + (1/β) * sin²(α * x)
#[derive(Module, Debug)]
pub struct SnakeBeta<B: Backend> {
    pub(crate) alpha: Tensor<B, 1>,
    pub(crate) beta: Tensor<B, 1>,
}

impl<B: Backend> SnakeBeta<B> {
    pub fn new(channels: usize, device: &B::Device) -> Self {
        // Initialize with zeros (exp(0) = 1)
        let alpha = Tensor::zeros([channels], device);
        let beta = Tensor::zeros([channels], device);
        Self { alpha, beta }
    }

    /// x + (1/β) * sin²(α * x)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Reshape for broadcasting: [C] -> [1, C, 1]
        let alpha = self
            .alpha
            .clone()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(2);
        let beta = self
            .beta
            .clone()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(2);

        // Exponentiate parameters
        let alpha = alpha.exp();
        let beta = beta.exp();

        // sin²(α * x)
        let scaled_x = x.clone() * alpha;
        let sin_term = scaled_x.sin().powf_scalar(2.0);

        // 1/(β + ε)
        let inv_beta = (beta + 1e-9).recip();

        // x + (1/β) * sin²(α * x)
        x + sin_term * inv_beta
    }
}

// ── ConvNeXt Block ──────────────────────────────────────────────────────

/// ConvNeXt block: depthwise conv + LayerNorm + pointwise convs + GELU + residual.
#[derive(Module, Debug)]
pub struct ConvNeXtBlock<B: Backend> {
    pub(crate) dwconv: CausalConv1d<B>,
    pub(crate) norm: LayerNorm<B>,
    pub(crate) pwconv1: Linear<B>,
    pub(crate) pwconv2: Linear<B>,
    pub(crate) gamma: Tensor<B, 1>,
}

impl<B: Backend> ConvNeXtBlock<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        // Depthwise causal conv with groups=dim
        let dwconv = CausalConv1d::new(dim, dim, 7, 1, dim, device);
        let norm = LayerNormConfig::new(dim).with_epsilon(1e-6).init(device);
        let pwconv1 = LinearConfig::new(dim, 4 * dim).with_bias(true).init(device);
        let pwconv2 = LinearConfig::new(4 * dim, dim).with_bias(true).init(device);
        let gamma = Tensor::ones([dim], device);
        Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();

        // Depthwise causal conv
        let hidden = self.dwconv.forward(x);

        // Transpose: [B, C, T] -> [B, T, C]
        let hidden = hidden.swap_dims(1, 2);

        // LayerNorm
        let hidden = self.norm.forward(hidden);

        // Pointwise conv 1 (expansion) + GELU
        let hidden = self.pwconv1.forward(hidden);
        let hidden = burn::tensor::activation::gelu(hidden);

        // Pointwise conv 2 (projection)
        let hidden = self.pwconv2.forward(hidden);

        // Gamma scaling: [dim] -> [1, 1, dim] for broadcasting with [B, T, dim]
        let gamma = self.gamma.clone().unsqueeze::<2>().unsqueeze::<3>();
        let hidden = hidden * gamma;

        // Transpose back: [B, T, C] -> [B, C, T]
        let hidden = hidden.swap_dims(1, 2);

        // Residual
        residual + hidden
    }
}

// ── Residual Unit ───────────────────────────────────────────────────────

/// Residual unit: SnakeBeta + dilated conv + SnakeBeta + 1×1 conv + residual.
#[derive(Module, Debug)]
pub struct ResidualUnit<B: Backend> {
    pub(crate) act1: SnakeBeta<B>,
    pub(crate) conv1: CausalConv1d<B>,
    pub(crate) act2: SnakeBeta<B>,
    pub(crate) conv2: CausalConv1d<B>,
}

impl<B: Backend> ResidualUnit<B> {
    pub fn new(dim: usize, dilation: usize, device: &B::Device) -> Self {
        Self {
            act1: SnakeBeta::new(dim, device),
            conv1: CausalConv1d::new(dim, dim, 7, dilation, 1, device),
            act2: SnakeBeta::new(dim, device),
            conv2: CausalConv1d::new(dim, dim, 1, 1, 1, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let hidden = self.act1.forward(x);
        let hidden = self.conv1.forward(hidden);
        let hidden = self.act2.forward(hidden);
        let hidden = self.conv2.forward(hidden);
        hidden + residual
    }
}

// ── Decoder Block ───────────────────────────────────────────────────────

/// BigVGAN-style decoder block: SnakeBeta + upsample + 3 residual units.
#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    pub(crate) snake: SnakeBeta<B>,
    pub(crate) upsample: CausalTransConv1d<B>,
    pub(crate) res1: ResidualUnit<B>,
    pub(crate) res2: ResidualUnit<B>,
    pub(crate) res3: ResidualUnit<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        upsample_rate: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            snake: SnakeBeta::new(in_channels, device),
            upsample: CausalTransConv1d::new(
                in_channels,
                out_channels,
                upsample_rate * 2,
                upsample_rate,
                device,
            ),
            res1: ResidualUnit::new(out_channels, 1, device),
            res2: ResidualUnit::new(out_channels, 3, device),
            res3: ResidualUnit::new(out_channels, 9, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.snake.forward(x);
        let hidden = self.upsample.forward(hidden);
        let hidden = self.res1.forward(hidden);
        let hidden = self.res2.forward(hidden);
        self.res3.forward(hidden)
    }
}

// ── Upsample Stage ──────────────────────────────────────────────────────

/// Upsample stage: CausalTransConv + ConvNeXtBlock.
#[derive(Module, Debug)]
pub struct UpsampleStage<B: Backend> {
    pub(crate) trans_conv: CausalTransConv1d<B>,
    pub(crate) convnext: ConvNeXtBlock<B>,
}

impl<B: Backend> UpsampleStage<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &B::Device) -> Self {
        let kernel_size = stride * 2;
        Self {
            trans_conv: CausalTransConv1d::new(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                device,
            ),
            convnext: ConvNeXtBlock::new(out_channels, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.trans_conv.forward(x);
        self.convnext.forward(hidden)
    }
}

// ── Decoder Transformer Layer ───────────────────────────────────────────

/// Internal transformer layer for the decoder (different from talker/code_predictor:
/// has layer scales, no QK normalization).
#[derive(Module, Debug)]
pub struct DecoderTransformerLayer<B: Backend> {
    pub(crate) input_layernorm: RmsNorm<B>,
    pub(crate) q_proj: Linear<B>,
    pub(crate) k_proj: Linear<B>,
    pub(crate) v_proj: Linear<B>,
    pub(crate) o_proj: Linear<B>,
    pub(crate) attn_layer_scale: Tensor<B, 1>,
    pub(crate) post_attention_layernorm: RmsNorm<B>,
    pub(crate) gate_proj: Linear<B>,
    pub(crate) up_proj: Linear<B>,
    pub(crate) down_proj: Linear<B>,
    pub(crate) mlp_layer_scale: Tensor<B, 1>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    scale: f64,
}

impl<B: Backend> DecoderTransformerLayer<B> {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
        layer_scale_init: f64,
        device: &B::Device,
    ) -> Self {
        let input_layernorm = RmsNormConfig::new(hidden_size)
            .with_epsilon(rms_norm_eps)
            .init(device);
        let q_proj = LinearConfig::new(hidden_size, num_heads * head_dim)
            .with_bias(false)
            .init(device);
        let k_proj = LinearConfig::new(hidden_size, num_heads * head_dim)
            .with_bias(false)
            .init(device);
        let v_proj = LinearConfig::new(hidden_size, num_heads * head_dim)
            .with_bias(false)
            .init(device);
        let o_proj = LinearConfig::new(num_heads * head_dim, hidden_size)
            .with_bias(false)
            .init(device);
        let attn_layer_scale = Tensor::full([hidden_size], layer_scale_init as f32, device);
        let post_attention_layernorm = RmsNormConfig::new(hidden_size)
            .with_epsilon(rms_norm_eps)
            .init(device);
        let gate_proj = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(false)
            .init(device);
        let up_proj = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(false)
            .init(device);
        let down_proj = LinearConfig::new(intermediate_size, hidden_size)
            .with_bias(false)
            .init(device);
        let mlp_layer_scale = Tensor::full([hidden_size], layer_scale_init as f32, device);

        Self {
            input_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            attn_layer_scale,
            post_attention_layernorm,
            gate_proj,
            up_proj,
            down_proj,
            mlp_layer_scale,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        mask: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = hidden_states.dims();

        // RMS Norm
        let normed = self.input_layernorm.forward(hidden_states.clone());

        // Self attention
        let q = self.q_proj.forward(normed.clone());
        let k = self.k_proj.forward(normed.clone());
        let v = self.v_proj.forward(normed);

        // Reshape for multi-head: [B, T, H*D] -> [B, H, T, D]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE (interleaved style for codec decoder)
        let q = self.apply_rope(q, cos.clone(), sin.clone());
        let k = self.apply_rope(k, cos, sin);

        // Scaled dot-product attention
        let attn = q.matmul(k.swap_dims(2, 3)) * self.scale;
        let attn = attn + mask;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let attn_out = attn.matmul(v);

        // Reshape back: [B, H, T, D] -> [B, T, H*D]
        let attn_out =
            attn_out
                .swap_dims(1, 2)
                .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        // Output projection + layer scale + residual
        let attn_out = self.o_proj.forward(attn_out);
        let attn_scale = self
            .attn_layer_scale
            .clone()
            .unsqueeze::<2>()
            .unsqueeze::<3>();
        let attn_out = attn_out * attn_scale;
        let hidden_states = hidden_states + attn_out;

        // MLP
        let normed = self.post_attention_layernorm.forward(hidden_states.clone());
        let gate = burn::tensor::activation::silu(self.gate_proj.forward(normed.clone()));
        let up = self.up_proj.forward(normed);
        let mlp_out = self.down_proj.forward(gate * up);

        // Layer scale + residual
        let mlp_scale = self
            .mlp_layer_scale
            .clone()
            .unsqueeze::<2>()
            .unsqueeze::<3>();
        let mlp_out = mlp_out * mlp_scale;
        hidden_states + mlp_out
    }

    /// Apply rotary embedding (interleaved cos/sin style).
    fn apply_rope(&self, x: Tensor<B, 4>, cos: Tensor<B, 2>, sin: Tensor<B, 2>) -> Tensor<B, 4> {
        let half = self.head_dim / 2;
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.narrow(3, half, half);

        // [seq, half] -> [1, 1, seq, half]
        let cos = cos.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        let sin = sin.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

        // Interleaved RoPE: [-x2 * sin + x1 * cos, x1 * sin + x2 * cos]
        let neg_x2 = x2.clone().neg();
        let part1 = x1.clone() * cos.clone() + neg_x2 * sin.clone();
        let part2 = x1 * sin + x2 * cos;
        Tensor::cat(vec![part1, part2], 3)
    }
}

// ── Full Decoder ────────────────────────────────────────────────────────

/// Full 12Hz decoder.
#[derive(Module, Debug)]
pub struct Decoder12Hz<B: Backend> {
    // Config fields stored individually (Burn's Module derive needs primitive types)
    #[module(skip)]
    codebook_size: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    rope_theta: f64,
    #[module(skip)]
    total_upsample_factor: usize,
    // Codebook embeddings
    pub(crate) first_codebook: Embedding<B>,
    pub(crate) rest_codebooks: Vec<Embedding<B>>,
    // Output projections (1×1 conv implemented as Linear)
    pub(crate) first_output_proj: Linear<B>,
    pub(crate) rest_output_proj: Linear<B>,
    // Pre-conv
    pub(crate) pre_conv: CausalConv1d<B>,
    // Transformer
    pub(crate) input_proj: Linear<B>,
    pub(crate) transformer_layers: Vec<DecoderTransformerLayer<B>>,
    pub(crate) final_norm: RmsNorm<B>,
    pub(crate) output_proj: Linear<B>,
    // Upsample stages
    pub(crate) upsample_stages: Vec<UpsampleStage<B>>,
    // Decoder
    pub(crate) decoder_init_conv: CausalConv1d<B>,
    pub(crate) decoder_blocks: Vec<DecoderBlock<B>>,
    pub(crate) final_snake: SnakeBeta<B>,
    pub(crate) final_conv: CausalConv1d<B>,
}

impl<B: Backend> Decoder12Hz<B> {
    /// Initialize decoder from config (with random weights — use weight loading to fill).
    pub fn init(config: Decoder12HzConfig, device: &B::Device) -> Self {
        // Codebook embeddings
        let codebook_embed_dim = 256; // Fixed in the architecture
        let first_codebook =
            EmbeddingConfig::new(config.codebook_size, codebook_embed_dim).init(device);
        let rest_codebooks = (0..15)
            .map(|_| EmbeddingConfig::new(config.codebook_size, codebook_embed_dim).init(device))
            .collect();

        // Output projections
        let first_output_proj = LinearConfig::new(codebook_embed_dim, config.codebook_dim)
            .with_bias(false)
            .init(device);
        let rest_output_proj = LinearConfig::new(codebook_embed_dim, config.codebook_dim)
            .with_bias(false)
            .init(device);

        // Pre-conv: [codebook_dim -> latent_dim, kernel=3]
        let pre_conv = CausalConv1d::new(config.codebook_dim, config.latent_dim, 3, 1, 1, device);

        // Transformer
        let input_proj = LinearConfig::new(config.latent_dim, config.hidden_size)
            .with_bias(true)
            .init(device);
        let transformer_layers = (0..config.num_layers)
            .map(|_| {
                DecoderTransformerLayer::new(
                    config.hidden_size,
                    config.num_heads,
                    config.head_dim,
                    config.intermediate_size,
                    config.rms_norm_eps,
                    config.layer_scale,
                    device,
                )
            })
            .collect();
        let final_norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);
        let output_proj = LinearConfig::new(config.hidden_size, config.latent_dim)
            .with_bias(true)
            .init(device);

        // Upsample stages
        let upsample_dim = config.latent_dim;
        let upsample_stages = config
            .upsampling_ratios
            .iter()
            .map(|&ratio| UpsampleStage::new(upsample_dim, upsample_dim, ratio, device))
            .collect();

        // Decoder blocks
        let decoder_init_conv =
            CausalConv1d::new(upsample_dim, config.decoder_dim, 7, 1, 1, device);

        let mut cur_dim = config.decoder_dim;
        let decoder_blocks = config
            .upsample_rates
            .iter()
            .map(|&rate| {
                let out_dim = cur_dim / 2;
                let block = DecoderBlock::new(cur_dim, out_dim, rate, device);
                cur_dim = out_dim;
                block
            })
            .collect();

        let final_snake = SnakeBeta::new(cur_dim, device);
        let final_conv = CausalConv1d::new(cur_dim, 1, 7, 1, 1, device);

        let pre_upsample: usize = config.upsampling_ratios.iter().product();
        let decoder_upsample: usize = config.upsample_rates.iter().product();
        let total_upsample_factor = pre_upsample * decoder_upsample;

        Self {
            codebook_size: config.codebook_size,
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
            total_upsample_factor,
            first_codebook,
            rest_codebooks,
            first_output_proj,
            rest_output_proj,
            pre_conv,
            input_proj,
            transformer_layers,
            final_norm,
            output_proj,
            upsample_stages,
            decoder_init_conv,
            decoder_blocks,
            final_snake,
            final_conv,
        }
    }

    /// Decode codec tokens to audio waveform.
    ///
    /// # Arguments
    /// * `codes` - Token indices of shape [batch, num_quantizers, seq_len] as Int tensor
    ///
    /// # Returns
    /// Audio tensor of shape [batch, 1, samples]
    pub fn decode(&self, codes: Tensor<B, 3, Int>) -> Tensor<B, 3> {
        let [batch_size, _num_quantizers, seq_len] = codes.dims();
        let device = codes.device();

        // 1. Quantizer decode
        // First quantizer (semantic)
        let first_codes = codes
            .clone()
            .narrow(1, 0, 1)
            .reshape([batch_size * seq_len]);
        // Apply modulo to map 3072 vocab → 2048 codebook
        // We do this on CPU since modulo might not be directly available
        // Convert via float for backend-agnostic Int element type (I32 on WGPU, I64 on NdArray)
        let first_codes_data: Vec<f32> = first_codes
            .float()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        let codebook_size = self.codebook_size as i32;
        let first_codes_mod: Vec<i32> = first_codes_data
            .iter()
            .map(|&c| (c as i32) % codebook_size)
            .collect();
        let first_codes_tensor =
            Tensor::<B, 1, Int>::from_ints(first_codes_mod.as_slice(), &device)
                .reshape([batch_size, seq_len]);
        let first_embed = self.first_codebook.forward(first_codes_tensor); // [B, T, 256]

        // Apply first output projection: [B, T, 256] -> [B, T, 512]
        let first_proj = self.first_output_proj.forward(first_embed);
        // Transpose: [B, T, 512] -> [B, 512, T]
        let first_proj = first_proj.swap_dims(1, 2);

        // Rest quantizers (acoustic) - sum embeddings then project
        let mut rest_embed = Tensor::<B, 3>::zeros([batch_size, seq_len, 256], &device);
        for i in 0..15 {
            let layer_codes = codes
                .clone()
                .narrow(1, i + 1, 1)
                .reshape([batch_size, seq_len]);
            let embed = self.rest_codebooks[i].forward(layer_codes); // [B, T, 256]
            rest_embed = rest_embed + embed;
        }
        let rest_proj = self.rest_output_proj.forward(rest_embed);
        let rest_proj = rest_proj.swap_dims(1, 2); // [B, 512, T]

        // Sum projected outputs
        let quantized = first_proj + rest_proj; // [B, 512, T]

        // 2. Pre-conv
        let hidden = self.pre_conv.forward(quantized); // [B, 1024, T]

        // 3. Pre-transformer
        let hidden = hidden.swap_dims(1, 2); // [B, T, 1024]
        let hidden = self.input_proj.forward(hidden); // [B, T, 512]

        // Build RoPE embeddings for transformer
        let (cos, sin) = self.build_rope_tables(seq_len, &device);
        let mask = self.build_causal_mask(seq_len, &device);

        let mut hidden = hidden;
        for layer in &self.transformer_layers {
            hidden = layer.forward(hidden, cos.clone(), sin.clone(), mask.clone());
        }

        // Final norm + output projection
        let hidden = self.final_norm.forward(hidden);
        let hidden = self.output_proj.forward(hidden); // [B, T, 1024]

        // 4. Transpose for conv: [B, T, 1024] -> [B, 1024, T]
        let mut hidden = hidden.swap_dims(1, 2);

        // 5. Upsample stages
        for stage in &self.upsample_stages {
            hidden = stage.forward(hidden);
        }

        // 6. Decoder init conv
        hidden = self.decoder_init_conv.forward(hidden);

        // 7. Decoder blocks
        for block in &self.decoder_blocks {
            hidden = block.forward(hidden);
        }

        // 8. Final SnakeBeta + conv
        hidden = self.final_snake.forward(hidden);
        hidden = self.final_conv.forward(hidden);

        // 9. Clamp to [-1, 1]
        hidden.clamp(-1.0, 1.0)
    }

    /// Build RoPE cos/sin tables for the decoder transformer.
    fn build_rope_tables(
        &self,
        seq_len: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let inv_freq: Vec<f32> = (0..self.head_dim)
            .step_by(2)
            .map(|i| 1.0 / (self.rope_theta as f32).powf(i as f32 / self.head_dim as f32))
            .collect();
        let half_dim = inv_freq.len();
        let inv_freq = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);

        let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let positions =
            Tensor::<B, 1>::from_floats(positions.as_slice(), device).unsqueeze_dim::<2>(1);

        // [seq_len, 1] @ [1, half_dim] -> [seq_len, half_dim]
        let freqs = positions.matmul(inv_freq.unsqueeze_dim::<2>(0));

        // Repeat for full head_dim (interleaved RoPE)
        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        debug_assert_eq!(cos.dims(), [seq_len, half_dim]);
        (cos, sin)
    }

    /// Build causal mask for the decoder transformer.
    fn build_causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::<B, 1>::from_floats(mask_data.as_slice(), device).reshape([1, 1, seq_len, seq_len])
    }

    /// Get total upsampling factor.
    pub fn total_upsample(&self) -> usize {
        self.total_upsample_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_config_default() {
        let config = Decoder12HzConfig::default();
        assert_eq!(config.codebook_dim, 512);
        assert_eq!(config.num_quantizers, 16);
        assert_eq!(config.upsample_rates, vec![8, 5, 4, 3]);
    }

    #[test]
    fn test_total_upsample() {
        let config = Decoder12HzConfig::default();
        let pre: usize = config.upsampling_ratios.iter().product();
        let dec: usize = config.upsample_rates.iter().product();
        assert_eq!(pre * dec, 1920);
    }

    #[test]
    fn test_causal_conv_shape() {
        let device = Default::default();
        let conv = CausalConv1d::<B>::new(4, 8, 3, 1, 1, &device);
        let input = Tensor::<B, 3>::zeros([1, 4, 10], &device);
        let output = conv.forward(input);
        assert_eq!(output.dims(), [1, 8, 10]);
    }

    #[test]
    fn test_snake_beta_shape() {
        let device = Default::default();
        let snake = SnakeBeta::<B>::new(64, &device);
        let input = Tensor::<B, 3>::zeros([2, 64, 100], &device);
        let output = snake.forward(input);
        assert_eq!(output.dims(), [2, 64, 100]);
    }

    #[test]
    fn test_causal_transconv_shape() {
        let device = Default::default();
        // UpsampleStage-style: stride=2, kernel=4
        let tc = CausalTransConv1d::<B>::new(4, 4, 4, 2, &device);
        for in_len in [1, 2, 3, 4, 5] {
            let input = Tensor::<B, 3>::zeros([1, 4, in_len], &device);
            let output = tc.forward(input);
            let out_len = output.dims()[2];
            assert_eq!(
                out_len,
                in_len * 2,
                "expected in*stride for in_len={in_len}"
            );
        }
    }
}
