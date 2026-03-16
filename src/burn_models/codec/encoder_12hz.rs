//! 12Hz Audio Encoder for Qwen3-TTS (Burn)
//!
//! Encodes raw audio into 12Hz, 16-codebook discrete codec codes for ICL
//! voice cloning. Mirrors the candle-based encoder using Burn abstractions.
//!
//! Components:
//! - SEANet encoder: hierarchical CNN with ELU activation
//! - Projected transformer: 8-layer with LayerNorm, GELU MLP, layer scales
//! - ConvDownsample: strided causal conv (25Hz → 12.5Hz)
//! - Split residual vector quantizer: 1 semantic + 15 acoustic codebooks

use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

use super::decoder_12hz::CausalConv1d;

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for the 12Hz encoder.
#[derive(Debug, Clone)]
pub struct Encoder12HzConfig {
    pub audio_channels: usize,
    pub num_filters: usize,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub compress: usize,
    pub num_residual_layers: usize,
    pub downsample_ratios: Vec<usize>,
    pub dimension: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_transformer_layers: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub num_semantic_quantizers: usize,
    pub num_acoustic_quantizers: usize,
    pub layer_scale: f64,
    pub rope_theta: f64,
    pub norm_eps: f64,
}

impl Default for Encoder12HzConfig {
    fn default() -> Self {
        Self {
            audio_channels: 1,
            num_filters: 64,
            kernel_size: 7,
            last_kernel_size: 3,
            residual_kernel_size: 3,
            compress: 2,
            num_residual_layers: 1,
            // Reversed from decoder ratios [8, 6, 5, 4]
            downsample_ratios: vec![4, 5, 6, 8],
            dimension: 512,
            hidden_size: 512,
            num_heads: 8,
            head_dim: 64,
            intermediate_size: 2048,
            num_transformer_layers: 8,
            codebook_dim: 256,
            codebook_size: 2048,
            num_semantic_quantizers: 1,
            // encoder_valid_num_quantizers (16) - num_semantic (1) = 15
            num_acoustic_quantizers: 15,
            layer_scale: 0.01,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        }
    }
}

// ── SEANet Encoder Residual Block ───────────────────────────────────────

/// Residual block for the SEANet encoder: ELU + dilated conv + ELU + 1×1 conv.
///
/// Unlike the decoder's SnakeBeta-based units, the encoder uses ELU activation.
/// The first conv compresses channels by `compress` ratio (typically 2).
#[derive(Module, Debug)]
pub struct EncoderResidualBlock<B: Backend> {
    pub(crate) conv1: CausalConv1d<B>,
    pub(crate) conv2: CausalConv1d<B>,
}

impl<B: Backend> EncoderResidualBlock<B> {
    pub fn new(
        dim: usize,
        compress: usize,
        kernel_size: usize,
        dilation: usize,
        device: &B::Device,
    ) -> Self {
        let hidden = dim / compress;
        Self {
            conv1: CausalConv1d::new(dim, hidden, kernel_size, dilation, 1, device),
            conv2: CausalConv1d::new(hidden, dim, 1, 1, 1, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let h = burn::tensor::activation::elu(x, 1.0);
        let h = self.conv1.forward(h);
        let h = burn::tensor::activation::elu(h, 1.0);
        let h = self.conv2.forward(h);
        h + residual
    }
}

// ── SEANet Encoder Stage ────────────────────────────────────────────────

/// One stage of the SEANet encoder: residual blocks + ELU + downsample conv.
#[derive(Module, Debug)]
pub struct EncoderStage<B: Backend> {
    pub(crate) residual_blocks: Vec<EncoderResidualBlock<B>>,
    pub(crate) downsample: CausalConv1d<B>,
}

impl<B: Backend> EncoderStage<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        ratio: usize,
        num_residual_layers: usize,
        compress: usize,
        residual_kernel_size: usize,
        dilation_growth_rate: usize,
        device: &B::Device,
    ) -> Self {
        let residual_blocks = (0..num_residual_layers)
            .map(|j| {
                let dilation = dilation_growth_rate.pow(j as u32);
                EncoderResidualBlock::new(
                    in_channels,
                    compress,
                    residual_kernel_size,
                    dilation,
                    device,
                )
            })
            .collect();

        // Downsample: causal conv with stride, kernel = 2 * ratio
        let downsample =
            CausalConv1d::new_strided(in_channels, out_channels, 2 * ratio, ratio, 1, 1, device);

        Self {
            residual_blocks,
            downsample,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = x;
        for block in &self.residual_blocks {
            h = block.forward(h);
        }
        let h = burn::tensor::activation::elu(h, 1.0);
        self.downsample.forward(h)
    }
}

// ── SEANet Encoder ──────────────────────────────────────────────────────

/// Hierarchical CNN encoder from the Mimi SEANet architecture.
///
/// Architecture: init_conv → N stages (residual + downsample) → ELU → final_conv.
/// Produces 25Hz features at `dimension` channels from raw audio.
#[derive(Module, Debug)]
pub struct SeaNetEncoder<B: Backend> {
    pub(crate) init_conv: CausalConv1d<B>,
    pub(crate) stages: Vec<EncoderStage<B>>,
    pub(crate) final_conv: CausalConv1d<B>,
}

impl<B: Backend> SeaNetEncoder<B> {
    pub fn new(config: &Encoder12HzConfig, device: &B::Device) -> Self {
        let init_conv = CausalConv1d::new(
            config.audio_channels,
            config.num_filters,
            config.kernel_size,
            1,
            1,
            device,
        );

        let mut cur_channels = config.num_filters;
        let stages = config
            .downsample_ratios
            .iter()
            .map(|&ratio| {
                let out_channels = cur_channels * 2;
                let stage = EncoderStage::new(
                    cur_channels,
                    out_channels,
                    ratio,
                    config.num_residual_layers,
                    config.compress,
                    config.residual_kernel_size,
                    2, // dilation_growth_rate
                    device,
                );
                cur_channels = out_channels;
                stage
            })
            .collect();

        // After all stages: channels = num_filters * 2^num_stages = 64 * 16 = 1024
        let final_conv = CausalConv1d::new(
            cur_channels,
            config.dimension,
            config.last_kernel_size,
            1,
            1,
            device,
        );

        Self {
            init_conv,
            stages,
            final_conv,
        }
    }

    /// Encode raw audio to features.
    ///
    /// Input: [B, 1, samples] at 24kHz.
    /// Output: [B, dimension, T_25hz] where T_25hz = samples / product(ratios).
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = self.init_conv.forward(x);
        for stage in &self.stages {
            h = stage.forward(h);
        }
        let h = burn::tensor::activation::elu(h, 1.0);
        self.final_conv.forward(h)
    }
}

// ── Encoder Transformer Layer ───────────────────────────────────────────

/// Transformer layer for the encoder (different from decoder transformer:
/// uses LayerNorm with bias, GELU MLP instead of SwiGLU, has layer scales).
#[derive(Module, Debug)]
pub struct EncoderTransformerLayer<B: Backend> {
    pub(crate) input_layernorm: LayerNorm<B>,
    pub(crate) q_proj: Linear<B>,
    pub(crate) k_proj: Linear<B>,
    pub(crate) v_proj: Linear<B>,
    pub(crate) o_proj: Linear<B>,
    pub(crate) attn_layer_scale: Tensor<B, 1>,
    pub(crate) post_attention_layernorm: LayerNorm<B>,
    pub(crate) fc1: Linear<B>,
    pub(crate) fc2: Linear<B>,
    pub(crate) mlp_layer_scale: Tensor<B, 1>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    scale: f64,
}

impl<B: Backend> EncoderTransformerLayer<B> {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        norm_eps: f64,
        layer_scale_init: f64,
        device: &B::Device,
    ) -> Self {
        let input_layernorm = LayerNormConfig::new(hidden_size)
            .with_epsilon(norm_eps)
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
        let post_attention_layernorm = LayerNormConfig::new(hidden_size)
            .with_epsilon(norm_eps)
            .init(device);
        let fc1 = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(false)
            .init(device);
        let fc2 = LinearConfig::new(intermediate_size, hidden_size)
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
            fc1,
            fc2,
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

        // LayerNorm
        let normed = self.input_layernorm.forward(hidden_states.clone());

        // Self attention
        let q = self.q_proj.forward(normed.clone());
        let k = self.k_proj.forward(normed.clone());
        let v = self.v_proj.forward(normed);

        // Reshape: [B, T, H*D] -> [B, H, T, D]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE (interleaved style)
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

        // MLP: GELU(fc1(x)) → fc2
        let normed = self.post_attention_layernorm.forward(hidden_states.clone());
        let mlp_out = self.fc1.forward(normed);
        let mlp_out = burn::tensor::activation::gelu(mlp_out);
        let mlp_out = self.fc2.forward(mlp_out);

        // Layer scale + residual
        let mlp_scale = self
            .mlp_layer_scale
            .clone()
            .unsqueeze::<2>()
            .unsqueeze::<3>();
        let mlp_out = mlp_out * mlp_scale;
        hidden_states + mlp_out
    }

    fn apply_rope(&self, x: Tensor<B, 4>, cos: Tensor<B, 2>, sin: Tensor<B, 2>) -> Tensor<B, 4> {
        let half = self.head_dim / 2;
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.narrow(3, half, half);

        let cos = cos.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        let sin = sin.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

        let neg_x2 = x2.clone().neg();
        let part1 = x1.clone() * cos.clone() + neg_x2 * sin.clone();
        let part2 = x1 * sin + x2 * cos;
        Tensor::cat(vec![part1, part2], 3)
    }
}

// ── Vector Quantizer (single codebook) ──────────────────────────────────

/// Single codebook with L2 nearest-neighbor encoding.
#[derive(Module, Debug)]
pub struct VectorQuantizer<B: Backend> {
    /// Normalized codebook embeddings: [codebook_size, dim].
    pub(crate) codebook: Tensor<B, 2>,
    #[module(skip)]
    codebook_size: usize,
    #[module(skip)]
    dim: usize,
}

impl<B: Backend> VectorQuantizer<B> {
    pub fn new(codebook_size: usize, dim: usize, device: &B::Device) -> Self {
        let codebook = Tensor::zeros([codebook_size, dim], device);
        Self {
            codebook,
            codebook_size,
            dim,
        }
    }

    /// Encode input to nearest codebook indices.
    ///
    /// Input: [B*T, dim]. Output: [B*T] Int tensor of codebook indices.
    pub fn encode(&self, x: Tensor<B, 2>) -> Tensor<B, 1, Int> {
        // L2 distance: ||x - cb||^2 = ||x||^2 - 2*x·cb^T + ||cb||^2
        // We can skip ||x||^2 since it's constant across codebook entries.
        let dot = x.matmul(self.codebook.clone().transpose()); // [B*T, codebook_size]
        let cb_sq = self
            .codebook
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .squeeze::<1>(); // [codebook_size]
        let cb_sq = cb_sq.unsqueeze_dim::<2>(0); // [1, codebook_size]
        let dist = dot.neg() * 2.0 + cb_sq; // [B*T, codebook_size]
        let n = dist.dims()[0];
        dist.argmin(1).reshape([n]) // [B*T]
    }

    /// Look up codebook entries by index.
    ///
    /// Input: [N] Int tensor. Output: [N, dim].
    pub fn decode(&self, codes: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        let n = codes.dims()[0];
        // Gather from codebook: for each code, select the corresponding row
        let codes_data: Vec<f32> = codes.float().into_data().convert::<f32>().to_vec().unwrap();
        let indices: Vec<i32> = codes_data.iter().map(|&c| c as i32).collect();
        let idx_tensor =
            Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &self.codebook.device());
        // select rows from codebook [codebook_size, dim] → [N, dim]
        self.codebook
            .clone()
            .select(0, idx_tensor)
            .reshape([n, self.dim])
    }
}

// ── Residual Vector Quantizer ───────────────────────────────────────────

/// Residual VQ with input/output projections and multiple codebook layers.
#[derive(Module, Debug)]
pub struct ResidualVectorQuantizer<B: Backend> {
    /// 1×1 conv projection: input_dim → codebook_dim (stored as Linear).
    pub(crate) input_proj: Linear<B>,
    /// 1×1 conv projection: codebook_dim → input_dim (stored as Linear).
    pub(crate) output_proj: Linear<B>,
    pub(crate) layers: Vec<VectorQuantizer<B>>,
    #[module(skip)]
    codebook_dim: usize,
}

impl<B: Backend> ResidualVectorQuantizer<B> {
    pub fn new(
        input_dim: usize,
        codebook_dim: usize,
        codebook_size: usize,
        num_layers: usize,
        device: &B::Device,
    ) -> Self {
        let input_proj = LinearConfig::new(input_dim, codebook_dim)
            .with_bias(false)
            .init(device);
        let output_proj = LinearConfig::new(codebook_dim, input_dim)
            .with_bias(false)
            .init(device);
        let layers = (0..num_layers)
            .map(|_| VectorQuantizer::new(codebook_size, codebook_dim, device))
            .collect();
        Self {
            input_proj,
            output_proj,
            layers,
            codebook_dim,
        }
    }

    /// Encode: project input, then residual VQ through all layers.
    ///
    /// Input: [B, input_dim, T] (conv layout).
    /// Output: [B, num_layers, T] Int tensor of codes.
    pub fn encode(&self, x: Tensor<B, 3>) -> Tensor<B, 3, Int> {
        let [batch, _dim, seq_len] = x.dims();

        // Project: [B, input_dim, T] → [B, T, input_dim] → Linear → [B, T, codebook_dim]
        let projected = x.swap_dims(1, 2);
        let projected = self.input_proj.forward(projected); // [B, T, codebook_dim]

        // Flatten for VQ: [B*T, codebook_dim]
        let mut residual = projected.reshape([batch * seq_len, self.codebook_dim]);

        let mut all_codes = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let codes = layer.encode(residual.clone()); // [B*T]
            let quantized = layer.decode(codes.clone()); // [B*T, codebook_dim]
            residual = residual - quantized;
            all_codes.push(codes.reshape([batch, 1, seq_len]));
        }

        // Stack: [B, num_layers, T]
        Tensor::cat(all_codes, 1)
    }

    /// Decode and project back to input space.
    ///
    /// Used internally to compute residuals during SplitRVQ encoding.
    /// Input: [B, input_dim, T]. Output: [B, input_dim, T].
    pub fn encode_and_project(&self, x: Tensor<B, 3>) -> (Tensor<B, 3, Int>, Tensor<B, 3>) {
        let [batch, _dim, seq_len] = x.dims();

        // Project to codebook space
        let projected = x.swap_dims(1, 2);
        let projected = self.input_proj.forward(projected); // [B, T, codebook_dim]
        let mut residual = projected.reshape([batch * seq_len, self.codebook_dim]);

        let mut all_codes = Vec::with_capacity(self.layers.len());
        let mut total_quantized =
            Tensor::<B, 2>::zeros([batch * seq_len, self.codebook_dim], &residual.device());

        for layer in &self.layers {
            let codes = layer.encode(residual.clone());
            let quantized = layer.decode(codes.clone());
            residual = residual.clone() - quantized.clone();
            total_quantized = total_quantized + quantized;
            all_codes.push(codes.reshape([batch, 1, seq_len]));
        }

        let codes = Tensor::cat(all_codes, 1);

        // Project quantized back to input space: [B*T, codebook_dim] → [B, T, codebook_dim] → Linear → [B, T, input_dim] → [B, input_dim, T]
        let quantized = total_quantized.reshape([batch, seq_len, self.codebook_dim]);
        let output = self.output_proj.forward(quantized); // [B, T, input_dim]
        let output = output.swap_dims(1, 2); // [B, input_dim, T]

        (codes, output)
    }
}

// ── Split Residual Vector Quantizer ─────────────────────────────────────

/// Split RVQ: 1 semantic codebook + N acoustic codebooks (residual).
#[derive(Module, Debug)]
pub struct SplitResidualVectorQuantizer<B: Backend> {
    pub(crate) semantic: ResidualVectorQuantizer<B>,
    pub(crate) acoustic: ResidualVectorQuantizer<B>,
}

impl<B: Backend> SplitResidualVectorQuantizer<B> {
    pub fn new(
        input_dim: usize,
        codebook_dim: usize,
        codebook_size: usize,
        num_semantic: usize,
        num_acoustic: usize,
        device: &B::Device,
    ) -> Self {
        let semantic = ResidualVectorQuantizer::new(
            input_dim,
            codebook_dim,
            codebook_size,
            num_semantic,
            device,
        );
        let acoustic = ResidualVectorQuantizer::new(
            input_dim,
            codebook_dim,
            codebook_size,
            num_acoustic,
            device,
        );
        Self { semantic, acoustic }
    }

    /// Encode to discrete codes.
    ///
    /// Input: [B, dim, T] (conv layout, dim=512).
    /// Output: [B, num_semantic + num_acoustic, T] Int tensor.
    pub fn encode(&self, x: Tensor<B, 3>) -> Tensor<B, 3, Int> {
        // Semantic: encode and get the projected-back output for residual
        let (semantic_codes, semantic_output) = self.semantic.encode_and_project(x.clone());

        // Acoustic: encode the residual
        let residual = x - semantic_output;
        let acoustic_codes = self.acoustic.encode(residual);

        // Concatenate: [B, 1, T] + [B, 15, T] = [B, 16, T]
        Tensor::cat(vec![semantic_codes, acoustic_codes], 1)
    }
}

// ── Full Encoder ────────────────────────────────────────────────────────

/// Full 12Hz encoder: SEANet → transformer → downsample → quantize.
#[derive(Module, Debug)]
pub struct Encoder12Hz<B: Backend> {
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    rope_theta: f64,
    pub(crate) seanet: SeaNetEncoder<B>,
    pub(crate) transformer_layers: Vec<EncoderTransformerLayer<B>>,
    pub(crate) downsample: CausalConv1d<B>,
    pub(crate) quantizer: SplitResidualVectorQuantizer<B>,
}

impl<B: Backend> Encoder12Hz<B> {
    pub fn init(config: Encoder12HzConfig, device: &B::Device) -> Self {
        let seanet = SeaNetEncoder::new(&config, device);

        let transformer_layers = (0..config.num_transformer_layers)
            .map(|_| {
                EncoderTransformerLayer::new(
                    config.hidden_size,
                    config.num_heads,
                    config.head_dim,
                    config.intermediate_size,
                    config.norm_eps,
                    config.layer_scale,
                    device,
                )
            })
            .collect();

        // Downsample 25Hz → 12.5Hz: stride=2, kernel=4
        let downsample =
            CausalConv1d::new_strided(config.dimension, config.dimension, 4, 2, 1, 1, device);

        let quantizer = SplitResidualVectorQuantizer::new(
            config.dimension,
            config.codebook_dim,
            config.codebook_size,
            config.num_semantic_quantizers,
            config.num_acoustic_quantizers,
            device,
        );

        Self {
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
            seanet,
            transformer_layers,
            downsample,
            quantizer,
        }
    }

    /// Encode audio samples to discrete codec codes.
    ///
    /// Input: raw f32 samples at 24kHz.
    /// Output: `Vec<Vec<u32>>` — outer vec is frames, inner vec is 16 codes per frame.
    pub fn encode(&self, samples: &[f32]) -> Vec<Vec<u32>> {
        let device = self.seanet.init_conv.conv.weight.device();

        // [1, 1, N_samples]
        let input = Tensor::<B, 1>::from_floats(samples, &device)
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0);

        // SEANet: [1, 1, N] → [1, 512, T_25hz]
        let features = self.seanet.forward(input);
        let t_25hz = features.dims()[2];

        // Transformer: [1, 512, T] → [1, T, 512] → layers → [1, 512, T]
        let mut hidden = features.swap_dims(1, 2); // [1, T, 512]
        let (cos, sin) = self.build_rope_tables(t_25hz, &device);
        let mask = self.build_causal_mask(t_25hz, &device);
        for layer in &self.transformer_layers {
            hidden = layer.forward(hidden, cos.clone(), sin.clone(), mask.clone());
        }
        let hidden = hidden.swap_dims(1, 2); // [1, 512, T]

        // Downsample: [1, 512, T_25hz] → [1, 512, T_12hz]
        let hidden = self.downsample.forward(hidden);

        // Quantize: [1, 512, T_12hz] → [1, 16, T_12hz]
        let codes = self.quantizer.encode(hidden);

        // Convert to Vec<Vec<u32>>: [1, 16, T] → T frames of 16 codes
        let [_batch, num_q, num_frames] = codes.dims();
        let codes_data: Vec<f32> = codes.float().into_data().convert::<f32>().to_vec().unwrap();

        let mut result = Vec::with_capacity(num_frames);
        for t in 0..num_frames {
            let mut frame_codes = Vec::with_capacity(num_q);
            for q in 0..num_q {
                let idx = q * num_frames + t;
                frame_codes.push(codes_data[idx] as u32);
            }
            result.push(frame_codes);
        }
        result
    }

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

        let freqs = positions.matmul(inv_freq.unsqueeze_dim::<2>(0));
        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        debug_assert_eq!(cos.dims(), [seq_len, half_dim]);
        (cos, sin)
    }

    fn build_causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::<B, 1>::from_floats(mask_data.as_slice(), device).reshape([1, 1, seq_len, seq_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_encoder_config_default() {
        let config = Encoder12HzConfig::default();
        assert_eq!(config.downsample_ratios, vec![4, 5, 6, 8]);
        assert_eq!(config.dimension, 512);
        assert_eq!(config.num_acoustic_quantizers, 15);
        let total_downsample: usize = config.downsample_ratios.iter().product();
        // 24000 / 960 = 25Hz, then /2 for downsample = 12.5Hz
        assert_eq!(total_downsample, 960);
    }

    #[test]
    fn test_seanet_shapes() {
        let device = Default::default();
        let config = Encoder12HzConfig::default();
        let seanet = SeaNetEncoder::<B>::new(&config, &device);

        // 960 samples = 1 frame at 25Hz (before downsample)
        let input = Tensor::<B, 3>::zeros([1, 1, 960], &device);
        let output = seanet.forward(input);
        assert_eq!(output.dims()[0], 1);
        assert_eq!(output.dims()[1], 512);
        // T_25hz should be 1
        assert_eq!(output.dims()[2], 1);
    }

    #[test]
    fn test_encoder_residual_block_shape() {
        let device = Default::default();
        let block = EncoderResidualBlock::<B>::new(64, 2, 3, 1, &device);
        let input = Tensor::<B, 3>::zeros([1, 64, 100], &device);
        let output = block.forward(input);
        assert_eq!(output.dims(), [1, 64, 100]);
    }

    #[test]
    fn test_vector_quantizer() {
        let device = Default::default();
        let mut vq = VectorQuantizer::<B>::new(8, 4, &device);
        // Set codebook to known values
        let cb_data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        vq.codebook =
            Tensor::from_data(burn::tensor::TensorData::new(cb_data, vec![8, 4]), &device);

        // Query that is closest to codebook entry 0
        let query = Tensor::<B, 2>::from_floats([[0.1, 0.1, 0.1, 0.1]], &device);
        let codes = vq.encode(query);
        let code_val: Vec<f32> = codes.float().into_data().to_vec().unwrap();
        assert_eq!(code_val[0] as u32, 0);
    }
}
