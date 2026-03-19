//! Speaker encoder (ECAPA-TDNN) for voice cloning (Burn)
//!
//! Full ECAPA-TDNN architecture for extracting speaker embeddings.
//! Only needed for Base model variant.

use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::prelude::*;

use crate::audio::{AudioBuffer, MelSpectrogram};
use crate::models::config::SpeakerEncoderConfig;

// ── Helpers ─────────────────────────────────────────────────────────────

/// Apply 1D reflect padding to a `[B, C, T]` tensor along the time dimension.
fn reflect_pad_1d<B: Backend>(x: Tensor<B, 3>, pad_left: usize, pad_right: usize) -> Tensor<B, 3> {
    if pad_left == 0 && pad_right == 0 {
        return x;
    }

    let [_b, _c, t] = x.dims();
    let device = x.device();

    // Build index vector for gather
    let mut indices = Vec::with_capacity(pad_left + t + pad_right);

    // Left reflection: mirror from position 1 outward
    for i in (1..=pad_left).rev() {
        indices.push(i as i32);
    }
    // Original signal
    for i in 0..t {
        indices.push(i as i32);
    }
    // Right reflection: mirror from position t-2 inward
    for i in 0..pad_right {
        indices.push((t - 2 - i) as i32);
    }

    let idx = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &device);
    // select along dim 2 (time)
    x.select(2, idx)
}

// ── Conv1d with reflect padding ─────────────────────────────────────────

/// Conv1d with "same" output length via reflect padding.
#[derive(Module, Debug)]
pub(crate) struct ReflectPadConv1d<B: Backend> {
    pub(crate) conv: Conv1d<B>,
    #[module(skip)]
    pad_left: usize,
    #[module(skip)]
    pad_right: usize,
}

impl<B: Backend> ReflectPadConv1d<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        device: &B::Device,
    ) -> Self {
        let total_pad = dilation * (kernel_size - 1);
        let pad_left = total_pad / 2;
        let pad_right = total_pad - pad_left;

        let conv = Conv1dConfig::new(in_channels, out_channels, kernel_size)
            .with_dilation(dilation)
            .with_bias(true)
            .with_padding(burn::nn::PaddingConfig1d::Explicit(0, 0))
            .init(device);

        Self {
            conv,
            pad_left,
            pad_right,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let padded = reflect_pad_1d(x, self.pad_left, self.pad_right);
        self.conv.forward(padded)
    }
}

// ── Building blocks ─────────────────────────────────────────────────────

/// Time-delay neural network block: Conv1d (reflect-padded) + ReLU.
#[derive(Module, Debug)]
pub(crate) struct TimeDelayNetBlock<B: Backend> {
    pub(crate) conv: ReflectPadConv1d<B>,
}

impl<B: Backend> TimeDelayNetBlock<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            conv: ReflectPadConv1d::new(in_channels, out_channels, kernel_size, dilation, device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        burn::tensor::activation::relu(self.conv.forward(x))
    }
}

/// Res2Net block with cascaded TDNNs.
#[derive(Module, Debug)]
pub(crate) struct Res2NetBlock<B: Backend> {
    pub(crate) blocks: Vec<TimeDelayNetBlock<B>>,
    #[module(skip)]
    scale: usize,
    #[module(skip)]
    chunk_size: usize,
}

impl<B: Backend> Res2NetBlock<B> {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        scale: usize,
        device: &B::Device,
    ) -> Self {
        let chunk_size = channels / scale;
        let blocks = (0..(scale - 1))
            .map(|_| TimeDelayNetBlock::new(chunk_size, chunk_size, kernel_size, dilation, device))
            .collect();
        Self {
            blocks,
            scale,
            chunk_size,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut outputs = Vec::with_capacity(self.scale);

        // First chunk passes through unchanged
        outputs.push(x.clone().narrow(1, 0, self.chunk_size));

        for (i, block) in self.blocks.iter().enumerate() {
            let chunk = x
                .clone()
                .narrow(1, (i + 1) * self.chunk_size, self.chunk_size);
            let input = if i == 0 {
                chunk
            } else {
                chunk + outputs.last().unwrap().clone()
            };
            outputs.push(block.forward(input));
        }

        Tensor::cat(outputs, 1)
    }
}

/// Squeeze-and-excitation block for channel attention.
#[derive(Module, Debug)]
pub(crate) struct SqueezeExcitationBlock<B: Backend> {
    pub(crate) conv1: Conv1d<B>,
    pub(crate) conv2: Conv1d<B>,
}

impl<B: Backend> SqueezeExcitationBlock<B> {
    fn new(channels: usize, se_channels: usize, device: &B::Device) -> Self {
        let conv1 = Conv1dConfig::new(channels, se_channels, 1)
            .with_bias(true)
            .with_padding(burn::nn::PaddingConfig1d::Explicit(0, 0))
            .init(device);
        let conv2 = Conv1dConfig::new(se_channels, channels, 1)
            .with_bias(true)
            .with_padding(burn::nn::PaddingConfig1d::Explicit(0, 0))
            .init(device);
        Self { conv1, conv2 }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Global average pool: [B, C, T] → [B, C, 1]
        let s = x.clone().mean_dim(2); // [B, C, 1]
        let s = burn::tensor::activation::relu(self.conv1.forward(s));
        let s = burn::tensor::activation::sigmoid(self.conv2.forward(s));
        x * s
    }
}

/// SE-Res2Net block: TDNN1 → Res2Net → TDNN2 → SE → residual add.
#[derive(Module, Debug)]
pub(crate) struct SqueezeExcitationRes2NetBlock<B: Backend> {
    pub(crate) tdnn1: TimeDelayNetBlock<B>,
    pub(crate) res2net_block: Res2NetBlock<B>,
    pub(crate) tdnn2: TimeDelayNetBlock<B>,
    pub(crate) se_block: SqueezeExcitationBlock<B>,
}

impl<B: Backend> SqueezeExcitationRes2NetBlock<B> {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        scale: usize,
        se_channels: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            tdnn1: TimeDelayNetBlock::new(channels, channels, 1, 1, device),
            res2net_block: Res2NetBlock::new(channels, kernel_size, dilation, scale, device),
            tdnn2: TimeDelayNetBlock::new(channels, channels, 1, 1, device),
            se_block: SqueezeExcitationBlock::new(channels, se_channels, device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let out = self.tdnn1.forward(x);
        let out = self.res2net_block.forward(out);
        let out = self.tdnn2.forward(out);
        let out = self.se_block.forward(out);
        out + residual
    }
}

/// Attentive statistics pooling.
#[derive(Module, Debug)]
pub(crate) struct AttentiveStatisticsPooling<B: Backend> {
    pub(crate) tdnn: TimeDelayNetBlock<B>,
    pub(crate) conv: Conv1d<B>,
}

impl<B: Backend> AttentiveStatisticsPooling<B> {
    fn new(channels: usize, attention_channels: usize, device: &B::Device) -> Self {
        Self {
            tdnn: TimeDelayNetBlock::new(channels * 3, attention_channels, 1, 1, device),
            conv: Conv1dConfig::new(attention_channels, channels, 1)
                .with_bias(true)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(0, 0))
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, c, t] = x.dims();

        // Global statistics
        let mean = x.clone().mean_dim(2); // [B, C, 1]
        let diff = x.clone() - mean.clone();
        let var = diff.clone().powf_scalar(2.0).mean_dim(2); // [B, C, 1]
        let std = (var + 1e-5).sqrt();

        let mean_exp = mean.expand([b, c, t]);
        let std_exp = std.expand([b, c, t]);

        // Concatenate [x, mean, std] along channel dim → [B, 3C, T]
        let attn_in = Tensor::cat(vec![x.clone(), mean_exp, std_exp], 1);

        // Attention: TDNN → Tanh → Conv → Softmax
        let attn = self.tdnn.forward(attn_in);
        let attn = attn.tanh();
        let attn = self.conv.forward(attn);
        let attn = burn::tensor::activation::softmax(attn, 2); // softmax over T

        // Weighted mean
        let w_mean = (x.clone() * attn.clone()).sum_dim(2); // [B, C, 1]

        // Weighted std
        let w_diff = x - w_mean.clone();
        let w_var = (w_diff.powf_scalar(2.0) * attn).sum_dim(2);
        let w_std = (w_var + 1e-5).sqrt();

        // Output: [B, 2C, 1]
        Tensor::cat(vec![w_mean, w_std], 1)
    }
}

// ── Main encoder ────────────────────────────────────────────────────────

/// Full ECAPA-TDNN speaker encoder.
#[derive(Module, Debug)]
pub struct SpeakerEncoder<B: Backend> {
    pub(crate) initial_tdnn: TimeDelayNetBlock<B>,
    pub(crate) se_res2net_blocks: Vec<SqueezeExcitationRes2NetBlock<B>>,
    pub(crate) mfa_tdnn: TimeDelayNetBlock<B>,
    pub(crate) asp: AttentiveStatisticsPooling<B>,
    pub(crate) fc: Conv1d<B>,
}

impl<B: Backend> SpeakerEncoder<B> {
    /// Initialize from config.
    pub fn init(config: SpeakerEncoderConfig, device: &B::Device) -> Self {
        let initial_tdnn = TimeDelayNetBlock::new(
            config.mel_dim,
            config.enc_channels[0],
            config.enc_kernel_sizes[0],
            config.enc_dilations[0],
            device,
        );

        let se_res2net_blocks = (1..4)
            .map(|i| {
                SqueezeExcitationRes2NetBlock::new(
                    config.enc_channels[i],
                    config.enc_kernel_sizes[i],
                    config.enc_dilations[i],
                    config.enc_res2net_scale,
                    config.enc_se_channels,
                    device,
                )
            })
            .collect();

        let mfa_in_channels: usize = config.enc_channels[1..4].iter().sum();
        let mfa_tdnn = TimeDelayNetBlock::new(
            mfa_in_channels,
            config.enc_channels[4],
            config.enc_kernel_sizes[4],
            config.enc_dilations[4],
            device,
        );

        let asp = AttentiveStatisticsPooling::new(
            config.enc_channels[4],
            config.enc_attention_channels,
            device,
        );

        let fc = Conv1dConfig::new(config.enc_channels[4] * 2, config.enc_dim, 1)
            .with_bias(true)
            .with_padding(burn::nn::PaddingConfig1d::Explicit(0, 0))
            .init(device);

        Self {
            initial_tdnn,
            se_res2net_blocks,
            mfa_tdnn,
            asp,
            fc,
        }
    }

    /// Extract a speaker embedding from reference audio.
    ///
    /// Returns an embedding of shape `[enc_dim]` (default 1024).
    pub fn encode(&self, audio: &AudioBuffer, device: &B::Device) -> Tensor<B, 1> {
        let mel_extractor = MelSpectrogram::new(MelSpectrogram::speaker_encoder());
        let mel = mel_extractor.compute_for_speaker_encoder_raw(&audio.samples);
        // mel is Vec<Vec<f32>> of shape [n_mels, T]
        let n_mels = mel.len();
        let t = mel[0].len();
        let flat: Vec<f32> = mel.into_iter().flatten().collect();
        let mel_tensor =
            Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([1, n_mels, t]);

        let embed = self.forward(mel_tensor); // [1, enc_dim]
        embed.squeeze_dim::<1>(0) // [enc_dim]
    }

    /// Forward pass on a batched mel spectrogram `[B, n_mels, T]`.
    ///
    /// Returns embeddings of shape `[B, enc_dim]`.
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 2> {
        // blocks[0]: initial TDNN
        let x = self.initial_tdnn.forward(mel);

        // blocks[1-3]: SE-Res2Net, collecting outputs for MFA
        let mut se_outputs = Vec::with_capacity(3);
        let mut h = x;
        for block in &self.se_res2net_blocks {
            h = block.forward(h);
            se_outputs.push(h.clone());
        }

        // MFA: concatenate SE-Res2Net outputs along channel dim
        let mfa_input = Tensor::cat(se_outputs, 1);
        let h = self.mfa_tdnn.forward(mfa_input);

        // ASP: attentive statistics pooling → [B, 2C, 1]
        let pooled = self.asp.forward(h);

        // FC: project to embedding dimension → [B, enc_dim, 1]
        let embed = self.fc.forward(pooled);
        embed.squeeze_dim::<2>(2) // [B, enc_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_reflect_pad_1d_no_pad() {
        let device = Default::default();
        let x = Tensor::<B, 3>::zeros([1, 1, 5], &device);
        let padded = reflect_pad_1d(x, 0, 0);
        assert_eq!(padded.dims(), [1, 1, 5]);
    }

    #[test]
    fn test_reflect_pad_1d_left() {
        let device = Default::default();
        let x =
            Tensor::<B, 1>::from_floats([0.0f32, 1.0, 2.0, 3.0, 4.0], &device).reshape([1, 1, 5]);
        let padded = reflect_pad_1d(x, 2, 0);
        let vals: Vec<f32> = padded.reshape([7]).into_data().to_vec().unwrap();
        assert_eq!(vals, vec![2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reflect_pad_1d_right() {
        let device = Default::default();
        let x =
            Tensor::<B, 1>::from_floats([0.0f32, 1.0, 2.0, 3.0, 4.0], &device).reshape([1, 1, 5]);
        let padded = reflect_pad_1d(x, 0, 2);
        let vals: Vec<f32> = padded.reshape([7]).into_data().to_vec().unwrap();
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }
}
