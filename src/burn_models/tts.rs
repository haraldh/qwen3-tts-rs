//! TTS-specific generation logic (Burn)
//!
//! Token suppression during sampling to prevent the model from generating
//! tokens in the reserved control range.

use burn::prelude::*;

/// Pre-computed boolean mask for token suppression.
///
/// Build once with [`build_suppression_mask`], then apply cheaply each frame
/// with [`apply_token_suppression_with_mask`].
pub struct SuppressionMask<B: Backend> {
    /// Float mask: -inf at positions to suppress, 0.0 elsewhere. Shape [1, vocab].
    mask: Tensor<B, 2>,
}

/// Build a reusable suppression mask for the given vocab/EOS config.
///
/// The mask is a [1, vocab] float tensor with -inf at suppressed positions
/// and 0.0 elsewhere, so it can be added directly to logits.
pub fn build_suppression_mask<B: Backend>(
    vocab_size: usize,
    eos_token_id: u32,
    device: &B::Device,
) -> SuppressionMask<B> {
    let suppress_start = vocab_size - 1024;
    let mut mask_data = vec![0.0f32; vocab_size];
    for (v, val) in mask_data
        .iter_mut()
        .enumerate()
        .skip(suppress_start)
        .take(1024)
    {
        if v as u32 != eos_token_id {
            *val = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::<B, 1>::from_floats(mask_data.as_slice(), device).reshape([1, vocab_size]);
    SuppressionMask { mask }
}

/// Apply a pre-built suppression mask to logits (cheap per-frame operation).
///
/// Adds the mask (0.0 or -inf) to logits, which zeroes out suppressed tokens
/// after softmax.
pub fn apply_token_suppression_with_mask<B: Backend>(
    logits: Tensor<B, 2>,
    suppression: &SuppressionMask<B>,
) -> Tensor<B, 2> {
    let [batch, vocab] = logits.dims();
    let mask = suppression.mask.clone().expand([batch, vocab]);
    logits + mask
}

/// Apply token suppression to logits (builds mask each call — use the mask
/// variant for hot loops).
///
/// Masks out tokens in range `[vocab_size - 1024, vocab_size)` except for the
/// EOS token, which is preserved.
pub fn apply_token_suppression<B: Backend>(
    logits: Tensor<B, 2>,
    vocab_size: usize,
    eos_token_id: u32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let suppression = build_suppression_mask(vocab_size, eos_token_id, device);
    apply_token_suppression_with_mask(logits, &suppression)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_suppression_masks_control_tokens() {
        let device = Default::default();
        let vocab_size = 3072;
        let eos_id = 2150u32;

        let logits = Tensor::<B, 2>::ones([1, vocab_size], &device);
        let result = apply_token_suppression(logits, vocab_size, eos_id, &device);
        let vals: Vec<f32> = result.into_data().to_vec().unwrap();

        // Non-control tokens should be unchanged
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[2047] - 1.0).abs() < 1e-6);

        // Control tokens (except EOS) should be -inf
        assert!(vals[2048].is_infinite() && vals[2048] < 0.0);
        assert!(vals[2149].is_infinite() && vals[2149] < 0.0);

        // EOS should be preserved
        assert!((vals[2150] - 1.0).abs() < 1e-6);

        // Other control tokens suppressed
        assert!(vals[2151].is_infinite() && vals[2151] < 0.0);
        assert!(vals[3071].is_infinite() && vals[3071] < 0.0);
    }

    #[test]
    fn test_suppression_batch() {
        let device = Default::default();
        let vocab_size = 3072;
        let eos_id = 2150u32;

        let logits = Tensor::<B, 2>::ones([2, vocab_size], &device);
        let result = apply_token_suppression(logits, vocab_size, eos_id, &device);
        let vals: Vec<f32> = result.into_data().to_vec().unwrap();

        // Both batches should have suppression
        assert!(vals[2048].is_infinite()); // batch 0
        assert!(vals[vocab_size + 2048].is_infinite()); // batch 1

        // EOS preserved in both
        assert!((vals[2150] - 1.0).abs() < 1e-6);
        assert!((vals[vocab_size + 2150] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_prebuilt_mask_reusable_across_batches() {
        let device = Default::default();
        let vocab_size = 3072;
        let eos_id = 2150u32;

        // Build mask once
        let mask = build_suppression_mask::<B>(vocab_size, eos_id, &device);

        // Apply to batch=1
        let logits1 = Tensor::<B, 2>::ones([1, vocab_size], &device);
        let r1 = apply_token_suppression_with_mask(logits1, &mask);
        let vals1: Vec<f32> = r1.into_data().to_vec().unwrap();
        assert!(vals1[2048].is_infinite());

        // Apply to batch=3 (same mask reused)
        let logits3 = Tensor::<B, 2>::ones([3, vocab_size], &device);
        let r3 = apply_token_suppression_with_mask(logits3, &mask);
        let vals: Vec<f32> = r3.into_data().to_vec().unwrap();
        for batch in 0..3 {
            assert!(vals[batch * vocab_size + 2048].is_infinite());
            assert!((vals[batch * vocab_size + 2150] - 1.0).abs() < 1e-6);
        }
    }
}
