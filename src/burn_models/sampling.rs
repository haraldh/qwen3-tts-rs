//! Token sampling strategies for autoregressive generation (Burn)
//!
//! Supports both deterministic (seeded) and non-deterministic random sampling.
//! Create a [`SamplingContext`] with an optional seed for reproducible outputs.

use burn::prelude::*;

/// RNG and sampling state for a single generation session.
///
/// Encapsulates all randomness so that multiple sessions can run
/// concurrently without interfering with each other.
pub struct SamplingContext {
    /// PCG state (only used when seeded)
    state: u64,
    /// Whether we're in seeded mode
    seeded: bool,
    /// Counter for unseeded fallback
    counter: u64,
}

impl SamplingContext {
    /// Create a new sampling context with an optional seed.
    pub fn new(seed: Option<u64>) -> Self {
        match seed {
            Some(s) => {
                let state = s
                    .wrapping_mul(2685821657736338717)
                    .wrapping_add(1442695040888963407);
                Self {
                    state,
                    seeded: true,
                    counter: 0,
                }
            }
            None => Self {
                state: 0,
                seeded: false,
                counter: 0,
            },
        }
    }

    /// Reset the RNG to its initial seeded state.
    pub fn reset(&mut self, seed: u64) {
        let state = seed
            .wrapping_mul(2685821657736338717)
            .wrapping_add(1442695040888963407);
        self.state = state;
        self.seeded = true;
    }

    /// Generate a random f32 in [0, 1).
    fn rand_f32(&mut self) -> f32 {
        if !self.seeded {
            use std::time::{SystemTime, UNIX_EPOCH};

            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as u64;
            let count = self.counter;
            self.counter += 1;

            let state = seed
                .wrapping_add(count)
                .wrapping_mul(1103515245)
                .wrapping_add(12345);
            return (state as f32) / (u64::MAX as f32);
        }

        // PCG XSH RR 64/32
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);

        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        let output = xorshifted.rotate_right(rot);

        (output as f32) / (u32::MAX as f32)
    }
}

/// Configuration for autoregressive generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Sampling temperature (1.0 = no change, <1.0 = more focused, >1.0 = more random)
    pub temperature: f64,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling threshold (1.0 = disabled)
    pub top_p: f64,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f64,
    /// End-of-sequence token ID (generation stops when this token is sampled)
    pub eos_token_id: Option<u32>,
    /// Minimum number of tokens before EOS is allowed (default: 2, matching Python)
    pub min_new_tokens: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 2048,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            eos_token_id: None,
            min_new_tokens: 2,
        }
    }
}

/// Sample next token from logits.
///
/// # Arguments
/// * `logits` - Logits tensor of shape [batch, vocab_size]
/// * `config` - Generation configuration
/// * `ctx` - Sampling context (owns RNG state)
///
/// # Returns
/// Token index as u32 (single batch only).
pub fn sample<B: Backend>(
    logits: Tensor<B, 2>,
    config: &GenerationConfig,
    ctx: &mut SamplingContext,
) -> u32 {
    // Transfer to CPU for sampling (sampling is a tiny operation)
    let [batch, vocab] = logits.dims();
    assert_eq!(batch, 1, "Only batch=1 supported for sampling");

    let mut logits_vec: Vec<f32> = logits.into_data().convert::<f32>().to_vec().unwrap();
    // Only use the first batch
    logits_vec.truncate(vocab);

    // Apply temperature
    if config.temperature != 1.0 && config.temperature > 0.0 {
        let inv_temp = 1.0 / config.temperature as f32;
        for v in &mut logits_vec {
            *v *= inv_temp;
        }
    }

    // Very low temperature → greedy
    if config.temperature < 0.01 {
        return argmax_vec(&logits_vec) as u32;
    }

    // Apply top-k filtering
    if config.top_k > 0 {
        top_k_filter_vec(&mut logits_vec, config.top_k);
    }

    // Apply top-p filtering
    if config.top_p < 1.0 && config.top_p > 0.0 {
        top_p_filter_vec(&mut logits_vec, config.top_p as f32);
    }

    // Softmax
    softmax_vec(&mut logits_vec);

    // Multinomial sample
    multinomial_sample_vec(&logits_vec, ctx)
}

/// Greedy sampling (argmax).
pub fn greedy_sample<B: Backend>(logits: Tensor<B, 2>) -> u32 {
    let vals: Vec<f32> = logits.into_data().convert::<f32>().to_vec().unwrap();
    argmax_vec(&vals) as u32
}

/// Apply repetition penalty to logits using a pre-built boolean mask.
///
/// The mask has `true` at positions of previously generated tokens.
pub fn apply_repetition_penalty_with_mask<B: Backend>(
    logits: Tensor<B, 2>,
    penalty_mask: &[bool],
    penalty: f64,
    device: &B::Device,
) -> Tensor<B, 2> {
    if (penalty - 1.0).abs() < 1e-9 {
        return logits;
    }

    let [batch, vocab] = logits.dims();
    let logits_vec: Vec<f32> = logits.into_data().convert::<f32>().to_vec().unwrap();
    let penalty_f32 = penalty as f32;

    let mut result = logits_vec;
    for b in 0..batch {
        for v in 0..vocab {
            let idx = b * vocab + v;
            if v < penalty_mask.len() && penalty_mask[v] {
                if result[idx] > 0.0 {
                    result[idx] /= penalty_f32;
                } else {
                    result[idx] *= penalty_f32;
                }
            }
        }
    }

    Tensor::<B, 1>::from_floats(result.as_slice(), device).reshape([batch, vocab])
}

// ── Vec-based helpers (CPU sampling path) ───────────────────────────────

fn argmax_vec(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn top_k_filter_vec(logits: &mut [f32], k: usize) {
    let k = k.min(logits.len());
    let mut sorted = logits.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn top_p_filter_vec(logits: &mut [f32], p: f32) {
    let vocab = logits.len();
    let mut indices: Vec<usize> = (0..vocab).collect();
    indices.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Softmax over sorted values
    let max_val = logits[indices[0]];
    let mut exp_sorted: Vec<f32> = indices
        .iter()
        .map(|&i| (logits[i] - max_val).exp())
        .collect();
    let sum: f32 = exp_sorted.iter().sum();
    for v in &mut exp_sorted {
        *v /= sum;
    }

    // Cumulative probability cutoff
    let mut cumsum = 0.0f32;
    let mut cutoff_idx = vocab;
    for (i, &prob) in exp_sorted.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Mask out tokens beyond cutoff
    let mut keep = vec![false; vocab];
    for &idx in &indices[..cutoff_idx] {
        keep[idx] = true;
    }
    for (i, k) in keep.iter().enumerate() {
        if !k {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

fn softmax_vec(logits: &mut [f32]) {
    let max_val = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

fn multinomial_sample_vec(probs: &[f32], ctx: &mut SamplingContext) -> u32 {
    let u = ctx.rand_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum > u {
            return i as u32;
        }
    }
    // Fallback: return last valid token
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 2048);
        assert!((config.temperature - 0.7).abs() < 1e-6);
        assert_eq!(config.top_k, 50);
    }

    #[test]
    fn test_greedy_sample() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[1.0f32, 2.0, 5.0, 1.0]], &device);
        let idx = greedy_sample(logits);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_sample_very_low_temperature() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[1.0f32, 10.0, 2.0, 1.0]], &device);
        let config = GenerationConfig {
            temperature: 0.001,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(Some(42));
        let idx = sample(logits, &config, &mut ctx);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_sample_normal_temperature() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[1.0f32, 1.0, 1.0, 1.0]], &device);
        let config = GenerationConfig::default();
        let mut ctx = SamplingContext::new(None);
        let idx = sample(logits, &config, &mut ctx);
        assert!(idx < 4);
    }

    #[test]
    fn test_seeded_deterministic() {
        let mut ctx1 = SamplingContext::new(Some(12345));
        let values1: Vec<f32> = (0..10).map(|_| ctx1.rand_f32()).collect();

        let mut ctx2 = SamplingContext::new(Some(12345));
        let values2: Vec<f32> = (0..10).map(|_| ctx2.rand_f32()).collect();

        for (a, b) in values1.iter().zip(values2.iter()) {
            assert!((a - b).abs() < 1e-9, "Seeded values should be identical");
        }
    }

    #[test]
    fn test_top_k_filter_keeps_top_values() {
        let mut logits = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        top_k_filter_vec(&mut logits, 3);
        assert!((logits[1] - 5.0).abs() < 1e-5);
        assert!((logits[4] - 4.0).abs() < 1e-5);
        assert!((logits[2] - 3.0).abs() < 1e-5);
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert!(logits[3].is_infinite() && logits[3] < 0.0);
    }

    #[test]
    fn test_reset() {
        let mut ctx = SamplingContext::new(Some(42));
        let _first = ctx.rand_f32();
        let second = ctx.rand_f32();

        ctx.reset(42);
        let _after_reset_first = ctx.rand_f32();
        let after_reset_second = ctx.rand_f32();

        assert!((after_reset_second - second).abs() < 1e-9);
    }

    #[test]
    fn test_repetition_penalty_no_penalty() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[1.0f32, 2.0, 3.0]], &device);
        let mask = vec![true, false, false];
        let result = apply_repetition_penalty_with_mask(logits, &mask, 1.0, &device);
        let vals: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
        assert!((vals[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_repetition_penalty_with_penalty() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[2.0f32, 3.0, 4.0]], &device);
        let mask = vec![true, false, false];
        let result = apply_repetition_penalty_with_mask(logits, &mask, 2.0, &device);
        let vals: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-5); // 2.0 / 2.0
        assert!((vals[1] - 3.0).abs() < 1e-5);
        assert!((vals[2] - 4.0).abs() < 1e-5);
    }
}
