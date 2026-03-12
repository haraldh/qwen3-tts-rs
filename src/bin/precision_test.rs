//! Precision analysis: where does BF16 diverge from F32 across transformer layers?
//!
//! Builds a stack of DecoderLayers with identical random weights, runs the same
//! input through three paths, and measures divergence at each layer:
//!
//!   A) Pure F32:  Rocm<f32> backend, everything F32
//!   B) Pure BF16: Rocm<half::bf16> backend, everything BF16
//!   C) Mixed:     Rocm<half::bf16> backend, F32 residual stream (current implementation)
//!
//! Also tests individual components to isolate precision-sensitive ops:
//!   - RMSNorm alone
//!   - Attention (QKV projections + dot-product + softmax)
//!   - MLP (SwiGLU)
//!   - Softmax alone
//!   - Residual accumulation over N additions
//!
//! Usage:
//!   cargo run --release --features rocm --bin precision_test 2>&1 > /tmp/precision_test.log

use burn::nn::{RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::DType;

// 0.6B model dimensions
const HIDDEN: usize = 1024;
const INTERMEDIATE: usize = 3072;
const NUM_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const RMS_EPS: f64 = 1e-6;
const NUM_LAYERS: usize = 28;
const SEQ_LEN: usize = 1;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Extract tensor values as Vec<f32> regardless of backend dtype.
fn to_f32_vec<B: Backend, const D: usize>(t: &Tensor<B, D>) -> Vec<f32> {
    let t_f32 = t.clone().cast(DType::F32);
    let data = t_f32.into_data();
    data.to_vec::<f32>().expect("f32 conversion")
}

/// Compute max absolute error between two f32 vectors.
fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compute mean absolute error.
fn mean_abs_error(a: &[f32], b: &[f32]) -> f32 {
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

/// Compute RMS of a vector (for norm comparison).
fn rms(a: &[f32]) -> f32 {
    let sum_sq: f32 = a.iter().map(|x| x * x).sum();
    (sum_sq / a.len() as f32).sqrt()
}

/// Compute relative error (MAE / RMS of reference).
fn relative_error(reference: &[f32], test: &[f32]) -> f32 {
    let mae = mean_abs_error(reference, test);
    let ref_rms = rms(reference);
    if ref_rms > 0.0 {
        mae / ref_rms
    } else {
        mae
    }
}

fn print_comparison(label: &str, reference: &[f32], test: &[f32]) {
    let mae = mean_abs_error(reference, test);
    let max_err = max_abs_error(reference, test);
    let rel_err = relative_error(reference, test);
    let ref_rms = rms(reference);
    let test_rms = rms(test);
    println!(
        "  {label:<45} MAE={mae:.2e}  MaxErr={max_err:.2e}  RelErr={rel_err:.2e}  RMS(ref)={ref_rms:.4}  RMS(test)={test_rms:.4}"
    );
}

// ── Component tests ─────────────────────────────────────────────────────────

/// Test RMSNorm precision: F32 vs BF16 input
fn test_rmsnorm_precision(device: &burn::backend::rocm::RocmDevice) {
    println!("\n=== RMSNorm Precision ===");

    type F32 = burn::backend::Rocm;
    type BF16 = burn::backend::Rocm<half::bf16>;

    let norm_f32: RmsNorm<F32> = RmsNormConfig::new(HIDDEN)
        .with_epsilon(RMS_EPS)
        .init(device);
    let norm_bf16: RmsNorm<BF16> = RmsNormConfig::new(HIDDEN)
        .with_epsilon(RMS_EPS)
        .init(device);

    // Create input with realistic hidden state magnitude
    let input_f32 = Tensor::<F32, 3>::random(
        [1, SEQ_LEN, HIDDEN],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    let out_f32 = norm_f32.forward(input_f32.clone());
    let ref_vals = to_f32_vec(&out_f32);

    // BF16: cast input, run norm, compare
    let input_bf16: Tensor<BF16, 3> = Tensor::from_data(
        input_f32.clone().into_data().convert::<half::bf16>(),
        device,
    );
    let out_bf16 = norm_bf16.forward(input_bf16);
    let bf16_vals = to_f32_vec(&out_bf16);

    // F32-on-BF16-backend: cast to F32, run norm (but norm weights are BF16)
    let input_f32_on_bf16: Tensor<BF16, 3> = Tensor::from_data(
        input_f32.into_data().convert::<half::bf16>(),
        device,
    );
    let out_f32_norm = norm_bf16.forward(input_f32_on_bf16.cast(DType::F32).cast(DType::BF16));
    let f32_norm_vals = to_f32_vec(&out_f32_norm);

    print_comparison("BF16 norm vs F32 norm", &ref_vals, &bf16_vals);
    print_comparison("BF16(cast from F32) vs F32", &ref_vals, &f32_norm_vals);
}

/// Test softmax precision
fn test_softmax_precision(device: &burn::backend::rocm::RocmDevice) {
    println!("\n=== Softmax Precision ===");

    type F32 = burn::backend::Rocm;
    type BF16 = burn::backend::Rocm<half::bf16>;

    // Simulate attention scores: [batch, heads, 1, seq_kv]
    for seq_kv in [16, 64, 256, 512] {
        let scores_f32 = Tensor::<F32, 4>::random(
            [1, NUM_HEADS, 1, seq_kv],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let out_f32 = burn::tensor::activation::softmax(scores_f32.clone(), 3);
        let ref_vals = to_f32_vec(&out_f32);

        let scores_bf16: Tensor<BF16, 4> = Tensor::from_data(
            scores_f32.clone().into_data().convert::<half::bf16>(),
            device,
        );
        let out_bf16 = burn::tensor::activation::softmax(scores_bf16, 3);
        let bf16_vals = to_f32_vec(&out_bf16);

        // F32 softmax on BF16 backend
        let scores_bf16_cast: Tensor<BF16, 4> = Tensor::from_data(
            scores_f32.into_data().convert::<half::bf16>(),
            device,
        );
        let out_f32_softmax =
            burn::tensor::activation::softmax(scores_bf16_cast.cast(DType::F32), 3);
        let f32_softmax_vals = to_f32_vec(&out_f32_softmax);

        let label_bf16 = format!("BF16 softmax (seq_kv={seq_kv})");
        let label_f32 = format!("F32 softmax on BF16 backend (seq_kv={seq_kv})");
        print_comparison(&label_bf16, &ref_vals, &bf16_vals);
        print_comparison(&label_f32, &ref_vals, &f32_softmax_vals);
    }
}

/// Test matmul precision (simulates QK^T)
fn test_matmul_precision(device: &burn::backend::rocm::RocmDevice) {
    println!("\n=== Matmul (QK^T) Precision ===");

    type F32 = burn::backend::Rocm;
    type BF16 = burn::backend::Rocm<half::bf16>;

    for seq_kv in [16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512] {
        // Q: [1, heads, 1, head_dim], K: [1, heads, seq_kv, head_dim]
        let q_f32 = Tensor::<F32, 4>::random(
            [1, NUM_HEADS, 1, HEAD_DIM],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        );
        let k_f32 = Tensor::<F32, 4>::random(
            [1, NUM_HEADS, seq_kv, HEAD_DIM],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        );

        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores_f32 = q_f32.clone().matmul(k_f32.clone().swap_dims(2, 3)) * scale;
        let ref_vals = to_f32_vec(&scores_f32);

        let q_bf16: Tensor<BF16, 4> =
            Tensor::from_data(q_f32.into_data().convert::<half::bf16>(), device);
        let k_bf16: Tensor<BF16, 4> =
            Tensor::from_data(k_f32.into_data().convert::<half::bf16>(), device);

        let scores_bf16 = q_bf16.clone().matmul(k_bf16.clone().swap_dims(2, 3)) * scale;
        let bf16_vals = to_f32_vec(&scores_bf16);

        // F32 matmul on BF16 backend
        let scores_f32_on_bf16 = q_bf16
            .cast(DType::F32)
            .matmul(k_bf16.cast(DType::F32).swap_dims(2, 3))
            * scale;
        let f32_matmul_vals = to_f32_vec(&scores_f32_on_bf16);

        let label = format!("BF16 QK^T (seq_kv={seq_kv})");
        let label2 = format!("F32 QK^T on BF16 backend (seq_kv={seq_kv})");
        print_comparison(&label, &ref_vals, &bf16_vals);
        print_comparison(&label2, &ref_vals, &f32_matmul_vals);
    }
}

/// Test residual accumulation: how much does F32 vs BF16 diverge over N additions?
fn test_residual_accumulation(device: &burn::backend::rocm::RocmDevice) {
    println!("\n=== Residual Accumulation ({NUM_LAYERS} layers) ===");
    println!("  Simulates adding N random perturbations to a hidden state");

    type F32 = burn::backend::Rocm;
    type BF16 = burn::backend::Rocm<half::bf16>;

    let hidden_f32 = Tensor::<F32, 3>::random(
        [1, SEQ_LEN, HIDDEN],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    // Generate perturbations (simulate attention/MLP outputs)
    let perturbations_f32: Vec<Tensor<F32, 3>> = (0..NUM_LAYERS)
        .map(|_| {
            Tensor::random(
                [1, SEQ_LEN, HIDDEN],
                burn::tensor::Distribution::Normal(0.0, 0.5),
                device,
            )
        })
        .collect();

    // F32 accumulation (reference)
    let mut accum_f32 = hidden_f32.clone();
    for p in &perturbations_f32 {
        accum_f32 = accum_f32 + p.clone();
    }
    let ref_vals = to_f32_vec(&accum_f32);

    // BF16 accumulation
    let mut accum_bf16: Tensor<BF16, 3> = Tensor::from_data(
        hidden_f32.clone().into_data().convert::<half::bf16>(),
        device,
    );
    for p in &perturbations_f32 {
        let p_bf16: Tensor<BF16, 3> =
            Tensor::from_data(p.clone().into_data().convert::<half::bf16>(), device);
        accum_bf16 = accum_bf16 + p_bf16;
    }
    let bf16_vals = to_f32_vec(&accum_bf16);

    // F32 accumulation on BF16 backend (what our mixed-precision does)
    let mut accum_mixed: Tensor<BF16, 3> = Tensor::from_data(
        hidden_f32.into_data().convert::<half::bf16>(),
        device,
    );
    accum_mixed = accum_mixed.cast(DType::F32); // promote to F32
    for p in &perturbations_f32 {
        let p_bf16: Tensor<BF16, 3> =
            Tensor::from_data(p.clone().into_data().convert::<half::bf16>(), device);
        // Simulate: perturbation arrives in BF16, cast to F32 for add
        accum_mixed = accum_mixed + p_bf16.cast(DType::F32);
    }
    let mixed_vals = to_f32_vec(&accum_mixed);

    print_comparison("BF16 accumulation vs F32", &ref_vals, &bf16_vals);
    print_comparison("F32-on-BF16 accumulation vs F32", &ref_vals, &mixed_vals);

    // Show per-layer divergence for BF16
    println!("\n  Per-layer BF16 divergence (cumulative):");
    let mut accum_f32_step = Tensor::<F32, 3>::random(
        [1, SEQ_LEN, HIDDEN],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let mut accum_bf16_step: Tensor<BF16, 3> = Tensor::from_data(
        accum_f32_step.clone().into_data().convert::<half::bf16>(),
        device,
    );
    let mut accum_mixed_step: Tensor<BF16, 3> = Tensor::from_data(
        accum_f32_step.clone().into_data().convert::<half::bf16>(),
        device,
    );
    accum_mixed_step = accum_mixed_step.cast(DType::F32);

    for (i, p) in perturbations_f32.iter().enumerate() {
        accum_f32_step = accum_f32_step + p.clone();
        let p_bf16: Tensor<BF16, 3> =
            Tensor::from_data(p.clone().into_data().convert::<half::bf16>(), device);
        accum_bf16_step = accum_bf16_step.clone() + p_bf16.clone();
        accum_mixed_step = accum_mixed_step + p_bf16.cast(DType::F32);

        if i == 0 || i == 4 || i == 9 || i == 13 || i == 19 || i == 27 {
            let ref_v = to_f32_vec(&accum_f32_step);
            let bf16_v = to_f32_vec(&accum_bf16_step);
            let mixed_v = to_f32_vec(&accum_mixed_step);
            let bf16_rel = relative_error(&ref_v, &bf16_v);
            let mixed_rel = relative_error(&ref_v, &mixed_v);
            println!(
                "    Layer {:>2}: BF16 RelErr={bf16_rel:.2e}  Mixed RelErr={mixed_rel:.2e}",
                i + 1
            );
        }
    }
}

// ── Full decoder layer test ─────────────────────────────────────────────────

/// Test a single DecoderLayer: F32 vs BF16 vs Mixed
fn test_single_layer(device: &burn::backend::rocm::RocmDevice) {
    println!("\n=== Single DecoderLayer Precision ===");

    use qwen3_tts::burn_models::transformer::{DecoderLayerConfig, RoPEType, RotaryEmbedding};


    type F32 = burn::backend::Rocm;
    type BF16 = burn::backend::Rocm<half::bf16>;

    let config = DecoderLayerConfig::new(HIDDEN, INTERMEDIATE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, RMS_EPS);

    let layer_f32: qwen3_tts::burn_models::transformer::DecoderLayer<F32> = config.init(device);
    let layer_bf16: qwen3_tts::burn_models::transformer::DecoderLayer<BF16> = config.init(device);

    let rope_f32 = RoPEType::Standard(RotaryEmbedding::<F32>::new(HEAD_DIM, 512, 10000.0, device));
    let rope_bf16 = RoPEType::Standard(RotaryEmbedding::<BF16>::new(HEAD_DIM, 512, 10000.0, device));

    let input_f32 = Tensor::<F32, 3>::random(
        [1, 8, HIDDEN],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    // F32 reference (no cache for simplicity)
    let out_f32 = layer_f32.forward(input_f32.clone(), &rope_f32, true, None, 0);
    let ref_vals = to_f32_vec(&out_f32);

    // Pure BF16
    let input_bf16: Tensor<BF16, 3> = Tensor::from_data(
        input_f32.clone().into_data().convert::<half::bf16>(),
        device,
    );
    let out_bf16 = layer_bf16.forward(input_bf16, &rope_bf16, true, None, 0);
    let bf16_vals = to_f32_vec(&out_bf16);

    // Mixed: input cast to F32 on BF16 backend (triggers mixed path)
    let input_mixed: Tensor<BF16, 3> = Tensor::from_data(
        input_f32.into_data().convert::<half::bf16>(),
        device,
    );
    let input_mixed = input_mixed.cast(DType::F32);
    let out_mixed = layer_bf16.forward(input_mixed, &rope_bf16, true, None, 0);
    let mixed_vals = to_f32_vec(&out_mixed);

    print_comparison("BF16 layer vs F32 layer", &ref_vals, &bf16_vals);
    print_comparison("Mixed layer vs F32 layer", &ref_vals, &mixed_vals);

    // Note: weights differ between F32/BF16 layers (random init), so absolute
    // error isn't meaningful. What matters is the relative magnitude.
    println!("  (Note: different random weights — compare BF16 vs Mixed relative errors)");
}

/// Test stacked layers: accumulation of error across N layers
fn test_stacked_layers(device: &burn::backend::rocm::RocmDevice) {
    println!("\n=== Stacked DecoderLayers ({NUM_LAYERS} layers) ===");
    println!("  Same weights for BF16 and Mixed paths; F32 is separate (different random init)");

    use qwen3_tts::burn_models::transformer::{DecoderLayerConfig, RoPEType, RotaryEmbedding};

    type BF16 = burn::backend::Rocm<half::bf16>;

    let config = DecoderLayerConfig::new(HIDDEN, INTERMEDIATE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, RMS_EPS);

    let layers: Vec<qwen3_tts::burn_models::transformer::DecoderLayer<BF16>> =
        (0..NUM_LAYERS).map(|_| config.init(device)).collect();

    let rope = RoPEType::Standard(RotaryEmbedding::<BF16>::new(HEAD_DIM, 512, 10000.0, device));

    let input_bf16 = Tensor::<BF16, 3>::random(
        [1, SEQ_LEN, HIDDEN],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    // Pure BF16 path
    let mut hidden_bf16 = input_bf16.clone();
    // Mixed path: same input, cast to F32
    let mut hidden_mixed = input_bf16.clone().cast(DType::F32);

    println!("  {:>5}  {:>12}  {:>12}  {:>12}  {:>12}", "Layer", "BF16 RMS", "Mixed RMS", "MAE(B,M)", "RelErr(B,M)");
    println!("  {}", "-".repeat(65));

    for (i, layer) in layers.iter().enumerate() {
        hidden_bf16 = layer.forward(hidden_bf16, &rope, false, None, 0);
        hidden_mixed = layer.forward(hidden_mixed, &rope, false, None, 0);

        let bf16_v = to_f32_vec(&hidden_bf16);
        let mixed_v = to_f32_vec(&hidden_mixed);
        let bf16_rms = rms(&bf16_v);
        let mixed_rms = rms(&mixed_v);
        let mae = mean_abs_error(&bf16_v, &mixed_v);
        let rel = relative_error(&bf16_v, &mixed_v);

        if i < 5 || i % 5 == 4 || i == NUM_LAYERS - 1 {
            println!(
                "  {:>5}  {:>12.6}  {:>12.6}  {:>12.2e}  {:>12.2e}",
                i + 1,
                bf16_rms,
                mixed_rms,
                mae,
                rel,
            );
        }
    }

    // Final comparison
    let bf16_final = to_f32_vec(&hidden_bf16);
    let mixed_final = to_f32_vec(&hidden_mixed);
    println!("\n  Final hidden state comparison (BF16 vs Mixed, same weights):");
    print_comparison("Pure BF16 vs F32-residual Mixed", &mixed_final, &bf16_final);
}

// ── Matmul shape sweep ──────────────────────────────────────────────────────

/// Systematic matmul shape sweep to isolate the MMA bug.
/// Tests 2D and 4D matmuls with varying M, N, K dimensions.
fn test_matmul_shape_sweep(device: &burn::backend::rocm::RocmDevice) {
    println!("\n=== Matmul Shape Sweep (isolating MMA bug) ===");

    type F32 = burn::backend::Rocm;
    type BF16 = burn::backend::Rocm<half::bf16>;

    // Test 1: 2D matmul [M, K] @ [K, N] — is the bug in 4D batching or raw matmul?
    println!("\n  --- 2D matmul [M, K] @ [K, N] ---");
    println!("  {:>4} {:>4} {:>4}  {:>12} {:>12}", "M", "K", "N", "BF16 RelErr", "F32-on-BF16");
    for m in [1, 2, 4, 8, 16, 32] {
        for n in [64, 128, 192, 256, 512] {
            let k = 128;
            let a_f32 = Tensor::<F32, 2>::random(
                [m, k],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                device,
            );
            let b_f32 = Tensor::<F32, 2>::random(
                [k, n],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                device,
            );
            let ref_out = a_f32.clone().matmul(b_f32.clone());
            let ref_vals = to_f32_vec(&ref_out);

            let a_bf16: Tensor<BF16, 2> =
                Tensor::from_data(a_f32.clone().into_data().convert::<half::bf16>(), device);
            let b_bf16: Tensor<BF16, 2> =
                Tensor::from_data(b_f32.clone().into_data().convert::<half::bf16>(), device);

            let out_bf16 = a_bf16.clone().matmul(b_bf16.clone());
            let bf16_vals = to_f32_vec(&out_bf16);
            let bf16_rel = relative_error(&ref_vals, &bf16_vals);

            let out_f32_on_bf16 = a_bf16.cast(DType::F32).matmul(b_bf16.cast(DType::F32));
            let f32_vals = to_f32_vec(&out_f32_on_bf16);
            let f32_rel = relative_error(&ref_vals, &f32_vals);

            let flag = if bf16_rel > 0.01 { " *** BUG" } else { "" };
            println!(
                "  {:>4} {:>4} {:>4}  {:>12.2e} {:>12.2e}{}",
                m, k, n, bf16_rel, f32_rel, flag
            );
        }
    }

    // Test 2: 4D batched matmul [B, H, M, K] @ [B, H, K, N] (attention-like)
    println!("\n  --- 4D batched matmul [1, 16, M, K] @ [1, 16, K, N] ---");
    println!("  {:>4} {:>4} {:>4}  {:>12} {:>12}", "M", "K", "N", "BF16 RelErr", "F32-on-BF16");
    for m in [1, 2, 4, 8, 16] {
        for n in [64, 128, 192, 256] {
            let k = 128;
            let a_f32 = Tensor::<F32, 4>::random(
                [1, NUM_HEADS, m, k],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                device,
            );
            let b_f32 = Tensor::<F32, 4>::random(
                [1, NUM_HEADS, k, n],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                device,
            );
            let ref_out = a_f32.clone().matmul(b_f32.clone());
            let ref_vals = to_f32_vec(&ref_out);

            let a_bf16: Tensor<BF16, 4> =
                Tensor::from_data(a_f32.clone().into_data().convert::<half::bf16>(), device);
            let b_bf16: Tensor<BF16, 4> =
                Tensor::from_data(b_f32.clone().into_data().convert::<half::bf16>(), device);

            let out_bf16 = a_bf16.clone().matmul(b_bf16.clone());
            let bf16_vals = to_f32_vec(&out_bf16);
            let bf16_rel = relative_error(&ref_vals, &bf16_vals);

            let out_f32_on_bf16 = a_bf16.cast(DType::F32).matmul(b_bf16.cast(DType::F32));
            let f32_vals = to_f32_vec(&out_f32_on_bf16);
            let f32_rel = relative_error(&ref_vals, &f32_vals);

            let flag = if bf16_rel > 0.01 { " *** BUG" } else { "" };
            println!(
                "  {:>4} {:>4} {:>4}  {:>12.2e} {:>12.2e}{}",
                m, k, n, bf16_rel, f32_rel, flag
            );
        }
    }

    // Test 3: Vary K to see if the bug depends on inner dimension
    println!("\n  --- 2D matmul M=1 varying K ---");
    println!("  {:>4} {:>4} {:>4}  {:>12}", "M", "K", "N", "BF16 RelErr");
    for k in [16, 32, 64, 128, 256, 512, 1024] {
        let n = 256;
        let a_f32 = Tensor::<F32, 2>::random(
            [1, k],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        );
        let b_f32 = Tensor::<F32, 2>::random(
            [k, n],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        );
        let ref_out = a_f32.clone().matmul(b_f32.clone());
        let ref_vals = to_f32_vec(&ref_out);

        let a_bf16: Tensor<BF16, 2> =
            Tensor::from_data(a_f32.into_data().convert::<half::bf16>(), device);
        let b_bf16: Tensor<BF16, 2> =
            Tensor::from_data(b_f32.into_data().convert::<half::bf16>(), device);

        let out_bf16 = a_bf16.matmul(b_bf16);
        let bf16_vals = to_f32_vec(&out_bf16);
        let bf16_rel = relative_error(&ref_vals, &bf16_vals);

        let flag = if bf16_rel > 0.01 { " *** BUG" } else { "" };
        println!("  {:>4} {:>4} {:>4}  {:>12.2e}{}", 1, k, n, bf16_rel, flag);
    }

    // Test 4: burn::tensor::module::attention (what actually runs during inference)
    println!("\n  --- burn attention module (Q[1,16,1,128] @ K/V) ---");
    println!("  {:>6}  {:>12} {:>12}", "seq_kv", "BF16 RelErr", "F32-on-BF16");
    for seq_kv in [16, 64, 128, 192, 256, 512] {
        let q_f32 = Tensor::<F32, 4>::random(
            [1, NUM_HEADS, 1, HEAD_DIM],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        );
        let k_f32 = Tensor::<F32, 4>::random(
            [1, NUM_HEADS, seq_kv, HEAD_DIM],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        );
        let v_f32 = Tensor::<F32, 4>::random(
            [1, NUM_HEADS, seq_kv, HEAD_DIM],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        );

        let opts = burn::tensor::ops::AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: false,
        };
        let ref_out = burn::tensor::module::attention(
            q_f32.clone(), k_f32.clone(), v_f32.clone(), None, None, opts.clone(),
        );
        let ref_vals = to_f32_vec(&ref_out);

        let q_bf16: Tensor<BF16, 4> =
            Tensor::from_data(q_f32.clone().into_data().convert::<half::bf16>(), device);
        let k_bf16: Tensor<BF16, 4> =
            Tensor::from_data(k_f32.clone().into_data().convert::<half::bf16>(), device);
        let v_bf16: Tensor<BF16, 4> =
            Tensor::from_data(v_f32.clone().into_data().convert::<half::bf16>(), device);

        let out_bf16 = burn::tensor::module::attention(
            q_bf16.clone(), k_bf16.clone(), v_bf16.clone(), None, None, opts.clone(),
        );
        let bf16_vals = to_f32_vec(&out_bf16);
        let bf16_rel = relative_error(&ref_vals, &bf16_vals);

        let out_f32_attn = burn::tensor::module::attention(
            q_bf16.cast(DType::F32), k_bf16.cast(DType::F32), v_bf16.cast(DType::F32),
            None, None, opts,
        );
        let f32_vals = to_f32_vec(&out_f32_attn);
        let f32_rel = relative_error(&ref_vals, &f32_vals);

        let flag = if bf16_rel > 0.01 { " *** BUG" } else { "" };
        println!("  {:>6}  {:>12.2e} {:>12.2e}{}", seq_kv, bf16_rel, f32_rel, flag);
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║         BF16 Precision Analysis for ROCm (gfx1151)         ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Model: 0.6B (hidden=1024, heads=16, kv_heads=2, 28 layers)║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        let device = burn::backend::rocm::RocmDevice::default();

        // 1. Component-level tests
        test_rmsnorm_precision(&device);
        test_softmax_precision(&device);
        test_matmul_precision(&device);
        test_residual_accumulation(&device);

        // 2. Matmul shape sweep (isolate the bug)
        test_matmul_shape_sweep(&device);

        // 3. Full layer tests (skip for now — focus on matmul bug)
        // test_single_layer(&device);
        // test_stacked_layers(&device);

        println!("\n=== Summary ===");
        println!("If Mixed RelErr << BF16 RelErr at layer 28, F32-residual helps.");
        println!("If both diverge similarly, the error source is within the layer (attention/MLP),");
        println!("not the residual accumulation. In that case, F32 attention may be needed.");
    }

    #[cfg(not(all(feature = "rocm", not(feature = "cuda"))))]
    {
        println!("This test requires --features rocm (without cuda)");
    }
}
