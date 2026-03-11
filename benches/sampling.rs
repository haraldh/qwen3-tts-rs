//! Micro-benchmarks for sampling and logit processing functions.
//!
//! Run with: `cargo bench -- sampling`

use burn::backend::NdArray;
use burn::prelude::*;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use qwen3_tts::burn_models::sampling::{
    apply_repetition_penalty_with_mask, sample, GenerationConfig, SamplingContext,
};
use qwen3_tts::burn_models::tts;
use std::hint::black_box;

type B = NdArray;

fn random_logits(vocab_size: usize) -> Tensor<B, 2> {
    let device: <B as Backend>::Device = Default::default();
    let data: Vec<f32> = (0..vocab_size)
        .map(|i| (i as f32 * 0.1).sin() * 5.0)
        .collect();
    Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape([1, vocab_size])
}

fn bench_sample_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_top_k");

    for vocab_size in [3072, 32000] {
        let logits = random_logits(vocab_size);
        let config = GenerationConfig {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0, // disable top-p to isolate top-k
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("vocab_{vocab_size}")),
            &vocab_size,
            |b, _| {
                let mut ctx = SamplingContext::new(Some(42));
                b.iter(|| sample::<B>(black_box(logits.clone()), black_box(&config), &mut ctx));
            },
        );
    }
    group.finish();
}

fn bench_sample_top_p(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_top_p");

    for p in [0.5, 0.9, 0.95] {
        let logits = random_logits(3072);
        let config = GenerationConfig {
            temperature: 0.9,
            top_k: 0, // disable top-k to isolate top-p
            top_p: p,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(format!("p_{p}")), &p, |b, _| {
            let mut ctx = SamplingContext::new(Some(42));
            b.iter(|| sample::<B>(black_box(logits.clone()), black_box(&config), &mut ctx));
        });
    }
    group.finish();
}

fn bench_repetition_penalty(c: &mut Criterion) {
    let device: <B as Backend>::Device = Default::default();
    let mut group = c.benchmark_group("repetition_penalty");

    for (penalty, n_prev) in [(1.05, 0), (1.05, 100), (1.05, 500), (1.5, 100), (1.5, 500)] {
        let logits = random_logits(3072);
        let mut penalty_mask = vec![false; 3072];
        for i in 0..n_prev {
            penalty_mask[i % 3072] = true;
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("pen_{penalty}_prev_{n_prev}")),
            &(penalty, n_prev),
            |b, _| {
                b.iter(|| {
                    apply_repetition_penalty_with_mask::<B>(
                        black_box(logits.clone()),
                        black_box(&penalty_mask),
                        penalty,
                        &device,
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_token_suppression(c: &mut Criterion) {
    let device: <B as Backend>::Device = Default::default();
    let logits = random_logits(3072);

    let mask = tts::build_suppression_mask::<B>(3072, 2150, &device);

    c.bench_function("token_suppression_codec", |b| {
        b.iter(|| tts::apply_token_suppression_with_mask(black_box(logits.clone()), &mask));
    });
}

criterion_group!(
    benches,
    bench_sample_top_k,
    bench_sample_top_p,
    bench_repetition_penalty,
    bench_token_suppression,
);
criterion_main!(benches);
