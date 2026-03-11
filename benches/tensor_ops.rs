//! Micro-benchmarks for tensor operations (codes_to_tensor).
//!
//! Run with: `cargo bench -- tensor_ops`

use burn::backend::NdArray;
use burn::prelude::*;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

type B = NdArray;

/// Build `n_frames` dummy codec frames (16 codebooks each).
fn make_frames(n_frames: usize) -> Vec<Vec<u32>> {
    (0..n_frames)
        .map(|f| (0..16).map(|q| ((f * 16 + q) % 3072) as u32).collect())
        .collect()
}

/// Burn equivalent of codes_to_tensor: frame codes → [1, 16, T] Int tensor.
fn codes_to_tensor(codes: &[Vec<u32>]) -> Tensor<B, 3, Int> {
    let device: <B as Backend>::Device = Default::default();
    let num_frames = codes.len();
    let mut data = vec![0i32; 16 * num_frames];
    for (frame, frame_codes) in codes.iter().enumerate() {
        for (q, &code) in frame_codes.iter().enumerate() {
            data[q * num_frames + frame] = code as i32;
        }
    }
    Tensor::<B, 1, Int>::from_ints(data.as_slice(), &device).reshape([1, 16, num_frames])
}

fn bench_codes_to_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("codes_to_tensor");

    // 12 frames ≈ 1s, 60 frames ≈ 5s, 240 frames ≈ 20s of audio at 12 Hz
    for n_frames in [12, 60, 240] {
        let frames = make_frames(n_frames);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_frames}_frames")),
            &n_frames,
            |b, _| {
                b.iter(|| codes_to_tensor(black_box(&frames)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_codes_to_tensor);
criterion_main!(benches);
