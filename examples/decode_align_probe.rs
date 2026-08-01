//! Probe: where does the ROCm decoder's sample shortfall land?
//!
//! On ROCm a decode of L frames comes out shorter than L * total_upsample.
//! Chunked streaming slices at frame boundaries, so it matters whether the
//! missing samples are at the head or the tail. This decodes the same codes at
//! two lengths and reports the lag that best aligns the shorter decode with the
//! longer one — lag 0 means the shortfall is at the tail.
//!
//! Throwaway diagnostic, not part of the library:
//!   cargo run --release --features rocm,cli --example decode_align_probe -- <model-dir>

#[cfg(feature = "cuda")]
type SelectedBackend = burn::backend::Cuda;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
type SelectedBackend = burn::backend::Rocm<half::bf16>;
#[cfg(all(feature = "wgpu", not(any(feature = "cuda", feature = "rocm"))))]
type SelectedBackend = burn::backend::Wgpu;
#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "wgpu")))]
type SelectedBackend = burn::backend::NdArray;

const TAIL_PAD: usize = 1200; // decode_codes' trailing silence

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test_data/models/0.6B-CustomVoice".to_string());

    let device = Default::default();
    let model = qwen3_tts::Qwen3TTS::<SelectedBackend>::from_pretrained(&model_dir, device)?;

    // Deterministic pseudo-random codes: content is irrelevant, alignment is not.
    let mut state = 12345u64;
    let mut rnd = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as u32
    };
    let codes: Vec<Vec<u32>> = (0..80)
        .map(|_| (0..16).map(|_| rnd() % 2048).collect())
        .collect();

    for len in [2usize, 3, 10, 20, 40, 80] {
        let audio = model.decode_codes(&codes[..len])?;
        let n = audio.samples.len() - TAIL_PAD;
        println!(
            "  {len:3} frames -> {n:7} samples   ideal {:7}   shortfall {:5}",
            len * 1920,
            len as i64 * 1920 - n as i64
        );
    }

    // Alignment: 20-frame decode vs 80-frame decode.
    let short = model.decode_codes(&codes[..20])?;
    let long = model.decode_codes(&codes[..80])?;
    let a = &short.samples[..short.samples.len() - TAIL_PAD];
    let b = &long.samples[..long.samples.len() - TAIL_PAD];

    let mut best = (0i64, f64::INFINITY);
    for lag in -2400i64..=2400 {
        let mut err = 0.0f64;
        let mut count = 0usize;
        for i in 0..a.len() {
            let j = i as i64 + lag;
            if j < 0 || j as usize >= b.len() {
                continue;
            }
            let d = (a[i] - b[j as usize]) as f64;
            err += d * d;
            count += 1;
        }
        if count > a.len() / 2 {
            let rms = (err / count as f64).sqrt();
            if rms < best.1 {
                best = (lag, rms);
            }
        }
    }
    let rms_signal =
        (a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / a.len() as f64).sqrt();
    println!(
        "\n  best lag = {} samples, RMS error {:.6} (signal RMS {:.5})",
        best.0, best.1, rms_signal
    );
    println!("  lag 0 => shortfall is at the tail; negative => samples missing at the head");
    Ok(())
}
