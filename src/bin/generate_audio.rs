//! CLI tool for generating audio with Qwen3-TTS
//!
//! Uses the Burn-based TTS pipeline with backend selection via feature flags:
//!   - `cpu` (default): NdArray backend
//!   - `cuda`: NVIDIA GPU via CubeCL
//!   - `rocm`: AMD GPU via CubeCL
//!   - `wgpu`: Cross-platform GPU via WebGPU/Vulkan
//!
//! Usage:
//!     cargo run --features cli --bin generate_audio -- --text "Hello" --seed 42
//!     cargo run --features cli,cuda --bin generate_audio -- --model-dir <path> --text "Hello"

use anyhow::Result;
use clap::Parser;
use std::path::Path;

use qwen3_tts::{AudioBuffer, Language, ModelType, Speaker, SynthesisOptions, VoiceClonePrompt};

// ── Backend selection ────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
type SelectedBackend = burn::backend::Cuda;

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
type SelectedBackend = burn::backend::Rocm<half::bf16>;

#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "rocm")))]
type SelectedBackend = burn::backend::Wgpu;

#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "wgpu")))]
type SelectedBackend = burn::backend::NdArray;

/// Generate speech audio from text using Qwen3-TTS
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text to synthesize
    #[arg(short, long, default_value = "Hello")]
    text: String,

    /// Random seed for reproducible generation
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Maximum number of frames to generate (default: 2048, ~164s). Generation
    /// stops early when the model emits an end-of-sequence token.
    #[arg(short, long, default_value_t = 2048)]
    frames: usize,

    /// Maximum duration in seconds (overrides --frames if specified).
    #[arg(short, long)]
    duration: Option<f64>,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.3)]
    temperature: f64,

    /// Top-k sampling parameter
    #[arg(long, default_value_t = 20)]
    top_k: usize,

    /// Top-p (nucleus) sampling parameter
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Repetition penalty (1.0 = disabled)
    #[arg(long, default_value_t = 1.2)]
    repetition_penalty: f64,

    /// Model directory containing model.safetensors
    #[arg(short, long, default_value = "test_data/model")]
    model_dir: String,

    /// Tokenizer directory (defaults to model_dir)
    #[arg(long)]
    tokenizer_dir: Option<String>,

    /// Speaker name for CustomVoice (ryan, serena, vivian, aiden, etc.)
    #[arg(long, default_value = "ryan")]
    speaker: String,

    /// Language for TTS (english, chinese, japanese, etc.)
    #[arg(long, default_value = "english")]
    language: String,

    /// Voice description for VoiceDesign model (e.g. "A cheerful young female voice")
    #[arg(long)]
    instruct: Option<String>,

    /// Reference audio WAV for voice cloning (Base models only)
    #[arg(long)]
    ref_audio: Option<String>,

    /// Reference text transcript (enables ICL voice cloning when used with --ref-audio)
    #[arg(long)]
    ref_text: Option<String>,

    /// Output WAV file path (default: output.wav)
    #[arg(short, long, default_value = "output.wav")]
    output: String,

    /// Use streaming mode (processes and outputs audio incrementally)
    #[arg(long)]
    streaming: bool,

    /// Frames per streaming chunk (default: 10 = ~800ms)
    #[arg(long, default_value_t = 10)]
    chunk_frames: usize,
}

fn backend_name() -> &'static str {
    if cfg!(feature = "cuda") {
        "CUDA"
    } else if cfg!(feature = "rocm") {
        "ROCm"
    } else if cfg!(feature = "vulkan") {
        "Vulkan (SPIR-V)"
    } else if cfg!(feature = "wgpu") {
        "WGPU"
    } else {
        "CPU (NdArray)"
    }
}

fn max_frames_from_args(args: &Args) -> usize {
    if let Some(duration) = args.duration {
        (duration * 12.5) as usize
    } else {
        args.frames
    }
}

fn build_options(args: &Args) -> SynthesisOptions {
    SynthesisOptions {
        max_length: max_frames_from_args(args),
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        seed: Some(args.seed),
        chunk_frames: args.chunk_frames,
        ..Default::default()
    }
}

fn run(args: &Args) -> Result<()> {
    let device = Default::default();

    println!("=== Qwen3-TTS (Burn) ===");
    println!("Backend: {}", backend_name());
    println!("Text: {}", args.text);
    println!("Seed: {}", args.seed);

    // Load model
    println!("\nLoading model from {}...", args.model_dir);
    let model = qwen3_tts::Qwen3TTS::<SelectedBackend>::from_pretrained_with_tokenizer(
        &args.model_dir,
        args.tokenizer_dir.as_deref(),
        device,
    )?;

    if let Some(mt) = model.model_type() {
        println!("Model variant: {:?}", mt);
    }

    let language: Language = args.language.parse()?;
    let options = build_options(args);

    // Select synthesis path based on model type and args
    let audio = if let Some(ref ref_audio_path) = args.ref_audio {
        // Voice cloning: Base models with reference audio
        let is_icl = args.ref_text.is_some();
        println!(
            "Mode: VoiceClone ({})",
            if is_icl { "ICL" } else { "x_vector_only" }
        );
        let ref_audio = AudioBuffer::load(ref_audio_path)?;
        println!(
            "Reference audio: {:.2}s ({} samples, {}Hz)",
            ref_audio.duration(),
            ref_audio.len(),
            ref_audio.sample_rate
        );

        let prompt = model.create_voice_clone_prompt(&ref_audio, args.ref_text.as_deref())?;

        if args.streaming {
            synthesize_voice_clone_streaming(&model, &args.text, &prompt, language, options)?
        } else {
            model.synthesize_voice_clone(&args.text, &prompt, language, Some(options))?
        }
    } else if let Some(ref instruct) = args.instruct {
        // VoiceDesign: text-described voice
        println!("Mode: VoiceDesign");
        println!("Instruct: {}", instruct);

        if args.streaming {
            synthesize_voice_design_streaming(&model, &args.text, instruct, language, options)?
        } else {
            model.synthesize_voice_design(&args.text, instruct, language, Some(options))?
        }
    } else {
        // CustomVoice: preset speaker
        let speaker: Speaker = args.speaker.parse()?;
        println!("Mode: CustomVoice");
        println!("Speaker: {:?}, Language: {:?}", speaker, language);

        if let Some(ModelType::Base) = model.model_type() {
            eprintln!(
                "  WARNING: This is a Base model (voice cloning). \
                 Using preset speaker fallback — voice may be unpredictable."
            );
        }

        if args.streaming {
            synthesize_streaming(&model, &args.text, speaker, language, options)?
        } else {
            model.synthesize_with_voice(&args.text, speaker, language, Some(options))?
        }
    };

    println!(
        "\nGenerated: {:.2}s, {} samples",
        audio.duration(),
        audio.len()
    );

    // Save output
    let output_path = Path::new(&args.output);
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    audio.save(output_path)?;
    println!("Saved WAV to: {}", output_path.display());

    Ok(())
}

/// Streaming synthesis with progress output.
fn synthesize_streaming(
    model: &qwen3_tts::Qwen3TTS<SelectedBackend>,
    text: &str,
    speaker: Speaker,
    language: Language,
    options: SynthesisOptions,
) -> Result<AudioBuffer> {
    let chunk_frames = options.chunk_frames;
    let mut session = model.synthesize_streaming(text, speaker, language, options)?;

    let mut all_samples = Vec::new();
    let mut chunk_count = 0;
    while let Some(chunk) = session.next_chunk()? {
        all_samples.extend_from_slice(&chunk.samples);
        chunk_count += 1;
        eprintln!(
            "  Chunk {}: {} frames, {:.2}s total",
            chunk_count,
            chunk_frames,
            all_samples.len() as f64 / 24000.0
        );
    }
    eprintln!(
        "  Streaming complete: {} chunks, {} frames",
        chunk_count,
        session.frames_generated()
    );

    Ok(AudioBuffer::new(all_samples, 24000))
}

/// Streaming voice clone synthesis with progress output.
fn synthesize_voice_clone_streaming(
    model: &qwen3_tts::Qwen3TTS<SelectedBackend>,
    text: &str,
    prompt: &VoiceClonePrompt<SelectedBackend>,
    language: Language,
    options: SynthesisOptions,
) -> Result<AudioBuffer> {
    let chunk_frames = options.chunk_frames;
    let mut session = model.synthesize_voice_clone_streaming(text, prompt, language, options)?;

    let mut all_samples = Vec::new();
    let mut chunk_count = 0;
    while let Some(chunk) = session.next_chunk()? {
        all_samples.extend_from_slice(&chunk.samples);
        chunk_count += 1;
        eprintln!(
            "  Chunk {}: {} frames, {:.2}s total",
            chunk_count,
            chunk_frames,
            all_samples.len() as f64 / 24000.0
        );
    }
    eprintln!(
        "  Streaming complete: {} chunks, {} frames",
        chunk_count,
        session.frames_generated()
    );

    Ok(AudioBuffer::new(all_samples, 24000))
}

/// Streaming VoiceDesign synthesis with progress output.
fn synthesize_voice_design_streaming(
    model: &qwen3_tts::Qwen3TTS<SelectedBackend>,
    text: &str,
    instruct: &str,
    language: Language,
    options: SynthesisOptions,
) -> Result<AudioBuffer> {
    let chunk_frames = options.chunk_frames;
    let mut session = model.synthesize_voice_design_streaming(text, instruct, language, options)?;

    let mut all_samples = Vec::new();
    let mut chunk_count = 0;
    while let Some(chunk) = session.next_chunk()? {
        all_samples.extend_from_slice(&chunk.samples);
        chunk_count += 1;
        eprintln!(
            "  Chunk {}: {} frames, {:.2}s total",
            chunk_count,
            chunk_frames,
            all_samples.len() as f64 / 24000.0
        );
    }
    eprintln!(
        "  Streaming complete: {} chunks, {} frames",
        chunk_count,
        session.frames_generated()
    );

    Ok(AudioBuffer::new(all_samples, 24000))
}

fn main() -> Result<()> {
    let _profiling_guard = qwen3_tts::profiling::init();
    if _profiling_guard.is_none() {
        tracing_subscriber::fmt::init();
    }

    let args = Args::parse();

    // Validate
    if args.instruct.is_some() && args.speaker != "ryan" {
        eprintln!("Note: --speaker is ignored when --instruct is used (VoiceDesign mode).");
    }

    run(&args)
}
