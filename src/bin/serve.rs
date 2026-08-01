//! OpenAI-compatible TTS HTTP server for Qwen3-TTS
//!
//! Provides `POST /v1/audio/speech` compatible with OpenAI's TTS API,
//! enabling drop-in use with Open WebUI, SillyTavern, and other clients.
//!
//! Usage:
//!     cargo run --release --features serve,rocm --bin serve -- \
//!       --model-dir test_data/models/0.6B-CustomVoice

use anyhow::Result;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::sync::mpsc as std_mpsc;
use tokio::sync::mpsc as tokio_mpsc;

use qwen3_tts::{AudioBuffer, Language, Speaker, SynthesisOptions, VoiceClonePrompt};

// ── Backend selection ────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
type SelectedBackend = burn::backend::Cuda;

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
type SelectedBackend = burn::backend::Rocm<half::bf16>;

#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "rocm")))]
type SelectedBackend = burn::backend::Wgpu;

#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "wgpu")))]
type SelectedBackend = burn::backend::NdArray;

// ── CLI args ─────────────────────────────────────────────────────────────

/// OpenAI-compatible TTS server powered by Qwen3-TTS
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Model directory containing model.safetensors and config.json
    #[arg(short, long, default_value = "test_data/model")]
    model_dir: String,

    /// Tokenizer directory (defaults to model_dir)
    #[arg(long)]
    tokenizer_dir: Option<String>,

    /// Listen address
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Listen port
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Default speaker name
    #[arg(long, default_value = "ryan")]
    default_speaker: String,

    /// Default language
    #[arg(long, default_value = "english")]
    default_language: String,

    /// Default sampling temperature
    #[arg(long, default_value_t = 0.3)]
    temperature: f64,

    /// Default top-k sampling parameter
    #[arg(long, default_value_t = 20)]
    top_k: usize,

    /// Default top-p (nucleus) sampling parameter
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Default repetition penalty (1.0 = disabled)
    #[arg(long, default_value_t = 1.2)]
    repetition_penalty: f64,

    /// Default max frames to generate (default: 2048, ~164s)
    #[arg(long, default_value_t = 2048)]
    max_frames: usize,

    /// Use streaming synthesis (per-chunk decode; may have boundary artifacts)
    #[arg(long)]
    streaming: bool,

    /// Reference audio file for voice cloning (Base models only)
    #[arg(long)]
    ref_audio: Option<String>,

    /// Reference text transcript (required with --ref-audio for ICL mode)
    #[arg(long)]
    ref_text: Option<String>,

    /// Language for voice cloning
    #[arg(long, default_value = "english")]
    language: String,

    /// Skip the startup warmup synthesis. Warmup costs a few seconds at boot
    /// but stops the first real request from paying CubeCL autotune.
    #[arg(long)]
    no_warmup: bool,
}

// ── Audio format ─────────────────────────────────────────────────────────

#[derive(Debug)]
enum AudioFormat {
    Wav,
    Pcm,
}

impl AudioFormat {
    fn from_str_opt(s: Option<&str>) -> Result<Self, String> {
        match s {
            None | Some("wav") => Ok(AudioFormat::Wav),
            Some("pcm") => Ok(AudioFormat::Pcm),
            Some(other) => Err(format!(
                "Unsupported response_format '{other}'. Supported: wav, pcm"
            )),
        }
    }

    fn content_type(&self) -> &'static str {
        match self {
            AudioFormat::Wav => "audio/wav",
            AudioFormat::Pcm => "audio/pcm",
        }
    }
}

// ── Internal channel types ───────────────────────────────────────────────

struct SynthesisRequest {
    text: String,
    speaker: Speaker,
    language: Language,
    options: SynthesisOptions,
    format: AudioFormat,
    streaming: bool,
    voice_clone: bool,
    /// Per-request reference voice. `Some` overrides the startup `--ref-audio`.
    ref_voice: Option<RefVoice>,
    response_tx: tokio_mpsc::Sender<Result<Vec<u8>, String>>,
}

/// Decoded reference audio for a single request, plus the cache key it hashes to.
struct RefVoice {
    audio: AudioBuffer,
    text: Option<String>,
    key: u64,
}

// ── OpenAI API types ─────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SpeechRequest {
    #[allow(dead_code)]
    model: String,
    input: String,
    /// Optional despite being required by OpenAI's schema: clients that only
    /// ever use one voice routinely omit it, and rejecting them is unhelpful
    /// when --default-speaker already says what to use.
    voice: Option<String>,
    response_format: Option<String>,
    #[allow(dead_code)]
    speed: Option<f64>,
    /// Non-standard extension: language override
    language: Option<String>,
    /// Non-standard extension: sampling temperature
    temperature: Option<f64>,
    /// Non-standard extension: top-k sampling
    top_k: Option<usize>,
    /// Non-standard extension: top-p (nucleus) sampling
    top_p: Option<f64>,
    /// Non-standard extension: repetition penalty
    repetition_penalty: Option<f64>,
    /// Non-standard extension: max frames to generate
    max_frames: Option<usize>,
    /// Non-standard extension: reference audio for voice cloning, as a base64
    /// WAV (optionally a `data:audio/wav;base64,...` URI). Base models only.
    ref_audio: Option<String>,
    /// Non-standard extension: transcript of `ref_audio`. Supplying it selects
    /// ICL cloning; omitting it falls back to speaker-embedding-only cloning.
    ref_text: Option<String>,
}

#[derive(Serialize)]
struct ApiError {
    error: ApiErrorBody,
}

#[derive(Serialize)]
struct ApiErrorBody {
    message: String,
    r#type: String,
    code: Option<String>,
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    let body = ApiError {
        error: ApiErrorBody {
            message: message.into(),
            r#type: "invalid_request_error".into(),
            code: None,
        },
    };
    (status, Json(body)).into_response()
}

// ── Voice mapping ────────────────────────────────────────────────────────

fn resolve_voice(name: &str) -> Result<Speaker, String> {
    // Try OpenAI aliases first, then native names
    let lowered = name.to_lowercase();
    let mapped = match lowered.as_str() {
        "alloy" => "ryan",
        "nova" => "serena",
        "echo" => "aiden",
        "fable" => "dylan",
        "onyx" => "eric",
        "shimmer" => "vivian",
        other => other,
    };
    mapped
        .parse::<Speaker>()
        .map_err(|_| format!("Unknown voice '{name}'. Available: ryan, serena, vivian, aiden, eric, dylan, uncle_fu, ono_anna, sohee, alloy, nova, echo, fable, onyx, shimmer"))
}

/// Decode a base64 WAV from a request into reference audio.
///
/// Accepts a bare base64 payload or a `data:audio/wav;base64,...` URI, since
/// clients differ on which they send.
fn decode_ref_audio(encoded: &str, ref_text: Option<&str>) -> Result<RefVoice, String> {
    use base64::Engine as _;
    use std::hash::{Hash, Hasher};

    let payload = match encoded.find("base64,") {
        Some(idx) if encoded.starts_with("data:") => &encoded[idx + "base64,".len()..],
        _ => encoded,
    };

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|e| format!("Field 'ref_audio' is not valid base64: {e}"))?;

    let audio = AudioBuffer::from_wav_bytes(&bytes)
        .map_err(|e| format!("Field 'ref_audio' is not a readable WAV: {e}"))?;

    // Hash the encoded form rather than the samples — same input, same key,
    // and it avoids walking a few hundred thousand floats per request.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    payload.hash(&mut hasher);
    ref_text.hash(&mut hasher);
    let key = hasher.finish();

    Ok(RefVoice {
        audio,
        text: ref_text.map(str::to_owned),
        key,
    })
}

// ── App state ────────────────────────────────────────────────────────────

struct AppState {
    command_tx: std_mpsc::SyncSender<SynthesisRequest>,
    default_speaker: Speaker,
    default_language: Language,
    default_options: SynthesisOptions,
    streaming: bool,
    voice_clone: bool,
}

impl Clone for AppState {
    fn clone(&self) -> Self {
        Self {
            command_tx: self.command_tx.clone(),
            default_speaker: self.default_speaker,
            default_language: self.default_language,
            default_options: self.default_options.clone(),
            streaming: self.streaming,
            voice_clone: self.voice_clone,
        }
    }
}

// ── Handlers ─────────────────────────────────────────────────────────────

async fn health() -> &'static str {
    "OK"
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

async fn list_models() -> Json<ModelsResponse> {
    let voices = [
        "ryan", "serena", "vivian", "aiden", "eric", "dylan", "uncle_fu", "ono_anna", "sohee",
    ];
    let mut data = vec![ModelEntry {
        id: "tts-1".into(),
        object: "model",
        owned_by: "qwen3-tts-rs",
    }];
    for voice in voices {
        data.push(ModelEntry {
            id: format!("voice-{voice}"),
            object: "model",
            owned_by: "qwen3-tts-rs",
        });
    }
    Json(ModelsResponse {
        object: "list",
        data,
    })
}

async fn speech_handler(State(state): State<AppState>, Json(req): Json<SpeechRequest>) -> Response {
    // Validate input
    if req.input.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "Field 'input' must not be empty");
    }

    // Resolve voice, falling back to --default-speaker when the client omits it
    let speaker = match req.voice.as_deref() {
        Some(v) => match resolve_voice(v) {
            Ok(s) => s,
            Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
        },
        None => state.default_speaker,
    };

    // Resolve format
    let format = match AudioFormat::from_str_opt(req.response_format.as_deref()) {
        Ok(f) => f,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
    };

    // Resolve language
    let language = if let Some(ref lang) = req.language {
        match lang.parse::<Language>() {
            Ok(l) => l,
            Err(_) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Unknown language '{lang}'"),
                )
            }
        }
    } else {
        state.default_language
    };

    let content_type = format.content_type();

    let options = SynthesisOptions {
        temperature: req.temperature.unwrap_or(state.default_options.temperature),
        top_k: req.top_k.unwrap_or(state.default_options.top_k),
        top_p: req.top_p.unwrap_or(state.default_options.top_p),
        repetition_penalty: req
            .repetition_penalty
            .unwrap_or(state.default_options.repetition_penalty),
        max_length: req.max_frames.unwrap_or(state.default_options.max_length),
        ..state.default_options.clone()
    };

    // Resolve a per-request reference voice, if the caller sent one
    let ref_voice = match req.ref_audio.as_deref() {
        Some(encoded) => match decode_ref_audio(encoded, req.ref_text.as_deref()) {
            Ok(v) => Some(v),
            Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
        },
        None => None,
    };

    // Cloning requires either a per-request reference or one pinned at startup
    let voice_clone = ref_voice.is_some() || state.voice_clone;

    // Create response channel
    let (response_tx, mut response_rx) = tokio_mpsc::channel::<Result<Vec<u8>, String>>(32);

    let synth_req = SynthesisRequest {
        text: req.input,
        speaker,
        language,
        options,
        format,
        streaming: state.streaming,
        voice_clone,
        ref_voice,
        response_tx,
    };

    // Send to model thread
    if state.command_tx.send(synth_req).is_err() {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Model thread unavailable",
        );
    }

    // Wait for complete response (synthesis decodes all codes at once to avoid gaps)
    match response_rx.recv().await {
        Some(Ok(bytes)) => {
            let mut headers = HeaderMap::new();
            headers.insert("content-type", content_type.parse().unwrap());
            (StatusCode::OK, headers, bytes).into_response()
        }
        Some(Err(e)) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
        None => error_response(StatusCode::INTERNAL_SERVER_ERROR, "Model thread dropped"),
    }
}

// ── Model thread ─────────────────────────────────────────────────────────

/// How many distinct reference voices to keep prompts for.
///
/// Building one runs the ECAPA-TDNN speaker encoder (and, for ICL, the speech
/// encoder) on CPU, which costs far more than a synthesis. Callers overwhelmingly
/// reuse a handful of voices, so a small cache removes that cost from all but the
/// first request per voice.
const PROMPT_CACHE_CAP: usize = 8;

fn run_model_thread(
    model: qwen3_tts::Qwen3TTS<SelectedBackend>,
    startup_prompt: Option<VoiceClonePrompt<SelectedBackend>>,
    rx: std_mpsc::Receiver<SynthesisRequest>,
) {
    // Insertion-ordered so eviction can drop the oldest entry.
    let mut cache: Vec<(u64, VoiceClonePrompt<SelectedBackend>)> = Vec::new();

    while let Ok(req) = rx.recv() {
        // Per-request reference voice wins over the one pinned at startup.
        let prompt = match &req.ref_voice {
            Some(rv) => {
                if !cache.iter().any(|(k, _)| *k == rv.key) {
                    eprintln!("Building voice clone prompt (ICL={})", rv.text.is_some());
                    match model.create_voice_clone_prompt(&rv.audio, rv.text.as_deref()) {
                        Ok(p) => {
                            if cache.len() >= PROMPT_CACHE_CAP {
                                cache.remove(0);
                            }
                            cache.push((rv.key, p));
                        }
                        Err(e) => {
                            let _ = req.response_tx.blocking_send(Err(e.to_string()));
                            continue;
                        }
                    }
                }
                cache.iter().find(|(k, _)| *k == rv.key).map(|(_, p)| p)
            }
            None => startup_prompt.as_ref(),
        };

        let result = process_request(&model, prompt, &req);
        if let Err(e) = result {
            let _ = req.response_tx.blocking_send(Err(e));
        }
    }
}

fn process_request(
    model: &qwen3_tts::Qwen3TTS<SelectedBackend>,
    voice_clone_prompt: Option<&VoiceClonePrompt<SelectedBackend>>,
    req: &SynthesisRequest,
) -> Result<(), String> {
    eprintln!(
        "Synthesizing: voice_clone={} lang={:?} streaming={} text={:?}",
        req.voice_clone, req.language, req.streaming, &req.text
    );
    let t0 = std::time::Instant::now();
    let audio = if req.voice_clone {
        let prompt = voice_clone_prompt.ok_or(
            "Voice cloning requires 'ref_audio' in the request, or --ref-audio at startup",
        )?;
        model
            .synthesize_voice_clone(&req.text, prompt, req.language, Some(req.options.clone()))
            .map_err(|e| e.to_string())?
    } else if req.streaming {
        let mut session = model
            .synthesize_streaming(&req.text, req.speaker, req.language, req.options.clone())
            .map_err(|e| e.to_string())?;
        let mut all_samples = Vec::new();
        while let Some(chunk) = session.next_chunk().map_err(|e| e.to_string())? {
            all_samples.extend_from_slice(&chunk.samples);
        }
        AudioBuffer::new(all_samples, 24000)
    } else {
        model
            .synthesize_with_voice(
                &req.text,
                req.speaker,
                req.language,
                Some(req.options.clone()),
            )
            .map_err(|e| e.to_string())?
    };
    let elapsed = t0.elapsed();

    let duration = audio.duration();
    let rtf = elapsed.as_secs_f64() / duration as f64;
    eprintln!(
        "Generated {:.2}s audio in {:.2}s (RTF={:.2}x, {:.1}x realtime)",
        duration,
        elapsed.as_secs_f64(),
        rtf,
        1.0 / rtf,
    );

    let bytes = match req.format {
        AudioFormat::Wav => audio.to_wav_bytes(),
        AudioFormat::Pcm => audio.to_pcm_i16_bytes(),
    };

    req.response_tx
        .blocking_send(Ok(bytes))
        .map_err(|_| "Client disconnected".to_string())?;

    Ok(())
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let default_language: Language = args.language.parse().expect("Invalid language");

    // Load model
    eprintln!("Loading model from {}...", args.model_dir);
    let device = Default::default();
    let model = qwen3_tts::Qwen3TTS::<SelectedBackend>::from_pretrained_with_tokenizer(
        &args.model_dir,
        args.tokenizer_dir.as_deref(),
        device,
    )?;
    eprintln!("Model loaded.");

    // Pre-compute voice clone prompt if --ref-audio provided
    let voice_clone_prompt = if let Some(ref ref_audio_path) = args.ref_audio {
        eprintln!("Loading reference audio from {ref_audio_path}...");
        let ref_audio = qwen3_tts::AudioBuffer::load(ref_audio_path)?;
        let prompt = model.create_voice_clone_prompt(&ref_audio, args.ref_text.as_deref())?;
        eprintln!(
            "Voice clone prompt ready (ICL={})",
            prompt.ref_codes.is_some()
        );
        Some(prompt)
    } else {
        None
    };
    let voice_clone = voice_clone_prompt.is_some();

    let default_speaker: Speaker = args.default_speaker.parse().unwrap_or(Speaker::Ryan);

    // Absorb the one-off GPU costs before the socket opens, so the first real
    // request isn't the one that pays them. CubeCL autotunes every new BF16
    // convolution and matmul shape on first use, and the HIP kernels compile
    // through HIPRTC — together worth tens of seconds. /health stays closed
    // until this finishes, so an orchestrator won't route traffic here early.
    if !args.no_warmup {
        eprintln!("Warming up...");
        let t0 = std::time::Instant::now();
        let opts = SynthesisOptions {
            max_length: 16, // just enough frames to touch every kernel
            ..Default::default()
        };
        let warm = match voice_clone_prompt.as_ref() {
            Some(prompt) => {
                model.synthesize_voice_clone("Warming up.", prompt, default_language, Some(opts))
            }
            None => model.synthesize_with_voice(
                "Warming up.",
                default_speaker,
                default_language,
                Some(opts),
            ),
        };
        match warm {
            // Don't abort startup on a warmup failure — the server is still
            // usable, it will just be slow on the first request.
            Err(e) => eprintln!("Warmup failed (continuing): {e}"),
            Ok(_) => eprintln!("Warmed up in {:.1}s", t0.elapsed().as_secs_f64()),
        }
    }

    // Create channel (bounded to prevent unbounded queue)
    let (command_tx, command_rx) = std_mpsc::sync_channel::<SynthesisRequest>(16);

    // Spawn model thread
    std::thread::spawn(move || run_model_thread(model, voice_clone_prompt, command_rx));

    let default_options = SynthesisOptions {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        max_length: args.max_frames,
        ..Default::default()
    };

    let state = AppState {
        command_tx,
        default_speaker,
        default_language,
        default_options,
        streaming: args.streaming,
        voice_clone,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/speech", post(speech_handler))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    eprintln!("Listening on http://{addr}");

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}
