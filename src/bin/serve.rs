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

use qwen3_tts::{Language, Speaker, SynthesisOptions};

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
    response_tx: tokio_mpsc::Sender<Result<Vec<u8>, String>>,
}

// ── OpenAI API types ─────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SpeechRequest {
    #[allow(dead_code)]
    model: String,
    input: String,
    voice: String,
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

// ── App state ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    command_tx: std_mpsc::SyncSender<SynthesisRequest>,
    default_language: Language,
    default_options: SynthesisOptions,
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

    // Resolve voice
    let speaker = match resolve_voice(&req.voice) {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
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

    // Create response channel
    let (response_tx, mut response_rx) = tokio_mpsc::channel::<Result<Vec<u8>, String>>(32);

    let synth_req = SynthesisRequest {
        text: req.input,
        speaker,
        language,
        options,
        format,
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

fn run_model_thread(
    model: qwen3_tts::Qwen3TTS<SelectedBackend>,
    rx: std_mpsc::Receiver<SynthesisRequest>,
) {
    while let Ok(req) = rx.recv() {
        let result = process_request(&model, &req);
        if let Err(e) = result {
            let _ = req.response_tx.blocking_send(Err(e));
        }
    }
}

fn process_request(
    model: &qwen3_tts::Qwen3TTS<SelectedBackend>,
    req: &SynthesisRequest,
) -> Result<(), String> {
    // Use non-streaming synthesis: collect all codes, decode once.
    // The decoder (ConvNeXt + transformer) has causal convolutions that need full
    // context — decoding chunks independently produces discontinuities at boundaries.
    eprintln!(
        "Synthesizing: speaker={:?} lang={:?} text={:?}",
        req.speaker, req.language, &req.text
    );
    let t0 = std::time::Instant::now();
    let audio = model
        .synthesize_with_voice(
            &req.text,
            req.speaker,
            req.language,
            Some(req.options.clone()),
        )
        .map_err(|e| e.to_string())?;
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

    // Validate default speaker arg early (used for documentation, could be a future fallback)
    let _: Speaker = args
        .default_speaker
        .parse()
        .expect("Invalid default speaker");
    let default_language: Language = args
        .default_language
        .parse()
        .expect("Invalid default language");

    // Load model
    eprintln!("Loading model from {}...", args.model_dir);
    let device = Default::default();
    let model = qwen3_tts::Qwen3TTS::<SelectedBackend>::from_pretrained_with_tokenizer(
        &args.model_dir,
        args.tokenizer_dir.as_deref(),
        device,
    )?;
    eprintln!("Model loaded.");

    // Create channel (bounded to prevent unbounded queue)
    let (command_tx, command_rx) = std_mpsc::sync_channel::<SynthesisRequest>(16);

    // Spawn model thread
    std::thread::spawn(move || run_model_thread(model, command_rx));

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
        default_language,
        default_options,
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
