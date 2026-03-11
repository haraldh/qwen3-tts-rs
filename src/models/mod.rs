//! Neural network models for Qwen3-TTS
//!
//! This module contains framework-independent configuration types.
//! The candle-based model implementations are behind `_candle_legacy` feature.
//! See `burn_models` for the Burn-based implementations.

// Config is pure serde — always available
pub mod config;

// Framework-independent types (enums, configs, token constants)
pub mod talker_types;

// Candle-based implementations — only available with legacy feature
#[cfg(feature = "_candle_legacy")]
pub mod code_predictor;
#[cfg(feature = "_candle_legacy")]
pub mod codec;
#[cfg(feature = "_candle_legacy")]
pub mod fused_ops;
#[cfg(feature = "_candle_legacy")]
pub mod kv_cache;
#[cfg(feature = "_candle_legacy")]
pub mod speaker;
#[cfg(feature = "_candle_legacy")]
pub mod talker;
#[cfg(feature = "_candle_legacy")]
pub mod transformer;

pub use config::{ModelType, ParsedModelConfig, Qwen3TTSConfig, SpeakerEncoderConfig};
pub use talker_types::{codec_tokens, special_tokens, tts_tokens, Language, Speaker, TalkerConfig};

#[cfg(feature = "_candle_legacy")]
pub use code_predictor::{CodePredictor, CodePredictorConfig};
#[cfg(feature = "_candle_legacy")]
pub use kv_cache::{AnyKVCache, KVCache, PreAllocKVCache};
#[cfg(feature = "_candle_legacy")]
pub use talker::TalkerModel;
#[cfg(feature = "_candle_legacy")]
pub use transformer::{MRoPE, RoPEType, RotaryEmbedding};
