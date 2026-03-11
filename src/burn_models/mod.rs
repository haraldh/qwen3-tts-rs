//! Burn-based neural network models for Qwen3-TTS
//!
//! This module contains the full model stack ported from Candle to Burn,
//! enabling multi-backend inference (CPU, CUDA, ROCm, WGPU).
//!
//! - `kv_cache`: KV cache for autoregressive generation
//! - `transformer`: Shared building blocks (RoPE, Attention, MLP, DecoderLayer)
//! - `talker`: TalkerModel for semantic token generation
//! - `code_predictor`: Acoustic token predictor
//! - `codec`: Audio codec decoder
//! - `speaker`: Speaker encoder (ECAPA-TDNN)
//! - `sampling`: Token sampling strategies
//! - `tts`: TTS-specific generation logic

pub mod code_predictor;
pub mod codec;
pub mod facade;
pub mod kv_cache;
pub mod sampling;
pub mod speaker;
pub mod talker;
pub mod transformer;
pub mod tts;
pub mod weight_loader;
