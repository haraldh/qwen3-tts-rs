//! Framework-independent types for the Talker model.
//!
//! This module contains enums, configs, and token constants that are shared
//! between the Candle and Burn implementations.

use std::str::FromStr;

use super::config::{ParsedModelConfig, Qwen3TTSConfig};

/// ChatML special token IDs
pub mod special_tokens {
    pub const IM_START: u32 = 151644;
    pub const IM_END: u32 = 151645;
    pub const ASSISTANT: u32 = 77091;
    pub const NEWLINE: u32 = 198;
}

/// TTS special token IDs (text vocabulary tokens for TTS generation)
pub mod tts_tokens {
    pub const TTS_PAD: u32 = 151671;
    pub const TTS_BOS: u32 = 151672;
    pub const TTS_EOS: u32 = 151673;
}

/// Codec special token IDs
pub mod codec_tokens {
    pub const CODEC_PAD: u32 = 2148;
    pub const CODEC_BOS: u32 = 2149;
    pub const CODEC_EOS: u32 = 2150;
    pub const CODEC_THINK: u32 = 2154;
    pub const CODEC_NOTHINK: u32 = 2155;
    pub const CODEC_THINK_BOS: u32 = 2156;
    pub const CODEC_THINK_EOS: u32 = 2157;
    /// Total codec vocabulary size (semantic + acoustic + control tokens)
    pub const CODEC_VOCAB_SIZE: usize = 3072;
}

/// Language IDs for codec prefix
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Chinese,
    English,
    Japanese,
    Korean,
    German,
    French,
    Russian,
    Portuguese,
    Spanish,
    Italian,
}

impl FromStr for Language {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "english" | "en" => Ok(Language::English),
            "chinese" | "zh" => Ok(Language::Chinese),
            "japanese" | "ja" => Ok(Language::Japanese),
            "korean" | "ko" => Ok(Language::Korean),
            "german" | "de" => Ok(Language::German),
            "french" | "fr" => Ok(Language::French),
            "russian" | "ru" => Ok(Language::Russian),
            "portuguese" | "pt" => Ok(Language::Portuguese),
            "spanish" | "es" => Ok(Language::Spanish),
            "italian" | "it" => Ok(Language::Italian),
            _ => anyhow::bail!("Unknown language: {}", s),
        }
    }
}

impl Language {
    /// Get the codec language token ID
    pub fn token_id(&self) -> u32 {
        match self {
            Language::Chinese => 2055,
            Language::English => 2050,
            Language::Japanese => 2058,
            Language::Korean => 2064,
            Language::German => 2053,
            Language::French => 2061,
            Language::Russian => 2069,
            Language::Portuguese => 2071,
            Language::Spanish => 2054,
            Language::Italian => 2070,
        }
    }
}

/// Speaker IDs for CustomVoice model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    Serena,
    Vivian,
    UncleFu,
    Ryan,
    Aiden,
    OnoAnna,
    Sohee,
    Eric,
    Dylan,
}

impl FromStr for Speaker {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ryan" => Ok(Speaker::Ryan),
            "serena" => Ok(Speaker::Serena),
            "vivian" => Ok(Speaker::Vivian),
            "aiden" => Ok(Speaker::Aiden),
            "uncle_fu" | "unclefu" => Ok(Speaker::UncleFu),
            "ono_anna" | "onoanna" => Ok(Speaker::OnoAnna),
            "sohee" => Ok(Speaker::Sohee),
            "eric" => Ok(Speaker::Eric),
            "dylan" => Ok(Speaker::Dylan),
            _ => anyhow::bail!("Unknown speaker: {}", s),
        }
    }
}

impl Speaker {
    /// Get the speaker token ID
    pub fn token_id(&self) -> u32 {
        match self {
            Speaker::Serena => 3066,
            Speaker::Vivian => 3065,
            Speaker::UncleFu => 3010,
            Speaker::Ryan => 3061,
            Speaker::Aiden => 2861,
            Speaker::OnoAnna => 2873,
            Speaker::Sohee => 2864,
            Speaker::Eric => 2875,
            Speaker::Dylan => 2878,
        }
    }

    /// Get the native language for this speaker
    pub fn native_language(&self) -> Language {
        match self {
            Speaker::Serena
            | Speaker::Vivian
            | Speaker::UncleFu
            | Speaker::Eric
            | Speaker::Dylan => Language::Chinese,
            Speaker::Ryan | Speaker::Aiden => Language::English,
            Speaker::OnoAnna => Language::Japanese,
            Speaker::Sohee => Language::Korean,
        }
    }
}

/// Talker model configuration
#[derive(Debug, Clone)]
pub struct TalkerConfig {
    /// Text vocabulary size (151936)
    pub text_vocab_size: usize,
    /// Text embedding dimension (2048)
    pub text_embed_dim: usize,
    /// Hidden dimension (1024)
    pub hidden_size: usize,
    /// Intermediate size for text projection (2048)
    pub text_proj_intermediate: usize,
    /// Intermediate size for MLP (3072)
    pub intermediate_size: usize,
    /// Number of transformer layers (28)
    pub num_hidden_layers: usize,
    /// Number of attention heads (16)
    pub num_attention_heads: usize,
    /// Number of KV heads for GQA (8)
    pub num_key_value_heads: usize,
    /// Head dimension (128)
    pub head_dim: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f64,
    /// RoPE theta
    pub rope_theta: f64,
    /// Max position embeddings
    pub max_position_embeddings: usize,
    /// Codec vocabulary size (3072 - includes special tokens)
    pub codec_vocab_size: usize,
    /// MRoPE section for multimodal rotary embedding [T, H, W]
    /// None = use standard RoPE, Some([24, 20, 20]) = use interleaved MRoPE
    pub mrope_section: Option<[usize; 3]>,
}

impl Default for TalkerConfig {
    /// Default config for 0.6B models (hidden=1024).
    ///
    /// All model variants use MRoPE with section `[24, 20, 20]`.
    fn default() -> Self {
        Self {
            text_vocab_size: 151936,
            text_embed_dim: 2048,
            hidden_size: 1024,
            text_proj_intermediate: 2048,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            max_position_embeddings: 32768,
            codec_vocab_size: 3072,
            mrope_section: Some([24, 20, 20]),
        }
    }
}

impl TalkerConfig {
    /// Create config from parsed HuggingFace config.json.
    pub fn from_parsed(parsed: &ParsedModelConfig) -> Self {
        Self {
            text_vocab_size: parsed.talker_text_vocab_size,
            text_embed_dim: parsed.talker_text_hidden_size,
            hidden_size: parsed.talker_hidden_size,
            text_proj_intermediate: parsed.talker_text_hidden_size,
            intermediate_size: parsed.talker_intermediate_size,
            num_hidden_layers: parsed.talker_num_hidden_layers,
            num_attention_heads: parsed.talker_num_attention_heads,
            num_key_value_heads: parsed.talker_num_key_value_heads,
            head_dim: parsed.talker_head_dim,
            rms_norm_eps: parsed.talker_rms_norm_eps,
            rope_theta: parsed.talker_rope_theta,
            max_position_embeddings: parsed.talker_max_position_embeddings,
            codec_vocab_size: parsed.talker_vocab_size,
            mrope_section: parsed.mrope_section,
        }
    }

    /// Create config for 1.7B models (larger hidden dimension, MRoPE)
    pub fn custom_voice() -> Self {
        Self {
            text_vocab_size: 151936,
            text_embed_dim: 2048,
            hidden_size: 2048,
            text_proj_intermediate: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            max_position_embeddings: 32768,
            codec_vocab_size: 3072,
            mrope_section: Some([24, 20, 20]),
        }
    }

    /// Convert to a Qwen3TTSConfig for building DecoderLayers
    pub fn to_layer_config(&self) -> Qwen3TTSConfig {
        Qwen3TTSConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: Some(self.num_key_value_heads),
            head_dim_override: Some(self.head_dim),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            ..Default::default()
        }
    }
}
