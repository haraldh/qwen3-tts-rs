//! Audio codec for Qwen3-TTS (Burn)

pub mod decoder_12hz;
pub mod encoder_12hz;

pub use decoder_12hz::{Decoder12Hz, Decoder12HzConfig};
pub use encoder_12hz::{Encoder12Hz, Encoder12HzConfig};
