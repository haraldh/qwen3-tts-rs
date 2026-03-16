//! Burn-based integration facade for Qwen3-TTS
//!
//! Provides [`Qwen3TTS`], the main entry point for speech synthesis using the
//! Burn framework. Ties together the talker, code predictor, and decoder models
//! and implements the autoregressive generation loop.

use burn::prelude::*;

use crate::audio::AudioBuffer;
use crate::models::config::ModelType;
use crate::tokenizer;
use crate::FrameCodes;

use super::code_predictor::CodePredictor;
use super::codec::decoder_12hz::Decoder12Hz;
use super::codec::encoder_12hz::Encoder12Hz;
use super::kv_cache::KVCache;
use super::sampling::{self, GenerationConfig, SamplingContext};
use super::speaker::SpeakerEncoder;
use super::talker::{codec_tokens, Language, Speaker, TalkerModel};
use super::transformer::RoPEType;
use super::tts::{self, SuppressionMask};

/// The codec end-of-sequence token ID (2150).
pub const CODEC_EOS_TOKEN_ID: u32 = codec_tokens::CODEC_EOS;

/// Set a single position in a Bool tensor to `true` via scatter.
fn set_penalty_bit<B: Backend>(
    mask: &Tensor<B, 1, Bool>,
    idx: usize,
    device: &B::Device,
) -> Tensor<B, 1, Bool> {
    #[allow(clippy::single_range_in_vec_init)]
    let one_hot = Tensor::<B, 1, Int>::zeros([mask.dims()[0]], device)
        .slice_assign([idx..idx + 1], Tensor::<B, 1, Int>::ones([1], device))
        .equal_elem(1);
    mask.clone().bool_or(one_hot)
}

/// Number of audio samples per codec frame at 24kHz (1920 = 80ms at 12Hz).
pub const SAMPLES_PER_FRAME: usize = 1920;

/// Minimum repetition penalty for ICL mode (matches mlx-audio).
const ICL_MIN_REPETITION_PENALTY: f64 = 1.2;

/// Minimum frame count for ICL generation.
const ICL_MIN_FRAMES: usize = 100;

/// Frames per input text token for ICL max_new_tokens estimation.
const ICL_FRAMES_PER_TOKEN: usize = 5;

/// Voice clone prompt holding speaker embedding and optional ICL data.
///
/// Created via [`Qwen3TTS::create_voice_clone_prompt`] or constructed directly
/// with pre-computed reference codes for ICL mode.
pub struct VoiceClonePrompt<B: Backend> {
    /// Speaker embedding from the ECAPA-TDNN encoder, shape `[enc_dim]` (typically 1024).
    pub speaker_embedding: Tensor<B, 1>,
    /// Reference audio codec codes for ICL mode. Each inner Vec is one frame of 16 codes.
    /// `None` = x_vector_only mode.
    pub ref_codes: Option<Vec<Vec<u32>>>,
    /// Tokenized reference text for ICL mode.
    pub ref_text_ids: Option<Vec<u32>>,
}

/// Options for speech synthesis.
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Maximum number of frames to generate
    pub max_length: usize,
    /// Sampling temperature (higher = more random)
    pub temperature: f64,
    /// Top-k sampling
    pub top_k: usize,
    /// Top-p (nucleus) sampling
    pub top_p: f64,
    /// Repetition penalty (1.0 = disabled, 1.05 = Python default)
    pub repetition_penalty: f64,
    /// End-of-sequence token ID (defaults to codec EOS token 2150)
    pub eos_token_id: Option<u32>,
    /// Frames per streaming chunk (default: 10 = ~800ms)
    pub chunk_frames: usize,
    /// Minimum tokens before EOS is allowed (default: 2, matching Python)
    pub min_new_tokens: usize,
    /// Random seed for deterministic generation. `None` = non-deterministic.
    pub seed: Option<u64>,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_length: 2048,
            temperature: 0.3,
            top_k: 20,
            top_p: 0.9,
            repetition_penalty: 1.2,
            eos_token_id: Some(CODEC_EOS_TOKEN_ID),
            chunk_frames: 10,
            min_new_tokens: 2,
            seed: None,
        }
    }
}

impl SynthesisOptions {
    /// Convert to a [`GenerationConfig`] for the generation loop.
    pub fn to_gen_config(&self) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: self.max_length,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            eos_token_id: self.eos_token_id,
            min_new_tokens: self.min_new_tokens,
        }
    }
}

// ── Main TTS facade ──────────────────────────────────────────────────────

/// Main TTS model facade for Burn.
///
/// Owns all model components (talker, code predictor, decoder) and implements
/// the full synthesis pipeline: text → semantic tokens → acoustic codes → audio.
pub struct Qwen3TTS<B: Backend> {
    talker: TalkerModel<B>,
    code_predictor: CodePredictor<B>,
    decoder: Decoder12Hz<B>,
    encoder: Encoder12Hz<B>,
    text_tokenizer: tokenizer::TextTokenizer,
    speaker_encoder: Option<SpeakerEncoder<B>>,
    rope: RoPEType<B>,
    cp_rope: RoPEType<B>,
    model_type: Option<ModelType>,
    device: B::Device,
}

impl<B: Backend> Qwen3TTS<B> {
    /// Load a model from a directory containing config.json and safetensors files.
    ///
    /// Expected directory structure:
    /// ```text
    /// model_dir/
    /// ├── config.json
    /// ├── model.safetensors
    /// ├── tokenizer.json          (or merges.txt + vocab.json)
    /// └── speech_tokenizer/
    ///     └── model.safetensors
    /// ```
    pub fn from_pretrained(model_dir: &str, device: B::Device) -> anyhow::Result<Self> {
        Self::from_pretrained_with_tokenizer(model_dir, None, device)
    }

    /// Load a model with an explicit tokenizer source.
    ///
    /// `tokenizer_id` can be a local directory or a file path containing
    /// tokenizer.json. If `None`, resolves from the model directory.
    pub fn from_pretrained_with_tokenizer(
        model_dir: &str,
        tokenizer_id: Option<&str>,
        device: B::Device,
    ) -> anyhow::Result<Self> {
        let model_path = std::path::Path::new(model_dir);

        tracing::info!("Loading Qwen3-TTS from: {}", model_dir);

        // Load text tokenizer
        let tok_source = tokenizer_id.unwrap_or(model_dir);
        let text_tokenizer = tokenizer::TextTokenizer::from_pretrained(tok_source)?;

        // Load all model components
        let components = super::weight_loader::load_all(model_path, &device)?;

        Ok(Self::build_from_components(
            components.talker,
            components.code_predictor,
            components.decoder,
            components.encoder,
            text_tokenizer,
            components.speaker_encoder,
            components.model_type,
            device,
        ))
    }

    /// Build from pre-constructed model components.
    pub fn build_from_components(
        talker: TalkerModel<B>,
        code_predictor: CodePredictor<B>,
        decoder: Decoder12Hz<B>,
        encoder: Encoder12Hz<B>,
        text_tokenizer: tokenizer::TextTokenizer,
        speaker_encoder: Option<SpeakerEncoder<B>>,
        model_type: Option<ModelType>,
        device: B::Device,
    ) -> Self {
        let rope = talker.create_rope(&device);
        let cp_rope = code_predictor.create_rope(&device);
        Self {
            talker,
            code_predictor,
            decoder,
            encoder,
            text_tokenizer,
            speaker_encoder,
            rope,
            cp_rope,
            model_type,
            device,
        }
    }

    /// Get the device this model is running on.
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Get the model type (Base, CustomVoice, VoiceDesign).
    pub fn model_type(&self) -> Option<&ModelType> {
        self.model_type.as_ref()
    }

    // ── CustomVoice synthesis ────────────────────────────────────────────

    /// Synthesize speech with a predefined speaker voice (CustomVoice models).
    pub fn synthesize_with_voice(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> anyhow::Result<AudioBuffer> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        if let Some(ModelType::Base) = &self.model_type {
            tracing::warn!(
                "Using preset speaker {:?} on a Base model. Use synthesize_voice_clone() instead.",
                speaker
            );
        } else if let Some(ModelType::VoiceDesign) = &self.model_type {
            tracing::warn!("Using preset speaker {:?} on a VoiceDesign model.", speaker);
        }

        let options = options.unwrap_or_default();
        let mut sampling_ctx = SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;
        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids);

        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        // Estimate max sequence: prefix (~15) + decode frames
        let max_seq = input_ids.len() + 20 + gen_config.max_new_tokens;
        let mut kv_caches = self.talker.new_kv_caches(max_seq, &self.device);
        let (hidden, logits) = self.talker.prefill_custom_voice(
            &input_ids,
            speaker,
            language,
            &self.rope,
            &mut kv_caches,
            &self.device,
        );
        let prefill_len = hidden.dims()[1];
        let last_hidden = hidden.narrow(1, prefill_len - 1, 1);

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            prefill_len,
            last_hidden,
            logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        );

        self.decode_codes(&all_codes)
    }

    /// Create a streaming synthesis session (CustomVoice).
    pub fn synthesize_streaming(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> anyhow::Result<StreamingSession<'_, B>> {
        let input_ids = self.text_tokenizer.encode(text)?;
        StreamingSession::new(self, &input_ids, speaker, language, options)
    }

    // ── VoiceDesign synthesis ────────────────────────────────────────────

    /// Synthesize speech using a text-described voice (VoiceDesign models).
    pub fn synthesize_voice_design(
        &self,
        text: &str,
        instruct: &str,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> anyhow::Result<AudioBuffer> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        if let Some(ref mt) = self.model_type {
            if *mt != ModelType::VoiceDesign {
                tracing::warn!("Using VoiceDesign synthesis on a {:?} model.", mt);
            }
        }

        let options = options.unwrap_or_default();
        let mut sampling_ctx = SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;

        let instruct_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let instruct_ids = self.text_tokenizer.encode(&instruct_text)?;
        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids);

        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        let max_seq = input_ids.len() + instruct_ids.len() + 20 + gen_config.max_new_tokens;
        let mut kv_caches = self.talker.new_kv_caches(max_seq, &self.device);
        let (hidden, logits) = self.talker.prefill_voice_design(
            &input_ids,
            &instruct_ids,
            language,
            &self.rope,
            &mut kv_caches,
            &self.device,
        );
        let prefill_len = hidden.dims()[1];
        let last_hidden = hidden.narrow(1, prefill_len - 1, 1);

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            prefill_len,
            last_hidden,
            logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        );

        self.decode_codes(&all_codes)
    }

    /// Create a streaming synthesis session (VoiceDesign).
    pub fn synthesize_voice_design_streaming(
        &self,
        text: &str,
        instruct: &str,
        language: Language,
        options: SynthesisOptions,
    ) -> anyhow::Result<StreamingSession<'_, B>> {
        let input_ids = self.text_tokenizer.encode(text)?;
        let instruct_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let instruct_ids = self.text_tokenizer.encode(&instruct_text)?;
        StreamingSession::new_voice_design(self, &input_ids, &instruct_ids, language, options)
    }

    // ── Voice cloning API ────────────────────────────────────────────────

    /// Returns `true` if a speaker encoder is loaded (voice cloning is available).
    pub fn has_speaker_encoder(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Create a voice clone prompt from reference audio.
    ///
    /// When `ref_text` is `None`, produces an **x_vector_only** prompt (speaker
    /// embedding only). When `Some`, produces an **ICL** prompt with reference
    /// codec codes and tokenized reference text.
    pub fn create_voice_clone_prompt(
        &self,
        ref_audio: &AudioBuffer,
        ref_text: Option<&str>,
    ) -> anyhow::Result<VoiceClonePrompt<B>> {
        let speaker_enc = self.speaker_encoder.as_ref().ok_or_else(|| {
            let hint = match &self.model_type {
                Some(ModelType::CustomVoice) => {
                    " CustomVoice models use preset speakers, not voice cloning. \
                     Use a Base model for voice cloning."
                }
                Some(ModelType::VoiceDesign) => {
                    " VoiceDesign models use text-described voices, not voice cloning. \
                     Use a Base model for voice cloning."
                }
                _ => {
                    " Ensure model weights contain `speaker_encoder.*` keys \
                     (only Base models include a speaker encoder)."
                }
            };
            anyhow::anyhow!("Speaker encoder not available.{}", hint)
        })?;

        // Resample to 24kHz if needed
        let ref_audio_24k;
        let ref_audio = if ref_audio.sample_rate != 24000 {
            tracing::info!(
                "Resampling reference audio from {}Hz to 24000Hz",
                ref_audio.sample_rate
            );
            ref_audio_24k = crate::audio::resample_to_24k(ref_audio)?;
            &ref_audio_24k
        } else {
            ref_audio
        };

        let speaker_embedding = speaker_enc.encode(ref_audio, &self.device);

        if let Some(ref_text) = ref_text {
            // ICL mode: encode reference audio to codec codes + tokenize ref text
            tracing::info!("Encoding reference audio for ICL voice cloning...");
            let ref_codes = self.encoder.encode(&ref_audio.samples);
            tracing::info!(
                "Encoded {} frames of reference codec codes",
                ref_codes.len()
            );

            let ref_text_ids = self.text_tokenizer.encode(ref_text)?;
            tracing::info!("Tokenized reference text: {} tokens", ref_text_ids.len());

            Ok(VoiceClonePrompt {
                speaker_embedding,
                ref_codes: Some(ref_codes),
                ref_text_ids: Some(ref_text_ids),
            })
        } else {
            // x_vector_only mode
            Ok(VoiceClonePrompt {
                speaker_embedding,
                ref_codes: None,
                ref_text_ids: None,
            })
        }
    }

    /// Synthesize speech using a cloned voice.
    ///
    /// Supports both x_vector_only mode (speaker embedding only) and ICL mode
    /// (with ref_codes and ref_text_ids in the prompt).
    pub fn synthesize_voice_clone(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt<B>,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> anyhow::Result<AudioBuffer> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize_voice_clone").entered();

        let options = options.unwrap_or_default();
        let mut sampling_ctx = SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;

        let is_icl = prompt.ref_codes.is_some() && prompt.ref_text_ids.is_some();

        // ICL mode adjustments
        let mut gen_config = options.to_gen_config();
        if is_icl {
            gen_config.repetition_penalty = gen_config
                .repetition_penalty
                .max(ICL_MIN_REPETITION_PENALTY);
            gen_config.max_new_tokens = gen_config
                .max_new_tokens
                .min(ICL_MIN_FRAMES.max(input_ids.len() * ICL_FRAMES_PER_TOKEN));
        }

        let speaker_embed = prompt
            .speaker_embedding
            .clone()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0); // [1, 1, enc_dim]

        let max_seq = input_ids.len() + 256 + gen_config.max_new_tokens;
        let mut kv_caches = self.talker.new_kv_caches(max_seq, &self.device);
        let (hidden, logits) = self.talker.prefill_voice_clone(
            &input_ids,
            speaker_embed,
            language,
            is_icl,
            &self.rope,
            &mut kv_caches,
            &self.device,
        );
        let prefill_len = hidden.dims()[1];
        let mut offset = prefill_len;
        let mut last_hidden = hidden.narrow(1, prefill_len - 1, 1);

        // ICL extension
        let (trailing_text_hidden, trailing_text_len, logits) =
            if let (Some(ref ref_codes), Some(ref ref_text_ids)) =
                (&prompt.ref_codes, &prompt.ref_text_ids)
            {
                let ref_codec_embeds = self.sum_ref_codec_embeddings(ref_codes);

                let (icl_embed, icl_trailing) = self.talker.build_icl_prompt(
                    &input_ids,
                    ref_text_ids,
                    &ref_codec_embeds,
                    &self.device,
                );

                let icl_len = icl_embed.dims()[1];
                if icl_len > 0 {
                    let (icl_last_hidden, icl_logits) =
                        self.talker
                            .run_icl_layers(icl_embed, &self.rope, &mut kv_caches, offset);
                    offset += icl_len;
                    last_hidden = icl_last_hidden;

                    let trailing_len = icl_trailing.dims()[1];
                    (icl_trailing, trailing_len, icl_logits)
                } else {
                    let (trailing, trailing_len, tts_pad) = self.build_trailing_text(&input_ids);
                    let _ = tts_pad; // unused in this branch
                    (trailing, trailing_len, logits)
                }
            } else {
                let (trailing, trailing_len, _tts_pad) = self.build_trailing_text(&input_ids);
                (trailing, trailing_len, logits)
            };

        let tts_pad_embed = self.talker.get_tts_pad_embed(&self.device);

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        );

        // ICL decode: prepend ref_codes and trim proportionally
        if let Some(ref ref_codes) = prompt.ref_codes {
            let ref_len = ref_codes.len();
            let mut combined = ref_codes.clone();
            combined.extend(all_codes.iter().cloned());

            let mut audio = self.decode_codes(&combined)?;
            let total_frames = combined.len();
            let cut_samples = ref_len * audio.len() / total_frames.max(1);
            audio.samples = audio.samples[cut_samples.min(audio.len())..].to_vec();
            Ok(audio)
        } else {
            self.decode_codes(&all_codes)
        }
    }

    /// Create a streaming voice clone synthesis session.
    pub fn synthesize_voice_clone_streaming(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt<B>,
        language: Language,
        options: SynthesisOptions,
    ) -> anyhow::Result<StreamingSession<'_, B>> {
        let input_ids = self.text_tokenizer.encode(text)?;
        StreamingSession::new_voice_clone(self, &input_ids, prompt, language, options)
    }

    /// Sum reference codec embeddings across all 16 codebooks.
    ///
    /// Group 0 (semantic) uses talker's codec_embedding.
    /// Groups 1-15 (acoustic) use code_predictor's codec_embeddings.
    ///
    /// Returns `[1, T_frames, hidden_size]`.
    fn sum_ref_codec_embeddings(&self, ref_codes: &[Vec<u32>]) -> Tensor<B, 3> {
        let n_frames = ref_codes.len();
        assert!(n_frames > 0, "ref_codes must not be empty");

        // Group 0: semantic codes → talker.codec_embedding
        let semantic_ids: Vec<i32> = ref_codes.iter().map(|f| f[0] as i32).collect();
        let semantic_tensor = Tensor::<B, 1, Int>::from_ints(semantic_ids.as_slice(), &self.device);
        let mut summed = self.talker.get_codec_embedding_batch(semantic_tensor); // [1, T, hidden]

        // Groups 1-15: acoustic codes → code_predictor.embed_codes_for_group
        for group in 1..16 {
            let group_ids: Vec<i32> = ref_codes.iter().map(|f| f[group] as i32).collect();
            let group_tensor = Tensor::<B, 1, Int>::from_ints(group_ids.as_slice(), &self.device);
            let group_embed = self
                .code_predictor
                .embed_codes_for_group(group - 1, group_tensor); // [1, T, embed_dim]
            summed = summed + group_embed;
        }

        summed
    }

    /// Convert frame codes to tensor [1, 16, T] for the decoder.
    pub fn codes_to_tensor(&self, codes: &[Vec<u32>]) -> Tensor<B, 3, Int> {
        let num_frames = codes.len();
        assert!(num_frames > 0, "Cannot decode empty codes");

        let mut data = vec![0i32; 16 * num_frames];
        for (frame, frame_codes) in codes.iter().enumerate() {
            for (q, &code) in frame_codes.iter().enumerate() {
                data[q * num_frames + frame] = code as i32;
            }
        }

        Tensor::<B, 1, Int>::from_ints(data.as_slice(), &self.device).reshape([1, 16, num_frames])
    }

    /// Decode frame codes to audio.
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> anyhow::Result<AudioBuffer> {
        if codes.is_empty() {
            return Ok(AudioBuffer::new(Vec::new(), 24000));
        }

        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        let tensor = self.codes_to_tensor(codes);
        let waveform = self.decoder.decode(tensor); // [1, 1, samples]
        let num_samples = waveform.dims()[2];
        let samples: Vec<f32> = waveform
            .reshape([num_samples])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        Ok(AudioBuffer::new(samples, 24000))
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Build trailing text embeddings from input_ids[1..] + tts_eos.
    fn build_trailing_text(&self, input_ids: &[u32]) -> (Tensor<B, 3>, usize, Tensor<B, 3>) {
        let trailing_text_hidden = if input_ids.len() > 1 {
            let remaining_proj = self
                .talker
                .get_projected_text_embeddings(&input_ids[1..], &self.device);
            let tts_eos_embed = self.talker.get_tts_eos_embed(&self.device);
            Tensor::cat(vec![remaining_proj, tts_eos_embed], 1)
        } else {
            self.talker.get_tts_eos_embed(&self.device)
        };
        let trailing_text_len = trailing_text_hidden.dims()[1];
        let tts_pad_embed = self.talker.get_tts_pad_embed(&self.device);
        (trailing_text_hidden, trailing_text_len, tts_pad_embed)
    }

    /// Core autoregressive generation loop.
    ///
    /// Shared by all synthesis methods. Callers handle prefill (which varies
    /// by model variant) and pass in the initial hidden state and logits.
    #[allow(clippy::too_many_arguments)]
    fn generate_codes(
        &self,
        gen_config: &GenerationConfig,
        sampling_ctx: &mut SamplingContext,
        kv_caches: &mut [KVCache<B>],
        mut offset: usize,
        mut last_hidden: Tensor<B, 3>,
        initial_logits: Tensor<B, 3>,
        trailing_text_hidden: &Tensor<B, 3>,
        trailing_text_len: usize,
        tts_pad_embed: &Tensor<B, 3>,
    ) -> FrameCodes {
        let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;

        // Pre-build suppression mask (reused every frame)
        let suppression =
            tts::build_suppression_mask::<B>(vocab_size, CODEC_EOS_TOKEN_ID, &self.device);

        // GPU-resident repetition penalty mask (all false initially)
        let mut penalty_mask = Tensor::<B, 1, Int>::zeros([vocab_size], &self.device).equal_elem(1);

        // Code predictor KV caches (reused + reset each frame)
        // CP processes: 2 prefill tokens + up to 14 autoregressive steps = 16
        let mut cp_kv_caches = self.code_predictor.new_kv_caches(17, &self.device);

        // Sample first semantic token from prefill logits
        let logits_2d = initial_logits.squeeze_dim::<2>(1);
        let logits_2d =
            self.apply_generation_penalties(logits_2d, &penalty_mask, gen_config, 0, &suppression);
        let mut semantic_token = sampling::sample(logits_2d, gen_config, sampling_ctx);
        if (semantic_token as usize) < vocab_size {
            penalty_mask =
                set_penalty_bit::<B>(&penalty_mask, semantic_token as usize, &self.device);
        }
        let mut token_count: usize = 1;

        let mut all_codes: FrameCodes = Vec::new();

        #[cfg(feature = "profiling")]
        let _gen_span = tracing::info_span!("generate_frames").entered();

        #[cfg(feature = "profiling")]
        let mut t_cp_total = std::time::Duration::ZERO;
        #[cfg(feature = "profiling")]
        let mut t_talker_total = std::time::Duration::ZERO;
        #[cfg(feature = "profiling")]
        let mut t_embed_total = std::time::Duration::ZERO;
        #[cfg(feature = "profiling")]
        let mut t_sample_total = std::time::Duration::ZERO;
        #[cfg(feature = "profiling")]
        let loop_start = std::time::Instant::now();

        for frame_idx in 0..gen_config.max_new_tokens {
            // Check EOS
            if let Some(eos_id) = gen_config.eos_token_id {
                if semantic_token == eos_id {
                    break;
                }
            }

            // Embed semantic token
            let semantic_embed = self
                .talker
                .get_codec_embedding(semantic_token, &self.device);

            // Generate 15 acoustic codes
            #[cfg(feature = "profiling")]
            let _cp_span = tracing::info_span!("code_predictor", frame = frame_idx).entered();

            #[cfg(feature = "profiling")]
            let t = std::time::Instant::now();
            let (acoustic_codes, acoustic_embed_sum) =
                self.code_predictor.generate_acoustic_codes(
                    last_hidden.clone(),
                    semantic_embed.clone(),
                    &self.cp_rope,
                    &mut cp_kv_caches,
                    &self.device,
                );
            #[cfg(feature = "profiling")]
            {
                t_cp_total += t.elapsed();
            }

            #[cfg(feature = "profiling")]
            drop(_cp_span);

            // Build frame: [semantic, acoustic_0..14]
            let mut frame = Vec::with_capacity(16);
            frame.push(semantic_token);
            frame.extend_from_slice(&acoustic_codes);
            all_codes.push(frame);

            // Residual VQ: sum semantic + all acoustic embeddings
            #[cfg(feature = "profiling")]
            let t = std::time::Instant::now();
            let summed = semantic_embed + acoustic_embed_sum;

            // Trailing text fusion
            let text_addition = if frame_idx < trailing_text_len {
                trailing_text_hidden.clone().narrow(1, frame_idx, 1)
            } else {
                tts_pad_embed.clone()
            };
            let step_input = summed + text_addition;
            #[cfg(feature = "profiling")]
            {
                t_embed_total += t.elapsed();
            }

            // Talker step
            #[cfg(feature = "profiling")]
            let _talker_span = tracing::info_span!("talker_step", frame = frame_idx).entered();

            #[cfg(feature = "profiling")]
            let t = std::time::Instant::now();
            let (h, new_logits) = self
                .talker
                .generate_step_with_embed(step_input, &self.rope, kv_caches, offset);
            offset += 1;
            last_hidden = h;
            #[cfg(feature = "profiling")]
            {
                t_talker_total += t.elapsed();
            }

            #[cfg(feature = "profiling")]
            drop(_talker_span);

            // Sample next semantic token
            #[cfg(feature = "profiling")]
            let t = std::time::Instant::now();
            let logits_2d = new_logits.squeeze_dim::<2>(1);
            let logits_2d = self.apply_generation_penalties(
                logits_2d,
                &penalty_mask,
                gen_config,
                token_count,
                &suppression,
            );
            semantic_token = sampling::sample(logits_2d, gen_config, sampling_ctx);
            if (semantic_token as usize) < vocab_size {
                penalty_mask =
                    set_penalty_bit::<B>(&penalty_mask, semantic_token as usize, &self.device);
            }
            token_count += 1;
            #[cfg(feature = "profiling")]
            {
                t_sample_total += t.elapsed();
            }
        }

        #[cfg(feature = "profiling")]
        {
            let loop_elapsed = loop_start.elapsed();
            let frames = all_codes.len();
            eprintln!(
                "Generation loop: {} frames in {:.1?} ({:.1}ms/frame)",
                frames,
                loop_elapsed,
                loop_elapsed.as_secs_f64() * 1000.0 / frames.max(1) as f64
            );
            eprintln!(
                "  Code predictor: {:.1?} ({:.1}%)",
                t_cp_total,
                t_cp_total.as_secs_f64() / loop_elapsed.as_secs_f64() * 100.0
            );
            eprintln!(
                "  Talker step:    {:.1?} ({:.1}%)",
                t_talker_total,
                t_talker_total.as_secs_f64() / loop_elapsed.as_secs_f64() * 100.0
            );
            eprintln!(
                "  Embed+fuse:     {:.1?} ({:.1}%)",
                t_embed_total,
                t_embed_total.as_secs_f64() / loop_elapsed.as_secs_f64() * 100.0
            );
            eprintln!(
                "  Sampling:       {:.1?} ({:.1}%)",
                t_sample_total,
                t_sample_total.as_secs_f64() / loop_elapsed.as_secs_f64() * 100.0
            );
        }

        all_codes
    }

    /// Apply repetition penalty, token suppression, and min_new_tokens EOS mask.
    fn apply_generation_penalties(
        &self,
        logits: Tensor<B, 2>,
        penalty_mask: &Tensor<B, 1, Bool>,
        config: &GenerationConfig,
        token_count: usize,
        suppression: &SuppressionMask<B>,
    ) -> Tensor<B, 2> {
        // 1. Repetition penalty (GPU-resident)
        let logits = if config.repetition_penalty != 1.0 {
            sampling::apply_repetition_penalty_with_mask(
                logits,
                penalty_mask,
                config.repetition_penalty,
                &self.device,
            )
        } else {
            logits
        };

        // 2. Token suppression (device-resident mask)
        let logits = tts::apply_token_suppression_with_mask(logits, suppression);

        // 3. Min new tokens EOS suppression
        if token_count < config.min_new_tokens {
            if let Some(eos_id) = config.eos_token_id {
                let [batch, vocab] = logits.dims();
                let mut mask_data = vec![0.0f32; vocab];
                mask_data[eos_id as usize] = f32::NEG_INFINITY;
                let eos_mask = Tensor::<B, 1>::from_floats(mask_data.as_slice(), &self.device)
                    .reshape([1, vocab])
                    .expand([batch, vocab]);
                return logits + eos_mask;
            }
        }

        logits
    }
}

// ── Streaming session ────────────────────────────────────────────────────

/// Streaming synthesis session.
///
/// Yields audio chunks as they are generated. Use with
/// [`Qwen3TTS::synthesize_streaming`] or [`Qwen3TTS::synthesize_voice_design_streaming`].
pub struct StreamingSession<'a, B: Backend> {
    model: &'a Qwen3TTS<B>,
    config: GenerationConfig,
    sampling_ctx: SamplingContext,
    kv_caches: Vec<KVCache<B>>,
    offset: usize,
    last_hidden: Tensor<B, 3>,
    current_token: Option<u32>,
    frames_generated: usize,
    frame_buffer: FrameCodes,
    chunk_frames: usize,
    done: bool,
    trailing_text_hidden: Tensor<B, 3>,
    trailing_text_len: usize,
    tts_pad_embed: Tensor<B, 3>,
    penalty_mask: Tensor<B, 1, Bool>,
    token_count: usize,
    suppression_mask: SuppressionMask<B>,
    cp_kv_caches: Vec<KVCache<B>>,
}

impl<'a, B: Backend> StreamingSession<'a, B> {
    fn new(
        model: &'a Qwen3TTS<B>,
        input_ids: &[u32],
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> anyhow::Result<Self> {
        let sampling_ctx = SamplingContext::new(options.seed);
        let config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            model.build_trailing_text(input_ids);

        let max_seq = input_ids.len() + 20 + config.max_new_tokens;
        let mut kv_caches = model.talker.new_kv_caches(max_seq, &model.device);
        let prefill_result = model.talker.prefill_custom_voice(
            input_ids,
            speaker,
            language,
            &model.rope,
            &mut kv_caches,
            &model.device,
        );

        Ok(Self::from_prefill(
            model,
            config,
            sampling_ctx,
            kv_caches,
            prefill_result,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            options.chunk_frames,
        ))
    }

    fn new_voice_design(
        model: &'a Qwen3TTS<B>,
        input_ids: &[u32],
        instruct_ids: &[u32],
        language: Language,
        options: SynthesisOptions,
    ) -> anyhow::Result<Self> {
        let sampling_ctx = SamplingContext::new(options.seed);
        let config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            model.build_trailing_text(input_ids);

        let max_seq = input_ids.len() + instruct_ids.len() + 20 + config.max_new_tokens;
        let mut kv_caches = model.talker.new_kv_caches(max_seq, &model.device);
        let prefill_result = model.talker.prefill_voice_design(
            input_ids,
            instruct_ids,
            language,
            &model.rope,
            &mut kv_caches,
            &model.device,
        );

        Ok(Self::from_prefill(
            model,
            config,
            sampling_ctx,
            kv_caches,
            prefill_result,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            options.chunk_frames,
        ))
    }

    fn new_voice_clone(
        model: &'a Qwen3TTS<B>,
        input_ids: &[u32],
        prompt: &VoiceClonePrompt<B>,
        language: Language,
        options: SynthesisOptions,
    ) -> anyhow::Result<Self> {
        let sampling_ctx = SamplingContext::new(options.seed);
        let is_icl = prompt.ref_codes.is_some() && prompt.ref_text_ids.is_some();

        let mut config = options.to_gen_config();
        if is_icl {
            config.repetition_penalty = config.repetition_penalty.max(ICL_MIN_REPETITION_PENALTY);
            config.max_new_tokens = config
                .max_new_tokens
                .min(ICL_MIN_FRAMES.max(input_ids.len() * ICL_FRAMES_PER_TOKEN));
        }

        let speaker_embed = prompt
            .speaker_embedding
            .clone()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0);

        let max_seq = input_ids.len() + 256 + config.max_new_tokens;
        let mut kv_caches = model.talker.new_kv_caches(max_seq, &model.device);
        let (hidden, logits) = model.talker.prefill_voice_clone(
            input_ids,
            speaker_embed,
            language,
            is_icl,
            &model.rope,
            &mut kv_caches,
            &model.device,
        );
        let prefill_len = hidden.dims()[1];
        let mut offset = prefill_len;
        let mut last_hidden = hidden.narrow(1, prefill_len - 1, 1);

        // ICL extension
        let (trailing_text_hidden, trailing_text_len, tts_pad_embed, logits) =
            if let (Some(ref ref_codes), Some(ref ref_text_ids)) =
                (&prompt.ref_codes, &prompt.ref_text_ids)
            {
                let ref_codec_embeds = model.sum_ref_codec_embeddings(ref_codes);

                let (icl_embed, icl_trailing) = model.talker.build_icl_prompt(
                    input_ids,
                    ref_text_ids,
                    &ref_codec_embeds,
                    &model.device,
                );

                let icl_len = icl_embed.dims()[1];
                if icl_len > 0 {
                    let (icl_last_hidden, icl_logits) =
                        model
                            .talker
                            .run_icl_layers(icl_embed, &model.rope, &mut kv_caches, offset);
                    offset += icl_len;
                    last_hidden = icl_last_hidden;

                    let trailing_len = icl_trailing.dims()[1];
                    let tts_pad = model.talker.get_tts_pad_embed(&model.device);
                    (icl_trailing, trailing_len, tts_pad, icl_logits)
                } else {
                    let (trailing, trailing_len, tts_pad) = model.build_trailing_text(input_ids);
                    (trailing, trailing_len, tts_pad, logits)
                }
            } else {
                let (trailing, trailing_len, tts_pad) = model.build_trailing_text(input_ids);
                (trailing, trailing_len, tts_pad, logits)
            };

        // Now build the session using the resolved state.
        // We can't use from_prefill because we've already extracted last_hidden
        // and may have modified offset via ICL.
        let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;
        let suppression_mask =
            tts::build_suppression_mask::<B>(vocab_size, CODEC_EOS_TOKEN_ID, &model.device);
        let mut penalty_mask =
            Tensor::<B, 1, Int>::zeros([vocab_size], &model.device).equal_elem(1);

        let logits_2d = logits.squeeze_dim::<2>(1);
        let logits_2d = model.apply_generation_penalties(
            logits_2d,
            &penalty_mask,
            &config,
            0,
            &suppression_mask,
        );
        let mut sampling_ctx = sampling_ctx;
        let first_token = sampling::sample(logits_2d, &config, &mut sampling_ctx);
        if (first_token as usize) < vocab_size {
            penalty_mask = set_penalty_bit::<B>(&penalty_mask, first_token as usize, &model.device);
        }

        let done = config.eos_token_id == Some(first_token);
        let cp_kv_caches = model.code_predictor.new_kv_caches(17, &model.device);

        Ok(Self {
            model,
            config,
            sampling_ctx,
            kv_caches,
            offset,
            last_hidden,
            current_token: if done { None } else { Some(first_token) },
            frames_generated: 0,
            frame_buffer: Vec::new(),
            chunk_frames: options.chunk_frames,
            done,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            penalty_mask,
            token_count: 1,
            suppression_mask,
            cp_kv_caches,
        })
    }

    /// Shared post-prefill constructor.
    #[allow(clippy::too_many_arguments)]
    fn from_prefill(
        model: &'a Qwen3TTS<B>,
        config: GenerationConfig,
        mut sampling_ctx: SamplingContext,
        kv_caches: Vec<KVCache<B>>,
        prefill_result: (Tensor<B, 3>, Tensor<B, 3>),
        trailing_text_hidden: Tensor<B, 3>,
        trailing_text_len: usize,
        tts_pad_embed: Tensor<B, 3>,
        chunk_frames: usize,
    ) -> Self {
        let (hidden, logits) = prefill_result;
        let prefill_len = hidden.dims()[1];
        let last_hidden = hidden.narrow(1, prefill_len - 1, 1);

        let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;
        let suppression_mask =
            tts::build_suppression_mask::<B>(vocab_size, CODEC_EOS_TOKEN_ID, &model.device);

        let mut penalty_mask =
            Tensor::<B, 1, Int>::zeros([vocab_size], &model.device).equal_elem(1);

        // Sample first semantic token
        let logits_2d = logits.squeeze_dim::<2>(1);
        let logits_2d = model.apply_generation_penalties(
            logits_2d,
            &penalty_mask,
            &config,
            0,
            &suppression_mask,
        );
        let first_token = sampling::sample(logits_2d, &config, &mut sampling_ctx);
        if (first_token as usize) < vocab_size {
            penalty_mask = set_penalty_bit::<B>(&penalty_mask, first_token as usize, &model.device);
        }

        let done = config.eos_token_id == Some(first_token);
        let cp_kv_caches = model.code_predictor.new_kv_caches(17, &model.device);

        Self {
            model,
            config,
            sampling_ctx,
            kv_caches,
            offset: prefill_len,
            last_hidden,
            current_token: if done { None } else { Some(first_token) },
            frames_generated: 0,
            frame_buffer: Vec::new(),
            chunk_frames,
            done,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            penalty_mask,
            token_count: 1,
            suppression_mask,
            cp_kv_caches,
        }
    }

    /// Generate the next chunk of audio.
    ///
    /// Returns `Some(AudioBuffer)` for each chunk, or `None` when generation is complete.
    pub fn next_chunk(&mut self) -> anyhow::Result<Option<AudioBuffer>> {
        if self.done {
            // Flush remaining buffer
            if !self.frame_buffer.is_empty() {
                let audio = self.model.decode_codes(&self.frame_buffer)?;
                self.frame_buffer.clear();
                return Ok(Some(audio));
            }
            return Ok(None);
        }

        // Generate frames until we have enough for a chunk
        while self.frame_buffer.len() < self.chunk_frames
            && self.frames_generated < self.config.max_new_tokens
        {
            let token_id = match self.current_token {
                Some(id) => id,
                None => {
                    self.done = true;
                    break;
                }
            };

            // Embed semantic token
            let semantic_embed = self
                .model
                .talker
                .get_codec_embedding(token_id, &self.model.device);

            // Generate 15 acoustic codes
            let (acoustic_codes, acoustic_embed_sum) =
                self.model.code_predictor.generate_acoustic_codes(
                    self.last_hidden.clone(),
                    semantic_embed.clone(),
                    &self.model.cp_rope,
                    &mut self.cp_kv_caches,
                    &self.model.device,
                );

            // Build frame
            let mut frame = Vec::with_capacity(16);
            frame.push(token_id);
            frame.extend_from_slice(&acoustic_codes);
            self.frame_buffer.push(frame);

            let frame_idx = self.frames_generated;
            self.frames_generated += 1;

            // Residual VQ sum + trailing text fusion
            let summed = semantic_embed + acoustic_embed_sum;

            let text_addition = if frame_idx < self.trailing_text_len {
                self.trailing_text_hidden.clone().narrow(1, frame_idx, 1)
            } else {
                self.tts_pad_embed.clone()
            };
            let step_input = summed + text_addition;

            // Talker step
            let (h, new_logits) = self.model.talker.generate_step_with_embed(
                step_input,
                &self.model.rope,
                &mut self.kv_caches,
                self.offset,
            );
            self.offset += 1;
            self.last_hidden = h;

            // Sample next semantic token
            let logits_2d = new_logits.squeeze_dim::<2>(1);
            let logits_2d = self.model.apply_generation_penalties(
                logits_2d,
                &self.penalty_mask,
                &self.config,
                self.token_count,
                &self.suppression_mask,
            );
            let next_token = sampling::sample(logits_2d, &self.config, &mut self.sampling_ctx);
            let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;
            if (next_token as usize) < vocab_size {
                self.penalty_mask = set_penalty_bit::<B>(
                    &self.penalty_mask,
                    next_token as usize,
                    &self.model.device,
                );
            }
            self.token_count += 1;

            if self.config.eos_token_id == Some(next_token) {
                self.current_token = None;
                self.done = true;
            } else {
                self.current_token = Some(next_token);
            }
        }

        // Decode buffered frames
        if self.frame_buffer.is_empty() {
            return Ok(None);
        }

        let audio = self.model.decode_codes(&self.frame_buffer)?;
        self.frame_buffer.clear();
        Ok(Some(audio))
    }

    /// Returns the total number of frames generated so far.
    pub fn frames_generated(&self) -> usize {
        self.frames_generated
    }

    /// Returns true if generation is complete.
    pub fn is_done(&self) -> bool {
        self.done && self.frame_buffer.is_empty()
    }
}

impl<'a, B: Backend> Iterator for StreamingSession<'a, B> {
    type Item = anyhow::Result<AudioBuffer>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(audio)) => Some(Ok(audio)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_options_default() {
        let options = SynthesisOptions::default();
        assert_eq!(options.max_length, 2048);
        assert!((options.temperature - 0.3).abs() < 1e-6);
        assert_eq!(options.top_k, 20);
        assert_eq!(options.eos_token_id, Some(CODEC_EOS_TOKEN_ID));
        assert_eq!(options.chunk_frames, 10);
    }

    #[test]
    fn test_synthesis_options_to_gen_config() {
        let options = SynthesisOptions {
            max_length: 100,
            temperature: 0.5,
            top_k: 20,
            top_p: 0.8,
            repetition_penalty: 1.2,
            eos_token_id: Some(42),
            min_new_tokens: 5,
            ..Default::default()
        };
        let config = options.to_gen_config();
        assert_eq!(config.max_new_tokens, 100);
        assert!((config.temperature - 0.5).abs() < 1e-6);
        assert_eq!(config.top_k, 20);
        assert_eq!(config.eos_token_id, Some(42));
        assert_eq!(config.min_new_tokens, 5);
    }
}
