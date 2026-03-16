//! TalkerModel for autoregressive semantic token generation (Burn)

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;

use super::kv_cache::KVCache;
use super::transformer::{DecoderLayer, DecoderLayerConfig, MRoPE, RoPEType, RotaryEmbedding};

// Re-export framework-independent types
pub use crate::models::talker_types::{
    codec_tokens, special_tokens, tts_tokens, Language, Speaker, TalkerConfig,
};

/// Text projection with SwiGLU activation.
/// Maps text embeddings (2048) to hidden dimension (1024).
#[derive(Module, Debug)]
pub struct TextProjection<B: Backend> {
    pub(crate) fc1: Linear<B>,
    pub(crate) fc2: Linear<B>,
}

impl<B: Backend> TextProjection<B> {
    pub fn new(config: &TalkerConfig, device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(config.text_embed_dim, config.text_proj_intermediate)
            .with_bias(true)
            .init(device);
        let fc2 = LinearConfig::new(config.text_proj_intermediate, config.hidden_size)
            .with_bias(true)
            .init(device);
        Self { fc1, fc2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.fc1.forward(x);
        let hidden = burn::tensor::activation::silu(hidden);
        self.fc2.forward(hidden)
    }
}

/// TalkerModel for autoregressive semantic token generation.
#[derive(Module, Debug)]
pub struct TalkerModel<B: Backend> {
    pub(crate) text_embedding: Embedding<B>,
    pub(crate) text_projection: TextProjection<B>,
    pub(crate) codec_embedding: Embedding<B>,
    pub(crate) layers: Vec<DecoderLayer<B>>,
    pub(crate) norm: RmsNorm<B>,
    pub(crate) codec_head: Linear<B>,
    // Config stored as individual fields to avoid Module derive issues
    #[module(skip)]
    text_vocab_size: usize,
    #[module(skip)]
    text_embed_dim: usize,
    #[module(skip)]
    hidden_size: usize,
    #[module(skip)]
    intermediate_size: usize,
    #[module(skip)]
    num_hidden_layers: usize,
    #[module(skip)]
    num_attention_heads: usize,
    #[module(skip)]
    num_key_value_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    rms_norm_eps: f64,
    #[module(skip)]
    rope_theta: f64,
    #[module(skip)]
    max_position_embeddings: usize,
    #[module(skip)]
    codec_vocab_size: usize,
    #[module(skip)]
    mrope_section: Option<[usize; 3]>,
}

impl<B: Backend> TalkerModel<B> {
    /// Initialize from config.
    pub fn init(config: TalkerConfig, device: &B::Device) -> Self {
        let text_embedding =
            EmbeddingConfig::new(config.text_vocab_size, config.text_embed_dim).init(device);
        let text_projection = TextProjection::new(&config, device);
        let codec_embedding =
            EmbeddingConfig::new(config.codec_vocab_size, config.hidden_size).init(device);
        let norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);
        let codec_head = LinearConfig::new(config.hidden_size, config.codec_vocab_size)
            .with_bias(false)
            .init(device);

        let layer_config = DecoderLayerConfig::new(
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
        );
        let layers = (0..config.num_hidden_layers)
            .map(|_| layer_config.init(device))
            .collect();

        Self {
            text_embedding,
            text_projection,
            codec_embedding,
            layers,
            norm,
            codec_head,
            text_vocab_size: config.text_vocab_size,
            text_embed_dim: config.text_embed_dim,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_position_embeddings: config.max_position_embeddings,
            codec_vocab_size: config.codec_vocab_size,
            mrope_section: config.mrope_section,
        }
    }

    /// Reconstruct config from stored fields.
    pub fn config(&self) -> TalkerConfig {
        TalkerConfig {
            text_vocab_size: self.text_vocab_size,
            text_embed_dim: self.text_embed_dim,
            hidden_size: self.hidden_size,
            text_proj_intermediate: self.text_embed_dim,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            codec_vocab_size: self.codec_vocab_size,
            mrope_section: self.mrope_section,
        }
    }

    // ── Embedding helpers (Burn Embedding::forward expects 2D input) ─────

    /// Embed text tokens: [seq_len] → [1, seq_len, text_embed_dim]
    fn embed_text(&self, ids: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        self.text_embedding.forward(ids.unsqueeze::<2>())
    }

    /// Embed codec tokens: [seq_len] → [1, seq_len, hidden_size]
    fn embed_codec(&self, ids: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        self.codec_embedding.forward(ids.unsqueeze::<2>())
    }

    // ── Public API ───────────────────────────────────────────────────────

    /// Create the appropriate RoPE for this model.
    pub fn create_rope(&self, device: &B::Device) -> RoPEType<B> {
        if let Some(mrope_section) = self.mrope_section {
            RoPEType::Multimodal(MRoPE::new(
                self.head_dim,
                self.rope_theta,
                mrope_section,
                device,
            ))
        } else {
            RoPEType::Standard(RotaryEmbedding::new(
                self.head_dim,
                self.max_position_embeddings,
                self.rope_theta,
                device,
            ))
        }
    }

    /// Create pre-allocated KV caches for generation (one per layer).
    ///
    /// `max_seq` is the maximum total sequence length (prefill + decode steps).
    pub fn new_kv_caches(&self, max_seq: usize, device: &B::Device) -> Vec<KVCache<B>> {
        (0..self.num_hidden_layers)
            .map(|_| KVCache::new(1, self.num_key_value_heads, max_seq, self.head_dim, device))
            .collect()
    }

    /// Get projected text embeddings for a sequence of token IDs.
    ///
    /// Returns `[1, seq_len, hidden_size]` tensor.
    pub fn get_projected_text_embeddings(
        &self,
        token_ids: &[u32],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        if token_ids.is_empty() {
            return Tensor::<B, 3>::zeros([1, 0, self.hidden_size], device);
        }

        let ids: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let ids_tensor = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device);
        let embeds = self.embed_text(ids_tensor); // [1, seq_len, text_embed_dim]
        self.text_projection.forward(embeds)
    }

    /// Get codec embedding for a token.
    ///
    /// Returns `[1, 1, hidden_size]` tensor.
    pub fn get_codec_embedding(&self, token_id: u32, device: &B::Device) -> Tensor<B, 3> {
        let token = Tensor::<B, 1, Int>::from_ints([token_id as i32], device);
        self.embed_codec(token) // [1, 1, hidden_size]
    }

    /// Get codec embedding from a tensor (avoids CPU->GPU roundtrip).
    pub fn get_codec_embedding_from_tensor(&self, token: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        self.embed_codec(token)
    }

    /// Get tts_pad text embedding (projected).
    pub fn get_tts_pad_embed(&self, device: &B::Device) -> Tensor<B, 3> {
        self.get_projected_special_embed(tts_tokens::TTS_PAD, device)
    }

    /// Get tts_eos text embedding (projected).
    pub fn get_tts_eos_embed(&self, device: &B::Device) -> Tensor<B, 3> {
        self.get_projected_special_embed(tts_tokens::TTS_EOS, device)
    }

    fn get_projected_special_embed(&self, token_id: u32, device: &B::Device) -> Tensor<B, 3> {
        let id = Tensor::<B, 1, Int>::from_ints([token_id as i32], device);
        let embed = self.embed_text(id); // [1, 1, text_embed_dim]
        self.text_projection.forward(embed)
    }

    /// Build role prefix: text_proj([im_start, assistant, newline]).
    ///
    /// Returns `[1, 3, hidden_size]`.
    fn build_role_prefix(&self, device: &B::Device) -> Tensor<B, 3> {
        use special_tokens::*;
        let ids = Tensor::<B, 1, Int>::from_ints(
            [IM_START as i32, ASSISTANT as i32, NEWLINE as i32],
            device,
        );
        let embed = self.embed_text(ids); // [1, 3, text_embed_dim]
        self.text_projection.forward(embed)
    }

    /// Build tts_pad (projected, count copies) and tts_bos (projected, 1 copy).
    ///
    /// Returns `[1, pad_count + 1, hidden_size]`.
    fn build_tts_pad_bos(&self, pad_count: usize, device: &B::Device) -> Tensor<B, 3> {
        let tts_pad_proj = self.get_projected_special_embed(tts_tokens::TTS_PAD, device);
        let tts_bos_proj = self.get_projected_special_embed(tts_tokens::TTS_BOS, device);

        let tts_pad_expanded = tts_pad_proj.expand([1, pad_count, self.hidden_size]);
        Tensor::cat(vec![tts_pad_expanded, tts_bos_proj], 1)
    }

    /// Build first text token combined with codec_bos embedding.
    fn build_first_text_combined(
        &self,
        text_tokens: &[u32],
        codec_bos_embed: Tensor<B, 3>,
        device: &B::Device,
    ) -> Option<Tensor<B, 3>> {
        if text_tokens.is_empty() {
            return None;
        }
        let first_text_id = Tensor::<B, 1, Int>::from_ints([text_tokens[0] as i32], device);
        let first_text_embed = self.embed_text(first_text_id); // [1, 1, text_embed_dim]
        let first_text_proj = self.text_projection.forward(first_text_embed);
        Some(first_text_proj + codec_bos_embed)
    }

    /// Run prefill through all layers: causal mask -> layers -> norm -> logits.
    ///
    /// Returns `(hidden_states, logits)` for the full sequence.
    fn run_prefill_layers(
        &self,
        mut hidden: Tensor<B, 3>,
        rope: &RoPEType<B>,
        kv_caches: &mut [KVCache<B>],
        _device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(hidden, rope, true, Some(&mut kv_caches[i]), 0);
        }

        hidden = self.norm.forward(hidden);

        let seq_len = hidden.dims()[1];
        let last_hidden = hidden.clone().narrow(1, seq_len - 1, 1);
        let logits = self.codec_head.forward(last_hidden);

        (hidden, logits)
    }

    /// Prefill for CustomVoice model with speaker and language.
    pub fn prefill_custom_voice(
        &self,
        text_tokens: &[u32],
        speaker: Speaker,
        language: Language,
        rope: &RoPEType<B>,
        kv_caches: &mut [KVCache<B>],
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        use codec_tokens::*;

        let role_prefix_hidden = self.build_role_prefix(device);

        let codec_ids = Tensor::<B, 1, Int>::from_ints(
            [
                CODEC_THINK as i32,
                CODEC_THINK_BOS as i32,
                language.token_id() as i32,
                CODEC_THINK_EOS as i32,
                speaker.token_id() as i32,
                CODEC_PAD as i32,
                CODEC_BOS as i32,
            ],
            device,
        );
        let codec_embed = self.embed_codec(codec_ids); // [1, 7, hidden_size]

        // 5 x tts_pad + 1 x tts_bos overlaid on first 6 codec tokens
        let tts_text_embed = self.build_tts_pad_bos(5, device);
        let codec_first6 = codec_embed.clone().narrow(1, 0, 6);
        let codec_hidden = tts_text_embed + codec_first6;

        let mut hidden = Tensor::cat(vec![role_prefix_hidden, codec_hidden], 1);

        // First text token + codec_bos
        let codec_bos_embed = codec_embed.narrow(1, 6, 1);
        if let Some(combined) = self.build_first_text_combined(text_tokens, codec_bos_embed, device)
        {
            hidden = Tensor::cat(vec![hidden, combined], 1);
        }

        self.run_prefill_layers(hidden, rope, kv_caches, device)
    }

    /// Prefill for voice cloning (x_vector_only mode).
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_voice_clone(
        &self,
        text_tokens: &[u32],
        speaker_embed: Tensor<B, 3>,
        language: Language,
        icl_mode: bool,
        rope: &RoPEType<B>,
        kv_caches: &mut [KVCache<B>],
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        use codec_tokens::*;

        let role_prefix_hidden = self.build_role_prefix(device);

        let codec_prefix_ids = Tensor::<B, 1, Int>::from_ints(
            [
                CODEC_THINK as i32,
                CODEC_THINK_BOS as i32,
                language.token_id() as i32,
                CODEC_THINK_EOS as i32,
            ],
            device,
        );
        let codec_prefix_embed = self.embed_codec(codec_prefix_ids); // [1, 4, hidden_size]

        let speaker = speaker_embed.reshape([1, 1, self.hidden_size]);

        let codec_suffix_ids =
            Tensor::<B, 1, Int>::from_ints([CODEC_PAD as i32, CODEC_BOS as i32], device);
        let codec_suffix_embed = self.embed_codec(codec_suffix_ids); // [1, 2, hidden_size]

        let codec_embed = Tensor::cat(vec![codec_prefix_embed, speaker, codec_suffix_embed], 1);

        let tts_text_embed = self.build_tts_pad_bos(5, device);
        let codec_first6 = codec_embed.clone().narrow(1, 0, 6);
        let codec_hidden = tts_text_embed + codec_first6;

        let mut hidden = Tensor::cat(vec![role_prefix_hidden, codec_hidden], 1);

        if !icl_mode {
            let codec_bos_embed = codec_embed.narrow(1, 6, 1);
            if let Some(combined) =
                self.build_first_text_combined(text_tokens, codec_bos_embed, device)
            {
                hidden = Tensor::cat(vec![hidden, combined], 1);
            }
        }

        self.run_prefill_layers(hidden, rope, kv_caches, device)
    }

    /// Prefill for VoiceDesign model.
    pub fn prefill_voice_design(
        &self,
        text_tokens: &[u32],
        instruct_tokens: &[u32],
        language: Language,
        rope: &RoPEType<B>,
        kv_caches: &mut [KVCache<B>],
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        use codec_tokens::*;

        let instruct_embed = self.get_projected_text_embeddings(instruct_tokens, device);
        let role_prefix_hidden = self.build_role_prefix(device);

        let codec_ids = Tensor::<B, 1, Int>::from_ints(
            [
                CODEC_THINK as i32,
                CODEC_THINK_BOS as i32,
                language.token_id() as i32,
                CODEC_THINK_EOS as i32,
                CODEC_PAD as i32,
                CODEC_BOS as i32,
            ],
            device,
        );
        let codec_embed = self.embed_codec(codec_ids); // [1, 6, hidden_size]

        let tts_text_embed = self.build_tts_pad_bos(4, device);
        let codec_first5 = codec_embed.clone().narrow(1, 0, 5);
        let codec_hidden = tts_text_embed + codec_first5;

        let mut hidden = Tensor::cat(vec![instruct_embed, role_prefix_hidden, codec_hidden], 1);

        let codec_bos_embed = codec_embed.narrow(1, 5, 1);
        if let Some(combined) = self.build_first_text_combined(text_tokens, codec_bos_embed, device)
        {
            hidden = Tensor::cat(vec![hidden, combined], 1);
        }

        self.run_prefill_layers(hidden, rope, kv_caches, device)
    }

    /// Build ICL (in-context learning) prompt for voice cloning.
    ///
    /// Combines reference text, target text, and reference codec embeddings into
    /// the ICL input sequence. Uses streaming layout: element-wise overlay of
    /// text and codec embeddings where they overlap.
    ///
    /// # Returns
    /// `(icl_embed, trailing_text_embed)` — the ICL sequence to process through
    /// layers, and the trailing text for the generation loop.
    pub fn build_icl_prompt(
        &self,
        target_text_ids: &[u32],
        ref_text_ids: &[u32],
        ref_codec_embeds: &Tensor<B, 3>, // [1, T_ref, hidden]
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        use codec_tokens::*;

        // 1. Text embeddings: [ref_text, target_text, tts_eos] projected
        let mut all_text_ids: Vec<u32> =
            Vec::with_capacity(ref_text_ids.len() + target_text_ids.len() + 1);
        all_text_ids.extend_from_slice(ref_text_ids);
        all_text_ids.extend_from_slice(target_text_ids);
        all_text_ids.push(tts_tokens::TTS_EOS);

        let text_embed = self.get_projected_text_embeddings(&all_text_ids, device); // [1, N_text, hidden]
        let n_text = text_embed.dims()[1];

        // 2. Codec embeddings: prepend codec_bos, then ref_codec_embeds
        let bos_embed = self.get_codec_embedding(CODEC_BOS, device); // [1, 1, hidden]
        let codec_embed = Tensor::cat(vec![bos_embed, ref_codec_embeds.clone()], 1); // [1, T_ref+1, hidden]
        let n_codec = codec_embed.dims()[1];

        let tts_pad_embed = self.get_tts_pad_embed(device); // [1, 1, hidden]

        // 3. Streaming layout: element-wise overlay
        if n_text > n_codec {
            let text_head = text_embed.clone().narrow(1, 0, n_codec);
            let icl_embed = text_head + codec_embed;
            let trailing = text_embed.narrow(1, n_codec, n_text - n_codec);
            (icl_embed, trailing)
        } else {
            let pad_count = n_codec - n_text;
            let padded_text = if pad_count > 0 {
                let pad_broadcast = tts_pad_embed
                    .clone()
                    .expand([1, pad_count, self.hidden_size]);
                Tensor::cat(vec![text_embed, pad_broadcast], 1)
            } else {
                text_embed
            };
            let icl_embed = padded_text + codec_embed;
            (icl_embed, tts_pad_embed)
        }
    }

    /// Run ICL prompt through transformer layers (not using run_prefill_layers
    /// because ICL starts at a non-zero offset after voice clone prefill).
    ///
    /// Returns `(last_hidden_state, logits)` from the final ICL position.
    pub fn run_icl_layers(
        &self,
        icl_embed: Tensor<B, 3>,
        rope: &RoPEType<B>,
        kv_caches: &mut [KVCache<B>],
        offset: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut hidden = icl_embed;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(hidden, rope, true, Some(&mut kv_caches[i]), offset);
        }

        hidden = self.norm.forward(hidden);

        let icl_len = hidden.dims()[1];
        let last_hidden = hidden.narrow(1, icl_len - 1, 1);
        let logits = self.codec_head.forward(last_hidden.clone());

        (last_hidden, logits)
    }

    /// Generate step with pre-built input embedding.
    pub fn generate_step_with_embed(
        &self,
        input_embed: Tensor<B, 3>,
        rope: &RoPEType<B>,
        kv_caches: &mut [KVCache<B>],
        offset: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut hidden = input_embed;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(hidden, rope, false, Some(&mut kv_caches[i]), offset);
        }

        hidden = self.norm.forward(hidden);
        let logits = self.codec_head.forward(hidden.clone());
        (hidden, logits)
    }

    /// Embed a batch of codec tokens.
    ///
    /// Returns `[1, T, hidden_size]`.
    pub fn get_codec_embedding_batch(&self, token_ids: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        self.embed_codec(token_ids)
    }

    /// Apply final RMS norm.
    pub fn apply_norm(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.norm.forward(hidden)
    }

    /// Apply codec head.
    pub fn apply_codec_head(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.codec_head.forward(hidden)
    }

    /// Get an iterator over transformer layers.
    pub fn layers_iter(&self) -> impl Iterator<Item = &DecoderLayer<B>> {
        self.layers.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_talker_config_default() {
        let config = TalkerConfig::default();
        assert_eq!(config.text_vocab_size, 151936);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 28);
    }
}
