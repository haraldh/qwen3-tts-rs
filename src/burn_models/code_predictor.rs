//! Code Predictor for Qwen3-TTS (Burn)
//!
//! Generates acoustic tokens (groups 2-16) given the semantic token (group 1)
//! and the hidden state from the talker model.

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;

use super::kv_cache::KVCache;
use super::transformer::{DecoderLayer, DecoderLayerConfig, RoPEType, RotaryEmbedding};

/// Code predictor configuration
#[derive(Debug, Clone)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub num_code_groups: usize,
    pub codec_embed_dim: Option<usize>,
}

impl Default for CodePredictorConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 5,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            vocab_size: 2048,
            num_code_groups: 16,
            codec_embed_dim: None,
        }
    }
}

impl CodePredictorConfig {
    /// Create config from parsed model config.
    pub fn from_parsed(parsed: &crate::models::config::ParsedModelConfig) -> Self {
        let codec_embed_dim = if parsed.talker_hidden_size != parsed.cp_hidden_size {
            Some(parsed.talker_hidden_size)
        } else {
            None
        };
        Self {
            hidden_size: parsed.cp_hidden_size,
            intermediate_size: parsed.cp_intermediate_size,
            num_hidden_layers: parsed.cp_num_hidden_layers,
            num_attention_heads: parsed.cp_num_attention_heads,
            num_key_value_heads: parsed.cp_num_key_value_heads,
            head_dim: parsed.cp_head_dim,
            rms_norm_eps: parsed.cp_rms_norm_eps,
            rope_theta: parsed.cp_rope_theta,
            vocab_size: parsed.cp_vocab_size,
            num_code_groups: parsed.cp_num_code_groups,
            codec_embed_dim,
        }
    }

    /// Get the codec embedding dimension (defaults to hidden_size).
    pub fn codec_embed_dim(&self) -> usize {
        self.codec_embed_dim.unwrap_or(self.hidden_size)
    }
}

/// Code predictor model
#[derive(Module, Debug)]
pub struct CodePredictor<B: Backend> {
    /// Codec embeddings for each acoustic group (0-14 for groups 2-16)
    pub(crate) codec_embeddings: Vec<Embedding<B>>,
    /// Projection from codec_embed_dim to hidden_size (for 1.7B models)
    pub(crate) small_to_mtp_projection: Option<Linear<B>>,
    /// Transformer layers
    pub(crate) layers: Vec<DecoderLayer<B>>,
    /// Final normalization
    pub(crate) norm: RmsNorm<B>,
    /// LM heads for each acoustic group (0-14 for groups 2-16)
    pub(crate) lm_heads: Vec<Linear<B>>,
    // Config stored as individual fields to avoid Module derive issues
    #[module(skip)]
    hidden_size: usize,
    #[module(skip)]
    num_hidden_layers: usize,
    #[module(skip)]
    num_code_groups: usize,
    #[module(skip)]
    num_key_value_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    rope_theta: f64,
}

impl<B: Backend> CodePredictor<B> {
    /// Initialize from config.
    pub fn init(config: CodePredictorConfig, device: &B::Device) -> Self {
        let num_acoustic_groups = config.num_code_groups - 1;
        let codec_embed_dim = config.codec_embed_dim();

        let codec_embeddings = (0..num_acoustic_groups)
            .map(|_| EmbeddingConfig::new(config.vocab_size, codec_embed_dim).init(device))
            .collect();

        let small_to_mtp_projection = if codec_embed_dim != config.hidden_size {
            Some(
                LinearConfig::new(codec_embed_dim, config.hidden_size)
                    .with_bias(true)
                    .init(device),
            )
        } else {
            None
        };

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

        let norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);

        let lm_heads = (0..num_acoustic_groups)
            .map(|_| {
                LinearConfig::new(config.hidden_size, config.vocab_size)
                    .with_bias(false)
                    .init(device)
            })
            .collect();

        Self {
            codec_embeddings,
            small_to_mtp_projection,
            layers,
            norm,
            lm_heads,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_code_groups: config.num_code_groups,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
        }
    }

    /// Create RoPE for the code predictor.
    pub fn create_rope(&self, device: &B::Device) -> RoPEType<B> {
        RoPEType::Standard(RotaryEmbedding::new(
            self.head_dim,
            1024,
            self.rope_theta,
            device,
        ))
    }

    /// Create pre-allocated KV caches for the code predictor (one per layer).
    ///
    /// The code predictor processes at most `prefill_len + 15` positions per frame
    /// (2 prefill tokens + up to 14 autoregressive steps).
    pub fn new_kv_caches(&self, max_seq: usize, device: &B::Device) -> Vec<KVCache<B>> {
        (0..self.num_hidden_layers)
            .map(|_| KVCache::new(1, self.num_key_value_heads, max_seq, self.head_dim, device))
            .collect()
    }

    /// Generate all 15 acoustic tokens autoregressively.
    pub fn generate_acoustic_codes(
        &self,
        talker_hidden: Tensor<B, 3>,
        semantic_embed: Tensor<B, 3>,
        rope: &RoPEType<B>,
        cp_kv_caches: &mut [KVCache<B>],
        device: &B::Device,
    ) -> Vec<u32> {
        for cache in cp_kv_caches.iter_mut() {
            cache.reset();
        }

        let num_acoustic = self.num_code_groups - 1;

        // Step 1: Prefill with [talker_hidden, semantic_embed]
        let input = Tensor::cat(vec![talker_hidden, semantic_embed], 1);

        let input = if let Some(proj) = &self.small_to_mtp_projection {
            proj.forward(input)
        } else {
            input
        };

        let seq_len = input.dims()[1];
        let mask = super::transformer::create_causal_mask::<B>(seq_len, 0, device);

        let mut hidden = input;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                hidden,
                rope,
                Some(mask.clone()),
                Some(&mut cp_kv_caches[i]),
                0,
            );
        }
        hidden = self.norm.forward(hidden);

        // Step 2: Predict first acoustic code from last position (stay on GPU)
        let last_hidden = hidden.narrow(1, seq_len - 1, 1);
        let logits = self.lm_heads[0].forward(last_hidden);
        // argmax(2) on [1,1,V] → [1,1,1]; squeeze to [1,1] for embedding lookup
        let mut prev_code_idx: Tensor<B, 2, Int> = logits.argmax(2).squeeze_dim::<2>(2);
        let mut code_tensors: Vec<Tensor<B, 2, Int>> = vec![prev_code_idx.clone()];

        // Step 3: Autoregressively generate remaining 14 codes (all on GPU)
        let mut offset = seq_len;
        for group_idx in 1..num_acoustic {
            // Use GPU-resident argmax result directly as embedding index
            let code_embed = self.codec_embeddings[group_idx - 1].forward(prev_code_idx);
            // code_embed is [1, 1, codec_embed_dim]

            let code_embed = if let Some(proj) = &self.small_to_mtp_projection {
                proj.forward(code_embed)
            } else {
                code_embed
            };

            let mut h = code_embed;
            for (i, layer) in self.layers.iter().enumerate() {
                h = layer.forward(h, rope, None, Some(&mut cp_kv_caches[i]), offset);
            }
            h = self.norm.forward(h);

            let logits = self.lm_heads[group_idx].forward(h);
            prev_code_idx = logits.argmax(2).squeeze_dim::<2>(2);
            code_tensors.push(prev_code_idx.clone());
            offset += 1;
        }

        // Single GPU→CPU sync: read all 15 codes at once
        let all_codes_tensor = Tensor::cat(code_tensors, 1); // [1, 15]
        let codes_data: Vec<f32> = all_codes_tensor
            .reshape([num_acoustic as i32])
            .float()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        codes_data.iter().map(|&c| c as u32).collect()
    }

    /// Get sum of all acoustic code embeddings.
    ///
    /// acoustic_codes: 15 acoustic codes for groups 2-16
    /// Returns: [1, 1, codec_embed_dim] tensor with summed embeddings
    pub fn get_acoustic_embeddings_sum(
        &self,
        acoustic_codes: &[u32],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        assert_eq!(
            acoustic_codes.len(),
            self.codec_embeddings.len(),
            "Expected {} acoustic codes, got {}",
            self.codec_embeddings.len(),
            acoustic_codes.len()
        );

        let first_code =
            Tensor::<B, 1, Int>::from_ints([acoustic_codes[0] as i32], device).unsqueeze::<2>();
        let mut sum = self.codec_embeddings[0].forward(first_code); // [1, 1, embed_dim]

        for (i, &code) in acoustic_codes[1..].iter().enumerate() {
            let code_tensor =
                Tensor::<B, 1, Int>::from_ints([code as i32], device).unsqueeze::<2>();
            let embed = self.codec_embeddings[i + 1].forward(code_tensor);
            sum = sum + embed;
        }

        sum
    }

    /// Embed a sequence of codes for a specific acoustic group.
    ///
    /// Used by ICL voice cloning to build reference codec embeddings.
    ///
    /// # Arguments
    /// * `group_idx` — acoustic group (0-14)
    /// * `codes` — 1-D Int tensor of codec token IDs
    ///
    /// # Returns
    /// Tensor of shape `[1, T, codec_embed_dim]`
    pub fn embed_codes_for_group(
        &self,
        group_idx: usize,
        codes: Tensor<B, 1, Int>,
    ) -> Tensor<B, 3> {
        assert!(
            group_idx < self.codec_embeddings.len(),
            "Invalid group_idx {} (max {})",
            group_idx,
            self.codec_embeddings.len() - 1
        );
        // Unsqueeze to [1, T] for Burn's Embedding, get [1, T, embed_dim]
        self.codec_embeddings[group_idx].forward(codes.unsqueeze::<2>())
    }

    /// Reconstruct config.
    pub fn config(&self) -> CodePredictorConfig {
        CodePredictorConfig {
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_code_groups: self.num_code_groups,
            head_dim: self.head_dim,
            rope_theta: self.rope_theta,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_config_default() {
        let config = CodePredictorConfig::default();
        assert_eq!(config.num_hidden_layers, 5);
        assert_eq!(config.num_code_groups, 16);
        assert_eq!(config.hidden_size, 1024);
    }

    #[test]
    fn test_code_predictor_construction() {
        let device = Default::default();
        let config = CodePredictorConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            vocab_size: 64,
            num_code_groups: 4,
            ..Default::default()
        };

        let predictor = CodePredictor::<B>::init(config, &device);
        assert_eq!(predictor.codec_embeddings.len(), 3);
        assert_eq!(predictor.layers.len(), 2);
        assert_eq!(predictor.lm_heads.len(), 3);
    }

    #[test]
    fn test_code_predictor_with_projection() {
        let device = Default::default();
        let config = CodePredictorConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            vocab_size: 64,
            num_code_groups: 4,
            codec_embed_dim: Some(64),
            ..Default::default()
        };

        let predictor = CodePredictor::<B>::init(config, &device);
        assert!(predictor.small_to_mtp_projection.is_some());
    }
}
