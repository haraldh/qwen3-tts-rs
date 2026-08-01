//! Weight loading from safetensors files into Burn modules.
//!
//! Loads HuggingFace Qwen3-TTS weights directly from safetensors files
//! and constructs Burn modules with the loaded parameters.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, RmsNorm,
    RmsNormConfig,
};
use burn::prelude::*;

use super::code_predictor::{CodePredictor, CodePredictorConfig};
use super::codec::decoder_12hz::{Decoder12Hz, Decoder12HzConfig};
use super::codec::encoder_12hz::{Encoder12Hz, Encoder12HzConfig};
use super::speaker::SpeakerEncoder;
use super::talker::{TalkerConfig, TalkerModel, TextProjection};
use super::transformer::{
    Attention, AttentionConfig, DecoderLayer, DecoderLayerConfig, MLPConfig, MLP,
};
use crate::models::config::{ParsedModelConfig, SpeakerEncoderConfig};

// ── Tensor data extraction ───────────────────────────────────────────────

/// Parsed tensor info from a safetensors file.
struct TensorInfo {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Load all tensors from a safetensors file, converting to f32.
fn load_safetensors_f32(path: &Path) -> Result<HashMap<String, TensorInfo>> {
    let file_data =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&file_data)
        .with_context(|| format!("Failed to parse safetensors: {}", path.display()))?;

    let mut tensors = HashMap::new();
    for (name, view) in st.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let data = convert_to_f32(view.data(), view.dtype());
        tensors.insert(name.to_string(), TensorInfo { shape, data });
    }
    Ok(tensors)
}

/// Convert raw tensor bytes to f32 based on dtype.
fn convert_to_f32(bytes: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|b| burn::tensor::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        safetensors::Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|b| burn::tensor::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        safetensors::Dtype::F64 => bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        other => panic!("Unsupported safetensors dtype: {:?}", other),
    }
}

/// Filter tensors by prefix, stripping the prefix from keys.
fn filter_by_prefix<'a>(
    tensors: &'a HashMap<String, TensorInfo>,
    prefix: &str,
) -> HashMap<String, &'a TensorInfo> {
    tensors
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix(prefix)
                .map(|stripped| (stripped.to_string(), v))
        })
        .collect()
}

// ── Tensor → Burn module helpers ─────────────────────────────────────────

fn make_tensor_2d<B: Backend>(info: &TensorInfo, device: &B::Device) -> Tensor<B, 2> {
    assert_eq!(
        info.shape.len(),
        2,
        "Expected 2D tensor, got {:?}",
        info.shape
    );
    Tensor::from_data(
        burn::tensor::TensorData::new(info.data.clone(), info.shape.clone())
            .convert::<B::FloatElem>(),
        device,
    )
}

fn make_tensor_1d<B: Backend>(info: &TensorInfo, device: &B::Device) -> Tensor<B, 1> {
    let numel: usize = info.shape.iter().product();
    Tensor::from_data(
        burn::tensor::TensorData::new(info.data.clone(), vec![numel]).convert::<B::FloatElem>(),
        device,
    )
}

fn make_tensor_3d<B: Backend>(info: &TensorInfo, device: &B::Device) -> Tensor<B, 3> {
    assert_eq!(
        info.shape.len(),
        3,
        "Expected 3D tensor, got {:?}",
        info.shape
    );
    Tensor::from_data(
        burn::tensor::TensorData::new(info.data.clone(), info.shape.clone())
            .convert::<B::FloatElem>(),
        device,
    )
}

/// Replace weights in a Conv1d (accessed through CausalConv1d.conv).
fn load_conv1d_weights<B: Backend>(
    conv: &mut burn::nn::conv::Conv1d<B>,
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    device: &B::Device,
) -> Result<()> {
    let w_key = format!("{prefix}weight");
    let w_info = weights
        .get(&w_key)
        .with_context(|| format!("Missing conv weight: {w_key}"))?;
    let w_tensor = make_tensor_3d(w_info, device);
    conv.weight = conv.weight.clone().map(|_| w_tensor);

    let b_key = format!("{prefix}bias");
    if let Some(b_info) = weights.get(&b_key) {
        let b_tensor = make_tensor_1d(b_info, device);
        conv.bias = conv.bias.take().map(|p| p.map(|_| b_tensor));
    }
    Ok(())
}

/// Replace weights in a ConvTranspose1d.
fn load_conv_transpose1d_weights<B: Backend>(
    conv: &mut burn::nn::conv::ConvTranspose1d<B>,
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    device: &B::Device,
) -> Result<()> {
    let w_key = format!("{prefix}weight");
    let w_info = weights
        .get(&w_key)
        .with_context(|| format!("Missing conv_t weight: {w_key}"))?;
    let w_tensor = make_tensor_3d(w_info, device);
    conv.weight = conv.weight.clone().map(|_| w_tensor);

    let b_key = format!("{prefix}bias");
    if let Some(b_info) = weights.get(&b_key) {
        let b_tensor = make_tensor_1d(b_info, device);
        conv.bias = conv.bias.take().map(|p| p.map(|_| b_tensor));
    }
    Ok(())
}

/// Load a LayerNorm from weight data (gamma + optional beta).
fn load_layer_norm<B: Backend>(
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    d_model: usize,
    device: &B::Device,
) -> Result<LayerNorm<B>> {
    let w_key = format!("{prefix}weight");
    let w_info = weights
        .get(&w_key)
        .with_context(|| format!("Missing layernorm weight: {w_key}"))?;

    let b_key = format!("{prefix}bias");
    let has_bias = weights.contains_key(&b_key);

    let mut norm = LayerNormConfig::new(d_model)
        .with_epsilon(1e-6)
        .with_bias(has_bias)
        .init(device);
    let w_tensor = make_tensor_1d(w_info, device);
    norm.gamma = norm.gamma.map(|_| w_tensor);

    if has_bias {
        let b_info = weights.get(&b_key).unwrap();
        let b_tensor = make_tensor_1d(b_info, device);
        norm.beta = norm.beta.map(|p| p.map(|_| b_tensor));
    }
    Ok(norm)
}

/// Load a Linear module from weight data.
fn load_linear<B: Backend>(
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    d_input: usize,
    d_output: usize,
    has_bias: bool,
    device: &B::Device,
) -> Result<Linear<B>> {
    let weight_key = format!("{prefix}weight");
    let weight_info = weights
        .get(&weight_key)
        .with_context(|| format!("Missing weight: {weight_key}"))?;

    let mut linear = LinearConfig::new(d_input, d_output)
        .with_bias(has_bias)
        .init(device);

    // Safetensors stores [d_output, d_input] (PyTorch convention),
    // but Burn Linear expects [d_input, d_output]. Transpose on load.
    let weight_tensor = make_tensor_2d(weight_info, device).transpose();
    linear.weight = linear.weight.map(|_| weight_tensor);

    if has_bias {
        let bias_key = format!("{prefix}bias");
        let bias_info = weights
            .get(&bias_key)
            .with_context(|| format!("Missing bias: {bias_key}"))?;
        let bias_tensor = make_tensor_1d(bias_info, device);
        linear.bias = linear.bias.map(|p| p.map(|_| bias_tensor));
    }

    Ok(linear)
}

/// Load an Embedding module from weight data.
fn load_embedding<B: Backend>(
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    n_embedding: usize,
    d_model: usize,
    device: &B::Device,
) -> Result<Embedding<B>> {
    let weight_key = format!("{prefix}weight");
    let weight_info = weights
        .get(&weight_key)
        .with_context(|| format!("Missing embedding weight: {weight_key}"))?;

    let mut embedding = EmbeddingConfig::new(n_embedding, d_model).init(device);
    let weight_tensor = make_tensor_2d(weight_info, device);
    embedding.weight = embedding.weight.map(|_| weight_tensor);
    Ok(embedding)
}

/// Load an RmsNorm module from weight data.
fn load_rms_norm<B: Backend>(
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    d_model: usize,
    eps: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>> {
    let weight_key = format!("{prefix}weight");
    let weight_info = weights
        .get(&weight_key)
        .with_context(|| format!("Missing norm weight: {weight_key}"))?;

    let mut norm = RmsNormConfig::new(d_model).with_epsilon(eps).init(device);
    let weight_tensor = make_tensor_1d(weight_info, device);
    norm.gamma = norm.gamma.map(|_| weight_tensor);
    Ok(norm)
}

/// Load an Attention module from weight data.
fn load_attention<B: Backend>(
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    config: &AttentionConfig,
    device: &B::Device,
) -> Result<Attention<B>> {
    let hidden = config.hidden_size;
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let mut attn = config.init(device);
    attn.q_proj = load_linear(
        weights,
        &format!("{prefix}q_proj."),
        hidden,
        q_dim,
        false,
        device,
    )?;
    attn.k_proj = load_linear(
        weights,
        &format!("{prefix}k_proj."),
        hidden,
        kv_dim,
        false,
        device,
    )?;
    attn.v_proj = load_linear(
        weights,
        &format!("{prefix}v_proj."),
        hidden,
        kv_dim,
        false,
        device,
    )?;
    attn.o_proj = load_linear(
        weights,
        &format!("{prefix}o_proj."),
        hidden,
        q_dim,
        false,
        device,
    )?;
    attn.q_norm = load_rms_norm(
        weights,
        &format!("{prefix}q_norm."),
        config.head_dim,
        config.rms_norm_eps,
        device,
    )?;
    attn.k_norm = load_rms_norm(
        weights,
        &format!("{prefix}k_norm."),
        config.head_dim,
        config.rms_norm_eps,
        device,
    )?;
    Ok(attn)
}

/// Load an MLP module from weight data.
fn load_mlp<B: Backend>(
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
    device: &B::Device,
) -> Result<MLP<B>> {
    let config = MLPConfig {
        hidden_size: hidden,
        intermediate_size: intermediate,
    };
    let mut mlp = config.init(device);
    mlp.gate_proj = load_linear(
        weights,
        &format!("{prefix}gate_proj."),
        hidden,
        intermediate,
        false,
        device,
    )?;
    mlp.up_proj = load_linear(
        weights,
        &format!("{prefix}up_proj."),
        hidden,
        intermediate,
        false,
        device,
    )?;
    mlp.down_proj = load_linear(
        weights,
        &format!("{prefix}down_proj."),
        intermediate,
        hidden,
        false,
        device,
    )?;
    Ok(mlp)
}

/// Load a DecoderLayer from weight data.
fn load_decoder_layer<B: Backend>(
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    config: &DecoderLayerConfig,
    device: &B::Device,
) -> Result<DecoderLayer<B>> {
    let attn_config = AttentionConfig::new(
        config.hidden_size,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        config.rms_norm_eps,
    );

    let mut layer = config.init(device);
    layer.self_attn = load_attention(
        weights,
        &format!("{prefix}self_attn."),
        &attn_config,
        device,
    )?;
    layer.mlp = load_mlp(
        weights,
        &format!("{prefix}mlp."),
        config.hidden_size,
        config.intermediate_size,
        device,
    )?;
    layer.input_layernorm = load_rms_norm(
        weights,
        &format!("{prefix}input_layernorm."),
        config.hidden_size,
        config.rms_norm_eps,
        device,
    )?;
    layer.post_attention_layernorm = load_rms_norm(
        weights,
        &format!("{prefix}post_attention_layernorm."),
        config.hidden_size,
        config.rms_norm_eps,
        device,
    )?;
    Ok(layer)
}

// ── Component loaders ────────────────────────────────────────────────────

/// Load a [`TalkerModel`] from a safetensors file.
pub fn load_talker<B: Backend>(
    safetensors_path: &Path,
    config: TalkerConfig,
    device: &B::Device,
) -> Result<TalkerModel<B>> {
    let all_tensors = load_safetensors_f32(safetensors_path)?;

    // Weight keys use "talker.model." prefix for transformer layers/embeddings,
    // "talker.text_projection." for projection, "talker.codec_head." for lm head.
    let model_weights = filter_by_prefix(&all_tensors, "talker.model.");
    let talker_weights: HashMap<String, &TensorInfo> = all_tensors
        .iter()
        .filter_map(|(k, v)| k.strip_prefix("talker.").map(|s| (s.to_string(), v)))
        .collect();

    let layer_config = DecoderLayerConfig::new(
        config.hidden_size,
        config.intermediate_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.rms_norm_eps,
    );

    let mut talker = TalkerModel::init(config.clone(), device);

    // Text embedding: talker.model.text_embedding
    talker.text_embedding = load_embedding(
        &model_weights,
        "text_embedding.",
        config.text_vocab_size,
        config.text_embed_dim,
        device,
    )?;

    // Text projection: talker.text_projection.linear_fc1/linear_fc2
    let tp_fc1 = load_linear(
        &talker_weights,
        "text_projection.linear_fc1.",
        config.text_embed_dim,
        config.text_proj_intermediate,
        true,
        device,
    )?;
    let tp_fc2 = load_linear(
        &talker_weights,
        "text_projection.linear_fc2.",
        config.text_proj_intermediate,
        config.hidden_size,
        true,
        device,
    )?;
    talker.text_projection = TextProjection {
        fc1: tp_fc1,
        fc2: tp_fc2,
    };

    // Codec embedding: talker.model.codec_embedding
    talker.codec_embedding = load_embedding(
        &model_weights,
        "codec_embedding.",
        config.codec_vocab_size,
        config.hidden_size,
        device,
    )?;

    // Transformer layers
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let prefix = format!("layers.{i}.");
        let layer = load_decoder_layer(&model_weights, &prefix, &layer_config, device)?;
        layers.push(layer);
    }
    talker.layers = layers;

    // Final norm
    talker.norm = load_rms_norm(
        &model_weights,
        "norm.",
        config.hidden_size,
        config.rms_norm_eps,
        device,
    )?;

    // Codec head: talker.codec_head
    talker.codec_head = load_linear(
        &talker_weights,
        "codec_head.",
        config.hidden_size,
        config.codec_vocab_size,
        false,
        device,
    )?;

    // Skip codec_head_f32 — use native GPU codec_head for performance.
    // The F32 CPU path causes a GPU sync every frame, killing throughput.

    tracing::info!("Loaded talker model ({} layers)", config.num_hidden_layers);
    Ok(talker)
}

/// Load a [`CodePredictor`] from a safetensors file.
pub fn load_code_predictor<B: Backend>(
    safetensors_path: &Path,
    config: CodePredictorConfig,
    device: &B::Device,
) -> Result<CodePredictor<B>> {
    let all_tensors = load_safetensors_f32(safetensors_path)?;
    // Weight keys: talker.code_predictor.model.{codec_embedding,layers,norm}
    //              talker.code_predictor.lm_head.{0..14}
    let cp_weights = filter_by_prefix(&all_tensors, "talker.code_predictor.");
    let cp_model_weights = filter_by_prefix(&all_tensors, "talker.code_predictor.model.");
    let codec_embed_dim = config.codec_embed_dim();
    let num_acoustic = config.num_code_groups - 1;

    let layer_config = DecoderLayerConfig::new(
        config.hidden_size,
        config.intermediate_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.rms_norm_eps,
    );

    let mut cp = CodePredictor::init(config.clone(), device);

    // Codec embeddings (15 acoustic groups): talker.code_predictor.model.codec_embedding.{0..14}
    let mut codec_embeddings = Vec::with_capacity(num_acoustic);
    for i in 0..num_acoustic {
        let emb = load_embedding(
            &cp_model_weights,
            &format!("codec_embedding.{i}."),
            config.vocab_size,
            codec_embed_dim,
            device,
        )?;
        codec_embeddings.push(emb);
    }
    cp.codec_embeddings = codec_embeddings;

    // small_to_mtp_projection (for 1.7B models)
    // Note: this weight lives under talker.code_predictor., not talker.code_predictor.model.
    if codec_embed_dim != config.hidden_size {
        let proj = load_linear(
            &cp_weights,
            "small_to_mtp_projection.",
            codec_embed_dim,
            config.hidden_size,
            true,
            device,
        )?;
        cp.small_to_mtp_projection = Some(proj);
    }

    // Transformer layers: talker.code_predictor.model.layers.{0..N}
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let prefix = format!("layers.{i}.");
        let layer = load_decoder_layer(&cp_model_weights, &prefix, &layer_config, device)?;
        layers.push(layer);
    }
    cp.layers = layers;

    // Final norm: talker.code_predictor.model.norm
    cp.norm = load_rms_norm(
        &cp_model_weights,
        "norm.",
        config.hidden_size,
        config.rms_norm_eps,
        device,
    )?;

    // LM heads (15 acoustic groups): talker.code_predictor.lm_head.{0..14}
    let mut lm_heads = Vec::with_capacity(num_acoustic);
    for i in 0..num_acoustic {
        let head = load_linear(
            &cp_weights,
            &format!("lm_head.{i}."),
            config.hidden_size,
            config.vocab_size,
            false,
            device,
        )?;
        lm_heads.push(head);
    }
    cp.lm_heads = lm_heads;

    tracing::info!(
        "Loaded code predictor ({} layers)",
        config.num_hidden_layers
    );
    Ok(cp)
}

/// Load a [`Decoder12Hz`] from a safetensors file (speech_tokenizer/model.safetensors).
pub fn load_decoder<B: Backend>(
    safetensors_path: &Path,
    config: Decoder12HzConfig,
    device: &B::Device,
) -> Result<Decoder12Hz<B>> {
    let all_tensors = load_safetensors_f32(safetensors_path)?;
    let dec = filter_by_prefix(&all_tensors, "decoder.");

    let mut decoder = Decoder12Hz::<B>::init(config.clone(), device);

    // ── Codebook embeddings (normalized: embedding_sum / cluster_usage) ──
    load_normalized_codebook(
        &mut decoder.first_codebook,
        &dec,
        "quantizer.rvq_first.vq.layers.0._codebook.",
        device,
    )?;

    for i in 0..15 {
        load_normalized_codebook(
            &mut decoder.rest_codebooks[i],
            &dec,
            &format!("quantizer.rvq_rest.vq.layers.{i}._codebook."),
            device,
        )?;
    }

    // ── Output projections (1×1 conv stored as 3D → load as Linear 2D) ──
    load_output_proj_from_conv(
        &mut decoder.first_output_proj,
        &dec,
        "quantizer.rvq_first.output_proj.",
        device,
    )?;
    load_output_proj_from_conv(
        &mut decoder.rest_output_proj,
        &dec,
        "quantizer.rvq_rest.output_proj.",
        device,
    )?;

    // ── Pre-conv ──
    load_conv1d_weights(&mut decoder.pre_conv.conv, &dec, "pre_conv.conv.", device)?;

    // ── Transformer ──
    decoder.input_proj = load_linear(
        &dec,
        "pre_transformer.input_proj.",
        config.latent_dim,
        config.hidden_size,
        true,
        device,
    )?;
    decoder.output_proj = load_linear(
        &dec,
        "pre_transformer.output_proj.",
        config.hidden_size,
        config.latent_dim,
        true,
        device,
    )?;
    decoder.final_norm = load_rms_norm(
        &dec,
        "pre_transformer.norm.",
        config.hidden_size,
        config.rms_norm_eps,
        device,
    )?;

    let hidden = config.hidden_size;
    let q_dim = config.num_heads * config.head_dim;
    let intermediate = config.intermediate_size;
    for i in 0..config.num_layers {
        let p = format!("pre_transformer.layers.{i}.");
        let layer = &mut decoder.transformer_layers[i];

        layer.input_layernorm = load_rms_norm(
            &dec,
            &format!("{p}input_layernorm."),
            hidden,
            config.rms_norm_eps,
            device,
        )?;
        layer.q_proj = load_linear(
            &dec,
            &format!("{p}self_attn.q_proj."),
            hidden,
            q_dim,
            false,
            device,
        )?;
        layer.k_proj = load_linear(
            &dec,
            &format!("{p}self_attn.k_proj."),
            hidden,
            q_dim,
            false,
            device,
        )?;
        layer.v_proj = load_linear(
            &dec,
            &format!("{p}self_attn.v_proj."),
            hidden,
            q_dim,
            false,
            device,
        )?;
        layer.o_proj = load_linear(
            &dec,
            &format!("{p}self_attn.o_proj."),
            q_dim,
            hidden,
            false,
            device,
        )?;

        // Layer scale
        let attn_scale_key = format!("{p}self_attn_layer_scale.scale");
        if let Some(info) = dec.get(&attn_scale_key) {
            layer.attn_layer_scale = make_tensor_1d(info, device);
        }

        layer.post_attention_layernorm = load_rms_norm(
            &dec,
            &format!("{p}post_attention_layernorm."),
            hidden,
            config.rms_norm_eps,
            device,
        )?;
        layer.gate_proj = load_linear(
            &dec,
            &format!("{p}mlp.gate_proj."),
            hidden,
            intermediate,
            false,
            device,
        )?;
        layer.up_proj = load_linear(
            &dec,
            &format!("{p}mlp.up_proj."),
            hidden,
            intermediate,
            false,
            device,
        )?;
        layer.down_proj = load_linear(
            &dec,
            &format!("{p}mlp.down_proj."),
            intermediate,
            hidden,
            false,
            device,
        )?;

        let mlp_scale_key = format!("{p}mlp_layer_scale.scale");
        if let Some(info) = dec.get(&mlp_scale_key) {
            layer.mlp_layer_scale = make_tensor_1d(info, device);
        }
    }

    // ── Upsample stages ──
    for (i, stage) in decoder.upsample_stages.iter_mut().enumerate() {
        let p = format!("upsample.{i}.");
        load_conv_transpose1d_weights(
            &mut stage.trans_conv.conv,
            &dec,
            &format!("{p}0.conv."),
            device,
        )?;

        // ConvNeXt block
        let cnp = format!("{p}1.");
        load_conv1d_weights(
            &mut stage.convnext.dwconv.conv,
            &dec,
            &format!("{cnp}dwconv.conv."),
            device,
        )?;
        stage.convnext.norm = load_layer_norm(
            &dec,
            &format!("{cnp}norm."),
            stage.convnext.norm.gamma.dims()[0],
            device,
        )?;
        stage.convnext.pwconv1 = load_linear(
            &dec,
            &format!("{cnp}pwconv1."),
            stage.convnext.pwconv1.weight.dims()[1],
            stage.convnext.pwconv1.weight.dims()[0],
            true,
            device,
        )?;
        stage.convnext.pwconv2 = load_linear(
            &dec,
            &format!("{cnp}pwconv2."),
            stage.convnext.pwconv2.weight.dims()[1],
            stage.convnext.pwconv2.weight.dims()[0],
            true,
            device,
        )?;

        let gamma_key = format!("{cnp}gamma");
        if let Some(info) = dec.get(&gamma_key) {
            stage.convnext.gamma = make_tensor_1d(info, device);
        }
    }

    // ── Decoder blocks ──
    // decoder.0 = init conv
    load_conv1d_weights(
        &mut decoder.decoder_init_conv.conv,
        &dec,
        "decoder.0.conv.",
        device,
    )?;

    // decoder.{1-4} = decoder blocks
    for (i, block) in decoder.decoder_blocks.iter_mut().enumerate() {
        let block_idx = i + 1;
        let p = format!("decoder.{block_idx}.block.");

        // SnakeBeta (block.0)
        load_snake_beta(&mut block.snake, &dec, &format!("{p}0."), device)?;

        // Upsample trans conv (block.1)
        load_conv_transpose1d_weights(
            &mut block.upsample.conv,
            &dec,
            &format!("{p}1.conv."),
            device,
        )?;

        // Residual units (block.2, block.3, block.4)
        load_residual_unit(&mut block.res1, &dec, &format!("{p}2."), device)?;
        load_residual_unit(&mut block.res2, &dec, &format!("{p}3."), device)?;
        load_residual_unit(&mut block.res3, &dec, &format!("{p}4."), device)?;
    }

    // decoder.5 = final snake
    load_snake_beta(&mut decoder.final_snake, &dec, "decoder.5.", device)?;

    // decoder.6 = final conv
    load_conv1d_weights(
        &mut decoder.final_conv.conv,
        &dec,
        "decoder.6.conv.",
        device,
    )?;

    tracing::info!("Loaded decoder (12Hz)");
    Ok(decoder)
}

/// Load normalized codebook embedding: weight = embedding_sum / cluster_usage.
fn load_normalized_codebook<B: Backend>(
    embedding: &mut Embedding<B>,
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    device: &B::Device,
) -> Result<()> {
    let sum_key = format!("{prefix}embedding_sum");
    let usage_key = format!("{prefix}cluster_usage");

    let sum_info = weights
        .get(&sum_key)
        .with_context(|| format!("Missing codebook embedding_sum: {sum_key}"))?;
    let usage_info = weights
        .get(&usage_key)
        .with_context(|| format!("Missing codebook cluster_usage: {usage_key}"))?;

    // Normalize: embedding = embedding_sum / clamp(cluster_usage, min=1e-7).unsqueeze(-1)
    let embedding_sum = make_tensor_2d::<B>(sum_info, device);
    let cluster_usage = make_tensor_1d::<B>(usage_info, device);
    let cluster_usage = cluster_usage.clamp_min(1e-7).unsqueeze_dim::<2>(1);
    let normalized = embedding_sum / cluster_usage;

    embedding.weight = embedding.weight.clone().map(|_| normalized);
    Ok(())
}

/// Load a 1×1 conv stored as 3D tensor [out, in, 1] into a Linear [out, in].
fn load_output_proj_from_conv<B: Backend>(
    linear: &mut Linear<B>,
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    device: &B::Device,
) -> Result<()> {
    let w_key = format!("{prefix}weight");
    let w_info = weights
        .get(&w_key)
        .with_context(|| format!("Missing output_proj weight: {w_key}"))?;

    // Shape [out_ch, in_ch, 1] → squeeze to [out_ch, in_ch] → transpose to [in_ch, out_ch]
    // (Burn Linear stores weights as [d_input, d_output])
    let t3 = make_tensor_3d::<B>(w_info, device);
    let t2 = t3.squeeze::<2>().transpose();
    linear.weight = linear.weight.clone().map(|_| t2);
    Ok(())
}

/// Load SnakeBeta activation weights (alpha, beta).
fn load_snake_beta<B: Backend>(
    snake: &mut super::codec::decoder_12hz::SnakeBeta<B>,
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    device: &B::Device,
) -> Result<()> {
    let alpha_key = format!("{prefix}alpha");
    let beta_key = format!("{prefix}beta");

    if let Some(info) = weights.get(&alpha_key) {
        snake.alpha = make_tensor_1d(info, device);
    }
    if let Some(info) = weights.get(&beta_key) {
        snake.beta = make_tensor_1d(info, device);
    }
    Ok(())
}

/// Load a ResidualUnit (2 SnakeBeta + 2 CausalConv1d).
fn load_residual_unit<B: Backend>(
    unit: &mut super::codec::decoder_12hz::ResidualUnit<B>,
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    device: &B::Device,
) -> Result<()> {
    load_snake_beta(&mut unit.act1, weights, &format!("{prefix}act1."), device)?;
    load_conv1d_weights(
        &mut unit.conv1.conv,
        weights,
        &format!("{prefix}conv1.conv."),
        device,
    )?;
    load_snake_beta(&mut unit.act2, weights, &format!("{prefix}act2."), device)?;
    load_conv1d_weights(
        &mut unit.conv2.conv,
        weights,
        &format!("{prefix}conv2.conv."),
        device,
    )?;
    Ok(())
}

/// Load an [`Encoder12Hz`] from a safetensors file (speech_tokenizer/model.safetensors).
pub fn load_encoder<B: Backend>(
    safetensors_path: &Path,
    config: Encoder12HzConfig,
    device: &B::Device,
) -> Result<Encoder12Hz<B>> {
    let all_tensors = load_safetensors_f32(safetensors_path)?;
    // Strip "encoder." prefix to get component-level keys
    let enc: HashMap<String, &TensorInfo> = all_tensors
        .iter()
        .filter_map(|(k, v)| k.strip_prefix("encoder.").map(|s| (s.to_string(), v)))
        .collect();

    let mut encoder = Encoder12Hz::<B>::init(config.clone(), device);

    // ── SEANet encoder ──

    // Init conv: encoder.encoder.layers.0
    load_conv1d_weights(
        &mut encoder.seanet.init_conv.conv,
        &enc,
        "encoder.layers.0.conv.",
        device,
    )?;

    // Stages: each has residual blocks + downsample conv
    // Layer numbering: for stage i, residual blocks start at layer_idx, downsample at layer_idx + n_res + 1
    // With n_residual_layers=1: layout is [init(0), res(1), elu(2), down(3), res(4), elu(5), down(6), ...]
    let mut layer_idx = 1; // after init conv
    for (i, stage) in encoder.seanet.stages.iter_mut().enumerate() {
        // Residual blocks
        for (j, block) in stage.residual_blocks.iter_mut().enumerate() {
            let block_layer = layer_idx + j;
            let p = format!("encoder.layers.{block_layer}.block.");
            // block.0 = ELU (no weights), block.1 = dilated conv, block.2 = ELU, block.3 = 1x1 conv
            load_conv1d_weights(&mut block.conv1.conv, &enc, &format!("{p}1.conv."), device)?;
            load_conv1d_weights(&mut block.conv2.conv, &enc, &format!("{p}3.conv."), device)?;
        }
        layer_idx += config.num_residual_layers;

        // ELU (no weights) — skip
        layer_idx += 1;

        // Downsample conv
        let down_layer = layer_idx;
        load_conv1d_weights(
            &mut stage.downsample.conv,
            &enc,
            &format!("encoder.layers.{down_layer}.conv."),
            device,
        )?;
        layer_idx += 1;

        tracing::trace!("Loaded SEANet encoder stage {i}");
    }

    // Final ELU (no weights) + final conv
    layer_idx += 1; // skip ELU
    load_conv1d_weights(
        &mut encoder.seanet.final_conv.conv,
        &enc,
        &format!("encoder.layers.{layer_idx}.conv."),
        device,
    )?;

    // ── Transformer layers ──
    let hidden = config.hidden_size;
    let q_dim = config.num_heads * config.head_dim;
    let intermediate = config.intermediate_size;
    for i in 0..config.num_transformer_layers {
        let p = format!("encoder_transformer.layers.{i}.");
        let layer = &mut encoder.transformer_layers[i];

        layer.input_layernorm =
            load_layer_norm(&enc, &format!("{p}input_layernorm."), hidden, device)?;
        layer.q_proj = load_linear(
            &enc,
            &format!("{p}self_attn.q_proj."),
            hidden,
            q_dim,
            false,
            device,
        )?;
        layer.k_proj = load_linear(
            &enc,
            &format!("{p}self_attn.k_proj."),
            hidden,
            q_dim,
            false,
            device,
        )?;
        layer.v_proj = load_linear(
            &enc,
            &format!("{p}self_attn.v_proj."),
            hidden,
            q_dim,
            false,
            device,
        )?;
        layer.o_proj = load_linear(
            &enc,
            &format!("{p}self_attn.o_proj."),
            q_dim,
            hidden,
            false,
            device,
        )?;

        let attn_scale_key = format!("{p}self_attn_layer_scale.scale");
        if let Some(info) = enc.get(&attn_scale_key) {
            layer.attn_layer_scale = make_tensor_1d(info, device);
        }

        layer.post_attention_layernorm = load_layer_norm(
            &enc,
            &format!("{p}post_attention_layernorm."),
            hidden,
            device,
        )?;

        // GELU MLP: fc1 + fc2 (no bias, no gate)
        layer.fc1 = load_linear(
            &enc,
            &format!("{p}mlp.fc1."),
            hidden,
            intermediate,
            false,
            device,
        )?;
        layer.fc2 = load_linear(
            &enc,
            &format!("{p}mlp.fc2."),
            intermediate,
            hidden,
            false,
            device,
        )?;

        let mlp_scale_key = format!("{p}mlp_layer_scale.scale");
        if let Some(info) = enc.get(&mlp_scale_key) {
            layer.mlp_layer_scale = make_tensor_1d(info, device);
        }
    }

    // ── Downsample conv ──
    load_conv1d_weights(
        &mut encoder.downsample.conv,
        &enc,
        "downsample.conv.",
        device,
    )?;

    // ── Quantizer ──

    // Semantic RVQ: 1 codebook
    load_output_proj_from_conv(
        &mut encoder.quantizer.semantic.input_proj,
        &enc,
        "quantizer.semantic_residual_vector_quantizer.input_proj.",
        device,
    )?;
    load_output_proj_from_conv(
        &mut encoder.quantizer.semantic.output_proj,
        &enc,
        "quantizer.semantic_residual_vector_quantizer.output_proj.",
        device,
    )?;
    load_normalized_codebook_for_vq(
        &mut encoder.quantizer.semantic.layers[0],
        &enc,
        "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.",
        device,
    )?;

    // Acoustic RVQ: 15 codebooks (layers 0-14 from the safetensors)
    load_output_proj_from_conv(
        &mut encoder.quantizer.acoustic.input_proj,
        &enc,
        "quantizer.acoustic_residual_vector_quantizer.input_proj.",
        device,
    )?;
    load_output_proj_from_conv(
        &mut encoder.quantizer.acoustic.output_proj,
        &enc,
        "quantizer.acoustic_residual_vector_quantizer.output_proj.",
        device,
    )?;
    for i in 0..config.num_acoustic_quantizers {
        load_normalized_codebook_for_vq(
            &mut encoder.quantizer.acoustic.layers[i],
            &enc,
            &format!("quantizer.acoustic_residual_vector_quantizer.layers.{i}.codebook."),
            device,
        )?;
    }

    tracing::info!("Loaded encoder (12Hz)");
    Ok(encoder)
}

/// Load normalized codebook into a VectorQuantizer.
fn load_normalized_codebook_for_vq<B: Backend>(
    vq: &mut super::codec::encoder_12hz::VectorQuantizer<B>,
    weights: &HashMap<String, &TensorInfo>,
    prefix: &str,
    device: &B::Device,
) -> Result<()> {
    let sum_key = format!("{prefix}embed_sum");
    let usage_key = format!("{prefix}cluster_usage");

    let sum_info = weights
        .get(&sum_key)
        .with_context(|| format!("Missing codebook embed_sum: {sum_key}"))?;
    let usage_info = weights
        .get(&usage_key)
        .with_context(|| format!("Missing codebook cluster_usage: {usage_key}"))?;

    let embedding_sum = make_tensor_2d::<B>(sum_info, device);
    let cluster_usage = make_tensor_1d::<B>(usage_info, device);
    let cluster_usage = cluster_usage.clamp_min(1e-7).unsqueeze_dim::<2>(1);
    let normalized = embedding_sum / cluster_usage;

    vq.codebook = normalized;
    Ok(())
}

/// Load an ECAPA-TDNN speaker encoder from a safetensors file.
///
/// The speaker encoder weights live under the `speaker_encoder.` prefix
/// in the main `model.safetensors` file (Base models only).
fn load_speaker_encoder<B: Backend>(
    safetensors_path: &Path,
    config: SpeakerEncoderConfig,
    device: &B::Device,
) -> Result<SpeakerEncoder<B>> {
    let all_tensors = load_safetensors_f32(safetensors_path)?;
    let se = filter_by_prefix(&all_tensors, "speaker_encoder.");

    let mut encoder = SpeakerEncoder::<B>::init(config, device);

    // Initial TDNN (blocks.0)
    load_conv1d_weights(
        &mut encoder.initial_tdnn.conv.conv,
        &se,
        "blocks.0.conv.",
        device,
    )?;

    // SE-Res2Net blocks (blocks.1, blocks.2, blocks.3 in safetensors → index 0,1,2)
    for i in 0..3 {
        let block = &mut encoder.se_res2net_blocks[i];
        let p = format!("blocks.{}.", i + 1);

        load_conv1d_weights(
            &mut block.tdnn1.conv.conv,
            &se,
            &format!("{p}tdnn1.conv."),
            device,
        )?;

        for k in 0..block.res2net_block.blocks.len() {
            load_conv1d_weights(
                &mut block.res2net_block.blocks[k].conv.conv,
                &se,
                &format!("{p}res2net_block.blocks.{k}.conv."),
                device,
            )?;
        }

        load_conv1d_weights(
            &mut block.tdnn2.conv.conv,
            &se,
            &format!("{p}tdnn2.conv."),
            device,
        )?;

        // SE block conv1/conv2 are direct Conv1d (not wrapped in ReflectPadConv1d)
        load_conv1d_weights(
            &mut block.se_block.conv1,
            &se,
            &format!("{p}se_block.conv1."),
            device,
        )?;
        load_conv1d_weights(
            &mut block.se_block.conv2,
            &se,
            &format!("{p}se_block.conv2."),
            device,
        )?;
    }

    // MFA TDNN
    load_conv1d_weights(&mut encoder.mfa_tdnn.conv.conv, &se, "mfa.conv.", device)?;

    // ASP (attentive statistics pooling)
    load_conv1d_weights(
        &mut encoder.asp.tdnn.conv.conv,
        &se,
        "asp.tdnn.conv.",
        device,
    )?;
    load_conv1d_weights(&mut encoder.asp.conv, &se, "asp.conv.", device)?;

    // Final FC projection
    load_conv1d_weights(&mut encoder.fc, &se, "fc.", device)?;

    Ok(encoder)
}

/// Load all model components from a model directory.
///
/// Expected directory structure:
/// ```text
/// model_dir/
/// ├── config.json
/// ├── model.safetensors
/// └── speech_tokenizer/
///     └── model.safetensors
/// ```
pub fn load_all<B: Backend>(model_dir: &Path, device: &B::Device) -> Result<LoadedComponents<B>> {
    let config_path = model_dir.join("config.json");
    let int4_path = model_dir.join("model_int4.safetensors");
    let model_path = model_dir.join("model.safetensors");
    let st_path = model_dir.join("speech_tokenizer/model.safetensors");
    let has_int4 = int4_path.exists();

    // Parse config
    let parsed =
        ParsedModelConfig::from_file(&config_path).context("Failed to parse config.json")?;
    tracing::info!("Detected model variant: {}", parsed.label());

    anyhow::ensure!(
        model_path.exists(),
        "Model weights not found at {}",
        model_path.display()
    );
    anyhow::ensure!(
        st_path.exists(),
        "Speech tokenizer weights not found at {}",
        st_path.display()
    );

    // Load talker
    tracing::info!("Loading talker model...");
    let talker_config = TalkerConfig::from_parsed(&parsed);
    let talker = load_talker(&model_path, talker_config, device)?;

    // Load code predictor
    tracing::info!("Loading code predictor...");
    let cp_config = CodePredictorConfig::from_parsed(&parsed);
    let code_predictor = load_code_predictor(&model_path, cp_config, device)?;

    // Load decoder
    tracing::info!("Loading decoder (12Hz)...");
    let decoder = load_decoder(&st_path, Decoder12HzConfig::default(), device)?;

    // Load encoder on CPU (for ICL voice cloning — all model types have it).
    // Runs once per voice clone call. On GPU, BF16 autotuning crashes on HIP/RDNA
    // due to deferred page faults from benchmark candidates across matmul, conv,
    // and reduce operations.
    tracing::info!("Loading encoder (12Hz) on CPU...");
    let cpu_device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let encoder = load_encoder::<burn::backend::NdArray>(
        &st_path,
        Encoder12HzConfig::default(),
        &cpu_device,
    )?;

    // Load speaker encoder on CPU (Base models only).
    // The speaker encoder is small (~10M params) and runs once per synthesis.
    let speaker_encoder = if let Some(ref se_config) = parsed.speaker_encoder_config {
        tracing::info!("Loading speaker encoder (ECAPA-TDNN) on CPU...");
        Some(load_speaker_encoder::<burn::backend::NdArray>(
            &model_path,
            se_config.clone(),
            &cpu_device,
        )?)
    } else {
        None
    };

    if has_int4 {
        tracing::info!(
            "Int4 quantized weights available at {}",
            int4_path.display()
        );
    }

    Ok(LoadedComponents {
        talker,
        code_predictor,
        decoder,
        encoder,
        speaker_encoder,
        model_type: Some(parsed.model_type),
        int4_weights_path: if has_int4 { Some(int4_path) } else { None },
    })
}

/// All loaded model components, ready to assemble into a [`super::facade::Qwen3TTS`].
pub struct LoadedComponents<B: Backend> {
    pub talker: TalkerModel<B>,
    pub code_predictor: CodePredictor<B>,
    pub decoder: Decoder12Hz<B>,
    pub encoder: Encoder12Hz<burn::backend::NdArray>,
    pub speaker_encoder: Option<SpeakerEncoder<burn::backend::NdArray>>,
    pub model_type: Option<crate::models::config::ModelType>,
    /// Path to int4 quantized weights (if model_int4.safetensors exists).
    pub int4_weights_path: Option<std::path::PathBuf>,
}

/// Load a [`CodePredictor`] on the NdArray (CPU) backend from a model directory.
///
/// Used for running the code predictor on CPU when the main backend is a GPU,
/// avoiding kernel launch overhead on tiny seq_len=1 operations.
#[cfg(feature = "cpu")]
pub fn load_cpu_code_predictor(model_dir: &Path) -> Result<CodePredictor<burn::backend::NdArray>> {
    let config_path = model_dir.join("config.json");
    let model_path = model_dir.join("model.safetensors");
    let parsed =
        ParsedModelConfig::from_file(&config_path).context("Failed to parse config.json")?;
    let cp_config = CodePredictorConfig::from_parsed(&parsed);
    let device = Default::default();
    load_code_predictor(&model_path, cp_config, &device)
}

/// Raw tensor data from an int4 quantized safetensors file.
///
/// For quantized weights, the file contains `{key}_int4` (packed u32) and
/// `{key}_scales` (f32). Non-quantized tensors are stored as-is (BF16/F32).
pub struct Int4SafeTensors {
    data: Vec<u8>,
    header: HashMap<String, serde_json::Value>,
}

impl Int4SafeTensors {
    /// Load an int4 safetensors file.
    pub fn load(path: &Path) -> Result<Self> {
        let data =
            std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
        let header_size = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&data[8..8 + header_size])
                .context("Failed to parse int4 safetensors header")?;
        Ok(Self { data, header })
    }

    /// Get raw bytes for a tensor key.
    pub fn raw_bytes(&self, key: &str) -> Result<&[u8]> {
        let info = self
            .header
            .get(key)
            .with_context(|| format!("Key not found in int4 safetensors: {key}"))?;
        let offsets = info["data_offsets"]
            .as_array()
            .context("Missing data_offsets")?;
        let start = offsets[0].as_u64().unwrap() as usize;
        let end = offsets[1].as_u64().unwrap() as usize;
        let header_size = u64::from_le_bytes(self.data[..8].try_into().unwrap()) as usize;
        let base = 8 + header_size;
        Ok(&self.data[base + start..base + end])
    }

    /// Check if a key exists.
    pub fn has_key(&self, key: &str) -> bool {
        self.header.contains_key(key)
    }

    /// Check if this file has int4 quantization metadata.
    pub fn is_quantized(&self) -> bool {
        if let Some(meta) = self.header.get("__metadata__") {
            meta.get("quantization").is_some()
        } else {
            false
        }
    }
}
