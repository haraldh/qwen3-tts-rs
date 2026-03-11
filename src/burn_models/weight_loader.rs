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
use super::speaker::SpeakerEncoder;
use super::talker::{TalkerConfig, TalkerModel, TextProjection};
use super::transformer::{
    Attention, AttentionConfig, DecoderLayer, DecoderLayerConfig, MLPConfig, MLP,
};
use crate::models::config::ParsedModelConfig;

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
        burn::tensor::TensorData::new(info.data.clone(), vec![numel])
            .convert::<B::FloatElem>(),
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

    // Store F32 copy of codec_head weights for mixed-precision decode.
    // The weight is [codec_vocab_size, hidden_size] after transposition in load_linear.
    // We store the original (pre-transpose) layout which is [codec_vocab_size, hidden_size].
    let codec_head_weight_key = "codec_head.weight";
    if let Some(info) = talker_weights.get(codec_head_weight_key) {
        // info.data is already F32; shape is [codec_vocab_size, hidden_size]
        talker.codec_head_f32 = Some(info.data.clone());
    }

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
    if codec_embed_dim != config.hidden_size {
        let proj = load_linear(
            &cp_model_weights,
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
    let model_path = model_dir.join("model.safetensors");
    let st_path = model_dir.join("speech_tokenizer/model.safetensors");

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

    // Load speaker encoder (Base models only)
    let speaker_encoder = if parsed.speaker_encoder_config.is_some() {
        tracing::warn!(
            "Speaker encoder weight loading not yet implemented — \
             voice cloning will not work."
        );
        None
    } else {
        None
    };

    Ok(LoadedComponents {
        talker,
        code_predictor,
        decoder,
        speaker_encoder,
        model_type: Some(parsed.model_type),
    })
}

/// All loaded model components, ready to assemble into a [`super::facade::Qwen3TTS`].
pub struct LoadedComponents<B: Backend> {
    pub talker: TalkerModel<B>,
    pub code_predictor: CodePredictor<B>,
    pub decoder: Decoder12Hz<B>,
    pub speaker_encoder: Option<SpeakerEncoder<B>>,
    pub model_type: Option<crate::models::config::ModelType>,
}
