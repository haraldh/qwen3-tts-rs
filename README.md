# qwen3-tts — Strix Halo / ROCm fork

Pure Rust inference for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), a text-to-speech model from Alibaba. No Python or ONNX runtime required.

> **This is a hardware-specific fork of [TrevorS/qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs).**
> It targets one machine — AMD Ryzen AI MAX+ 395 w/ Radeon 8060S (gfx1151, RDNA 3.5) on ROCm 7.2 — and
> replaces upstream's candle/CUDA stack with [Burn](https://burn.dev) plus hand-written HIP kernels.
> Making BF16 work on this GPU required patching Burn, CubeCL and cubek themselves, so those are
> vendored as submodules pointing at patched forks. **Upstream is the better starting point for CUDA,
> Metal, or CPU users.**

All code in this repo was written with [Claude Code](https://claude.ai/code). This is an experiment -- not a production library.

## What's different in this fork

**Model stack.** Upstream runs on candle. Here the active stack is `src/burn_models/` on
[Burn](https://burn.dev), which is multi-backend (ROCm/HIP, CUDA, WGPU/Vulkan, CPU). The candle code
survives in `src/models/` behind the `_candle_legacy` feature as a migration reference.

**Patched GPU toolchain.** BF16 on gfx1151 hit bugs at every layer, so the fixes live in forks
vendored as submodules on branch `qwen3-tts-rs`:

| Submodule | Fork of | Carries |
|-----------|---------|---------|
| `burn/` | `tracel-ai/burn` | RMSNorm in F32 for BF16/F16 numerical stability |
| `cubecl/` | `tracel-ai/cubecl` | HIP 7.2 BF16 WMMA type fixes, `max_bfloat16`/`min_bfloat16`, comparison-operator promotion, sub-int narrowing |
| `cubek/` | `tracel-ai/cubek` | F32 SoftmaxLhs for BF16 attention, argmax/argmin OOB clamping, reduce bound checks |
| `cubecl-hip-sys/` | `tracel-ai/cubecl-hip-sys` | HIP 7.2 bindings |

**Raw HIP hot paths.** The code predictor and talker decode step bypass CubeCL entirely —
`src/burn_models/hip/` is ~3k lines of HIP compiled through HIPRTC. It includes a flat-parallel
decode-attention kernel (`attention_decode_long_bf16`) where each thread computes full dot products
for its own KV positions, dropping the ~7 barrier syncs per KV position that the cooperative
reduction needed — worth tens of thousands of syncs at seq_kv 2048+. This is what
took the 0.6B CustomVoice model from RTF ~1.84 to **0.29** on the 8060S (see below).

**Also added:** an OpenAI-compatible HTTP server (`serve` feature, `POST /v1/audio/speech`),
int4 weight quantization via asymmetric HQQ, and ECAPA-TDNN speaker-encoder voice cloning for
Base models.

### Building this fork

```bash
git clone --recurse-submodules -b strix-halo https://github.com/haraldh/qwen3-tts-rs.git
cd qwen3-tts-rs
cargo build --release --features rocm,serve
```

The submodules are mandatory — `[patch.crates-io]` points at them by relative path, so the build
cannot resolve without them. Cloning without `--recurse-submodules` fails at dependency resolution.

### Measured on this hardware

0.6B-CustomVoice, speaker Ryan, seed 42, 79 characters of English text producing 5.85 s of
24 kHz mono audio. Measured 2026-08-01 on the Radeon 8060S:

| Run | Wall time | RTF |
|-----|-----------|-----|
| Cold (first process, includes autotune) | 25.28 s | 4.32 |
| Warm | 1.70 s | **0.29** |
| Warm (repeat) | 1.69 s | **0.29** |

RTF = wall-clock ÷ audio duration; lower is better, < 1.0 is faster than real-time. So steady
state is roughly **3.4× real-time**.

**Cold start costs ~23 seconds.** The first synthesis in a fresh process spends that time in
CubeCL autotune, benchmarking BF16 convolution and matmul variants for the decoder's shapes.
The results cache, so every later run is ~15× faster — but it means the HTTP server's first
request after boot is dramatically slower than the rest. Warm it up before serving traffic.

**Output is deterministic**: identical MD5 across runs at the same seed.

**Quality check**: transcribing the output with whisper large-v3 returns the input sentence
verbatim — 100% word overlap (`scripts/verify_audio.py`, see [CLAUDE.md](CLAUDE.md)).

**The decoder loses 2880 samples off the tail** of every decode on ROCm — a constant, not a
proportion, and always at the end (`examples/decode_align_probe.rs` measures both facts). The
50 ms of silence appended to a finished utterance covers it. NdArray does not do this, so it
is a backend bug rather than the architecture; worth revisiting if the last 120 ms of an
utterance ever matters.

**Streaming replays decoder context.** The decoder is causal but not stateless — an 8-layer
transformer runs over the code sequence before the conv stack — so `--streaming` decodes each
chunk with the preceding 64 frames prepended and discards their samples. Without that,
chunks drift apart and long utterances come out audibly scrambled. Streamed output is
sample-count identical to buffered and tracks it to about 21 dB; the residual is attention
reaching back further than the replayed window. The reference decoder config caps that with a
`sliding_window` of 72, which this implementation does not yet apply — doing so would make
chunked decoding exact.

## Changelog

### 0.4.0

- Pre-allocated KV cache with InplaceOp2 (zero-copy CUDA writes, no Tensor::cat)
- GPU-side repetition penalty mask (incremental slice_assign, eliminates growing CPU transfer)
- Deferred acoustic codes transfer (single bulk GPU→CPU at end of generation)
- Fused residual + RMSNorm CUDA kernel
- GPU→CPU syncs reduced from 3/frame to 1/frame (4-byte EOS check)
- Non-streaming RTF: 0.48–0.67 across all variants (97-100% of theoretical throughput)

### 0.3.0

- GPU-side sampling: batched argmax, on-device top-k/top-p/repetition penalty
- Eliminated 15 of 16 GPU→CPU syncs per frame in code predictor
- Cached token suppression mask in streaming sessions
- Tokenizer fallback from vocab.json + merges.txt when tokenizer.json is unavailable
- Profiling infrastructure: Chrome tracing, flamegraph, Nsight Systems via Makefile
- Benchmarked all 4 model variants (0.6B Base, 1.7B Base/CustomVoice/VoiceDesign)
- Self-contained model directories (removed tokenizer symlinks)
- Enhanced waveform plots with stats annotation bar

### 0.2.0

- ICL voice cloning now works correctly with proper reference audio
- Fixed WAV output format (WAVEX/float32 → standard WAV/PCM16) — resolves playback speed issues in some players
- Improved tokenizer path resolution with explicit `--tokenizer-dir` override
- Added benchmarking suite (Criterion micro-benchmarks + E2E speed tests)
- Automatic resampling of reference audio to 24kHz for voice cloning
- Docker base image updated to NGC pytorch:25.11 (CUDA 13.0)

Thanks to [u/rngesius](https://www.reddit.com/r/LocalLLaMA/comments/1qqvb79/comment/o2nv6qm/) for feedback on playback speed and tokenizer issues.

## Acknowledgements

- [Qwen Team (Alibaba)](https://github.com/QwenLM) — [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) model, weights, and [technical report](https://arxiv.org/abs/2601.15621)
- [TrevorS/qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs) — the upstream this forked from
- [Burn](https://burn.dev), [CubeCL and cubek](https://github.com/tracel-ai) — Rust ML framework and GPU kernel stack by [Tracel AI](https://tracel.ai/), which this fork's model code runs on
- [candle](https://github.com/huggingface/candle) — Rust ML framework by [Hugging Face](https://huggingface.co/), used by the legacy `src/models/` stack
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — reference implementation that helped clarify model details
- [Claude Code](https://claude.ai/code) — wrote the code

## Samples

All samples generated with 1.7B models, seed 42. Text: *"The sun set behind the mountains, painting the sky in shades of gold and violet."*

### CustomVoice — Ryan

![CustomVoice Ryan](assets/images/customvoice-ryan.png)
[🔊 Listen](assets/audio/customvoice-ryan.wav)

### CustomVoice — Serena

![CustomVoice Serena](assets/images/customvoice-serena.png)
[🔊 Listen](assets/audio/customvoice-serena.wav)

### Voice Clone — ICL

![Voice Clone ICL](assets/images/voiceclone-icl.png)
[🔊 Listen](assets/audio/voiceclone-icl.wav)

### VoiceDesign — Radio Announcer

![VoiceDesign Radio](assets/images/voicedesign-radio.png)
[🔊 Listen](assets/audio/voicedesign-radio.wav)

### VoiceDesign — Storyteller

![VoiceDesign Storyteller](assets/images/voicedesign-storyteller.png)
[🔊 Listen](assets/audio/voicedesign-storyteller.wav)

### VoiceDesign — Sportscaster

![VoiceDesign Sportscaster](assets/images/voicedesign-sportscaster.png)
[🔊 Listen](assets/audio/voicedesign-sportscaster.wav)

## Model Variants

Five official model variants exist across two size classes. Each variant supports a different speaker conditioning method:

| Variant | Params | Speaker Conditioning | Use Case |
|---------|--------|---------------------|----------|
| **0.6B Base** | 1.8 GB | Voice cloning from reference audio | Clone any voice from a WAV file |
| **0.6B CustomVoice** | 1.8 GB | 9 preset speakers | Pick from built-in voices |
| **1.7B Base** | 3.9 GB | Voice cloning from reference audio | Higher quality voice cloning |
| **1.7B CustomVoice** | 3.9 GB | 9 preset speakers | Higher quality preset voices |
| **1.7B VoiceDesign** | 3.8 GB | Text description | Describe a voice in natural language |

### Which model should I use?

- **Want to clone a specific voice?** Use a **Base** model with `--ref-audio` plus `--ref-text` (ICL mode, best quality), or `--ref-audio` alone for speaker-embedding-only cloning (faster, lower quality).
- **Want a quick preset voice?** Use a **CustomVoice** model with `--speaker`.
- **Want to describe a voice in text?** Use **1.7B VoiceDesign** with `--instruct`.
- **Unsure?** Start with **0.6B CustomVoice** for the fastest results.

### Valid combinations

| | Preset speakers | Voice clone (x_vector) | Voice clone (ICL) | Text-described voice |
|---|:-:|:-:|:-:|:-:|
| **Base** | | x | x | |
| **CustomVoice** | x | | | |
| **VoiceDesign** | | | | x |

Using the wrong combination (e.g. preset speakers on a Base model) won't crash, but produces unpredictable voice output. The library and CLI warn when this happens.

## Architecture

The TTS pipeline consists of three stages:

1. **TalkerModel**: 28-layer transformer generating semantic tokens from text autoregressively. Uses MRoPE (multimodal rotary position encoding) across all variants.

1. **CodePredictor**: 5-layer decoder that generates 15 acoustic tokens per semantic token. Always 1024 hidden dim; 1.7B models use a projection layer to bridge from the talker's 2048-dim space.

1. **Decoder12Hz**: Converts 16-codebook tokens to 24kHz audio via ConvNeXt blocks and transposed convolution upsampling. Shared across all model variants.

```
Text --> TalkerModel --> Semantic Token --> CodePredictor --> [16 codes] --> Decoder --> Audio
              ^                                  ^
         (autoregressive,                  (per frame,
          one per frame)                    15 acoustic codes)
```

## CLI

The model variant is auto-detected from `config.json`. The CLI warns if your flags don't match the model type.

The backend is chosen at compile time, not by a flag — build with `--features rocm` for the
8060S. There is no `--device` option.

```bash
# CustomVoice: preset speaker
cargo run --release --features rocm,cli --bin generate_audio -- \
  --model-dir test_data/models/0.6B-CustomVoice \
  --text "Hello world" \
  --speaker ryan \
  --language english \
  --output /var/tmp/hello.wav

# Base: voice cloning (ICL — best quality, requires reference text)
cargo run --release --features rocm,cli --bin generate_audio -- \
  --model-dir test_data/models/0.6B-Base \
  --text "Hello world" \
  --ref-audio reference.wav \
  --ref-text "transcript of the reference audio"

# Base: voice cloning without a transcript (speaker embedding only)
cargo run --release --features rocm,cli --bin generate_audio -- \
  --model-dir test_data/models/0.6B-Base \
  --text "Hello world" \
  --ref-audio reference.wav

# VoiceDesign: describe the voice you want
cargo run --release --features rocm,cli --bin generate_audio -- \
  --model-dir path/to/voicedesign \
  --text "Hello world" \
  --instruct "A cheerful young female voice with high pitch and energetic tone" \
  --language english

# Streaming: emit audio incrementally in ~800 ms chunks
cargo run --release --features rocm,cli --bin generate_audio -- \
  --model-dir test_data/models/0.6B-CustomVoice \
  --text "Hello world" \
  --streaming --chunk-frames 10
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir`, `-m` | `test_data/model` | Path to model directory |
| `--text`, `-t` | `"Hello"` | Text to synthesize |
| `--tokenizer-dir` | *model-dir* | Tokenizer directory, if separate |
| `--speaker` | `ryan` | Preset speaker (CustomVoice only) |
| `--language` | `english` | Target language |
| `--instruct` | | Voice description for VoiceDesign models |
| `--ref-audio` | | Reference audio WAV for voice cloning (Base only) |
| `--ref-text` | | Reference transcript; enables ICL when used with `--ref-audio` |
| `--output`, `-o` | `output.wav` | Output WAV file path |
| `--streaming` | off | Emit audio incrementally |
| `--chunk-frames` | `10` | Frames per streaming chunk (~800 ms) |
| `--duration`, `-d` | | Max duration in seconds (overrides `--frames`) |
| `--frames`, `-f` | `2048` | Max frames (~164 s); generation stops at EOS |
| `--temperature` | `0.3` | Sampling temperature |
| `--top-k` | `20` | Top-k sampling |
| `--top-p` | `0.9` | Nucleus sampling threshold |
| `--repetition-penalty` | `1.2` | Repetition penalty (1.0 = disabled) |
| `--seed`, `-s` | `42` | Random seed for reproducibility |

## Model Files

All models share the same speech tokenizer and text tokenizer.

| Component | HuggingFace Repo | Size |
|-----------|------------------|------|
| 0.6B Base | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 1.8 GB |
| 0.6B CustomVoice | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 1.8 GB |
| 1.7B Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 3.9 GB |
| 1.7B CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 3.9 GB |
| 1.7B VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 3.8 GB |
| Speech Tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | 682 MB |
| Text Tokenizer | `Qwen/Qwen2-0.5B` | 7 MB |

### Supported languages

English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## Sample Rate

Output audio is always 24kHz mono. Use `audio::resample()` for other rates:

```rust
use qwen3_tts::audio;

let audio_48k = audio::resample(&audio, 48000)?;
```

## License

MIT License. See the main Qwen3-TTS repository for model license information.
