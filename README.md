# omnivoice-rs

A Rust implementation of [OmniVoice](https://github.com/k2-fsa/OmniVoice) TTS inference using the [Candle](https://github.com/huggingface/candle) ML framework. Generates speech from text with voice cloning, voice design, and automatic speech recognition support.

## Features

- **Voice cloning** -- clone a voice from a reference audio file
- **Voice design** -- describe the desired voice style (gender, age, pitch, accent)
- **Auto voice** -- let the model pick a voice automatically
- **Whisper ASR** -- auto-transcribe reference audio when `--ref-text` is omitted
- **Chunked generation** -- long texts are split at sentence boundaries and cross-faded
- **GPU acceleration** -- Metal (macOS) and CUDA support
- **600+ languages** -- language codes resolved via ISO 639

## Installation

```bash
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice/omnivoice-rs

# CPU
cargo build --release

# macOS GPU (Metal)
cargo build --release --features metal

# NVIDIA GPU (CUDA)
cargo build --release --features cuda
```

## Quick Start

Models are downloaded automatically from HuggingFace on first run.

### Voice design (no reference audio)

```bash
omnivoice-rs \
    --text "Hello, this is a test of the OmniVoice text to speech system." \
    --output output.wav
```

### Voice design with style

```bash
omnivoice-rs \
    --text "The quick brown fox jumps over the lazy dog." \
    --instruct "female, british accent, high pitch" \
    --language en \
    --output styled.wav
```

### Voice cloning

```bash
omnivoice-rs \
    --text "This sentence will be spoken in the cloned voice." \
    --ref-audio reference.wav \
    --ref-text "Transcript of the reference audio." \
    --language en \
    --output cloned.wav
```

### Voice cloning with auto-transcription

When `--ref-text` is omitted, Whisper automatically transcribes the reference audio:

```bash
omnivoice-rs \
    --text "This sentence will be spoken in the cloned voice." \
    --ref-audio reference.wav \
    --language en \
    --output cloned.wav
```

The Whisper model (`openai/whisper-large-v3-turbo` by default) is downloaded only when needed. Use `--asr-model` to specify a different model.

## CLI Reference

```
omnivoice-rs [OPTIONS] --text <TEXT> --output <OUTPUT>
```

### Required

| Flag | Description |
|------|-------------|
| `--text` | Text to synthesize |
| `--output` | Output WAV file path |

### Voice mode

| Flag | Description |
|------|-------------|
| `--ref-audio` | Reference audio for voice cloning |
| `--ref-text` | Transcript of reference audio (auto-transcribed if omitted) |
| `--instruct` | Voice style instruction (e.g. `"male, low pitch, whisper"`) |
| `--language` | Language of the **output** text (name or ISO 639 code) |

### Generation parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--num-step` | 32 | Iterative decoding steps |
| `--guidance-scale` | 2.0 | Classifier-free guidance scale |
| `--speed` | 1.0 | Speaking speed (>1 = faster) |
| `--duration` | auto | Fixed output duration in seconds |
| `--t-shift` | 0.1 | Noise schedule time shift |
| `--denoise` | true | Prepend denoise conditioning token |
| `--postprocess-output` | true | Remove silence, apply fade-in/out |
| `--position-temperature` | 5.0 | Gumbel noise for position selection |
| `--class-temperature` | 0.0 | Token sampling temperature (0 = greedy) |

### System

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `k2-fsa/OmniVoice` | Model path or HuggingFace repo |
| `--asr-model` | `openai/whisper-large-v3-turbo` | Whisper model for ASR |
| `--device` | auto | `cpu`, `cuda`, or `metal` |

## Voice Design Instructions

The `--instruct` flag accepts comma-separated attributes:

| Category | English | Chinese |
|----------|---------|---------|
| Gender | `male`, `female` | `男`, `女` |
| Age | `child`, `teenager`, `young adult`, `middle-aged`, `elderly` | `儿童`, `少年`, `青年`, `中年`, `老年` |
| Pitch | `very low pitch` ... `very high pitch` | `极低音调` ... `极高音调` |
| Style | `whisper` | `耳语` |
| Accent | `american accent`, `british accent`, `indian accent`, ... | -- |
| Dialect | -- | `四川话`, `东北话`, `河南话`, ... |

## Architecture

| Component | Description |
|-----------|-------------|
| **LLM backbone** | Qwen3-0.6B (28 layers, bidirectional attention) |
| **Audio codec** | HiggsAudioV2 (DAC + HuBERT, 8 codebooks, 25 fps) |
| **Generation** | Iterative masked discrete diffusion (32 steps, CFG) |
| **ASR** | Whisper large-v3-turbo (on-demand) |
| **Output** | 24 kHz mono WAV |

The model runs on GPU (Metal/CUDA) in FP16, while the audio tokenizer and Whisper always run on CPU in FP32 for numerical stability.

## Project Structure

```
src/
  main.rs                        CLI entry point
  config.rs                      Config deserialization
  models/
    omnivoice.rs                 Core model + iterative generation
    qwen3_bidirectional.rs       Qwen3 backbone (bidirectional)
    dac.rs                       DAC encoder/decoder
    hubert.rs                    HuBERT feature extractor
    higgs_audio_v2.rs            Audio tokenizer (encode/decode)
    rvq.rs                       Residual vector quantization
    semantic_codec.rs            Semantic encoder/decoder
    whisper_transcribe.rs        Whisper ASR integration
  utils/
    audio.rs                     WAV I/O, resampling, silence removal
    text.rs                      Text chunking, punctuation
    duration.rs                  Duration estimation
    sampling.rs                  Gumbel sampling, top-k filtering
    voice_design.rs              Instruct validation
```

## License

Apache-2.0
