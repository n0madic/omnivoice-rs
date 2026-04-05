use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use tracing::info;

use omnivoice_rs::config::{HiggsAudioV2Config, OmniVoiceConfig};
use omnivoice_rs::models::higgs_audio_v2::HiggsAudioV2Tokenizer;
use omnivoice_rs::models::omnivoice::{GenerateRequest, GenerationConfig, OmniVoice};
use omnivoice_rs::utils::audio::{fade_and_pad, load_wav, remove_silence, save_wav};
use omnivoice_rs::utils::duration::RuleDurationEstimator;
use omnivoice_rs::utils::text::{add_punctuation, combine_text, is_cjk};

#[derive(Parser, Debug)]
#[command(name = "omnivoice-rs", about = "OmniVoice TTS inference (Rust/Candle)")]
struct Args {
    /// Model checkpoint path or HuggingFace repo id
    #[arg(long, default_value = "k2-fsa/OmniVoice")]
    model: String,

    /// Text to synthesize
    #[arg(long)]
    text: String,

    /// Output WAV file path
    #[arg(long)]
    output: String,

    /// Reference audio file path for voice cloning
    #[arg(long)]
    ref_audio: Option<String>,

    /// Reference text describing the reference audio
    #[arg(long)]
    ref_text: Option<String>,

    /// Style instruction for voice design mode
    #[arg(long)]
    instruct: Option<String>,

    /// Language name or code
    #[arg(long)]
    language: Option<String>,

    /// Whisper model for auto-transcription (downloaded only when --ref-text is omitted)
    #[arg(long, default_value = "openai/whisper-large-v3-turbo")]
    asr_model: String,

    /// Number of iterative decoding steps
    #[arg(long, default_value_t = 32)]
    num_step: usize,

    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 2.0)]
    guidance_scale: f64,

    /// Speaking speed factor
    #[arg(long, default_value_t = 1.0)]
    speed: f64,

    /// Fixed output duration in seconds
    #[arg(long)]
    duration: Option<f64>,

    /// Noise schedule time shift
    #[arg(long, default_value_t = 0.1)]
    t_shift: f64,

    /// Whether to prepend the denoise token
    #[arg(long, default_value_t = true, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set)]
    denoise: bool,

    /// Post-process output (silence removal, fade)
    #[arg(long, default_value_t = true, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set)]
    postprocess_output: bool,

    /// Layer penalty factor (encourage lower codebook layers first)
    #[arg(long, default_value_t = 5.0)]
    layer_penalty_factor: f64,

    /// Temperature for position selection (Gumbel noise)
    #[arg(long, default_value_t = 5.0)]
    position_temperature: f64,

    /// Temperature for token sampling (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    class_temperature: f64,

    /// Device: cpu, cuda, or metal (auto-detect if not specified)
    #[arg(long)]
    device: Option<String>,
}

/// Resolve language name or code to ISO 639-1 code using `isolang`.
/// Accepts: ISO 639-1 ("en"), ISO 639-3 ("eng"), or full English name ("English").
fn resolve_language(lang: Option<&str>) -> Option<String> {
    let lang = lang?;
    if lang.eq_ignore_ascii_case("none") {
        return None;
    }
    // Try ISO 639-1 code first (e.g. "en")
    if let Some(l) = isolang::Language::from_639_1(lang) {
        return l.to_639_1().map(|s| s.to_string());
    }
    // Try ISO 639-3 code (e.g. "eng")
    if let Some(l) = isolang::Language::from_639_3(lang) {
        return l.to_639_1().map(|s| s.to_string());
    }
    // Try full English name (e.g. "English", "Chinese")
    if let Some(l) = isolang::Language::from_name(lang) {
        if let Some(code) = l.to_639_1() {
            return Some(code.to_string());
        }
    }
    // Case-insensitive name lookup via FromStr (requires lowercase_names feature)
    if let Ok(l) = lang.parse::<isolang::Language>() {
        if let Some(code) = l.to_639_1() {
            return Some(code.to_string());
        }
    }
    tracing::warn!(
        "Unknown language '{lang}'. Use ISO 639-1 (en, zh, ja), \
         ISO 639-3 (eng, zho, jpn), or English name (English, Chinese)."
    );
    Some(lang.to_string())
}

fn get_device(requested: Option<&str>) -> Result<Device> {
    match requested {
        Some("cpu") => Ok(Device::Cpu),
        Some("cuda") => Ok(Device::new_cuda(0)?),
        Some("metal") => Ok(Device::new_metal(0)?),
        Some(other) => bail!("Unknown device: {other}"),
        None => {
            if candle_core::utils::cuda_is_available() {
                Ok(Device::new_cuda(0)?)
            } else if candle_core::utils::metal_is_available() {
                Ok(Device::new_metal(0)?)
            } else {
                Ok(Device::Cpu)
            }
        }
    }
}

/// Resolve model path: local directory or download from HuggingFace Hub.
fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    let local = PathBuf::from(model_id);
    if local.is_dir() {
        return Ok(local);
    }
    info!("Downloading model from HuggingFace: {model_id}");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    // Download essential files
    let files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "audio_tokenizer/config.json",
        "audio_tokenizer/model.safetensors",
        "audio_tokenizer/preprocessor_config.json",
    ];
    let mut base_dir = None;
    for f in &files {
        let path = repo
            .get(f)
            .with_context(|| format!("Failed to download {f}"))?;
        if base_dir.is_none() {
            // The parent of config.json is the model directory
            if *f == "config.json" {
                base_dir = path.parent().map(|p| p.to_path_buf());
            }
        }
    }
    base_dir.context("Failed to resolve model directory")
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let device = get_device(args.device.as_deref())?;
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    };

    info!("Using device: {:?}", device);

    // 1. Resolve model path
    let model_dir = resolve_model_path(&args.model)?;
    info!("Model directory: {}", model_dir.display());

    // 2. Load configs
    let config_path = model_dir.join("config.json");
    let config: OmniVoiceConfig = serde_json::from_reader(std::fs::File::open(&config_path)?)?;

    let audio_config_path = model_dir.join("audio_tokenizer/config.json");
    let audio_config: HiggsAudioV2Config =
        serde_json::from_reader(std::fs::File::open(&audio_config_path)?)?;

    // 3. Load text tokenizer
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    // 4. Load OmniVoice model
    info!("Loading OmniVoice model...");
    let model_weights = model_dir.join("model.safetensors");
    // SAFETY: The safetensors file was either downloaded from HuggingFace Hub
    // (validated by hf-hub) or provided as a local path by the user. The file
    // must remain unmodified while memory-mapped. This is the standard candle
    // pattern for loading model weights.
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[model_weights], dtype, &device)?
    };
    let model = OmniVoice::new(&config, vb)?;

    // 5. Load HiggsAudioV2 tokenizer
    // Audio tokenizer always runs on CPU in F32 — Metal/MPS don't support the
    // Snake1d activations and conv_transpose1d at F16 precision, matching the
    // Python behavior (which also forces audio_tokenizer to CPU for MPS).
    info!("Loading HiggsAudioV2 audio tokenizer (CPU, F32)...");
    let audio_weights = model_dir.join("audio_tokenizer/model.safetensors");
    // SAFETY: Same as above — file is from HuggingFace or user-supplied local path.
    let audio_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[audio_weights], DType::F32, &Device::Cpu)?
    };
    let audio_tokenizer = HiggsAudioV2Tokenizer::new(&audio_config, audio_vb)?;

    // 6. Process reference audio if provided
    let (ref_audio_tokens, ref_text, ref_rms) = if let Some(ref_audio_path) = &args.ref_audio {
        info!("Processing reference audio: {ref_audio_path}");
        let sampling_rate = audio_config.sample_rate();
        let wav = load_wav(ref_audio_path, sampling_rate)?;

        // Compute RMS for volume normalization
        let rms = (wav.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?) as f64;

        // Normalize quiet audio before encoding (matches Python create_voice_clone_prompt)
        let wav = if rms > 0.0 && rms < 0.1 {
            (&wav * (0.1 / rms))?
        } else {
            wav
        };

        // Trim long audio (>20s) — only when ref_text not provided (auto-transcribe case)
        // When ref_text IS provided, warn but don't trim (matches Python behavior)
        let max_ref_seconds = 20.0;
        let wav_dur = wav.dim(1)? as f64 / sampling_rate as f64;
        let wav = if wav_dur > max_ref_seconds && args.ref_text.is_none() {
            let max_samples = (max_ref_seconds * sampling_rate as f64) as usize;
            tracing::info!("Trimming reference audio from {wav_dur:.1}s to {max_ref_seconds}s");
            wav.narrow(1, 0, max_samples.min(wav.dim(1)?))?
        } else if wav_dur > max_ref_seconds {
            tracing::warn!(
                "Reference audio is {wav_dur:.1}s (>{max_ref_seconds}s). \
                 Long references may degrade quality."
            );
            wav
        } else {
            wav
        };

        // Silence removal
        let wav = remove_silence(&wav, sampling_rate, 200, 100, 200)?;

        // Clip to multiple of hop_length
        let hop_length = audio_config.hop_length();
        let len = wav.dim(1)?;
        let clip = len % hop_length;
        let wav = if clip > 0 {
            wav.narrow(1, 0, len - clip)?
        } else {
            wav
        };

        // Encode reference audio to tokens (audio tokenizer is on CPU)
        let tokens = audio_tokenizer.encode(&wav.unsqueeze(0)?)?;
        let tokens = tokens.squeeze(0)?; // (8, T)

        let ref_text_str = args.ref_text.clone().unwrap_or_default();
        let ref_text_str = if ref_text_str.is_empty() {
            // Auto-transcribe with Whisper (model downloaded on first use)
            info!("Loading Whisper ASR model: {} ...", args.asr_model);
            let mut transcriber =
                omnivoice_rs::models::whisper_transcribe::WhisperTranscriber::new(
                    &args.asr_model,
                    &Device::Cpu,
                )?;
            info!("Transcribing reference audio...");
            let transcript = transcriber.transcribe(&wav, sampling_rate)?;
            info!("Transcribed: {transcript}");
            add_punctuation(&transcript)
        } else {
            add_punctuation(&ref_text_str)
        };

        (Some(tokens), Some(ref_text_str), Some(rms))
    } else {
        (None, None, None)
    };

    // 7. Estimate target duration
    let duration_estimator = RuleDurationEstimator::new();
    let frame_rate = audio_config.frame_rate();

    let num_target_tokens = if let Some(dur) = args.duration {
        (dur * frame_rate as f64).ceil() as usize
    } else {
        let (est_ref_text, est_ref_tokens) =
            if let (Some(rt), Some(rat)) = (&ref_text, &ref_audio_tokens) {
                (rt.as_str(), rat.dim(1)?)
            } else {
                ("Nice to meet you.", 25)
            };
        let est =
            duration_estimator.estimate_duration(&args.text, est_ref_text, est_ref_tokens as f64);
        let est = if args.speed > 0.0 && args.speed != 1.0 {
            est / args.speed
        } else {
            est
        };
        est.max(1.0) as usize // truncation, matching Python's int()
    };

    info!("Target audio tokens: {num_target_tokens}");

    // 8. Validate instruct and resolve language
    let use_zh = args.text.chars().any(is_cjk);
    let instruct = match &args.instruct {
        Some(s) => omnivoice_rs::utils::voice_design::resolve_instruct(Some(s), use_zh)?,
        None => None,
    };
    let language = resolve_language(args.language.as_deref());

    let gen_config = GenerationConfig {
        num_step: args.num_step,
        guidance_scale: args.guidance_scale,
        t_shift: args.t_shift,
        layer_penalty_factor: args.layer_penalty_factor,
        position_temperature: args.position_temperature,
        class_temperature: args.class_temperature,
        denoise: args.denoise,
        ..GenerationConfig::default()
    };

    let full_text = combine_text(&args.text, ref_text.as_deref());

    info!(
        "Generating audio for: {}...",
        &args.text.chars().take(80).collect::<String>()
    );
    let chunk_tokens = model.generate(&GenerateRequest {
        tokenizer: &tokenizer,
        full_text: &full_text,
        num_target_tokens,
        ref_audio_tokens: ref_audio_tokens.as_ref(),
        ref_text: ref_text.as_deref(),
        lang: language.as_deref(),
        instruct: instruct.as_deref(),
        gen_config: &gen_config,
        frame_rate,
        speed: args.speed,
        duration_estimator: &duration_estimator,
        device: &device,
        dtype,
    })?;

    // 9. Decode token chunks to waveform (audio tokenizer runs on CPU)
    info!("Decoding {} audio chunk(s)...", chunk_tokens.len());
    let chunk_audios: Vec<candle_core::Tensor> = chunk_tokens
        .iter()
        .map(|t| {
            let cpu_tokens = t.to_device(&Device::Cpu)?;
            let decoded = audio_tokenizer.decode(&cpu_tokens.unsqueeze(0)?)?;
            decoded.squeeze(0)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let audio = if chunk_audios.len() == 1 {
        chunk_audios.into_iter().next().unwrap()
    } else {
        use omnivoice_rs::utils::audio::cross_fade_chunks;
        cross_fade_chunks(&chunk_audios, audio_config.sample_rate(), 0.3)?
    };

    // 10. Post-process
    let sampling_rate = audio_config.sample_rate();
    let audio = if args.postprocess_output {
        remove_silence(&audio, sampling_rate, 500, 100, 100)?
    } else {
        audio
    };

    // Volume normalization
    let audio = if let Some(rms) = ref_rms {
        if rms < 0.1 {
            (audio * (rms / 0.1))?
        } else {
            audio
        }
    } else {
        // Voice design mode: peak-normalize to 0.5
        let peak = audio.abs()?.max(1)?.max(0)?.to_scalar::<f32>()?;
        if peak > 1e-6 {
            ((audio / (peak as f64))? * 0.5)?
        } else {
            audio
        }
    };

    let audio = fade_and_pad(&audio, 0.1, 0.1, sampling_rate)?;

    // 11. Save WAV
    let audio_f32 = audio.to_dtype(DType::F32)?;
    save_wav(&args.output, &audio_f32, sampling_rate)?;
    info!("Saved to {}", args.output);

    Ok(())
}
