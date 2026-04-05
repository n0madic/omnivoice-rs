//! Whisper-based automatic speech recognition for reference audio transcription.
//!
//! Loads a Whisper model on-demand (only when `--ref-text` is omitted) and
//! transcribes reference audio to text. Runs on CPU in F32.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper;

use crate::utils::audio::resample;

/// Whisper ASR transcriber.
pub struct WhisperTranscriber {
    model: whisper::model::Whisper,
    tokenizer: tokenizers::Tokenizer,
    mel_filters: Vec<f32>,
    device: Device,
    // Special token IDs
    sot_token: u32,
    transcribe_token: u32,
    no_timestamps_token: u32,
    eot_token: u32,
}

impl WhisperTranscriber {
    /// Load a Whisper model from HuggingFace.
    ///
    /// Model is downloaded on first use and cached locally.
    pub fn new(model_id: &str, device: &Device) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::new(
            model_id.to_string(),
            hf_hub::RepoType::Model,
        ));

        let config_path = repo.get("config.json").context("download config.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .context("download model.safetensors")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("download tokenizer.json")?;

        let config: whisper::Config = serde_json::from_reader(std::fs::File::open(&config_path)?)?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };
        let model = whisper::model::Whisper::load(&vb, config.clone())?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        let mel_filters =
            compute_mel_filters(whisper::SAMPLE_RATE, whisper::N_FFT, config.num_mel_bins);

        let sot_token = tokenizer
            .token_to_id(whisper::SOT_TOKEN)
            .context("missing SOT token")?;
        let transcribe_token = tokenizer
            .token_to_id(whisper::TRANSCRIBE_TOKEN)
            .context("missing TRANSCRIBE token")?;
        let no_timestamps_token = tokenizer
            .token_to_id(whisper::NO_TIMESTAMPS_TOKEN)
            .context("missing NO_TIMESTAMPS token")?;
        let eot_token = tokenizer
            .token_to_id(whisper::EOT_TOKEN)
            .context("missing EOT token")?;

        Ok(Self {
            model,
            tokenizer,
            mel_filters,
            device: device.clone(),
            sot_token,
            transcribe_token,
            no_timestamps_token,
            eot_token,
        })
    }

    /// Transcribe audio tensor to text.
    ///
    /// `audio`: tensor of shape `(1, T)` at any sample rate (will be resampled to 16kHz).
    /// `sample_rate`: sample rate of input audio.
    pub fn transcribe(&mut self, audio: &Tensor, sample_rate: usize) -> Result<String> {
        // 1. Resample to 16kHz
        let samples_f32: Vec<f32> = audio.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let samples = if sample_rate != whisper::SAMPLE_RATE {
            resample(&samples_f32, sample_rate, whisper::SAMPLE_RATE)?
        } else {
            samples_f32
        };

        // 2. Compute mel spectrogram
        let mel = whisper::audio::pcm_to_mel(&self.model.config, &samples, &self.mel_filters);
        let n_mels = self.model.config.num_mel_bins;
        let n_frames = mel.len() / n_mels;
        let mel_tensor = Tensor::from_vec(mel, (1, n_mels, n_frames), &self.device)?;

        // 3. Encode audio
        let encoded = self.model.encoder.forward(&mel_tensor, true)?;

        // 4. Greedy decode
        let mut tokens: Vec<u32> = vec![
            self.sot_token,
            self.transcribe_token,
            self.no_timestamps_token,
        ];

        let max_decode_steps = self.model.config.max_target_positions / 2;
        let suppress = &self.model.config.suppress_tokens;

        for _step in 0..max_decode_steps {
            let token_tensor = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &self.device)?;

            let logits = {
                let decoder_out = self.model.decoder.forward(
                    &token_tensor,
                    &encoded,
                    tokens.len() == 3, // flush KV cache on first step
                )?;
                self.model.decoder.final_linear(&decoder_out)?
            };

            // Get logits for last position
            let last_logits = logits.i((0, logits.dim(1)? - 1))?;
            let mut logits_vec: Vec<f32> = last_logits.to_vec1()?;

            // Suppress tokens
            for &t in suppress {
                if (t as usize) < logits_vec.len() {
                    logits_vec[t as usize] = f32::NEG_INFINITY;
                }
            }

            // Argmax
            let next_token = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(self.eot_token);

            if next_token == self.eot_token {
                break;
            }

            tokens.push(next_token);
        }

        // 5. Reset KV cache for next use
        self.model.reset_kv_cache();

        // 6. Decode tokens to text (skip prompt tokens)
        let text_tokens: Vec<u32> = tokens[3..].to_vec(); // skip SOT, transcribe, no_timestamps
        let text = self
            .tokenizer
            .decode(&text_tokens, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;

        Ok(text.trim().to_string())
    }
}

// ---------------------------------------------------------------------------
// Mel filter bank computation (standard librosa triangular filters)
// ---------------------------------------------------------------------------

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// Compute triangular mel filter banks with Slaney normalization.
///
/// Returns a flat array of `n_mels * n_freqs` where `n_freqs = n_fft / 2 + 1`.
fn compute_mel_filters(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let fmax = sample_rate as f64 / 2.0;

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 equally spaced points in mel scale
    let mels: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();
    let freqs: Vec<f64> = mels.iter().map(|&m| mel_to_hz(m)).collect();

    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| fmax * i as f64 / (n_freqs - 1) as f64)
        .collect();

    let mut filters = vec![0.0f32; n_mels * n_freqs];

    for i in 0..n_mels {
        let lower = freqs[i];
        let center = freqs[i + 1];
        let upper = freqs[i + 2];

        // Slaney normalization factor
        let enorm = 2.0 / (upper - lower);

        for (j, &ff) in fft_freqs.iter().enumerate() {
            let w = if ff >= lower && ff <= center && center > lower {
                (ff - lower) / (center - lower)
            } else if ff > center && ff <= upper && upper > center {
                (upper - ff) / (upper - center)
            } else {
                0.0
            };
            filters[i * n_freqs + j] = (w * enorm) as f32;
        }
    }

    filters
}
