// Copyright 2026 Xiaomi Corp. (authors: Han Zhu)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Timestep schedule and sampling utilities for iterative decoding.
//!
//! Provides the noise schedule, Gumbel-noise sampling, and top-k filtering
//! used by `OmniVoice::generate()`.

use candle_core::{DType, Result, Tensor, D};

/// Compute the shifted timestep schedule for iterative decoding.
///
/// Returns `num_step + 1` values from `t_start` to `t_end`, with a time-shift
/// transformation applied: `t_shifted = t_shift * t / (1 + (t_shift - 1) * t)`.
///
/// # Example
///
/// ```
/// use omnivoice_rs::utils::sampling::get_time_steps;
///
/// let steps = get_time_steps(0.0, 1.0, 10, 0.1);
/// assert_eq!(steps.len(), 11);
/// assert!((steps[0] - 0.0).abs() < 1e-9);
/// assert!((steps[10] - 1.0).abs() < 1e-9);
/// ```
pub fn get_time_steps(t_start: f64, t_end: f64, num_step: usize, t_shift: f64) -> Vec<f64> {
    let n = num_step + 1;
    let mut steps = Vec::with_capacity(n);

    for i in 0..n {
        let t = if num_step == 0 {
            t_start
        } else {
            t_start + (t_end - t_start) * i as f64 / num_step as f64
        };
        // Apply time-shift transformation.
        let shifted = t_shift * t / (1.0 + (t_shift - 1.0) * t);
        steps.push(shifted);
    }

    steps
}

/// Add Gumbel noise to logits scaled by `temperature`.
///
/// Computes: `logits / temperature + gumbel_noise` where
/// `gumbel_noise = -log(-log(u + eps) + eps)` with `u ~ Uniform(0,1)`.
///
/// The result has the same shape as `logits`.
pub fn gumbel_sample(logits: &Tensor, temperature: f64) -> Result<Tensor> {
    let scaled = (logits / temperature)?;

    // Generate uniform random noise in (0, 1).
    let u = Tensor::rand_like(&scaled, 0.0, 1.0)?;

    // gumbel = -log(-log(u + eps) + eps)
    let gumbel_noise = ((u + 1e-10)?.log()?.neg()? + 1e-10)?.log()?.neg()?;

    scaled + gumbel_noise
}

/// Filter logits to keep only the top-k values (by ratio of vocabulary size).
///
/// All positions outside the top `ceil(ratio * vocab_size)` are set to
/// negative infinity.
///
/// # Arguments
///
/// * `logits` -- tensor of shape `(..., vocab_size)`.
/// * `ratio` -- fraction of vocabulary to keep (e.g. 0.1 = top 10%).
pub fn filter_top_k(logits: &Tensor, ratio: f64) -> Result<Tensor> {
    let vocab_size = logits.dim(D::Minus1)?;
    let k = ((ratio * vocab_size as f64).ceil() as usize)
        .max(1)
        .min(vocab_size);

    // Flatten leading dimensions to work on 2-D.
    let shape = logits.shape().clone();
    let leading: usize = shape.dims().iter().rev().skip(1).product();
    let flat = logits.reshape((leading, vocab_size))?;

    // Sort descending along last dim to find the k-th threshold value.
    let sorted_indices = flat.arg_sort_last_dim(false)?;

    // Build output: positions in the top-k keep their value, rest get -inf.
    let flat_f32 = flat.to_dtype(DType::F32)?;
    let mut output_data = vec![f32::NEG_INFINITY; leading * vocab_size];

    for row in 0..leading {
        let row_vals = flat_f32.get(row)?.to_vec1::<f32>()?;
        let row_indices = sorted_indices.get(row)?.to_vec1::<u32>()?;

        for &idx in row_indices.iter().take(k) {
            let idx = idx as usize;
            output_data[row * vocab_size + idx] = row_vals[idx];
        }
    }

    let result = Tensor::from_vec(output_data, (leading, vocab_size), logits.device())?;
    let result = result.to_dtype(logits.dtype())?;
    result.reshape(shape)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_time_steps_endpoints() {
        let steps = get_time_steps(0.0, 1.0, 10, 1.0);
        assert_eq!(steps.len(), 11);
        assert!((steps[0] - 0.0).abs() < 1e-12);
        assert!((steps[10] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_time_steps_monotonic() {
        let steps = get_time_steps(0.0, 1.0, 32, 0.1);
        for i in 1..steps.len() {
            assert!(
                steps[i] >= steps[i - 1],
                "Steps must be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_time_steps_shift_identity() {
        // t_shift = 1.0 => no shift: t * 1 / (1 + 0*t) = t
        let steps = get_time_steps(0.0, 1.0, 5, 1.0);
        for (i, &s) in steps.iter().enumerate() {
            let expected = i as f64 / 5.0;
            assert!((s - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_time_steps_single_step() {
        let steps = get_time_steps(0.0, 1.0, 1, 0.5);
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_gumbel_sample_shape() {
        let logits = Tensor::randn(0f32, 1f32, (2, 10), &Device::Cpu).unwrap();
        let result = gumbel_sample(&logits, 1.0).unwrap();
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_filter_top_k_shape() {
        let logits = Tensor::randn(0f32, 1f32, (2, 100), &Device::Cpu).unwrap();
        let result = filter_top_k(&logits, 0.1).unwrap();
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_filter_top_k_sparsity() {
        let logits = Tensor::randn(0f32, 1f32, (1, 100), &Device::Cpu).unwrap();
        let result = filter_top_k(&logits, 0.1).unwrap();
        let vals = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let finite_count = vals.iter().filter(|&&v| v.is_finite()).count();
        // top 10% of 100 = 10
        assert_eq!(finite_count, 10);
    }

    #[test]
    fn test_filter_top_k_preserves_values() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let logits = Tensor::from_vec(data, (1, 20), &Device::Cpu).unwrap();
        let result = filter_top_k(&logits, 0.25).unwrap(); // top 5
        let vals = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // The top 5 values (15..=19) should be preserved exactly.
        for i in 15..20 {
            assert!((vals[i] - i as f32).abs() < 1e-6);
        }
        // All others should be -inf.
        for i in 0..15 {
            assert!(vals[i].is_infinite() && vals[i] < 0.0);
        }
    }
}
