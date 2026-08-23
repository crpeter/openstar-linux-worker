use crate::protocol::{Dataset, LombPayload, LombResult};
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ValidationError {
    #[error("dataset is empty")]
    Empty,
    #[error("dataset arrays have inconsistent lengths")]
    Length,
    #[error("input contains a non-finite number")]
    NonFinite,
    #[error("frequency count must be positive")]
    Count,
    #[error("frequency step must be positive")]
    Step,
    #[error("dataset has zero variance")]
    ZeroVariance,
}

pub fn execute(dataset: &Dataset, p: &LombPayload) -> Result<LombResult, ValidationError> {
    let y = dataset.values().ok_or(ValidationError::Length)?;
    if y.is_empty() {
        return Err(ValidationError::Empty);
    }
    if dataset.times.len() != y.len() {
        return Err(ValidationError::Length);
    }
    if p.frequency_count == 0 {
        return Err(ValidationError::Count);
    }
    if !p.frequency_start.is_finite()
        || !p.frequency_step.is_finite()
        || p.frequency_step <= 0.0
        || dataset.times.iter().chain(y).any(|v| !v.is_finite())
    {
        return Err(if p.frequency_step <= 0.0 {
            ValidationError::Step
        } else {
            ValidationError::NonFinite
        });
    }
    let mean = y.iter().copied().sum::<f32>() / y.len() as f32;
    let variance = y
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>();
    if variance == 0.0 {
        return Err(ValidationError::ZeroVariance);
    }
    let powers: Vec<f32> = (0..p.frequency_count)
        .into_par_iter()
        .map(|i| {
            power(
                &dataset.times,
                y,
                mean,
                variance,
                p.frequency_start + p.frequency_step * i as f32,
            )
        })
        .collect();
    let local = powers
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|x| x.0)
        .unwrap_or(0);
    Ok(LombResult {
        powers,
        best_frequency_index: p.frequency_start_index + local,
    })
}

fn power(t: &[f32], y: &[f32], mean: f32, variance: f32, frequency: f32) -> f32 {
    let omega = 2.0_f32 * std::f32::consts::PI * frequency;
    let (mut s2, mut c2) = (0.0_f32, 0.0_f32);
    for &x in t {
        s2 += (2.0 * omega * x).sin();
        c2 += (2.0 * omega * x).cos();
    }
    let tau = s2.atan2(c2) / (2.0 * omega);
    let (mut yc, mut ys, mut cc, mut ss) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for (&x, &v) in t.iter().zip(y) {
        let a = omega * (x - tau);
        let c = a.cos();
        let s = a.sin();
        let d = v - mean;
        yc += d * c;
        ys += d * s;
        cc += c * c;
        ss += s * s;
    }
    0.5 * (yc * yc / cc + ys * ys / ss) / variance
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn peak_and_global_index() {
        let d = Dataset {
            times: (0..20).map(|x| x as f32 / 10.0).collect(),
            flux: Some(
                (0..20)
                    .map(|x| (2.0 * std::f32::consts::PI * 2.0 * x as f32 / 10.0).sin())
                    .collect(),
            ),
            values: None,
        };
        let r = execute(
            &d,
            &LombPayload {
                frequency_start: 1.0,
                frequency_step: 0.5,
                frequency_count: 5,
                frequency_start_index: 10,
            },
        )
        .unwrap();
        assert_eq!(r.best_frequency_index, 12);
    }
    #[test]
    fn rejects_bad_data() {
        let d = Dataset {
            times: vec![0.],
            flux: Some(vec![f32::NAN]),
            values: None,
        };
        assert_eq!(
            execute(
                &d,
                &LombPayload {
                    frequency_start: 1.,
                    frequency_step: 1.,
                    frequency_count: 1,
                    frequency_start_index: 0
                }
            ),
            Err(ValidationError::NonFinite)
        );
    }
}
