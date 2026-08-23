use crate::protocol::{Dataset, LombPayload, LombResult};
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ComputeError {
    #[error("dataset must contain matching, nonempty coordinate and value arrays")]
    InvalidSeries,
    #[error("input contains a non-finite number")]
    NonFinite,
    #[error("frequency count must be positive")]
    InvalidCount,
    #[error("all frequencies must be finite and positive")]
    InvalidFrequency,
    #[error("value-squared normalization must be finite and positive")]
    InvalidNormalization,
    #[error("Lomb-Scargle denominator or result is invalid")]
    InvalidResult,
}

pub fn execute(dataset: &Dataset, p: &LombPayload) -> Result<LombResult, ComputeError> {
    let (coordinates, values) = dataset.series().ok_or(ComputeError::InvalidSeries)?;
    if coordinates.is_empty() || coordinates.len() != values.len() {
        return Err(ComputeError::InvalidSeries);
    }
    if coordinates.iter().chain(values).any(|v| !v.is_finite()) {
        return Err(ComputeError::NonFinite);
    }
    if p.frequency_count == 0 {
        return Err(ComputeError::InvalidCount);
    }
    if !p.start_frequency.is_finite() || !p.frequency_step.is_finite() {
        return Err(ComputeError::InvalidFrequency);
    }
    let normalization = values
        .iter()
        .fold(0.0_f32, |sum, value| sum + value * value);
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err(ComputeError::InvalidNormalization);
    }
    let (winner, best_frequency, best_power) = (0..p.frequency_count)
        .into_par_iter()
        .map(|index| {
            let frequency = p.start_frequency + index as f32 * p.frequency_step;
            if !frequency.is_finite() || frequency <= 0.0 {
                return Err(ComputeError::InvalidFrequency);
            }
            Ok((
                index,
                frequency,
                power(coordinates, values, normalization, frequency)?,
            ))
        })
        .try_reduce_with(|left, right| {
            let ordering = right.2.total_cmp(&left.2);
            if ordering.is_gt() || (ordering.is_eq() && right.0 < left.0) {
                right
            } else {
                left
            }
        })?
        .ok_or(ComputeError::InvalidCount)?;
    let best_frequency_index = p
        .frequency_start_index
        .checked_add(winner)
        .ok_or(ComputeError::InvalidResult)?;
    Ok(LombResult {
        best_frequency,
        best_period_days: 1.0 / best_frequency,
        best_power,
        best_frequency_index,
    })
}

fn power(x: &[f32], y: &[f32], normalization: f32, frequency: f32) -> Result<f32, ComputeError> {
    let omega = 2.0_f32 * std::f32::consts::PI * frequency;
    let (mut sum_sin_2wt, mut sum_cos_2wt) = (0.0_f32, 0.0_f32);
    for &coordinate in x {
        sum_sin_2wt += (2.0 * omega * coordinate).sin();
        sum_cos_2wt += (2.0 * omega * coordinate).cos();
    }
    let tau = sum_sin_2wt.atan2(sum_cos_2wt) / (2.0 * omega);
    let (mut sum_y_cos, mut sum_y_sin, mut sum_cos_squared, mut sum_sin_squared) =
        (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for (&coordinate, &value) in x.iter().zip(y) {
        let shifted = omega * (coordinate - tau);
        let cosine = shifted.cos();
        let sine = shifted.sin();
        sum_y_cos += value * cosine;
        sum_y_sin += value * sine;
        sum_cos_squared += cosine * cosine;
        sum_sin_squared += sine * sine;
    }
    if !sum_cos_squared.is_finite()
        || !sum_sin_squared.is_finite()
        || sum_cos_squared <= 0.0
        || sum_sin_squared <= 0.0
    {
        return Err(ComputeError::InvalidResult);
    }
    let result = ((sum_y_cos * sum_y_cos / sum_cos_squared)
        + (sum_y_sin * sum_y_sin / sum_sin_squared))
        / normalization;
    if !result.is_finite() {
        return Err(ComputeError::InvalidResult);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset() -> Dataset {
        Dataset {
            coordinates: Some(vec![0.0, 0.37, 1.11, 1.8, 2.73]),
            values: Some(vec![2.0, 3.2, 1.4, 2.8, 1.9]),
            times: None,
            flux: None,
        }
    }

    #[test]
    fn apple_scalar_golden_nonzero_mean_irregular_coordinates() {
        let result = execute(
            &dataset(),
            &LombPayload {
                start_frequency: 0.2,
                frequency_step: 0.15,
                frequency_count: 5,
                frequency_start_index: 40,
            },
        )
        .unwrap();
        assert_eq!(result.best_frequency_index, 44);
        assert!((result.best_frequency - 0.8).abs() < 1e-7);
        assert!((result.best_power - 0.43365732).abs() < 1e-6);
        assert!((result.best_period_days - 1.25).abs() < 1e-6);
    }

    #[test]
    fn regression_does_not_mean_center_or_half_power() {
        let d = Dataset {
            coordinates: Some(vec![0.0, 0.25, 0.5, 0.75]),
            values: Some(vec![2.0, 3.0, 2.0, 1.0]),
            times: None,
            flux: None,
        };
        let result = execute(
            &d,
            &LombPayload {
                start_frequency: 1.0,
                frequency_step: 1.0,
                frequency_count: 1,
                frequency_start_index: 5,
            },
        )
        .unwrap();
        assert!((result.best_power - 0.11111111).abs() < 1e-6);
        assert_eq!(result.best_frequency_index, 5);
    }

    #[test]
    fn rejects_invalid_inputs() {
        let mut d = dataset();
        d.values.as_mut().unwrap()[0] = f32::NAN;
        assert_eq!(
            execute(
                &d,
                &LombPayload {
                    start_frequency: 1.0,
                    frequency_step: 1.0,
                    frequency_count: 1,
                    frequency_start_index: 0
                }
            ),
            Err(ComputeError::NonFinite)
        );
    }

    #[test]
    fn legacy_dataset_and_frequency_chunk_boundary() {
        let d = Dataset {
            coordinates: None,
            values: None,
            times: Some(vec![0.0, 0.25, 0.5, 0.75]),
            flux: Some(vec![2.0, 3.0, 2.0, 1.0]),
        };
        let result = execute(
            &d,
            &LombPayload {
                start_frequency: 1.0,
                frequency_step: 0.25,
                frequency_count: 1,
                frequency_start_index: 4_096,
            },
        )
        .unwrap();
        assert_eq!(result.best_frequency_index, 4_096);
    }

    #[test]
    fn frequency_ties_choose_first_global_index() {
        let result = execute(
            &dataset(),
            &LombPayload {
                start_frequency: 0.8,
                frequency_step: 0.0,
                frequency_count: 8,
                frequency_start_index: 90,
            },
        )
        .unwrap();
        assert_eq!(result.best_frequency_index, 90);
    }

    #[test]
    fn rejects_mismatched_and_nonpositive_frequency_inputs() {
        let mismatch = Dataset {
            coordinates: Some(vec![0.0]),
            values: Some(vec![1.0, 2.0]),
            times: None,
            flux: None,
        };
        assert_eq!(
            execute(
                &mismatch,
                &LombPayload {
                    start_frequency: 1.0,
                    frequency_step: 1.0,
                    frequency_count: 1,
                    frequency_start_index: 0,
                }
            ),
            Err(ComputeError::InvalidSeries)
        );
        assert_eq!(
            execute(
                &dataset(),
                &LombPayload {
                    start_frequency: 0.0,
                    frequency_step: 1.0,
                    frequency_count: 1,
                    frequency_start_index: 0,
                }
            ),
            Err(ComputeError::InvalidFrequency)
        );
    }
}
