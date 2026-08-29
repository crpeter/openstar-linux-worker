use crate::protocol::{BoxPeriodPayload, BoxPeriodResult, Dataset};
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum BoxPeriodError {
    #[error("dataset must contain matching, nonempty coordinate and value arrays")]
    InvalidSeries,
    #[error("input contains a non-finite number")]
    NonFinite,
    #[error("frequency grid must be finite, positive, and nonempty")]
    InvalidFrequencyGrid,
    #[error("phase bin count must be at least two")]
    InvalidPhaseBinCount,
    #[error("duration fractions must be finite, nonempty, and between zero and one")]
    InvalidDurations,
    #[error("sample-count gates cannot be satisfied by this dataset")]
    InvalidSampleGates,
    #[error("no box window satisfied the sample-count gates")]
    NoValidWindow,
}

#[derive(Clone, Debug)]
struct Candidate {
    score: f32,
    frequency_index: usize,
    duration_index: usize,
    phase_bin: usize,
    duration_bins: usize,
    in_box_samples: usize,
    out_of_box_samples: usize,
}

pub fn execute(
    dataset: &Dataset,
    payload: &BoxPeriodPayload,
) -> Result<BoxPeriodResult, BoxPeriodError> {
    let (coordinates, values) = validate(dataset, payload)?;
    let candidates = (0..payload.frequency_count)
        .into_par_iter()
        .map(|frequency_index| score_frequency(coordinates, values, payload, frequency_index))
        .collect::<Result<Vec<_>, _>>()?;
    let winner = candidates
        .into_iter()
        .flatten()
        .reduce(select_better)
        .ok_or(BoxPeriodError::NoValidWindow)?;
    let best_frequency =
        payload.start_frequency + winner.frequency_index as f32 * payload.frequency_step;
    Ok(BoxPeriodResult {
        best_frequency,
        best_score: winner.score,
        best_phase: winner.phase_bin as f32 / payload.phase_bin_count as f32,
        best_duration_fraction: winner.duration_bins as f32 / payload.phase_bin_count as f32,
        best_frequency_index: payload.frequency_start_index + winner.frequency_index,
        best_duration_index: winner.duration_index,
        best_phase_bin: winner.phase_bin,
        in_box_samples: winner.in_box_samples,
        out_of_box_samples: winner.out_of_box_samples,
    })
}

fn validate<'a>(
    dataset: &'a Dataset,
    payload: &BoxPeriodPayload,
) -> Result<(&'a [f32], &'a [f32]), BoxPeriodError> {
    let (coordinates, values) = dataset.series().ok_or(BoxPeriodError::InvalidSeries)?;
    if coordinates.is_empty() || coordinates.len() != values.len() {
        return Err(BoxPeriodError::InvalidSeries);
    }
    if coordinates
        .iter()
        .chain(values)
        .any(|value| !value.is_finite())
    {
        return Err(BoxPeriodError::NonFinite);
    }
    if payload.frequency_count == 0
        || !payload.start_frequency.is_finite()
        || payload.start_frequency <= 0.0
        || !payload.frequency_step.is_finite()
        || payload.frequency_step <= 0.0
    {
        return Err(BoxPeriodError::InvalidFrequencyGrid);
    }
    let last =
        payload.start_frequency + (payload.frequency_count - 1) as f32 * payload.frequency_step;
    if !last.is_finite() || last <= 0.0 {
        return Err(BoxPeriodError::InvalidFrequencyGrid);
    }
    if payload.phase_bin_count < 2 {
        return Err(BoxPeriodError::InvalidPhaseBinCount);
    }
    if payload.duration_fractions.is_empty()
        || payload
            .duration_fractions
            .iter()
            .any(|duration| !duration.is_finite() || *duration <= 0.0 || *duration >= 1.0)
    {
        return Err(BoxPeriodError::InvalidDurations);
    }
    if payload.minimum_in_box_samples == 0
        || payload.minimum_out_of_box_samples == 0
        || payload.minimum_in_box_samples + payload.minimum_out_of_box_samples > coordinates.len()
    {
        return Err(BoxPeriodError::InvalidSampleGates);
    }
    Ok((coordinates, values))
}

fn score_frequency(
    coordinates: &[f32],
    values: &[f32],
    payload: &BoxPeriodPayload,
    frequency_index: usize,
) -> Result<Option<Candidate>, BoxPeriodError> {
    let frequency = payload.start_frequency + frequency_index as f32 * payload.frequency_step;
    let mut sums = vec![0.0_f32; payload.phase_bin_count];
    let mut counts = vec![0_usize; payload.phase_bin_count];
    let mut total_sum = 0.0_f32;
    for (&coordinate, &value) in coordinates.iter().zip(values) {
        let cycles = coordinate * frequency;
        let phase = cycles - cycles.floor();
        let bin =
            ((phase * payload.phase_bin_count as f32) as usize).min(payload.phase_bin_count - 1);
        sums[bin] += value;
        counts[bin] += 1;
        total_sum += value;
    }
    let total_count = coordinates.len();
    let mut best: Option<Candidate> = None;
    for (duration_index, duration) in payload.duration_fractions.iter().enumerate() {
        let duration_bins = ((*duration * payload.phase_bin_count as f32 + 0.5).floor() as usize)
            .clamp(1, payload.phase_bin_count - 1);
        let mut inside_sum: f32 = sums[..duration_bins].iter().sum();
        let mut inside_count: usize = counts[..duration_bins].iter().sum();
        for phase_bin in 0..payload.phase_bin_count {
            let outside_count = total_count - inside_count;
            if inside_count >= payload.minimum_in_box_samples
                && outside_count >= payload.minimum_out_of_box_samples
            {
                let outside_sum = total_sum - inside_sum;
                let contrast =
                    outside_sum / outside_count as f32 - inside_sum / inside_count as f32;
                let score = contrast
                    * ((inside_count as f32 * outside_count as f32) / total_count as f32).sqrt();
                if score.is_finite() {
                    let candidate = Candidate {
                        score,
                        frequency_index,
                        duration_index,
                        phase_bin,
                        duration_bins,
                        in_box_samples: inside_count,
                        out_of_box_samples: outside_count,
                    };
                    best = Some(match best {
                        None => candidate,
                        Some(current) => select_better(current, candidate),
                    });
                }
            }
            let leaving = phase_bin;
            let entering = (phase_bin + duration_bins) % payload.phase_bin_count;
            inside_sum += sums[entering] - sums[leaving];
            inside_count = inside_count + counts[entering] - counts[leaving];
        }
    }
    Ok(best)
}

fn select_better(current: Candidate, candidate: Candidate) -> Candidate {
    let candidate_key = (
        candidate.frequency_index,
        candidate.duration_index,
        candidate.phase_bin,
    );
    let current_key = (
        current.frequency_index,
        current.duration_index,
        current.phase_bin,
    );
    if candidate.score.total_cmp(&current.score).is_gt()
        || (candidate.score.total_cmp(&current.score).is_eq() && candidate_key < current_key)
    {
        candidate
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Dataset, BoxPeriodPayload) {
        let coordinates = (0..80).map(|index| index as f32 * 0.25).collect::<Vec<_>>();
        let values = coordinates
            .iter()
            .map(|time| {
                let phase = (*time * 0.5) - (*time * 0.5).floor();
                if phase < 0.15 {
                    -2.0
                } else {
                    0.25
                }
            })
            .collect();
        (
            Dataset {
                coordinates: Some(coordinates),
                values: Some(values),
                times: None,
                flux: None,
            },
            BoxPeriodPayload {
                start_frequency: 0.4,
                frequency_step: 0.05,
                frequency_count: 5,
                frequency_start_index: 20,
                phase_bin_count: 20,
                duration_fractions: vec![0.1, 0.15],
                minimum_in_box_samples: 4,
                minimum_out_of_box_samples: 20,
            },
        )
    }

    #[test]
    fn deterministic_box_golden_vector() {
        let (dataset, payload) = fixture();
        let result = execute(&dataset, &payload).unwrap();
        assert_eq!(result.best_frequency_index, 22);
        assert!((result.best_frequency - 0.5).abs() < 1e-7);
        assert_eq!(result.best_duration_index, 1);
        assert_eq!(result.best_phase_bin, 0);
        assert!((result.best_phase - 0.0).abs() < 1e-7);
        assert!((result.best_duration_fraction - 0.15).abs() < 1e-7);
        assert_eq!(result.in_box_samples, 20);
        assert_eq!(result.out_of_box_samples, 60);
        assert!((result.best_score - 8.714213).abs() < 1e-5);
    }

    #[test]
    fn rejects_invalid_contract_values() {
        let (dataset, mut payload) = fixture();
        payload.phase_bin_count = 1;
        assert_eq!(
            execute(&dataset, &payload),
            Err(BoxPeriodError::InvalidPhaseBinCount)
        );
    }
}
