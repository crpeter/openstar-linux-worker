use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const LOMB_SCARGLE_V1: &str = "openstar.lomb-scargle.v1";
pub const BOX_PERIOD_SEARCH_V1: &str = "openstar.box-period-search.v1";

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Registration<'a> {
    #[serde(rename = "nodeID")]
    pub node_id: &'a str,
    pub capabilities: Capabilities,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Capabilities {
    pub platform: &'static str,
    pub hardware_identifier: String,
    pub gpu_name: String,
    pub processor_count: usize,
    #[serde(rename = "memoryGB")]
    pub memory_gb: f32,
    pub compute_backends: Vec<Backend>,
    pub workloads: Vec<WorkloadCapability>,
}

#[derive(Debug, Serialize)]
pub struct Backend {
    pub id: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkloadCapability {
    #[serde(rename = "workloadID")]
    pub workload_id: &'static str,
    pub execution_backends: Vec<Backend>,
    #[serde(rename = "validatorID")]
    pub validator_id: Option<&'static str>,
}

#[derive(Debug, Deserialize)]
pub struct RegistrationReceipt {
    pub accepted: bool,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ClaimRequest<'a> {
    #[serde(rename = "nodeID")]
    pub node_id: &'a str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchClaimRequest<'a> {
    #[serde(rename = "nodeID")]
    pub node_id: &'a str,
    pub max_work_units: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkUnit {
    pub id: String,
    #[serde(rename = "projectID")]
    pub project_id: String,
    #[serde(rename = "workloadID")]
    pub workload_id: String,
    #[serde(rename = "datasetID")]
    pub dataset_id: String,
    #[serde(default)]
    pub payload: Option<Value>,
    #[serde(default)]
    pub start_frequency: Option<f32>,
    #[serde(default)]
    pub frequency_step: Option<f32>,
    #[serde(default)]
    pub frequency_count: Option<usize>,
    #[serde(default)]
    pub frequency_start_index: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct Dataset {
    #[serde(default)]
    pub coordinates: Option<Vec<f32>>,
    #[serde(default)]
    pub values: Option<Vec<f32>>,
    #[serde(default)]
    pub times: Option<Vec<f32>>,
    #[serde(default)]
    pub flux: Option<Vec<f32>>,
}

impl Dataset {
    pub fn series(&self) -> Option<(&[f32], &[f32])> {
        match (&self.coordinates, &self.values) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => self.times.as_deref().zip(self.flux.as_deref()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LombPayload {
    pub start_frequency: f32,
    pub frequency_step: f32,
    pub frequency_count: usize,
    #[serde(default)]
    pub frequency_start_index: usize,
}

impl WorkUnit {
    pub fn lomb_payload(&self) -> Result<LombPayload, String> {
        if let Some(value) = &self.payload {
            serde_json::from_value(value.clone()).map_err(|e| format!("invalid payload: {e}"))
        } else {
            Ok(LombPayload {
                start_frequency: self.start_frequency.ok_or("missing startFrequency")?,
                frequency_step: self.frequency_step.ok_or("missing frequencyStep")?,
                frequency_count: self.frequency_count.ok_or("missing frequencyCount")?,
                frequency_start_index: self.frequency_start_index.unwrap_or(0),
            })
        }
    }

    pub fn box_period_payload(&self) -> Result<BoxPeriodPayload, String> {
        let value = self.payload.clone().ok_or("missing payload")?;
        serde_json::from_value(value).map_err(|e| format!("invalid payload: {e}"))
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoxPeriodPayload {
    pub start_frequency: f32,
    pub frequency_step: f32,
    pub frequency_count: usize,
    #[serde(default)]
    pub frequency_start_index: usize,
    pub phase_bin_count: usize,
    pub duration_fractions: Vec<f32>,
    pub minimum_in_box_samples: usize,
    pub minimum_out_of_box_samples: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoxPeriodResult {
    pub best_frequency: f32,
    pub best_score: f32,
    pub best_phase: f32,
    pub best_duration_fraction: f32,
    pub best_frequency_index: usize,
    pub best_duration_index: usize,
    pub best_phase_bin: usize,
    pub in_box_samples: usize,
    pub out_of_box_samples: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LombResult {
    pub best_frequency: f32,
    pub best_period_days: f32,
    pub best_power: f32,
    pub best_frequency_index: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkResult {
    #[serde(rename = "workUnitID")]
    pub work_unit_id: String,
    #[serde(rename = "nodeID")]
    pub node_id: String,
    pub status: ResultStatus,
    pub duration: f64,
    pub payload: Option<WorkResultPayload>,
    pub error_message: Option<String>,
    pub failure_kind: Option<FailureKind>,
    pub best_frequency: Option<f32>,
    pub best_period_days: Option<f32>,
    pub best_power: Option<f32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ResultStatus {
    Completed,
    Failed,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResultPayload {
    pub best_frequency: f32,
    pub best_period_days: f32,
    pub best_power: f32,
    pub best_frequency_index: usize,
    // This legacy wire field is emitted only for CPU execution. The existing
    // top-level and total-workload durations remain backend-neutral.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_duration_seconds: Option<f64>,
    pub total_workload_duration_seconds: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BoxPeriodResultPayload {
    pub best_frequency: f32,
    pub best_score: f32,
    pub best_phase: f32,
    pub best_duration_fraction: f32,
    pub best_frequency_index: usize,
    pub best_duration_index: usize,
    pub best_phase_bin: usize,
    pub in_box_samples: usize,
    pub out_of_box_samples: usize,
    pub cpu_duration_seconds: f64,
    pub total_workload_duration_seconds: f64,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum WorkResultPayload {
    Lomb(ResultPayload),
    BoxPeriod(BoxPeriodResultPayload),
}

#[derive(Clone, Copy, Debug)]
pub struct ExecutionDuration {
    pub backend: &'static str,
    pub seconds: f64,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum FailureKind {
    Execution,
    WorkloadValidation,
    InvalidInput,
    EnvironmentUnavailable,
    TransportUnavailable,
    UnsupportedWorkload,
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct ResultReceipt {
    pub accepted: bool,
    #[serde(default)]
    pub message: Option<String>,
}

impl WorkResult {
    pub fn completed(
        work_id: &str,
        node_id: &str,
        result: LombResult,
        execution_duration: ExecutionDuration,
        total_duration: f64,
    ) -> Self {
        let payload = ResultPayload {
            best_frequency: result.best_frequency,
            best_period_days: result.best_period_days,
            best_power: result.best_power,
            best_frequency_index: result.best_frequency_index,
            cpu_duration_seconds: (execution_duration.backend == "cpu")
                .then_some(execution_duration.seconds),
            total_workload_duration_seconds: total_duration,
        };
        Self {
            work_unit_id: work_id.into(),
            node_id: node_id.into(),
            status: ResultStatus::Completed,
            duration: total_duration,
            payload: Some(WorkResultPayload::Lomb(payload)),
            error_message: None,
            failure_kind: None,
            best_frequency: Some(result.best_frequency),
            best_period_days: Some(result.best_period_days),
            best_power: Some(result.best_power),
        }
    }

    pub fn completed_box_period(
        work_id: &str,
        node_id: &str,
        result: BoxPeriodResult,
        execution_duration: ExecutionDuration,
        total_duration: f64,
    ) -> Self {
        debug_assert_eq!(execution_duration.backend, "cpu");
        let payload = BoxPeriodResultPayload {
            best_frequency: result.best_frequency,
            best_score: result.best_score,
            best_phase: result.best_phase,
            best_duration_fraction: result.best_duration_fraction,
            best_frequency_index: result.best_frequency_index,
            best_duration_index: result.best_duration_index,
            best_phase_bin: result.best_phase_bin,
            in_box_samples: result.in_box_samples,
            out_of_box_samples: result.out_of_box_samples,
            cpu_duration_seconds: execution_duration.seconds,
            total_workload_duration_seconds: total_duration,
        };
        Self {
            work_unit_id: work_id.into(),
            node_id: node_id.into(),
            status: ResultStatus::Completed,
            duration: total_duration,
            payload: Some(WorkResultPayload::BoxPeriod(payload)),
            error_message: None,
            failure_kind: None,
            best_frequency: None,
            best_period_days: None,
            best_power: None,
        }
    }

    pub fn failed(
        work_id: &str,
        node_id: &str,
        duration: f64,
        kind: FailureKind,
        message: String,
    ) -> Self {
        Self {
            work_unit_id: work_id.into(),
            node_id: node_id.into(),
            status: ResultStatus::Failed,
            duration,
            payload: None,
            error_message: Some(message),
            failure_kind: Some(kind),
            best_frequency: None,
            best_period_days: None,
            best_power: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_kinds_are_exactly_hyphenated() {
        let kinds = [
            FailureKind::Execution,
            FailureKind::WorkloadValidation,
            FailureKind::InvalidInput,
            FailureKind::EnvironmentUnavailable,
            FailureKind::TransportUnavailable,
            FailureKind::UnsupportedWorkload,
            FailureKind::Unknown,
        ];
        assert_eq!(
            serde_json::to_value(kinds).unwrap(),
            serde_json::json!([
                "execution",
                "workload-validation",
                "invalid-input",
                "environment-unavailable",
                "transport-unavailable",
                "unsupported-workload",
                "unknown"
            ])
        );
    }

    #[test]
    fn generic_dataset_wins_over_legacy() {
        let d: Dataset = serde_json::from_value(serde_json::json!({
            "coordinates":[1.0], "values":[2.0], "times":[3.0], "flux":[4.0]
        }))
        .unwrap();
        assert_eq!(d.series(), Some((&[1.0][..], &[2.0][..])));
    }

    #[test]
    fn nested_and_flattened_payloads_use_start_frequency() {
        let mut w: WorkUnit = serde_json::from_value(serde_json::json!({"id":"w","projectID":"p",
            "workloadID":LOMB_SCARGLE_V1,"datasetID":"d","startFrequency":0.1,
            "frequencyStep":0.01,"frequencyCount":3,"frequencyStartIndex":7}))
        .unwrap();
        assert_eq!(w.lomb_payload().unwrap().start_frequency, 0.1);
        w.payload = Some(
            serde_json::json!({"startFrequency":0.2,"frequencyStep":0.02,
            "frequencyCount":4,"frequencyStartIndex":8}),
        );
        assert_eq!(w.lomb_payload().unwrap().start_frequency, 0.2);
    }

    #[test]
    fn completed_result_has_real_envelope_and_flattened_fields() {
        let value = serde_json::to_value(WorkResult::completed(
            "work",
            "node",
            LombResult {
                best_frequency: 2.0,
                best_period_days: 0.5,
                best_power: 0.875,
                best_frequency_index: 42,
            },
            ExecutionDuration {
                backend: "cpu",
                seconds: 1.25,
            },
            1.25,
        ))
        .unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "workUnitID":"work", "nodeID":"node", "status":"completed", "duration":1.25,
                "payload":{"bestFrequency":2.0,"bestPeriodDays":0.5,"bestPower":0.875,
                    "bestFrequencyIndex":42,"cpuDurationSeconds":1.25,"totalWorkloadDurationSeconds":1.25},
                "errorMessage":null,"failureKind":null,"bestFrequency":2.0,
                "bestPeriodDays":0.5,"bestPower":0.875
            })
        );
    }

    #[test]
    fn vulkan_duration_is_not_labeled_as_cpu_time() {
        let value = serde_json::to_value(WorkResult::completed(
            "work",
            "node",
            LombResult {
                best_frequency: 2.0,
                best_period_days: 0.5,
                best_power: 0.875,
                best_frequency_index: 42,
            },
            ExecutionDuration {
                backend: "vulkan",
                seconds: 0.25,
            },
            0.5,
        ))
        .unwrap();
        let payload = value["payload"].as_object().unwrap();
        assert!(!payload.contains_key("cpuDurationSeconds"));
        assert_eq!(payload["totalWorkloadDurationSeconds"], 0.5);
        assert_eq!(value["duration"], 0.5);
    }

    #[test]
    fn completed_box_result_uses_only_generic_payload_fields() {
        let value = serde_json::to_value(WorkResult::completed_box_period(
            "work",
            "node",
            BoxPeriodResult {
                best_frequency: 0.5,
                best_score: 8.714213,
                best_phase: 0.0,
                best_duration_fraction: 0.15,
                best_frequency_index: 22,
                best_duration_index: 1,
                best_phase_bin: 0,
                in_box_samples: 20,
                out_of_box_samples: 60,
            },
            ExecutionDuration {
                backend: "cpu",
                seconds: 0.25,
            },
            0.5,
        ))
        .unwrap();
        assert_eq!(value["payload"]["bestFrequencyIndex"], 22);
        assert_eq!(
            value["payload"]["bestScore"],
            serde_json::json!(8.714213_f32)
        );
        assert_eq!(value["payload"]["cpuDurationSeconds"], 0.25);
        assert!(value["bestFrequency"].is_null());
        assert!(value["bestPower"].is_null());
    }
}
