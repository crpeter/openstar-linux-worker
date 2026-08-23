use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const LOMB_SCARGLE_V1: &str = "openstar.lomb-scargle.v1";

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Registration<'a> {
    pub name: &'a str,
    pub capabilities: Capabilities,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Capabilities {
    pub workloads: Vec<&'static str>,
    pub cpu_threads: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegistrationResponse {
    #[serde(alias = "id")]
    pub node_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaimRequest<'a> {
    pub node_id: &'a str,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkUnit {
    pub id: String,
    pub project_id: String,
    pub workload_id: String,
    pub dataset_id: String,
    #[serde(default)]
    pub payload: Option<Value>,
    #[serde(default)]
    pub frequency_start: Option<f32>,
    #[serde(default)]
    pub frequency_step: Option<f32>,
    #[serde(default)]
    pub frequency_count: Option<usize>,
    #[serde(default)]
    pub frequency_start_index: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ClaimResponse {
    Empty(Option<WorkUnit>),
    Wrapped {
        #[serde(rename = "workUnit")]
        #[serde(default)]
        work_unit: Option<WorkUnit>,
    },
    Direct(WorkUnit),
}
impl ClaimResponse {
    pub fn work(self) -> Option<WorkUnit> {
        match self {
            Self::Empty(v) => v,
            Self::Wrapped { work_unit } => work_unit,
            Self::Direct(v) => Some(v),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Dataset {
    #[serde(alias = "timestamps")]
    pub times: Vec<f32>,
    #[serde(default)]
    pub flux: Option<Vec<f32>>,
    #[serde(default)]
    pub values: Option<Vec<f32>>,
}
impl Dataset {
    pub fn values(&self) -> Option<&[f32]> {
        self.flux.as_deref().or(self.values.as_deref())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LombPayload {
    pub frequency_start: f32,
    pub frequency_step: f32,
    pub frequency_count: usize,
    #[serde(default)]
    pub frequency_start_index: usize,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LombResult {
    pub powers: Vec<f32>,
    pub best_frequency_index: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResultEnvelope<T> {
    pub workload_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<Failure>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Failure {
    pub code: &'static str,
    pub message: String,
}

impl WorkUnit {
    pub fn lomb_payload(&self) -> Result<LombPayload, String> {
        if let Some(value) = &self.payload {
            serde_json::from_value(value.clone()).map_err(|e| format!("invalid payload: {e}"))
        } else {
            Ok(LombPayload {
                frequency_start: self.frequency_start.ok_or("missing frequencyStart")?,
                frequency_step: self.frequency_step.ok_or("missing frequencyStep")?,
                frequency_count: self.frequency_count.ok_or("missing frequencyCount")?,
                frequency_start_index: self.frequency_start_index.unwrap_or(0),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work(payload: Option<Value>) -> WorkUnit {
        WorkUnit {
            id: "w1".into(),
            project_id: "p1".into(),
            workload_id: LOMB_SCARGLE_V1.into(),
            dataset_id: "d1".into(),
            payload,
            frequency_start: Some(1.0),
            frequency_step: Some(0.25),
            frequency_count: Some(8),
            frequency_start_index: Some(40),
        }
    }

    #[test]
    fn modern_and_legacy_payloads_decode() {
        let modern = work(Some(serde_json::json!({"frequencyStart":2.0,"frequencyStep":0.5,"frequencyCount":4,"frequencyStartIndex":9}))).lomb_payload().unwrap();
        assert_eq!(
            (
                modern.frequency_start,
                modern.frequency_count,
                modern.frequency_start_index
            ),
            (2.0, 4, 9)
        );
        let legacy = work(None).lomb_payload().unwrap();
        assert_eq!(
            (
                legacy.frequency_step,
                legacy.frequency_count,
                legacy.frequency_start_index
            ),
            (0.25, 8, 40)
        );
    }

    #[test]
    fn dataset_accepts_flux_values_and_timestamps() {
        let flux: Dataset =
            serde_json::from_value(serde_json::json!({"times":[0.0],"flux":[1.0]})).unwrap();
        let values: Dataset =
            serde_json::from_value(serde_json::json!({"timestamps":[0.0],"values":[2.0]})).unwrap();
        assert_eq!(flux.values(), Some(&[1.0][..]));
        assert_eq!(values.values(), Some(&[2.0][..]));
    }

    #[test]
    fn registration_claim_and_result_shapes() {
        let registration: RegistrationResponse =
            serde_json::from_str(r#"{"nodeId":"n1"}"#).unwrap();
        assert_eq!(registration.node_id, "n1");
        let claim: ClaimResponse = serde_json::from_str("{}").unwrap();
        assert!(claim.work().is_none());
        let body = serde_json::to_vec(&ResultEnvelope {
            workload_id: LOMB_SCARGLE_V1.into(),
            result: Some(LombResult {
                powers: vec![0.5],
                best_frequency_index: 7,
            }),
            failure: None,
        })
        .unwrap();
        assert_eq!(
            String::from_utf8(body).unwrap(),
            r#"{"workloadId":"openstar.lomb-scargle.v1","result":{"powers":[0.5],"bestFrequencyIndex":7}}"#
        );
    }
}
