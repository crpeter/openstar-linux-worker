use crate::protocol::*;
use anyhow::{Context, Result};
use reqwest::{Client, StatusCode};
use std::time::Duration;
use thiserror::Error;
use url::Url;

#[derive(Clone)]
pub struct Coordinator {
    base: Url,
    http: Client,
}

#[derive(Debug, Error)]
pub enum RequestError {
    #[error("coordinator transport error: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("coordinator rejected request with HTTP {0}")]
    Permanent(StatusCode),
    #[error("invalid coordinator response: {0}")]
    InvalidResponse(String),
    #[error("invalid coordinator URL: {0}")]
    Url(#[from] url::ParseError),
}

impl RequestError {
    pub fn transient(&self) -> bool {
        matches!(self, Self::Transport(error) if error.is_timeout() || error.is_connect() || error.is_request()
            || error.status().is_some_and(|status| status.is_server_error()))
    }
}

impl Coordinator {
    pub fn new(base: Url, timeout: Duration) -> Result<Self> {
        Ok(Self {
            base,
            http: Client::builder().timeout(timeout).build()?,
        })
    }
    fn url(&self, path: &str) -> Result<Url> {
        self.base.join(path).context("invalid coordinator URL")
    }

    pub async fn register(&self, body: &Registration<'_>) -> Result<RegistrationReceipt> {
        Ok(self
            .http
            .post(self.url("v1/nodes/register")?)
            .json(body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn claim(&self, node_id: &str) -> Result<Option<WorkUnit>> {
        let response = self
            .http
            .post(self.url("v1/work/claim")?)
            .json(&ClaimRequest { node_id })
            .send()
            .await?;
        if response.status() == StatusCode::NO_CONTENT {
            return Ok(None);
        }
        Ok(Some(response.error_for_status()?.json().await?))
    }

    pub async fn claim_batch(&self, node_id: &str, max_work_units: usize) -> Result<Vec<WorkUnit>> {
        let response = self
            .http
            .post(self.url("v1/work/claim")?)
            .json(&BatchClaimRequest {
                node_id,
                max_work_units,
            })
            .send()
            .await?;
        if response.status() == StatusCode::NO_CONTENT {
            return Ok(Vec::new());
        }
        Ok(response.error_for_status()?.json().await?)
    }

    pub async fn dataset(&self, work: &WorkUnit) -> Result<Dataset, RequestError> {
        let url = self.base.join(&format!(
            "v1/projects/{}/datasets/{}",
            work.project_id, work.dataset_id
        ))?;
        let response = self.http.get(url).send().await?;
        if response.status().is_server_error() {
            return Err(RequestError::Transport(
                response.error_for_status().unwrap_err(),
            ));
        }
        if !response.status().is_success() {
            return Err(RequestError::Permanent(response.status()));
        }
        response
            .json()
            .await
            .map_err(|error| RequestError::InvalidResponse(error.to_string()))
    }

    /// The caller serializes once, ensuring byte-identical bodies across retries.
    pub async fn submit_bytes(
        &self,
        work_id: &str,
        body: &[u8],
    ) -> Result<ResultReceipt, RequestError> {
        let url = self.base.join(&format!("v1/work/{work_id}/result"))?;
        let response = self
            .http
            .post(url)
            .header("content-type", "application/json")
            .body(body.to_vec())
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            if status.is_client_error() {
                return Err(RequestError::Permanent(status));
            }
            return Err(RequestError::Transport(
                response.error_for_status().unwrap_err(),
            ));
        }
        Ok(response.json().await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel;
    use httpmock::{Method::POST, MockServer};

    #[tokio::test]
    async fn claim_shapes_match_the_selected_mode() {
        let server = MockServer::start_async().await;
        let legacy = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/work/claim")
                    .json_body(serde_json::json!({"nodeID":"n"}));
                then.status(204);
            })
            .await;
        let client = Coordinator::new(
            Url::parse(&format!("{}/", server.base_url())).unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        assert!(client.claim("n").await.unwrap().is_none());
        legacy.assert_async().await;
        legacy.delete_async().await;

        let batch = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/work/claim")
                    .json_body(serde_json::json!({"nodeID":"n","maxWorkUnits":2}));
                then.status(200).json_body(serde_json::json!([{
                    "id":"w", "projectID":"p", "workloadID":LOMB_SCARGLE_V1,
                    "datasetID":"d", "payload":null
                }]));
            })
            .await;
        assert_eq!(client.claim_batch("n", 2).await.unwrap().len(), 1);
        batch.assert_async().await;
    }

    #[tokio::test]
    async fn accepted_false_is_still_transport_success_and_conflict_is_permanent() {
        let server = MockServer::start_async().await;
        let receipt = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/work/w/result");
                then.status(200)
                    .json_body(serde_json::json!({"accepted":false,"message":"duplicate"}));
            })
            .await;
        let client = Coordinator::new(
            Url::parse(&format!("{}/", server.base_url())).unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        assert!(!client.submit_bytes("w", b"{}").await.unwrap().accepted);
        receipt.delete_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/work/w/result");
                then.status(409);
            })
            .await;
        assert!(matches!(
            client.submit_bytes("w", b"{}").await,
            Err(RequestError::Permanent(StatusCode::CONFLICT))
        ));
    }

    #[tokio::test]
    async fn complete_coordinator_cycle_uses_verified_contract() {
        let server = MockServer::start_async().await;
        let registration_json = serde_json::json!({"nodeID":"00000000-0000-4000-8000-000000000001",
            "capabilities":{"platform":"linux","hardwareIdentifier":"x86_64 Linux CPU","gpuName":"none",
            "processorCount":2,"memoryGB":16.0,"computeBackends":[{"id":"cpu"}],"workloads":[{
                "workloadID":LOMB_SCARGLE_V1,"executionBackends":[{"id":"cpu"}],"validatorID":null}]}});
        let register = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/nodes/register")
                    .json_body(registration_json.clone());
                then.status(200)
                    .json_body(serde_json::json!({"accepted":true,"message":"Node registered."}));
            })
            .await;
        let claim = server.mock_async(|when, then| { when.method(POST).path("/v1/work/claim")
            .json_body(serde_json::json!({"nodeID":"00000000-0000-4000-8000-000000000001"}));
            then.status(200).json_body(serde_json::json!({"id":"w","projectID":"p","workloadID":LOMB_SCARGLE_V1,
                "datasetID":"d","payload":{"frequencyStartIndex":40,"startFrequency":0.2,"frequencyStep":0.15,"frequencyCount":5}})); }).await;
        let dataset = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET)
                    .path("/v1/projects/p/datasets/d");
                then.status(200).json_body(serde_json::json!({
                "coordinates":[0.0,0.37,1.11,1.8,2.73],"values":[2.0,3.2,1.4,2.8,1.9]}));
            })
            .await;
        let submit = server.mock_async(|when, then| { when.method(POST).path("/v1/work/w/result")
            .json_body(serde_json::json!({"workUnitID":"w","nodeID":"00000000-0000-4000-8000-000000000001",
                "status":"completed","duration":1.25,"payload":{"bestFrequency":0.8,"bestPeriodDays":1.25,
                "bestPower":0.43365732,"bestFrequencyIndex":44,"cpuDurationSeconds":1.25,
                "totalWorkloadDurationSeconds":1.25},"errorMessage":null,"failureKind":null,
                "bestFrequency":0.8,"bestPeriodDays":1.25,"bestPower":0.43365732}));
            then.status(200).json_body(serde_json::json!({"accepted":true,"message":"Result accepted."})); }).await;
        let client = Coordinator::new(
            Url::parse(&format!("{}/", server.base_url())).unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        let registration = Registration {
            node_id: "00000000-0000-4000-8000-000000000001",
            capabilities: Capabilities {
                platform: "linux",
                hardware_identifier: "x86_64 Linux CPU".into(),
                gpu_name: "none".to_owned(),
                processor_count: 2,
                memory_gb: 16.0,
                compute_backends: vec![Backend { id: "cpu" }],
                workloads: vec![WorkloadCapability {
                    workload_id: LOMB_SCARGLE_V1,
                    execution_backends: vec![Backend { id: "cpu" }],
                    validator_id: None,
                }],
            },
        };
        assert!(client.register(&registration).await.unwrap().accepted);
        let work = client.claim(registration.node_id).await.unwrap().unwrap();
        let result = kernel::execute(
            &client.dataset(&work).await.unwrap(),
            &work.lomb_payload().unwrap(),
        )
        .unwrap();
        let body = serde_json::to_vec(&WorkResult::completed(
            &work.id,
            registration.node_id,
            result,
            ExecutionDuration {
                backend: "cpu",
                seconds: 1.25,
            },
            1.25,
        ))
        .unwrap();
        assert!(client.submit_bytes(&work.id, &body).await.unwrap().accepted);
        register.assert_async().await;
        claim.assert_async().await;
        dataset.assert_async().await;
        submit.assert_async().await;

        claim.delete_async().await;
        let no_work = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/work/claim");
                then.status(204);
            })
            .await;
        assert!(client.claim(registration.node_id).await.unwrap().is_none());
        no_work.assert_async().await;
    }
}
