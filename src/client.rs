use crate::protocol::*;
use anyhow::{Context, Result};
use reqwest::{Client, StatusCode};
use std::time::Duration;
use url::Url;

#[derive(Clone)]
pub struct Coordinator {
    base: Url,
    http: Client,
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
    pub async fn register(&self, body: &Registration<'_>) -> Result<String> {
        Ok(self
            .http
            .post(self.url("v1/nodes/register")?)
            .json(body)
            .send()
            .await?
            .error_for_status()?
            .json::<RegistrationResponse>()
            .await?
            .node_id)
    }
    pub async fn claim(&self, node: &str) -> Result<Option<WorkUnit>> {
        let response = self
            .http
            .post(self.url("v1/work/claim")?)
            .json(&ClaimRequest { node_id: node })
            .send()
            .await?;
        if response.status() == StatusCode::NO_CONTENT {
            return Ok(None);
        }
        Ok(response
            .error_for_status()?
            .json::<ClaimResponse>()
            .await?
            .work())
    }
    pub async fn dataset(&self, work: &WorkUnit) -> Result<Dataset> {
        let path = format!(
            "v1/projects/{}/datasets/{}",
            work.project_id, work.dataset_id
        );
        Ok(self
            .http
            .get(self.url(&path)?)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }
    /// Serializes once so every retry carries byte-identical JSON.
    pub async fn submit_bytes(&self, work_id: &str, body: &[u8]) -> Result<()> {
        let path = format!("v1/work/{work_id}/result");
        self.http
            .post(self.url(&path)?)
            .header("content-type", "application/json")
            .body(body.to_vec())
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}
