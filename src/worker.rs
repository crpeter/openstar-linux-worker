use crate::{
    backend::{self, ComputeBackend},
    client::{Coordinator, RequestError},
    config::Config,
    protocol::*,
};
use anyhow::Result;
use rand::Rng;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{watch, Semaphore};
use tracing::{info, warn};

pub struct Worker {
    config: Config,
    client: Coordinator,
    backend: Arc<dyn ComputeBackend>,
    node_id: String,
}

impl Worker {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let node_id = config.node_identity()?.to_string();
        let threads = config.cpu_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        });
        let backend = backend::initialize(config.compute_backend, threads)?;
        let client = Coordinator::new(config.coordinator_url.clone(), config.timeout())?;
        Ok(Self {
            config,
            client,
            backend,
            node_id,
        })
    }

    pub async fn run(self, mut stop: watch::Receiver<bool>) -> Result<()> {
        let processor_count = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let receipt = self
            .client
            .register(&Registration {
                node_id: &self.node_id,
                capabilities: Capabilities {
                    platform: "linux",
                    hardware_identifier: format!("{} Linux CPU", std::env::consts::ARCH),
                    gpu_name: self.backend.gpu_name().unwrap_or("none").to_owned(),
                    processor_count,
                    memory_gb: memory_gb(),
                    compute_backends: vec![Backend {
                        id: self.backend.id(),
                    }],
                    workloads: vec![WorkloadCapability {
                        workload_id: LOMB_SCARGLE_V1,
                        execution_backends: vec![Backend {
                            id: self.backend.id(),
                        }],
                        validator_id: None,
                    }],
                },
            })
            .await?;
        anyhow::ensure!(
            receipt.accepted,
            "registration rejected: {}",
            receipt.message
        );
        info!(node_id=%self.node_id, message=%receipt.message, "registered");

        let permits = Arc::new(Semaphore::new(self.config.work_concurrency));
        let mut backoff = self.config.poll_interval_ms;
        loop {
            if *stop.borrow() {
                break;
            }
            let permit = tokio::select! {
                permit = permits.clone().acquire_owned() => permit?,
                changed = stop.changed() => { changed?; break; }
            };
            let claim = if self.config.work_batch_size == 1 {
                self.client
                    .claim(&self.node_id)
                    .await
                    .map(|work| work.into_iter().collect())
            } else {
                self.client
                    .claim_batch(&self.node_id, self.config.work_batch_size)
                    .await
            };
            match claim {
                Ok(batch) if !batch.is_empty() => {
                    backoff = self.config.poll_interval_ms;
                    let (client, backend, node_id) = (
                        self.client.clone(),
                        self.backend.clone(),
                        self.node_id.clone(),
                    );
                    let max = self.config.max_backoff_ms;
                    drop(tokio::spawn(async move {
                        let _permit = permit;
                        process_batch(client, backend, node_id, batch, max).await;
                    }));
                }
                Ok(_) => {
                    drop(permit);
                    sleep_or_stop(
                        Duration::from_millis(self.config.poll_interval_ms),
                        &mut stop,
                    )
                    .await;
                }
                Err(error) => {
                    drop(permit);
                    warn!(%error, "claim failed");
                    let jitter = rand::thread_rng().gen_range(0..=(backoff / 4));
                    sleep_or_stop(Duration::from_millis(backoff + jitter), &mut stop).await;
                    backoff = backoff.saturating_mul(2).min(self.config.max_backoff_ms);
                }
            }
        }
        let count = u32::try_from(self.config.work_concurrency)?;
        let _drain = permits.acquire_many(count).await?;
        Ok(())
    }
}

fn memory_gb() -> f32 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("MemTotal:"))
                .and_then(|line| line.split_whitespace().nth(1))?
                .parse::<f32>()
                .ok()
        })
        .map(|kib| kib / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

async fn sleep_or_stop(duration: Duration, stop: &mut watch::Receiver<bool>) {
    tokio::select! { _ = tokio::time::sleep(duration) => {}, _ = stop.changed() => {} }
}

async fn process(
    client: Coordinator,
    backend: Arc<dyn ComputeBackend>,
    node_id: String,
    work: WorkUnit,
    max_backoff: u64,
) {
    process_with_dataset(
        client,
        backend,
        node_id,
        work,
        max_backoff,
        None,
        Instant::now(),
    )
    .await;
}

async fn process_batch(
    client: Coordinator,
    backend: Arc<dyn ComputeBackend>,
    node_id: String,
    batch: Vec<WorkUnit>,
    max_backoff: u64,
) {
    let homogeneous = batch.first().is_some_and(|first| {
        batch.iter().all(|work| {
            work.project_id == first.project_id
                && work.dataset_id == first.dataset_id
                && work.workload_id == first.workload_id
        })
    });
    if !homogeneous || batch[0].workload_id != LOMB_SCARGLE_V1 {
        warn!("non-homogeneous or unsupported batch; processing work units individually");
        for work in batch {
            process(
                client.clone(),
                backend.clone(),
                node_id.clone(),
                work,
                max_backoff,
            )
            .await;
        }
        return;
    }

    let shared_fetch_started = Instant::now();
    let dataset = match fetch_with_retry(&client, &batch[0], max_backoff).await {
        Ok(dataset) => Arc::new(dataset),
        Err(error) => {
            warn!(%error, "shared dataset fetch failed; processing work units individually");
            for work in batch {
                process(
                    client.clone(),
                    backend.clone(),
                    node_id.clone(),
                    work,
                    max_backoff,
                )
                .await;
            }
            return;
        }
    };
    for (index, work) in batch.into_iter().enumerate() {
        // Charge the one shared transfer to exactly one result so summed worker
        // durations retain the same aggregate throughput meaning as legacy mode.
        let started = if index == 0 {
            shared_fetch_started
        } else {
            Instant::now()
        };
        process_with_dataset(
            client.clone(),
            backend.clone(),
            node_id.clone(),
            work,
            max_backoff,
            Some(dataset.clone()),
            started,
        )
        .await;
    }
}

async fn process_with_dataset(
    client: Coordinator,
    backend: Arc<dyn ComputeBackend>,
    node_id: String,
    work: WorkUnit,
    max_backoff: u64,
    shared_dataset: Option<Arc<Dataset>>,
    started: Instant,
) {
    let result = if work.workload_id != LOMB_SCARGLE_V1 {
        WorkResult::failed(
            &work.id,
            &node_id,
            started.elapsed().as_secs_f64(),
            FailureKind::UnsupportedWorkload,
            format!("unsupported workload: {}", work.workload_id),
        )
    } else {
        let dataset = match shared_dataset {
            Some(dataset) => Ok(dataset),
            None => fetch_with_retry(&client, &work, max_backoff)
                .await
                .map(Arc::new),
        };
        match dataset {
            Ok(dataset) => match work.lomb_payload() {
                Err(message) => WorkResult::failed(
                    &work.id,
                    &node_id,
                    started.elapsed().as_secs_f64(),
                    FailureKind::InvalidInput,
                    message,
                ),
                Ok(payload) => {
                    let backend_id = backend.id();
                    let computation = tokio::task::spawn_blocking(move || {
                        let execution_started = Instant::now();
                        let output = backend.execute(&dataset, &payload);
                        (output, execution_started.elapsed().as_secs_f64())
                    })
                    .await;
                    match computation {
                        Ok((Ok(output), execution_duration)) => WorkResult::completed(
                            &work.id,
                            &node_id,
                            output,
                            ExecutionDuration {
                                backend: backend_id,
                                seconds: execution_duration,
                            },
                            started.elapsed().as_secs_f64(),
                        ),
                        Ok((Err(error), _)) => {
                            let kind = backend_failure_kind(&error);
                            WorkResult::failed(
                                &work.id,
                                &node_id,
                                started.elapsed().as_secs_f64(),
                                kind,
                                error.to_string(),
                            )
                        }
                        Err(error) => WorkResult::failed(
                            &work.id,
                            &node_id,
                            started.elapsed().as_secs_f64(),
                            FailureKind::Execution,
                            error.to_string(),
                        ),
                    }
                }
            },
            Err(error) => {
                let kind = if matches!(
                    &error,
                    RequestError::InvalidResponse(_) | RequestError::Permanent(_)
                ) {
                    FailureKind::InvalidInput
                } else {
                    FailureKind::TransportUnavailable
                };
                WorkResult::failed(
                    &work.id,
                    &node_id,
                    started.elapsed().as_secs_f64(),
                    kind,
                    error.to_string(),
                )
            }
        }
    };
    let body = serde_json::to_vec(&result).expect("work result is serializable");
    submit_with_retry(&client, &work.id, &body, max_backoff).await;
}

fn backend_failure_kind(error: &crate::backend::BackendError) -> FailureKind {
    match error {
        crate::backend::BackendError::InvalidInput(_) => FailureKind::InvalidInput,
        crate::backend::BackendError::Execution(_) => FailureKind::Execution,
    }
}

async fn fetch_with_retry(
    client: &Coordinator,
    work: &WorkUnit,
    max: u64,
) -> Result<Dataset, RequestError> {
    let mut delay = 250;
    for attempt in 0..5 {
        match client.dataset(work).await {
            Ok(dataset) => return Ok(dataset),
            Err(error) if error.transient() && attempt < 4 => {
                tokio::time::sleep(Duration::from_millis(delay)).await;
                delay = (delay * 2).min(max);
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!()
}

async fn submit_with_retry(client: &Coordinator, work_id: &str, body: &[u8], max: u64) {
    let mut delay = 500;
    loop {
        match client.submit_bytes(work_id, body).await {
            Ok(receipt) => {
                info!(accepted=receipt.accepted, message=?receipt.message, "result receipt");
                break;
            }
            Err(error) if error.transient() => {
                warn!(%error, "result submission failed; retrying");
                tokio::time::sleep(Duration::from_millis(delay)).await;
                delay = (delay * 2).min(max.max(500));
            }
            Err(error) => {
                warn!(%error, "permanent result submission failure");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::{Method::POST, MockServer};
    use url::Url;
    use uuid::Uuid;

    #[test]
    fn backend_runtime_failure_is_not_invalid_input() {
        let error = crate::backend::BackendError::Execution(anyhow::anyhow!("device lost"));
        assert_eq!(backend_failure_kind(&error), FailureKind::Execution);
        let error =
            crate::backend::BackendError::InvalidInput(crate::kernel::ComputeError::InvalidCount);
        assert_eq!(backend_failure_kind(&error), FailureKind::InvalidInput);
    }

    #[tokio::test]
    async fn transient_submission_reuses_identical_bytes() {
        let server = MockServer::start_async().await;
        let body = String::from(r#"{"stable":"body"}"#);
        let bytes = body.as_bytes().to_vec();
        let first = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/work/w/result")
                    .header("content-type", "application/json")
                    .body(body.clone());
                then.status(503);
            })
            .await;
        let client = Coordinator::new(
            Url::parse(&format!("{}/", server.base_url())).unwrap(),
            Duration::from_secs(2),
        )
        .unwrap();
        let task = tokio::spawn({
            let client = client.clone();
            let bytes = bytes.clone();
            async move {
                submit_with_retry(&client, "w", &bytes, 500).await;
            }
        });
        while first.hits_async().await == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        first.delete_async().await;
        let second = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/work/w/result")
                    .header("content-type", "application/json")
                    .body(body.clone());
                then.status(200)
                    .json_body(serde_json::json!({"accepted":true}));
            })
            .await;
        task.await.unwrap();
        second.assert_async().await;
    }

    #[tokio::test]
    async fn permanent_conflict_is_not_retried() {
        let server = MockServer::start_async().await;
        let conflict = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/work/w/result");
                then.status(409);
            })
            .await;
        let client = Coordinator::new(
            Url::parse(&format!("{}/", server.base_url())).unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        tokio::time::timeout(
            Duration::from_secs(1),
            submit_with_retry(&client, "w", b"{}", 500),
        )
        .await
        .unwrap();
        assert_eq!(conflict.hits_async().await, 1);
    }

    #[tokio::test]
    async fn unsupported_workload_reports_failure_without_dataset_fetch() {
        let server = MockServer::start_async().await;
        let submission = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/work/w/result")
                    .body_contains(r#""failureKind":"unsupported-workload""#)
                    .body_contains(r#""errorMessage":"unsupported workload: other""#);
                then.status(200)
                    .json_body(serde_json::json!({"accepted":true}));
            })
            .await;
        let client = Coordinator::new(
            Url::parse(&format!("{}/", server.base_url())).unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        let backend = backend::initialize(crate::backend::BackendChoice::Cpu, 1).unwrap();
        let work = WorkUnit {
            id: "w".into(),
            project_id: "p".into(),
            workload_id: "other".into(),
            dataset_id: "d".into(),
            payload: None,
            start_frequency: None,
            frequency_step: None,
            frequency_count: None,
            frequency_start_index: None,
        };
        process(client, backend, "n".into(), work, 1).await;
        submission.assert_async().await;
    }

    fn test_work(id: &str, project: &str, dataset: &str) -> WorkUnit {
        WorkUnit {
            id: id.into(),
            project_id: project.into(),
            workload_id: LOMB_SCARGLE_V1.into(),
            dataset_id: dataset.into(),
            payload: Some(serde_json::json!({
                "startFrequency": 1.0, "frequencyStep": 1.0, "frequencyCount": 1
            })),
            start_frequency: None,
            frequency_step: None,
            frequency_count: None,
            frequency_start_index: None,
        }
    }

    #[tokio::test]
    async fn homogeneous_batch_fetches_once_and_submits_each_unit() {
        let server = MockServer::start_async().await;
        let dataset = server.mock_async(|when, then| { when.method(httpmock::Method::GET).path("/v1/projects/p/datasets/d");
            then.status(200).json_body(serde_json::json!({"coordinates":[0.0,0.25,0.5,0.75],"values":[2.0,3.0,2.0,1.0]})); }).await;
        let first = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/work/w1/result")
                    .body_contains("\"workUnitID\":\"w1\"");
                then.status(200)
                    .json_body(serde_json::json!({"accepted":true}));
            })
            .await;
        let second = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/work/w2/result")
                    .body_contains("\"workUnitID\":\"w2\"");
                then.status(200)
                    .json_body(serde_json::json!({"accepted":true}));
            })
            .await;
        let client = Coordinator::new(
            Url::parse(&format!("{}/", server.base_url())).unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        let backend = backend::initialize(crate::backend::BackendChoice::Cpu, 1).unwrap();
        process_batch(
            client,
            backend,
            "n".into(),
            vec![test_work("w1", "p", "d"), test_work("w2", "p", "d")],
            1,
        )
        .await;
        assert_eq!(dataset.hits_async().await, 1);
        first.assert_async().await;
        second.assert_async().await;
    }

    #[tokio::test]
    async fn non_homogeneous_batch_uses_individual_dataset_processing() {
        let server = MockServer::start_async().await;
        let d1 = server.mock_async(|when, then| { when.method(httpmock::Method::GET).path("/v1/projects/p/datasets/d1");
            then.status(200).json_body(serde_json::json!({"coordinates":[0.0,0.25,0.5,0.75],"values":[2.0,3.0,2.0,1.0]})); }).await;
        let d2 = server.mock_async(|when, then| { when.method(httpmock::Method::GET).path("/v1/projects/p/datasets/d2");
            then.status(200).json_body(serde_json::json!({"coordinates":[0.0,0.25,0.5,0.75],"values":[2.0,3.0,2.0,1.0]})); }).await;
        for id in ["w1", "w2"] {
            server
                .mock_async(move |when, then| {
                    when.method(POST).path(format!("/v1/work/{id}/result"));
                    then.status(200)
                        .json_body(serde_json::json!({"accepted":true}));
                })
                .await;
        }
        let client = Coordinator::new(
            Url::parse(&format!("{}/", server.base_url())).unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        let backend = backend::initialize(crate::backend::BackendChoice::Cpu, 1).unwrap();
        process_batch(
            client,
            backend,
            "n".into(),
            vec![test_work("w1", "p", "d1"), test_work("w2", "p", "d2")],
            1,
        )
        .await;
        d1.assert_async().await;
        d2.assert_async().await;
    }

    #[tokio::test]
    async fn shutdown_drains_an_entire_active_batch() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/nodes/register");
                then.status(200)
                    .json_body(serde_json::json!({"accepted":true,"message":"ok"}));
            })
            .await;
        server.mock_async(|when, then| { when.method(POST).path("/v1/work/claim")
            .json_body(serde_json::json!({"nodeID":"00000000-0000-4000-8000-000000000001","maxWorkUnits":2}));
            then.status(200).json_body(serde_json::json!([
                {"id":"w1","projectID":"p","workloadID":LOMB_SCARGLE_V1,"datasetID":"d",
                    "payload":{"startFrequency":1.0,"frequencyStep":1.0,"frequencyCount":1}},
                {"id":"w2","projectID":"p","workloadID":LOMB_SCARGLE_V1,"datasetID":"d",
                    "payload":{"startFrequency":1.0,"frequencyStep":1.0,"frequencyCount":1}}
            ])); }).await;
        let dataset = server.mock_async(|when, then| { when.method(httpmock::Method::GET).path("/v1/projects/p/datasets/d");
            then.status(200).json_body(serde_json::json!({"coordinates":[0.0,0.25,0.5,0.75],"values":[2.0,3.0,2.0,1.0]})); }).await;
        let first_result = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/work/w1/result");
                then.delay(Duration::from_millis(100))
                    .status(200)
                    .json_body(serde_json::json!({"accepted":true}));
            })
            .await;
        let second_result = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/work/w2/result");
                then.delay(Duration::from_millis(100))
                    .status(200)
                    .json_body(serde_json::json!({"accepted":true}));
            })
            .await;
        let config = Config {
            coordinator_url: Url::parse(&format!("{}/", server.base_url())).unwrap(),
            node_id: Some(Uuid::parse_str("00000000-0000-4000-8000-000000000001").unwrap()),
            state_dir: std::env::temp_dir(),
            work_concurrency: 1,
            work_batch_size: 2,
            cpu_threads: Some(1),
            compute_backend: crate::backend::BackendChoice::Cpu,
            poll_interval_ms: 10,
            max_backoff_ms: 100,
            request_timeout_secs: 2,
            log_level: "info".into(),
        };
        let worker = Worker::new(config).unwrap();
        let (stop_tx, stop_rx) = watch::channel(false);
        let mut task = tokio::spawn(worker.run(stop_rx));
        while dataset.hits_async().await == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        stop_tx.send(true).unwrap();
        assert!(tokio::time::timeout(Duration::from_millis(20), &mut task)
            .await
            .is_err());
        task.await.unwrap().unwrap();
        assert_eq!(dataset.hits_async().await, 1);
        first_result.assert_async().await;
        second_result.assert_async().await;
    }
}
