use crate::{client::Coordinator, config::Config, kernel, protocol::*};
use anyhow::Result;
use rand::Rng;
use std::{future::Future, sync::Arc, time::Duration};
use tokio::sync::{watch, Semaphore};
use tracing::{info, warn};

pub struct Worker {
    config: Config,
    client: Coordinator,
    pool: Arc<rayon::ThreadPool>,
}
impl Worker {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let threads = config.cpu_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        });
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        let client = Coordinator::new(config.coordinator_url.clone(), config.timeout())?;
        Ok(Self {
            config,
            client,
            pool: Arc::new(pool),
        })
    }
    pub async fn run(self, mut stop: watch::Receiver<bool>) -> Result<()> {
        let threads = self.pool.current_num_threads();
        let node = self
            .client
            .register(&Registration {
                name: &self.config.node_name,
                capabilities: Capabilities {
                    workloads: vec![LOMB_SCARGLE_V1],
                    cpu_threads: threads,
                },
            })
            .await?;
        info!(node_id=%node,"registered");
        let permits = Arc::new(Semaphore::new(self.config.work_concurrency));
        let mut backoff = self.config.poll_interval_ms;
        loop {
            if *stop.borrow() {
                break;
            }
            let permit = permits.clone().acquire_owned().await?;
            match self.client.claim(&node).await {
                Ok(Some(work)) => {
                    backoff = self.config.poll_interval_ms;
                    let client = self.client.clone();
                    let pool = self.pool.clone();
                    let max = self.config.max_backoff_ms;
                    tokio::spawn(async move {
                        let _permit = permit;
                        process(client, pool, work, max).await;
                    });
                }
                Ok(None) => {
                    drop(permit);
                    sleep_or_stop(
                        Duration::from_millis(self.config.poll_interval_ms),
                        &mut stop,
                    )
                    .await;
                }
                Err(e) => {
                    drop(permit);
                    warn!(error=%e,"claim failed");
                    let jitter = rand::thread_rng().gen_range(0..=(backoff / 4));
                    sleep_or_stop(Duration::from_millis(backoff + jitter), &mut stop).await;
                    backoff = (backoff.saturating_mul(2)).min(self.config.max_backoff_ms);
                }
            }
        }
        // Acquiring all permits waits for processing, including result retries, to finish.
        let _drain = permits
            .acquire_many(self.config.work_concurrency as u32)
            .await?;
        Ok(())
    }
}
async fn sleep_or_stop(d: Duration, stop: &mut watch::Receiver<bool>) {
    tokio::select! { _=tokio::time::sleep(d)=>{}, _=stop.changed()=>{} }
}
async fn process(
    client: Coordinator,
    pool: Arc<rayon::ThreadPool>,
    work: WorkUnit,
    max_backoff: u64,
) {
    let envelope = if work.workload_id != LOMB_SCARGLE_V1 {
        ResultEnvelope::<LombResult> {
            workload_id: work.workload_id.clone(),
            result: None,
            failure: Some(Failure {
                code: "unsupported_workload",
                message: format!("unsupported workload: {}", work.workload_id),
            }),
        }
    } else {
        match client.dataset(&work).await.and_then(|d| {
            let p = work.lomb_payload().map_err(anyhow::Error::msg)?;
            Ok(pool.install(|| kernel::execute(&d, &p)))
        }) {
            Ok(Ok(result)) => ResultEnvelope {
                workload_id: work.workload_id.clone(),
                result: Some(result),
                failure: None,
            },
            Ok(Err(e)) => ResultEnvelope {
                workload_id: work.workload_id.clone(),
                result: None,
                failure: Some(Failure {
                    code: "invalid_input",
                    message: e.to_string(),
                }),
            },
            Err(e) => ResultEnvelope {
                workload_id: work.workload_id.clone(),
                result: None,
                failure: Some(Failure {
                    code: "dataset_error",
                    message: e.to_string(),
                }),
            },
        }
    };
    let body = serde_json::to_vec(&envelope).expect("serializable result");
    retry(|| client.submit_bytes(&work.id, &body), max_backoff).await;
}
async fn retry<F, Fut>(mut f: F, max: u64)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<()>>,
{
    let mut delay = 500;
    loop {
        match f().await {
            Ok(()) => break,
            Err(e) => {
                warn!(error=%e,"result submission failed; retrying");
                tokio::time::sleep(Duration::from_millis(delay)).await;
                delay = (delay * 2).min(max.max(500));
            }
        }
    }
}
