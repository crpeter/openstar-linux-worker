use clap::Parser;
use std::time::Duration;
use url::Url;

#[derive(Debug, Clone, Parser)]
#[command(version, about = "Generic OpenStar CPU compute worker")]
pub struct Config {
    #[arg(long, env = "OPENSTAR_COORDINATOR_URL")]
    pub coordinator_url: Url,
    #[arg(
        long,
        env = "OPENSTAR_NODE_NAME",
        default_value = "openstar-linux-worker"
    )]
    pub node_name: String,
    #[arg(long, env = "OPENSTAR_WORK_CONCURRENCY", default_value_t = 1)]
    pub work_concurrency: usize,
    #[arg(long, env = "OPENSTAR_CPU_THREADS")]
    pub cpu_threads: Option<usize>,
    #[arg(long, env = "OPENSTAR_POLL_INTERVAL_MS", default_value_t = 2_000)]
    pub poll_interval_ms: u64,
    #[arg(long, env = "OPENSTAR_MAX_BACKOFF_MS", default_value_t = 30_000)]
    pub max_backoff_ms: u64,
    #[arg(long, env = "OPENSTAR_REQUEST_TIMEOUT_SECS", default_value_t = 30)]
    pub request_timeout_secs: u64,
    #[arg(long, env = "OPENSTAR_LOG", default_value = "info")]
    pub log_level: String,
}

impl Config {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.work_concurrency > 0,
            "work concurrency must be positive"
        );
        anyhow::ensure!(
            self.cpu_threads.unwrap_or(1) > 0,
            "CPU threads must be positive"
        );
        anyhow::ensure!(self.poll_interval_ms > 0, "poll interval must be positive");
        anyhow::ensure!(self.max_backoff_ms > 0, "maximum backoff must be positive");
        anyhow::ensure!(
            self.request_timeout_secs > 0,
            "request timeout must be positive"
        );
        Ok(())
    }
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs)
    }
}
