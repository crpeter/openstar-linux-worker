use crate::backend::BackendChoice;
use clap::Parser;
use std::{path::PathBuf, time::Duration};
use url::Url;
use uuid::Uuid;

#[derive(Debug, Clone, Parser)]
#[command(version, about = "Generic OpenStar compute worker")]
pub struct Config {
    #[arg(long, env = "OPENSTAR_COORDINATOR_URL")]
    pub coordinator_url: Url,
    #[arg(long, env = "OPENSTAR_NODE_ID")]
    pub node_id: Option<Uuid>,
    #[arg(
        long,
        env = "OPENSTAR_STATE_DIR",
        default_value = "/var/lib/openstar-worker"
    )]
    pub state_dir: PathBuf,
    #[arg(long, env = "OPENSTAR_WORK_CONCURRENCY", default_value_t = 1)]
    pub work_concurrency: usize,
    #[arg(long, env = "OPENSTAR_CPU_THREADS")]
    pub cpu_threads: Option<usize>,
    #[arg(long, env = "OPENSTAR_COMPUTE_BACKEND", default_value_t = BackendChoice::Auto)]
    pub compute_backend: BackendChoice,
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
            u32::try_from(self.work_concurrency).is_ok(),
            "work concurrency is too large"
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

    pub fn node_identity(&self) -> anyhow::Result<Uuid> {
        if let Some(id) = self.node_id {
            return Ok(id);
        }
        let path = self.state_dir.join("node-id");
        match std::fs::read_to_string(&path) {
            Ok(value) => return Ok(value.trim().parse()?),
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error.into()),
            Err(_) => {}
        }
        std::fs::create_dir_all(&self.state_dir)?;
        let id = Uuid::new_v4();
        use std::io::Write;
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                file.write_all(format!("{id}\n").as_bytes())?;
                file.sync_all()?;
                Ok(id)
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                Ok(std::fs::read_to_string(path)?.trim().parse()?)
            }
            Err(error) => Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generated_identity_is_stable() {
        let directory = std::env::temp_dir().join(format!("openstar-identity-{}", Uuid::new_v4()));
        let config = Config {
            coordinator_url: Url::parse("http://localhost/").unwrap(),
            node_id: None,
            state_dir: directory.clone(),
            work_concurrency: 1,
            cpu_threads: Some(1),
            compute_backend: BackendChoice::Cpu,
            poll_interval_ms: 1,
            max_backoff_ms: 2,
            request_timeout_secs: 1,
            log_level: "info".into(),
        };
        assert_eq!(
            config.node_identity().unwrap(),
            config.node_identity().unwrap()
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
