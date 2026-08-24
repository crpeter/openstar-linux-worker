use crate::{
    kernel::{self, ComputeError},
    protocol::{Dataset, LombPayload, LombResult},
    vulkan::VulkanBackend,
};
use anyhow::{Context, Result};
use clap::ValueEnum;
use std::{fmt, sync::Arc};
use tracing::{info, warn};

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error(transparent)]
    InvalidInput(#[from] ComputeError),
    #[error("compute backend execution failed: {0:#}")]
    Execution(#[source] anyhow::Error),
}

pub trait ComputeBackend: Send + Sync {
    fn execute(
        &self,
        dataset: &Dataset,
        payload: &LombPayload,
    ) -> std::result::Result<LombResult, BackendError>;
    fn id(&self) -> &'static str;
    fn gpu_name(&self) -> Option<&str> {
        None
    }
}

pub struct CpuBackend {
    pool: rayon::ThreadPool,
}
impl CpuBackend {
    pub fn new(threads: usize) -> Result<Self> {
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?,
        })
    }
}
impl ComputeBackend for CpuBackend {
    fn execute(
        &self,
        dataset: &Dataset,
        payload: &LombPayload,
    ) -> std::result::Result<LombResult, BackendError> {
        self.pool
            .install(|| kernel::execute(dataset, payload))
            .map_err(BackendError::InvalidInput)
    }
    fn id(&self) -> &'static str {
        "cpu"
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum BackendChoice {
    #[default]
    Auto,
    Cpu,
    Vulkan,
}

pub fn initialize(choice: BackendChoice, threads: usize) -> Result<Arc<dyn ComputeBackend>> {
    match choice {
        BackendChoice::Cpu => selected(Arc::new(CpuBackend::new(threads)?)),
        BackendChoice::Vulkan => selected(Arc::new(
            VulkanBackend::new(threads).context("Vulkan backend initialization failed")?,
        )),
        BackendChoice::Auto => match VulkanBackend::new(threads) {
            Ok(vulkan) => selected(Arc::new(vulkan)),
            Err(error) => {
                warn!(%error, "Vulkan unavailable; falling back to CPU");
                selected(Arc::new(CpuBackend::new(threads)?))
            }
        },
    }
}
fn selected(backend: Arc<dyn ComputeBackend>) -> Result<Arc<dyn ComputeBackend>> {
    info!(
        backend = backend.id(),
        gpu = backend.gpu_name().unwrap_or("none"),
        "compute backend selected"
    );
    Ok(backend)
}
impl fmt::Display for BackendChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Auto => "auto",
                Self::Cpu => "cpu",
                Self::Vulkan => "vulkan",
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backend_choice_parses() {
        use clap::ValueEnum;
        assert_eq!(
            BackendChoice::from_str("auto", false).unwrap(),
            BackendChoice::Auto
        );
        assert_eq!(
            BackendChoice::from_str("cpu", false).unwrap(),
            BackendChoice::Cpu
        );
        assert_eq!(
            BackendChoice::from_str("vulkan", false).unwrap(),
            BackendChoice::Vulkan
        );
    }
    #[test]
    fn explicit_cpu_never_requires_vulkan() {
        assert_eq!(initialize(BackendChoice::Cpu, 1).unwrap().id(), "cpu");
    }

    #[test]
    fn configured_cpu_backend_result_is_unchanged() {
        let dataset = Dataset {
            coordinates: Some(vec![0.0, 0.37, 1.11, 1.8, 2.73]),
            values: Some(vec![2.0, 3.2, 1.4, 2.8, 1.9]),
            times: None,
            flux: None,
        };
        let payload = LombPayload {
            start_frequency: 0.2,
            frequency_step: 0.15,
            frequency_count: 5,
            frequency_start_index: 40,
        };
        let expected = kernel::execute(&dataset, &payload).unwrap();
        let actual = CpuBackend::new(2)
            .unwrap()
            .execute(&dataset, &payload)
            .unwrap();
        assert_eq!(actual, expected);
    }
}
