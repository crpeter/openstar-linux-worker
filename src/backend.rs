use crate::{
    kernel::{self, ComputeError},
    protocol::{Dataset, LombPayload, LombResult},
    vulkan::VulkanBackend,
};
use anyhow::{Context, Result};
use clap::ValueEnum;
use std::{fmt, sync::Arc};
use tracing::{info, warn};

pub trait ComputeBackend: Send + Sync {
    fn execute(&self, dataset: &Dataset, payload: &LombPayload)
        -> Result<LombResult, ComputeError>;
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
    ) -> Result<LombResult, ComputeError> {
        self.pool.install(|| kernel::execute(dataset, payload))
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
            VulkanBackend::new().context("Vulkan backend initialization failed")?,
        )),
        BackendChoice::Auto => match VulkanBackend::new() {
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
}
