use crate::{
    backend::{BackendError, ComputeBackend},
    kernel::{self, ComputeError},
    protocol::{Dataset, LombPayload, LombResult},
};
use anyhow::{anyhow, Context as AnyhowContext, Result};
use ash::{vk, Entry};
use std::{
    ffi::{CStr, CString},
    mem::size_of_val,
    ptr,
    sync::Mutex,
};
use tracing::{debug, info};

pub struct VulkanBackend {
    inner: Mutex<Context>,
    name: String,
    refinement_pool: rayon::ThreadPool,
}
struct Context {
    _entry: Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    queue: vk::Queue,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    resources: Option<ExecutionResources>,
}

pub(crate) fn choose_queue(flags: &[vk::QueueFlags]) -> Option<u32> {
    flags
        .iter()
        .position(|f| f.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
        .and_then(|i| u32::try_from(i).ok())
}

fn refinement_edge_diagnostics(
    frequency_count: usize,
    refinement_start: usize,
    refinement_end: usize,
    refined_winner: usize,
) -> (bool, bool, bool) {
    let final_chunk_index = frequency_count - 1;
    let clamped_at_chunk_start = refinement_start == 0;
    let clamped_at_chunk_end = refinement_end == final_chunk_index;
    let winner_on_radius_edge = (refined_winner == refinement_start && refinement_start > 0)
        || (refined_winner == refinement_end && refinement_end < final_chunk_index);
    (
        clamped_at_chunk_start,
        clamped_at_chunk_end,
        winner_on_radius_edge,
    )
}

fn device_rank(kind: vk::PhysicalDeviceType) -> u8 {
    match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => 2,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        _ => 0,
    }
}

fn build_refinement_pool(threads: usize) -> Result<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("create Vulkan CPU refinement thread pool")
}

impl VulkanBackend {
    pub fn new(threads: usize) -> Result<Self> {
        let refinement_pool = build_refinement_pool(threads)?;
        let entry = unsafe { Entry::load() }.context("load Vulkan loader")?;
        let app = CString::new("openstar-linux-worker")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app)
            .api_version(vk::API_VERSION_1_0);
        let create = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance =
            unsafe { entry.create_instance(&create, None) }.context("create Vulkan instance")?;
        let physicals =
            unsafe { instance.enumerate_physical_devices() }.context("enumerate Vulkan devices")?;
        let mut choices = Vec::new();
        for physical in physicals {
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            let queues = unsafe { instance.get_physical_device_queue_family_properties(physical) };
            let flags: Vec<_> = queues.iter().map(|q| q.queue_flags).collect();
            if let Some(family) = choose_queue(&flags) {
                choices.push((
                    device_rank(properties.device_type),
                    physical,
                    family,
                    properties,
                ));
            }
        }
        choices.sort_by_key(|c| std::cmp::Reverse(c.0));
        let (_, physical, queue_family, properties) = choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no Vulkan device with a graphics+compute queue"))?;
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
                None,
            )
        }
        .context("create Vulkan device")?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let bindings = [0, 1, 2].map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(20)];
        let layouts = [descriptor_layout];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&push),
                None,
            )
        }?;
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/lomb_scargle.spv"));
        let code = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
        let shader = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
        }?;
        let main = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(main);
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];
        let pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
        }
        .map_err(|(_, e)| e)?[0];
        unsafe { device.destroy_shader_module(shader, None) };
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;
        info!(device=%name, api_version=properties.api_version, driver_version=properties.driver_version, queue_family, "Vulkan device selected");
        Ok(Self {
            name,
            refinement_pool,
            inner: Mutex::new(Context {
                _entry: entry,
                instance,
                device,
                physical,
                queue,
                descriptor_layout,
                pipeline_layout,
                pipeline,
                command_pool,
                resources: None,
            }),
        })
    }
}

struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    mapped: *mut u8,
}

// Access is serialized by VulkanBackend::inner, and the mapping remains valid until cleanup.
unsafe impl Send for Buffer {}

struct ExecutionResources {
    x: Buffer,
    y: Buffer,
    output: Buffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_buffer: vk::CommandBuffer,
}

fn capacity_sufficient(capacity: vk::DeviceSize, required: usize) -> bool {
    u64::try_from(required.max(4)).is_ok_and(|required| capacity >= required)
}

impl Context {
    fn buffer(&self, size: usize) -> Result<Buffer> {
        let size = u64::try_from(size.max(4))?;
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
                None,
            )
        }?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical)
        };
        let memory_type_index = (0..props.memory_type_count)
            .find(|&i| {
                req.memory_type_bits & (1 << i) != 0
                    && props.memory_types[i as usize].property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .ok_or_else(|| anyhow!("no host-visible coherent Vulkan memory"))?;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memory_type_index),
                None,
            )
        }?;
        unsafe {
            self.device.bind_buffer_memory(buffer, memory, 0)?;
        }
        let mapped = unsafe {
            self.device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?
                .cast()
        };
        Ok(Buffer {
            buffer,
            memory,
            size,
            mapped,
        })
    }
    unsafe fn write<T: Copy>(b: &Buffer, values: &[T]) {
        let bytes = size_of_val(values);
        unsafe {
            ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), b.mapped, bytes);
        }
    }
    unsafe fn read_f32(b: &Buffer, count: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(b.mapped.cast::<f32>(), count) }.to_vec()
    }
    unsafe fn destroy_buffer(&self, b: Buffer) {
        unsafe {
            self.device.unmap_memory(b.memory);
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }
    fn create_resources(
        &self,
        x_size: usize,
        y_size: usize,
        output_size: usize,
    ) -> Result<ExecutionResources> {
        unsafe {
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3)];
            let descriptor_pool = self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )?;
            let descriptor_set = self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[self.descriptor_layout]),
            )?[0];
            let command_buffer = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            Ok(ExecutionResources {
                x: self.buffer(x_size)?,
                y: self.buffer(y_size)?,
                output: self.buffer(output_size)?,
                descriptor_pool,
                descriptor_set,
                command_buffer,
            })
        }
    }

    fn grow_buffer(&self, buffer: &mut Buffer, required: usize) -> Result<()> {
        if !capacity_sufficient(buffer.size, required) {
            let replacement = self.buffer(required)?;
            let old = std::mem::replace(buffer, replacement);
            unsafe { self.destroy_buffer(old) };
        }
        Ok(())
    }

    fn execute(
        &mut self,
        refinement_pool: &rayon::ThreadPool,
        dataset: &Dataset,
        p: &LombPayload,
    ) -> std::result::Result<LombResult, BackendError> {
        let (x, y, normalization) = kernel::validate(dataset, p)?;
        if u32::try_from(x.len()).is_err() || u32::try_from(p.frequency_count).is_err() {
            return Err(ComputeError::InvalidCount.into());
        }
        let run = || -> Result<Vec<f32>> {
            let x_size = size_of_val(x);
            let y_size = size_of_val(y);
            let output_size = p.frequency_count * 4;
            let mut resources = match self.resources.take() {
                Some(resources) => resources,
                None => self.create_resources(x_size, y_size, output_size)?,
            };
            let result = (|| -> Result<Vec<f32>> {
                self.grow_buffer(&mut resources.x, x_size)?;
                self.grow_buffer(&mut resources.y, y_size)?;
                self.grow_buffer(&mut resources.output, output_size)?;
                unsafe {
                    Self::write(&resources.x, x);
                    Self::write(&resources.y, y);
                    let infos = [
                        resources.x.buffer,
                        resources.y.buffer,
                        resources.output.buffer,
                    ]
                    .map(|buffer| {
                        [vk::DescriptorBufferInfo::default()
                            .buffer(buffer)
                            .range(vk::WHOLE_SIZE)]
                    });
                    let writes = [0, 1, 2].map(|i| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(resources.descriptor_set)
                            .dst_binding(i as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&infos[i])
                    });
                    self.device.update_descriptor_sets(&writes, &[]);
                    let cmd = resources.command_buffer;
                    self.device
                        .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
                    self.device.begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )?;
                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipeline,
                    );
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipeline_layout,
                        0,
                        &[resources.descriptor_set],
                        &[],
                    );
                    let push = [
                        x.len() as u32,
                        p.frequency_count as u32,
                        p.start_frequency.to_bits(),
                        p.frequency_step.to_bits(),
                        normalization.to_bits(),
                    ];
                    self.device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        std::slice::from_raw_parts(push.as_ptr().cast::<u8>(), 20),
                    );
                    self.device
                        .cmd_dispatch(cmd, (p.frequency_count as u32).div_ceil(64), 1, 1);
                    self.device.end_command_buffer(cmd)?;
                    let cmds = [cmd];
                    self.device.queue_submit(
                        self.queue,
                        &[vk::SubmitInfo::default().command_buffers(&cmds)],
                        vk::Fence::null(),
                    )?;
                    self.device.queue_wait_idle(self.queue)?;
                    Ok(Self::read_f32(&resources.output, p.frequency_count))
                }
            })();
            self.resources = Some(resources);
            result
        };
        let powers = run().map_err(BackendError::Execution)?;
        let gpu_winner = kernel::select_winner(p, &powers).map_err(BackendError::InvalidInput)?;
        let chunk_winner = gpu_winner
            .best_frequency_index
            .checked_sub(p.frequency_start_index)
            .ok_or(ComputeError::InvalidResult)?;
        let refinement_range = kernel::refinement_range(p.frequency_count, chunk_winner)
            .map_err(BackendError::InvalidInput)?;
        let refinement_start = *refinement_range.start();
        let refinement_end = *refinement_range.end();
        let refined = refinement_pool
            .install(|| kernel::refine_winner(dataset, p, chunk_winner))
            .map_err(BackendError::InvalidInput)?;
        let refined_chunk_winner = refined
            .best_frequency_index
            .checked_sub(p.frequency_start_index)
            .ok_or(ComputeError::InvalidResult)
            .map_err(BackendError::InvalidInput)?;
        let (
            refinement_range_clamped_at_chunk_start,
            refinement_range_clamped_at_chunk_end,
            refined_winner_on_refinement_radius_edge,
        ) = refinement_edge_diagnostics(
            p.frequency_count,
            refinement_start,
            refinement_end,
            refined_chunk_winner,
        );
        debug!(
            raw_gpu_winner_index = chunk_winner,
            raw_gpu_winner_power = gpu_winner.best_power,
            cpu_refinement_range_start = refinement_start,
            cpu_refinement_range_end = refinement_end,
            refined_winner_index = refined_chunk_winner,
            refined_winner_power = refined.best_power,
            raw_to_refined_distance_bins = chunk_winner.abs_diff(refined_chunk_winner),
            refinement_range_clamped_at_chunk_start,
            refinement_range_clamped_at_chunk_end,
            refined_winner_on_refinement_radius_edge,
            "Vulkan work unit CPU refinement completed"
        );
        Ok(refined)
    }
}
impl ComputeBackend for VulkanBackend {
    fn execute(
        &self,
        d: &Dataset,
        p: &LombPayload,
    ) -> std::result::Result<LombResult, BackendError> {
        self.inner
            .lock()
            .map_err(|_| BackendError::Execution(anyhow!("Vulkan context mutex poisoned")))?
            .execute(&self.refinement_pool, d, p)
    }
    fn id(&self) -> &'static str {
        "vulkan"
    }
    fn gpu_name(&self) -> Option<&str> {
        Some(&self.name)
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(resources) = self.resources.take() {
                self.device.unmap_memory(resources.x.memory);
                self.device.unmap_memory(resources.y.memory);
                self.device.unmap_memory(resources.output.memory);
                self.device.destroy_buffer(resources.x.buffer, None);
                self.device.destroy_buffer(resources.y.buffer, None);
                self.device.destroy_buffer(resources.output.buffer, None);
                self.device.free_memory(resources.x.memory, None);
                self.device.free_memory(resources.y.memory, None);
                self.device.free_memory(resources.output.memory, None);
                self.device
                    .destroy_descriptor_pool(resources.descriptor_pool, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Dataset, LombPayload};
    #[test]
    fn graphics_compute_is_required_even_with_compute_only_first() {
        let flags = [
            vk::QueueFlags::COMPUTE,
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
        ];
        assert_eq!(choose_queue(&flags), Some(1));
    }
    #[test]
    fn compute_only_is_rejected() {
        assert_eq!(choose_queue(&[vk::QueueFlags::COMPUTE]), None);
    }

    #[test]
    fn buffer_capacity_is_reused_until_growth_is_required() {
        assert!(capacity_sufficient(1024, 512));
        assert!(capacity_sufficient(1024, 1024));
        assert!(!capacity_sufficient(1024, 1025));
        assert!(capacity_sufficient(4, 0));
    }

    #[test]
    fn refinement_diagnostics_distinguish_chunk_boundaries_from_radius_edges() {
        assert_eq!(
            refinement_edge_diagnostics(4096, 0, 128, 0),
            (true, false, false)
        );
        assert_eq!(
            refinement_edge_diagnostics(4096, 3967, 4095, 4095),
            (false, true, false)
        );
        assert_eq!(
            refinement_edge_diagnostics(4096, 1872, 2128, 1872),
            (false, false, true)
        );
        assert_eq!(
            refinement_edge_diagnostics(4096, 1872, 2128, 2128),
            (false, false, true)
        );
        assert_eq!(
            refinement_edge_diagnostics(4096, 1872, 2128, 2000),
            (false, false, false)
        );
    }

    #[test]
    fn matches_cpu_when_vulkan_is_available() {
        let Ok(vulkan) = VulkanBackend::new(2) else {
            eprintln!("skipping: no usable Vulkan GPU");
            return;
        };
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
        let cpu = kernel::execute(&dataset, &payload).unwrap();
        let gpu = vulkan.execute(&dataset, &payload).unwrap();
        let repeated = vulkan.execute(&dataset, &payload).unwrap();
        assert_eq!(gpu.best_frequency_index, cpu.best_frequency_index);
        assert_eq!(gpu.best_frequency, cpu.best_frequency);
        assert_eq!(gpu.best_period_days, cpu.best_period_days);
        assert!((gpu.best_power - cpu.best_power).abs() <= 1.0e-5);
        assert_eq!(repeated, gpu);
    }

    #[test]
    fn refinement_pool_honors_configured_thread_count_without_vulkan() {
        let pool = build_refinement_pool(3).unwrap();
        assert_eq!(pool.current_num_threads(), 3);
    }
}
