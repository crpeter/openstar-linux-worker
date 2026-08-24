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
use tracing::info;

pub struct VulkanBackend {
    inner: Mutex<Context>,
    name: String,
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
}

pub(crate) fn choose_queue(flags: &[vk::QueueFlags]) -> Option<u32> {
    flags
        .iter()
        .position(|f| f.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
        .and_then(|i| u32::try_from(i).ok())
}
fn device_rank(kind: vk::PhysicalDeviceType) -> u8 {
    match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => 2,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        _ => 0,
    }
}

impl VulkanBackend {
    pub fn new() -> Result<Self> {
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
                &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family),
                None,
            )
        }?;
        info!(device=%name, api_version=properties.api_version, driver_version=properties.driver_version, queue_family, "Vulkan device selected");
        Ok(Self {
            name,
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
            }),
        })
    }
}

struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
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
        Ok(Buffer {
            buffer,
            memory,
            size,
        })
    }
    unsafe fn write<T: Copy>(&self, b: &Buffer, values: &[T]) -> Result<()> {
        let bytes = size_of_val(values);
        let mapped = unsafe {
            self.device
                .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())?
        };
        unsafe {
            ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), mapped.cast(), bytes);
            self.device.unmap_memory(b.memory);
        }
        Ok(())
    }
    unsafe fn read_f32(&self, b: &Buffer, count: usize) -> Result<Vec<f32>> {
        let mapped = unsafe {
            self.device
                .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())?
        };
        let result = unsafe { std::slice::from_raw_parts(mapped.cast::<f32>(), count) }.to_vec();
        unsafe {
            self.device.unmap_memory(b.memory);
        }
        Ok(result)
    }
    fn execute(
        &mut self,
        dataset: &Dataset,
        p: &LombPayload,
    ) -> std::result::Result<LombResult, BackendError> {
        let (x, y, normalization) = kernel::validate(dataset, p)?;
        if u32::try_from(x.len()).is_err() || u32::try_from(p.frequency_count).is_err() {
            return Err(ComputeError::InvalidCount.into());
        }
        let run = || -> Result<Vec<f32>> {
            unsafe {
                let xb = self.buffer(size_of_val(x))?;
                let yb = self.buffer(size_of_val(y))?;
                let out = self.buffer(p.frequency_count * 4)?;
                self.write(&xb, x)?;
                self.write(&yb, y)?;
                let pool_sizes = [vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(3)];
                let pool = self.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )?;
                let layouts = [self.descriptor_layout];
                let set = self.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )?[0];
                let infos = [xb.buffer, yb.buffer, out.buffer].map(|buffer| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .range(vk::WHOLE_SIZE)]
                });
                let writes = [0, 1, 2].map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&infos[i])
                });
                self.device.update_descriptor_sets(&writes, &[]);
                let cmd = self.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0];
                self.device.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[set],
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
                let result = self.read_f32(&out, p.frequency_count)?;
                self.device.free_command_buffers(self.command_pool, &cmds);
                self.device.destroy_descriptor_pool(pool, None);
                for b in [xb, yb, out] {
                    self.device.destroy_buffer(b.buffer, None);
                    self.device.free_memory(b.memory, None);
                }
                Ok(result)
            }
        };
        let powers = run().map_err(BackendError::Execution)?;
        let gpu_winner = kernel::select_winner(p, &powers).map_err(BackendError::InvalidInput)?;
        let chunk_winner = gpu_winner
            .best_frequency_index
            .checked_sub(p.frequency_start_index)
            .ok_or(ComputeError::InvalidResult)?;
        kernel::refine_winner(dataset, p, chunk_winner).map_err(BackendError::InvalidInput)
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
            .execute(d, p)
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
    fn matches_cpu_when_vulkan_is_available() {
        let Ok(vulkan) = VulkanBackend::new() else {
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
        assert_eq!(gpu.best_frequency_index, cpu.best_frequency_index);
        assert_eq!(gpu.best_frequency, cpu.best_frequency);
        assert_eq!(gpu.best_period_days, cpu.best_period_days);
        assert!((gpu.best_power - cpu.best_power).abs() <= 1.0e-5);
    }
}
