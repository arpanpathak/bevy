//! Zero-copy CUDA to Vulkan buffer interop.
//!
//! # Why we allocate the buffer ourselves
//!
//! wgpu-hal keeps the `VkDeviceMemory` behind a buffer private, so there is no way
//! to ask wgpu to export one of *its* buffers to CUDA. This module therefore works
//! the other way round:
//!
//! 1. Allocate a `VkBuffer` and its memory here, marked as exportable.
//! 2. Export that memory as a file descriptor and import it into CUDA.
//! 3. Wrap the *same* `VkBuffer` for wgpu with [`wgpu::Device::create_buffer_from_hal`],
//!    using `wgpu_hal::vulkan::Buffer::from_raw_externally_owned` so that wgpu knows
//!    it must not destroy the handle.
//!
//! Both APIs end up addressing a single allocation, so no copy ever happens.
//!
//! # Caveat: do not read these buffers from the CPU
//!
//! Measured on Jetson Orin NX (L4T R36.4.3, CUDA 12.6): once CUDA writes through the
//! imported pointer, the Vulkan *GPU* observes the new data, but a CPU mapping of the
//! same memory stays stale even after `vkInvalidateMappedMemoryRanges`. Consume the
//! buffer on the GPU; do not `map_async` it and expect CUDA's writes to appear.
//!
//! # Synchronisation
//!
//! CUDA work is asynchronous. Launch kernels through [`CudaBuffer::with_device_ptr`],
//! then call [`CudaBuffer::synchronize`] before the renderer reads the buffer, or the
//! two will race. That device-wide wait is a placeholder: the intended design is an
//! exported timeline semaphore (`VK_KHR_external_semaphore_fd`), so the render graph
//! can wait on CUDA without ever stalling the CPU.

use alloc::sync::Arc;
use std::os::fd::RawFd;

use ash::vk;
use cudarc::driver::{sys as cu, CudaContext};
use wgpu::hal::api::Vulkan;

use crate::capabilities::CudaSupport;
use crate::error::CudaInteropError;

/// The handle type exported by Vulkan and imported by CUDA.
///
/// CUDA 12.6 has no dma-buf handle type (`CU_EXTERNAL_MEMORY_HANDLE_TYPE_*` stops at
/// `NVSCIBUF`), so the pairing is Vulkan's `OPAQUE_FD` bit against
/// `CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD`. On Linux that descriptor is a dma-buf,
/// which is what makes the import work at all.
const HANDLE_TYPE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;

/// How to size and use a CUDA-shareable buffer.
#[derive(Clone, Debug)]
pub struct CudaBufferDescriptor {
    /// Debug label, applied to the wgpu buffer.
    pub label: Option<&'static str>,
    /// Size in bytes. Must be non-zero.
    pub size: u64,
    /// How the renderer will use the buffer. Mapped onto Vulkan usage flags.
    pub usage: wgpu::BufferUsages,
}

/// A buffer that CUDA writes and the renderer reads, with no copies between them.
///
/// Dropping this releases the CUDA import and the Vulkan allocation. Make sure
/// in-flight GPU work has finished first, for example by polling the device or by
/// waiting on a submitted-work callback.
pub struct CudaBuffer {
    /// wgpu's view of the buffer.
    ///
    /// Declared *before* `raw` on purpose: Rust drops fields in declaration order,
    /// so wgpu releases the buffer before the raw Vulkan objects behind it are
    /// destroyed. Reordering these two fields would be a use-after-free.
    wgpu_buffer: wgpu::Buffer,
    raw: RawCudaBuffer,
}

/// The half of a [`CudaBuffer`] that we own outright and must tear down by hand.
struct RawCudaBuffer {
    /// Held purely for its lifetime: it keeps the underlying `VkDevice` alive so
    /// that `drop` can still destroy our Vulkan objects. Nothing ever reads it,
    /// hence the underscore.
    _wgpu_device: wgpu::Device,
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    external_memory: cu::CUexternalMemory,
    device_ptr: cu::CUdeviceptr,
    context: Arc<CudaContext>,
    size: u64,
}

impl CudaSupport {
    /// Allocate a buffer that CUDA and the renderer both address directly.
    ///
    /// `instance`, `adapter` and `device` must be the renderer's own, so that the
    /// buffer ends up on the device that will actually draw with it.
    pub fn create_buffer(
        &self,
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        descriptor: &CudaBufferDescriptor,
    ) -> Result<CudaBuffer, CudaInteropError> {
        if descriptor.size == 0 {
            return Err(CudaInteropError::ZeroSized);
        }

        // SAFETY: the raw handles are used only for the duration of this call, and
        // the `wgpu` objects that own them outlive it.
        let (Some(hal_instance), Some(hal_adapter), Some(hal_device)) = (
            unsafe { instance.as_hal::<Vulkan>() },
            unsafe { adapter.as_hal::<Vulkan>() },
            unsafe { device.as_hal::<Vulkan>() },
        ) else {
            return Err(CudaInteropError::NotVulkan);
        };

        let raw_instance = hal_instance.shared_instance().raw_instance();
        let raw_physical_device = hal_adapter.raw_physical_device();
        let raw_device = hal_device.raw_device();

        let usage = to_vk_usage(descriptor.usage);
        ensure_exportable(raw_instance, raw_physical_device, usage)?;

        // Owns the Vulkan objects until they are handed to the `CudaBuffer` below,
        // so every early return frees them without hand-written cleanup.
        let allocation = RawAllocation::create(
            raw_device,
            raw_instance,
            raw_physical_device,
            usage,
            descriptor.size,
        )?;

        let fd_ext = ash::khr::external_memory_fd::Device::new(raw_instance, raw_device);
        let fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(allocation.memory)
            .handle_type(HANDLE_TYPE);
        // SAFETY: `allocation.memory` was allocated as exportable with `HANDLE_TYPE`.
        let fd: RawFd = unsafe { fd_ext.get_memory_fd(&fd_info)? };

        self.context.bind_to_thread()?;
        // SAFETY: `fd` came from `vkGetMemoryFdKHR`; CUDA closes it for us on success.
        let external_memory = unsafe { import_opaque_fd(fd, descriptor.size)? };
        // Past this point the descriptor belongs to CUDA, so a mapping failure still
        // has to release the import. This is the one hand-written cleanup path.
        // SAFETY: `external_memory` was just imported and is live, which is exactly
        // what `map_whole_buffer` requires.
        let device_ptr = match unsafe { map_whole_buffer(external_memory, descriptor.size) } {
            Ok(device_ptr) => device_ptr,
            Err(error) => {
                // SAFETY: the import succeeded, so this handle is live and owned here.
                unsafe {
                    let _ = cu::cuDestroyExternalMemory(external_memory).result();
                }
                return Err(error);
            }
        };

        // Hand the *same* `VkBuffer` to wgpu. `from_raw_externally_owned` tells
        // wgpu-hal not to destroy the handle, which is why `RawCudaBuffer` does it.
        let drop_callback: wgpu::hal::DropCallback = Box::new(|| {});
        // SAFETY: `allocation.buffer` is a live `VkBuffer` created on this device. The
        // empty drop callback is correct because `RawCudaBuffer` takes over the handle
        // and destroys it itself.
        let hal_buffer = unsafe {
            wgpu::hal::vulkan::Buffer::from_raw_externally_owned(allocation.buffer, drop_callback)
        };
        // SAFETY: the descriptor matches how the Vulkan buffer was created, and the
        // buffer is already backed by bound memory.
        let wgpu_buffer = unsafe {
            device.create_buffer_from_hal::<Vulkan>(
                hal_buffer,
                &wgpu::BufferDescriptor {
                    label: descriptor.label,
                    size: descriptor.size,
                    usage: descriptor.usage,
                    mapped_at_creation: false,
                },
            )
        };

        let (buffer, memory) = allocation.into_raw();

        Ok(CudaBuffer {
            wgpu_buffer,
            raw: RawCudaBuffer {
                _wgpu_device: device.clone(),
                device: raw_device.clone(),
                buffer,
                memory,
                external_memory,
                device_ptr,
                context: self.context.clone(),
                size: descriptor.size,
            },
        })
    }
}

impl CudaBuffer {
    /// The wgpu view of this buffer. Bind it, or copy from it, like any other.
    pub fn wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.wgpu_buffer
    }

    /// Size of the buffer in bytes.
    pub fn size(&self) -> u64 {
        self.raw.size
    }

    /// The pointer CUDA code should write through.
    ///
    /// Prefer [`CudaBuffer::with_device_ptr`] unless you have already bound the
    /// context to this thread yourself.
    pub fn device_ptr(&self) -> cu::CUdeviceptr {
        self.raw.device_ptr
    }

    /// Run `f` with this buffer's CUDA pointer, with the CUDA context made current
    /// on the calling thread.
    ///
    /// The closure is where a kernel launch or a `cudaMemcpy` belongs.
    ///
    /// This does **not** wait for the work to finish. Anything that launched an
    /// asynchronous kernel inside `f` must be followed by [`CudaBuffer::synchronize`]
    /// before the renderer reads the buffer, otherwise the two race. Proper
    /// semaphore-based signalling is the next thing to build here; until then the
    /// synchronisation is a device-wide wait and costs more than it should.
    pub fn with_device_ptr<R>(
        &self,
        f: impl FnOnce(cu::CUdeviceptr) -> R,
    ) -> Result<R, CudaInteropError> {
        self.raw.context.bind_to_thread()?;
        Ok(f(self.raw.device_ptr))
    }

    /// Block until all work previously submitted to this CUDA context has finished.
    ///
    /// Call this after launching kernels and before the renderer consumes the
    /// buffer. It is a device-wide wait, so it stalls the caller; replacing it with
    /// an exported timeline semaphore is the intended follow-up.
    pub fn synchronize(&self) -> Result<(), CudaInteropError> {
        self.raw.context.bind_to_thread()?;
        // SAFETY: the context is current on this thread, and the call takes no
        // pointers that could outlive it.
        unsafe { cu::cuCtxSynchronize().result()? };
        Ok(())
    }
}

// SAFETY: `CudaBuffer`'s only non-`Send`/`Sync` members are the raw CUDA handles
// (`CUexternalMemory` and `CUdeviceptr`), which are opaque driver values rather
// than pointers into Rust memory. Every use of them binds the context to the
// calling thread first (`with_device_ptr`, `synchronize`, and `RawCudaBuffer::drop`),
// and `CudaContext` is itself `Send + Sync`, so moving the buffer across threads or
// sharing it between them is sound.
unsafe impl Send for CudaBuffer {}
// SAFETY: as above; shared access only exposes read-only handles and the
// thread-binding accessors, which serialise through CUDA's own context rules.
unsafe impl Sync for CudaBuffer {}

impl Drop for RawCudaBuffer {
    fn drop(&mut self) {
        // 1. Release the CUDA import. CUDA owns the descriptor now, so there is
        //    nothing to close on our side.
        if let Err(error) = self.context.bind_to_thread() {
            bevy_log::error!("could not bind the CUDA context while dropping a buffer: {error}");
        }
        // SAFETY: the handle is live, and the wgpu buffer that referenced the same
        // memory has already been dropped (see the field order on `CudaBuffer`).
        unsafe {
            if let Err(error) = cu::cuDestroyExternalMemory(self.external_memory).result() {
                bevy_log::error!("could not destroy imported CUDA external memory: {error}");
            }
        }

        // 2. Destroy the raw Vulkan objects. `self._wgpu_device` is held precisely so
        //    that the device is still alive at this point.
        // SAFETY: both handles were created on `device` and are destroyed exactly once.
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// Owns a raw Vulkan buffer and its memory until they are handed to a
/// [`CudaBuffer`]. Frees them if that hand-off never happens.
struct RawAllocation {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl RawAllocation {
    fn create(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        usage: vk::BufferUsageFlags,
        size: u64,
    ) -> Result<Self, CudaInteropError> {
        // Tell Vulkan up front that this buffer's memory is going to be exported.
        let mut external_info =
            vk::ExternalMemoryBufferCreateInfo::default().handle_types(HANDLE_TYPE);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_info);
        // SAFETY: `buffer_info` is fully initialised, and the `push_next`ed
        // `external_info` outlives the call.
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        // From here on we bail out through the guard, so `buffer` is never leaked.
        let mut allocation = RawAllocation {
            device: device.clone(),
            buffer,
            memory: vk::DeviceMemory::null(),
        };

        // SAFETY: `buffer` was just created on `device`.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        // SAFETY: `physical_device` belongs to `instance`.
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let memory_type_index = find_memory_type(
            &memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or(CudaInteropError::NoSuitableMemoryType)?;

        // Exportability is a property of the *allocation*, not the buffer, and
        // NVIDIA requires an exported allocation to be dedicated to its buffer.
        let mut export_info = vk::ExportMemoryAllocateInfo::default().handle_types(HANDLE_TYPE);
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export_info)
            .push_next(&mut dedicated_info);
        // SAFETY: the memory type supports the usage bits, and the chain is intact.
        allocation.memory = unsafe { device.allocate_memory(&allocate_info, None)? };
        // SAFETY: the memory was allocated for this buffer, at offset 0.
        unsafe { device.bind_buffer_memory(buffer, allocation.memory, 0)? };

        Ok(allocation)
    }

    /// Give up ownership so the caller becomes responsible for destruction.
    fn into_raw(self) -> (vk::Buffer, vk::DeviceMemory) {
        // `ManuallyDrop` keeps our `Drop` from running on the way out.
        let this = std::mem::ManuallyDrop::new(self);
        (this.buffer, this.memory)
    }
}

impl Drop for RawAllocation {
    fn drop(&mut self) {
        // SAFETY: both handles belong to `device` and are destroyed exactly once.
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            if self.memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.memory, None);
            }
        }
    }
}

/// Translate wgpu usage flags into the Vulkan ones a buffer must be created with.
fn to_vk_usage(usage: wgpu::BufferUsages) -> vk::BufferUsageFlags {
    let mut vk_usage = vk::BufferUsageFlags::empty();
    for (wgpu_flag, vk_flag) in [
        (
            wgpu::BufferUsages::STORAGE,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        ),
        (
            wgpu::BufferUsages::UNIFORM,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        ),
        (
            wgpu::BufferUsages::COPY_SRC,
            vk::BufferUsageFlags::TRANSFER_SRC,
        ),
        (
            wgpu::BufferUsages::COPY_DST,
            vk::BufferUsageFlags::TRANSFER_DST,
        ),
        (
            wgpu::BufferUsages::INDEX,
            vk::BufferUsageFlags::INDEX_BUFFER,
        ),
        (
            wgpu::BufferUsages::VERTEX,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        ),
        (
            wgpu::BufferUsages::INDIRECT,
            vk::BufferUsageFlags::INDIRECT_BUFFER,
        ),
    ] {
        if usage.contains(wgpu_flag) {
            vk_usage |= vk_flag;
        }
    }
    vk_usage
}

/// First memory type that satisfies `type_bits` and has every `required` property.
fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|index| {
        type_bits & (1 << index) != 0
            && properties.memory_types[*index as usize]
                .property_flags
                .contains(required)
    })
}

/// Check that the device will export this buffer before we bother allocating it,
/// so a failure is reported as an unsupported configuration instead of a raw
/// allocation error.
fn ensure_exportable(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    usage: vk::BufferUsageFlags,
) -> Result<(), CudaInteropError> {
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage)
        .handle_type(HANDLE_TYPE);
    let mut properties = vk::ExternalBufferProperties::default();
    // SAFETY: both handles are live, and the out parameter is initialised.
    unsafe {
        instance.get_physical_device_external_buffer_properties(
            physical_device,
            &info,
            &mut properties,
        );
    };

    if properties
        .external_memory_properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
    {
        Ok(())
    } else {
        Err(CudaInteropError::HandleTypeNotExportable {
            handle_type: "VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT",
        })
    }
}

/// Import an exported descriptor as CUDA external memory.
///
/// # Safety
/// `fd` must have been exported by Vulkan with [`HANDLE_TYPE`]. On success CUDA
/// takes ownership of the descriptor, so the caller must not close it.
unsafe fn import_opaque_fd(fd: RawFd, size: u64) -> Result<cu::CUexternalMemory, CudaInteropError> {
    let mut external_memory = std::mem::MaybeUninit::uninit();
    let handle = cu::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        type_: cu::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
        handle: cu::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 { fd },
        size,
        flags: 0,
        reserved: [0; 16],
    };
    // SAFETY: `handle` is fully initialised for the lifetime of the call.
    unsafe { cu::cuImportExternalMemory(external_memory.as_mut_ptr(), &handle).result()? };
    // SAFETY: the driver initialises the out parameter when it reports success.
    Ok(unsafe { external_memory.assume_init() })
}

/// Map an imported handle onto a device pointer CUDA kernels can use.
///
/// # Safety
/// `external_memory` must be a live handle from [`import_opaque_fd`].
unsafe fn map_whole_buffer(
    external_memory: cu::CUexternalMemory,
    size: u64,
) -> Result<cu::CUdeviceptr, CudaInteropError> {
    let mut device_ptr = 0;
    let buffer = cu::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
        offset: 0,
        size,
        flags: 0,
        reserved: [0; 16],
    };
    // SAFETY: `external_memory` is live, and `buffer` describes a valid range.
    unsafe {
        cu::cuExternalMemoryGetMappedBuffer(&mut device_ptr, external_memory, &buffer).result()?;
    };
    Ok(device_ptr)
}
