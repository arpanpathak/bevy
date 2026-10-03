//! Runtime detection of CUDA interop support.
//!
//! Interop needs three things at once: a Vulkan renderer, two device extensions
//! that let Vulkan hand memory out as a file descriptor, and a working CUDA
//! driver. Rather than assume, we probe all three once at `RenderStartup` and
//! report a specific reason when something is missing.

use bevy_ecs::resource::Resource;
use thiserror::Error;

#[cfg(target_os = "linux")]
use alloc::sync::Arc;

#[cfg(target_os = "linux")]
use cudarc::driver::CudaContext;

/// Why CUDA interop cannot be used on this machine.
///
/// Each variant names the concrete blocker, because "CUDA unavailable" is not an
/// actionable thing to see in a log.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CudaUnavailable {
    /// Interop is only implemented for Linux.
    #[error("CUDA interop is only implemented for Linux, and this target is `{target_os}`")]
    UnsupportedPlatform {
        /// The platform this build was compiled for.
        target_os: &'static str,
    },

    /// The renderer is not on Vulkan, so there is no `VkDeviceMemory` to export.
    #[error("the renderer is using the {backend} backend; CUDA interop requires Vulkan")]
    NotVulkan {
        /// The backend that is actually in use.
        backend: String,
    },

    /// A Vulkan extension we depend on is not exposed by the physical device.
    #[error("the Vulkan device is missing the `{extension}` extension, which CUDA interop needs")]
    MissingExtension {
        /// Name of the missing extension.
        extension: &'static str,
    },

    /// Asking the driver about its capabilities failed.
    #[error("could not query the Vulkan device: {message}")]
    VulkanQueryFailed {
        /// The underlying driver error.
        message: String,
    },

    /// The CUDA driver library or device could not be brought up.
    #[error("the CUDA driver could not be initialised: {message}")]
    CudaDriverUnavailable {
        /// The underlying driver error.
        message: String,
    },
}

/// The CUDA objects needed to share buffers with the renderer.
///
/// Built once during `RenderStartup` and reused for every interop buffer, so the
/// CUDA context is created exactly once per app.
#[cfg(target_os = "linux")]
#[derive(Debug, Clone)]
pub struct CudaSupport {
    pub(crate) context: Arc<CudaContext>,
    device_name: String,
}

#[cfg(target_os = "linux")]
impl CudaSupport {
    /// The CUDA context that buffers are imported into.
    ///
    /// Interop buffers must be created, and kernels launched, with this context
    /// current on the calling thread. [`crate::CudaBuffer`] handles that for the
    /// import path and for [`crate::CudaBuffer::with_device_ptr`].
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Name of the CUDA device backing this context, for diagnostics.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

/// Whether zero-copy CUDA to wgpu buffer sharing is usable in this app.
///
/// Inserted into the render world by [`crate::CudaPlugin`]; read it to decide
/// whether to take a CUDA path or fall back.
#[derive(Debug, Clone, Resource)]
pub enum CudaCapabilities {
    /// Interop is usable; the payload owns the CUDA context to use.
    #[cfg(target_os = "linux")]
    Available(CudaSupport),

    /// Interop is not usable; the payload says why.
    Unavailable(CudaUnavailable),
}

impl CudaCapabilities {
    /// Probe the renderer and the CUDA driver.
    ///
    /// This never fails: an unusable configuration is reported as
    /// `CudaCapabilities::Unavailable` rather than as an error, so callers can
    /// decide whether to degrade or to stop.
    ///
    /// Note that this creates a CUDA context as a side effect when interop *is*
    /// available, which is the expensive part of the probe and is why we do it
    /// once rather than per buffer.
    #[cfg(not(target_os = "linux"))]
    pub fn detect(_instance: &wgpu::Instance, _adapter: &wgpu::Adapter) -> Self {
        Self::Unavailable(CudaUnavailable::UnsupportedPlatform {
            target_os: std::env::consts::OS,
        })
    }

    /// Probe the renderer and the CUDA driver.
    ///
    /// See the other `detect` for the platform-independent contract.
    #[cfg(target_os = "linux")]
    pub fn detect(instance: &wgpu::Instance, adapter: &wgpu::Adapter) -> Self {
        use std::ffi::CStr;
        use wgpu::hal::api::Vulkan;

        let backend = adapter.get_info().backend;
        // SAFETY: `as_hal` is unsafe because the returned handles may be used
        // concurrently with wgpu; we only read static device properties here.
        let (Some(hal_instance), Some(hal_adapter)) =
            (unsafe { instance.as_hal::<Vulkan>() }, unsafe {
                adapter.as_hal::<Vulkan>()
            })
        else {
            return Self::Unavailable(CudaUnavailable::NotVulkan {
                backend: format!("{backend:?}"),
            });
        };

        let raw_instance = hal_instance.shared_instance().raw_instance();
        let raw_physical_device = hal_adapter.raw_physical_device();

        // SAFETY: both handles are live for the duration of this call, and the
        // enumeration only reads static device properties.
        let extensions = match unsafe {
            raw_instance.enumerate_device_extension_properties(raw_physical_device)
        } {
            Ok(extensions) => extensions,
            Err(error) => {
                return Self::Unavailable(CudaUnavailable::VulkanQueryFailed {
                    message: error.to_string(),
                });
            }
        };

        // `external_memory_fd` is what lets us hand a `VkDeviceMemory` to another
        // API as a file descriptor; `external_semaphore_fd` is what will let us
        // synchronise with CUDA without stalling the CPU.
        for required in [
            ash::khr::external_memory_fd::NAME,
            ash::khr::external_semaphore_fd::NAME,
        ] {
            let present = extensions.iter().any(|extension| {
                // SAFETY: Vulkan promises `extension_name` is a NUL-terminated string.
                let name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                name == required
            });
            if !present {
                return Self::Unavailable(CudaUnavailable::MissingExtension {
                    extension: required.to_str().unwrap_or("<unknown>"),
                });
            }
        }

        match CudaContext::new(0) {
            Ok(context) => Self::Available(CudaSupport {
                device_name: context.name().unwrap_or_else(|_| "unknown".to_owned()),
                context,
            }),
            Err(error) => Self::Unavailable(CudaUnavailable::CudaDriverUnavailable {
                message: error.to_string(),
            }),
        }
    }

    /// Whether interop can be used.
    pub fn is_available(&self) -> bool {
        match self {
            #[cfg(target_os = "linux")]
            Self::Available(_) => true,
            Self::Unavailable(_) => false,
        }
    }

    /// The reason interop is unavailable, if it is.
    pub fn unavailable_reason(&self) -> Option<&CudaUnavailable> {
        match self {
            #[cfg(target_os = "linux")]
            Self::Available(_) => None,
            Self::Unavailable(reason) => Some(reason),
        }
    }

    /// The CUDA context to import buffers into, if interop is available.
    #[cfg(target_os = "linux")]
    pub fn support(&self) -> Option<&CudaSupport> {
        match self {
            Self::Available(support) => Some(support),
            Self::Unavailable(_) => None,
        }
    }
}
