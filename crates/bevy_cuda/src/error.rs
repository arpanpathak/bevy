//! Errors produced while setting up CUDA interop.

use thiserror::Error;

/// Something went wrong while creating a CUDA-shareable buffer.
#[derive(Debug, Error)]
pub enum CudaInteropError {
    /// The renderer is not using Vulkan, so there is no `VkDeviceMemory` to export.
    #[error("CUDA interop requires the Vulkan backend, but the renderer is not using it")]
    NotVulkan,

    /// A zero-sized buffer has no memory to export.
    #[error("a CUDA interop buffer must be larger than zero bytes")]
    ZeroSized,

    /// The driver refuses to export this buffer with the handle type we need.
    #[error(
        "the Vulkan device will not export buffers using {handle_type}, so CUDA cannot import them"
    )]
    HandleTypeNotExportable {
        /// Name of the handle type that was refused.
        handle_type: &'static str,
    },

    /// No memory type satisfies both the export requirement and our usage flags.
    #[error("no memory type can back an exportable device-local buffer of this size")]
    NoSuitableMemoryType,

    /// A Vulkan call failed.
    #[cfg(target_os = "linux")]
    #[error("Vulkan call failed: {0}")]
    Vulkan(#[from] ash::vk::Result),

    /// A CUDA driver call failed.
    #[cfg(target_os = "linux")]
    #[error("CUDA driver call failed: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),
}
