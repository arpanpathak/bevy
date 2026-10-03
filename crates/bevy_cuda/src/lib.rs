//! Zero-copy CUDA interop for Bevy.
//!
//! This crate lets CUDA kernels write directly into buffers the renderer draws
//! with, without a single copy between the two APIs. It is aimed at hardware where
//! that is cheap, notably Jetson boards with unified memory.
//!
//! # What CUDA can and cannot do here
//!
//! CUDA has no rasteriser and no swapchain, so it cannot replace the renderer and
//! it cannot make Bevy's drawing "CUDA accelerated". What it *can* do is act as a
//! compute sidecar: run the kernels that are a poor fit for WGSL, or that need CUDA
//! only libraries such as `cuBLAS`, `cuDNN` or `TensorRT`, and hand the results
//! straight to the renderer.
//!
//! Note that on an integrated part like Orin, CUDA and Vulkan drive the same SMs
//! over the same memory, so moving work across is only worth it when CUDA buys you
//! something a compute shader cannot do.
//!
//! # Getting started
//!
//! ```ignore
//! use bevy::prelude::*;
//! use bevy_cuda::{CudaBufferDescriptor, CudaCapabilities, CudaPlugin};
//!
//! App::new()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugins(CudaPlugin)
//!     .finish();
//! ```
//!
//! Then, from a render world system, check [`CudaCapabilities`] and use
//! [`CudaSupport::create_buffer`] to make a buffer, and
//! [`CudaBuffer::with_device_ptr`] to launch a kernel over it.
//!
//! # Platform support
//!
//! Interop is implemented for Linux only, and needs the Vulkan backend plus the
//! `VK_KHR_external_memory_fd` and `VK_KHR_external_semaphore_fd` device
//! extensions. Everywhere else the crate still compiles, [`CudaCapabilities`]
//! reports [`CudaUnavailable::UnsupportedPlatform`], and nothing else is exposed.

#![allow(
    unsafe_code,
    reason = "This crate exists to make raw Vulkan and CUDA driver calls. Every unsafe block is \
              audited and carries a SAFETY comment. The code is Linux-only, so an `expect` here \
              would be unfulfilled on other platforms."
)]

extern crate alloc;

mod capabilities;

#[cfg(target_os = "linux")]
mod error;
#[cfg(target_os = "linux")]
mod interop;

pub use capabilities::{CudaCapabilities, CudaUnavailable};

#[cfg(target_os = "linux")]
pub use capabilities::CudaSupport;
#[cfg(target_os = "linux")]
pub use error::CudaInteropError;
#[cfg(target_os = "linux")]
pub use interop::{CudaBuffer, CudaBufferDescriptor};

/// Re-exported so downstream code can name CUDA driver types without taking its
/// own direct dependency on `cudarc`.
#[cfg(target_os = "linux")]
pub use cudarc;

use bevy_app::{App, Plugin};
use bevy_ecs::world::{FromWorld, World};
use bevy_render::{
    renderer::{RenderAdapter, RenderInstance},
    GpuResourceAppExt, RenderApp,
};

/// Detects CUDA interop support and makes the result available to the render world.
///
/// Add this after [`bevy_render::RenderPlugin`]. If the renderer is not on Vulkan,
/// or CUDA is not usable, this logs a specific reason and does nothing else;
/// [`CudaCapabilities`] reports the same reason to your own code.
pub struct CudaPlugin;

impl Plugin for CudaPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            bevy_log::warn!(
                "`CudaPlugin` needs the render sub-app. Add it after `RenderPlugin`, with the \
                 `bevy_render` feature enabled."
            );
            return;
        };

        // Probing at `RenderStartup` rather than here means we see the real adapter
        // and device, and the probe re-runs if the render device is ever lost.
        render_app.init_gpu_resource::<CudaCapabilities>();
    }
}

impl FromWorld for CudaCapabilities {
    fn from_world(world: &mut World) -> Self {
        // `RenderInstance` and `RenderAdapter` deref to their wgpu counterparts.
        let instance = world.resource::<RenderInstance>();
        let adapter = world.resource::<RenderAdapter>();
        let capabilities = Self::detect(instance, adapter);
        let adapter_name = adapter.get_info().name;

        match capabilities.unavailable_reason() {
            Some(reason) => {
                bevy_log::debug!("CUDA interop is unavailable on `{adapter_name}`: {reason}");
            }
            None => {
                bevy_log::info!("CUDA interop is available on `{adapter_name}`");
            }
        }

        capabilities
    }
}
