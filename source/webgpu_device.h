/*
 * Minimal headless WebGPU/Dawn device wrapper.
 *
 * This is the only WebGPU surface the headless server and the AVBD GPU
 * solver depend on: instance + adapter + device + queue, a status string,
 * and a tiny compute smoke test. The legacy WebGpuContext in
 * webgpu_backend.h (preview rendering, experimental kernels, diagnostics)
 * is used only by the OpenGL debug shell.
 */

#pragma once

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
#include <dawn/webgpu_cpp.h>
#endif

struct WebGpuDevice
{
    bool deviceReady = false;
    char status[512] = "WebGPU not initialized";

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
#endif

    // Creates instance/adapter/device/queue without any window surface.
    // Safe to call repeatedly; returns existing device when already ready.
    bool initialize();

    // Dispatches a trivial compute shader and validates the result via
    // readback. Returns false (and clears deviceReady) if compute does not
    // round-trip, so callers never trust a device that cannot execute work.
    bool runComputeSmokeTest();

    const char *statusText() const { return status; }
};
