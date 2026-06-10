#include "webgpu_device.h"

#include <cstdio>
#include <cstring>
#include <string>

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN

namespace
{

std::string viewToString(wgpu::StringView view)
{
    if (view.data == nullptr || view.length == 0)
        return std::string();
    return std::string(view.data, view.length);
}

const char *backendTypeName(wgpu::BackendType type)
{
    switch (type)
    {
    case wgpu::BackendType::D3D11:
        return "D3D11";
    case wgpu::BackendType::D3D12:
        return "D3D12";
    case wgpu::BackendType::Metal:
        return "Metal";
    case wgpu::BackendType::Vulkan:
        return "Vulkan";
    case wgpu::BackendType::OpenGL:
        return "OpenGL";
    case wgpu::BackendType::OpenGLES:
        return "OpenGLES";
    default:
        return "WebGPU";
    }
}

} // namespace

bool WebGpuDevice::initialize()
{
    if (deviceReady)
        return true;

    wgpu::InstanceDescriptor instanceDesc = {};
    static constexpr wgpu::InstanceFeatureName requiredFeatures[] = {
        wgpu::InstanceFeatureName::TimedWaitAny,
    };
    instanceDesc.requiredFeatureCount = 1;
    instanceDesc.requiredFeatures = requiredFeatures;
    instance = wgpu::CreateInstance(&instanceDesc);
    if (instance == nullptr)
    {
        snprintf(status, sizeof(status), "WebGPU unavailable: failed to create Dawn instance");
        return false;
    }

    wgpu::RequestAdapterOptions adapterOptions = {};
    adapterOptions.backendType = wgpu::BackendType::D3D12;
    adapterOptions.powerPreference = wgpu::PowerPreference::HighPerformance;

    wgpu::RequestAdapterStatus adapterStatus = wgpu::RequestAdapterStatus::Error;
    std::string adapterMessage;
    wgpu::Future adapterFuture = instance.RequestAdapter(
        &adapterOptions, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestAdapterStatus requestStatus, wgpu::Adapter requestAdapter, wgpu::StringView message)
        {
            adapterStatus = requestStatus;
            adapterMessage = viewToString(message);
            if (requestStatus == wgpu::RequestAdapterStatus::Success)
                adapter = requestAdapter;
        });
    if (instance.WaitAny(adapterFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        adapterStatus != wgpu::RequestAdapterStatus::Success || adapter == nullptr)
    {
        snprintf(status, sizeof(status), "WebGPU unavailable: adapter request failed%s%s",
                 adapterMessage.empty() ? "" : ": ", adapterMessage.c_str());
        return false;
    }

    wgpu::DeviceDescriptor deviceDesc = {};
    deviceDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device &, wgpu::ErrorType, wgpu::StringView message, WebGpuDevice *self)
        {
            snprintf(self->status, sizeof(self->status), "WebGPU validation error: %s", viewToString(message).c_str());
        },
        this);
    deviceDesc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device &, wgpu::DeviceLostReason, wgpu::StringView message, WebGpuDevice *self)
        {
            self->deviceReady = false;
            snprintf(self->status, sizeof(self->status), "WebGPU device lost: %s", viewToString(message).c_str());
        },
        this);

    wgpu::RequestDeviceStatus deviceStatus = wgpu::RequestDeviceStatus::Error;
    std::string deviceMessage;
    wgpu::Future deviceFuture = adapter.RequestDevice(
        &deviceDesc, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestDeviceStatus requestStatus, wgpu::Device requestDevice, wgpu::StringView message)
        {
            deviceStatus = requestStatus;
            deviceMessage = viewToString(message);
            if (requestStatus == wgpu::RequestDeviceStatus::Success)
                device = requestDevice;
        });
    if (instance.WaitAny(deviceFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        deviceStatus != wgpu::RequestDeviceStatus::Success || device == nullptr)
    {
        snprintf(status, sizeof(status), "WebGPU unavailable: device request failed%s%s",
                 deviceMessage.empty() ? "" : ": ", deviceMessage.c_str());
        return false;
    }

    queue = device.GetQueue();
    if (queue == nullptr)
    {
        snprintf(status, sizeof(status), "WebGPU unavailable: failed to get device queue");
        return false;
    }

    wgpu::AdapterInfo adapterInfo = {};
    adapter.GetInfo(&adapterInfo);
    std::string deviceName = viewToString(adapterInfo.device);
    if (deviceName.empty())
        deviceName = viewToString(adapterInfo.description);
    if (deviceName.empty())
        deviceName = "unknown adapter";

    deviceReady = true;
    snprintf(status, sizeof(status), "WebGPU ready: %s %s", backendTypeName(adapterInfo.backendType), deviceName.c_str());
    return true;
}

bool WebGpuDevice::runComputeSmokeTest()
{
    if (!deviceReady)
        return false;

    const char *shaderSource = R"(
@group(0) @binding(0) var<storage, read_write> data : array<u32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    if (id.x < arrayLength(&data)) {
        data[id.x] = data[id.x] + 10u;
    }
}
)";

    wgpu::ShaderSourceWGSL wgsl = {};
    wgsl.code = shaderSource;
    wgpu::ShaderModuleDescriptor shaderDesc = {};
    shaderDesc.nextInChain = &wgsl;
    wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
    if (shader == nullptr)
    {
        snprintf(status, sizeof(status), "WebGPU compute smoke failed: shader module");
        deviceReady = false;
        return false;
    }

    wgpu::ComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.compute.module = shader;
    pipelineDesc.compute.entryPoint = "main";
    wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&pipelineDesc);
    if (pipeline == nullptr)
    {
        snprintf(status, sizeof(status), "WebGPU compute smoke failed: pipeline");
        deviceReady = false;
        return false;
    }

    const uint32_t input[4] = {1, 2, 3, 4};
    wgpu::BufferDescriptor storageDesc = {};
    storageDesc.size = sizeof(input);
    storageDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer storage = device.CreateBuffer(&storageDesc);

    wgpu::BufferDescriptor readbackDesc = {};
    readbackDesc.size = sizeof(input);
    readbackDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer readback = device.CreateBuffer(&readbackDesc);

    if (storage == nullptr || readback == nullptr)
    {
        snprintf(status, sizeof(status), "WebGPU compute smoke failed: buffers");
        deviceReady = false;
        return false;
    }

    queue.WriteBuffer(storage, 0, input, sizeof(input));

    wgpu::BindGroupEntry entry = {};
    entry.binding = 0;
    entry.buffer = storage;
    entry.size = sizeof(input);
    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = pipeline.GetBindGroupLayout(0);
    bindGroupDesc.entryCount = 1;
    bindGroupDesc.entries = &entry;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups(1);
    pass.End();
    encoder.CopyBufferToBuffer(storage, 0, readback, 0, sizeof(input));
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = readback.MapAsync(
        wgpu::MapMode::Read, 0, sizeof(input), wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus mapped, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = mapped;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
    {
        snprintf(status, sizeof(status), "WebGPU compute smoke failed: readback map");
        deviceReady = false;
        return false;
    }

    const uint32_t *result = (const uint32_t *)readback.GetConstMappedRange(0, sizeof(input));
    bool ok = result != nullptr;
    for (int i = 0; ok && i < 4; ++i)
        ok = result[i] == input[i] + 10;
    readback.Unmap();

    if (!ok)
    {
        snprintf(status, sizeof(status), "WebGPU compute smoke failed: wrong results");
        deviceReady = false;
        return false;
    }
    return true;
}

#else // !(AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN)

bool WebGpuDevice::initialize()
{
    snprintf(status, sizeof(status), "WebGPU disabled at build time");
    return false;
}

bool WebGpuDevice::runComputeSmokeTest()
{
    return false;
}

#endif
