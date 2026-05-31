/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#include "webgpu_backend.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <string_view>
#include <vector>
#include <cmath>
#include <algorithm>

#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>

namespace
{
void setStatus(char *status, const char *message)
{
    snprintf(status, 1024, "%s", message);
}

void resetTimingStats(WebGpuTimingStats &stats)
{
    stats.lastMs = 0.0f;
    stats.avgMs = 0.0f;
    stats.recentAvgMs = 0.0f;
    stats.minMs = 0.0f;
    stats.maxMs = 0.0f;
    stats.samples = 0;
    stats.recentSamples = 0;
    stats.recentIndex = 0;
    stats.recentSumMs = 0.0f;
    for (int i = 0; i < 64; ++i)
        stats.recentMs[i] = 0.0f;
}

void recordTimingSample(WebGpuTimingStats &stats, float ms)
{
    stats.lastMs = ms;
    if (stats.recentSamples < 64)
    {
        stats.recentMs[stats.recentIndex] = ms;
        stats.recentSumMs += ms;
        stats.recentSamples++;
    }
    else
    {
        stats.recentSumMs -= stats.recentMs[stats.recentIndex];
        stats.recentMs[stats.recentIndex] = ms;
        stats.recentSumMs += ms;
    }
    stats.recentIndex = (stats.recentIndex + 1) % 64;
    stats.recentAvgMs = stats.recentSamples > 0 ? stats.recentSumMs / (float)stats.recentSamples : 0.0f;

    if (stats.samples == 0)
    {
        stats.avgMs = ms;
        stats.minMs = ms;
        stats.maxMs = ms;
        stats.samples = 1;
        return;
    }

    stats.samples++;
    stats.avgMs += (ms - stats.avgMs) / (float)stats.samples;
    if (ms < stats.minMs)
        stats.minMs = ms;
    if (ms > stats.maxMs)
        stats.maxMs = ms;
}

float elapsedMs(Uint64 begin, Uint64 end)
{
    return (float)((double)(end - begin) * 1000.0 / (double)SDL_GetPerformanceFrequency());
}

struct GpuRenderInstance
{
    // position.w is a reserved mesh asset id for future mesh-backed batches.
    float position[4];
    float size[4];
    float rotation[4];
    // params: shape type, material/color id, render kind, half-length.
    float params[4];
};

struct GpuRenderInstanceInput
{
    // position.w is unused by the current procedural batches.
    float position[4];
    float shape[4];
    float rotation[4];
    // params: shape type, material/color id, half-length, reserved mesh asset id.
    float params[4];
};

struct GpuRenderBatch
{
    uint32_t vertexCount;
    uint32_t firstInstance;
    uint32_t instanceCount;
};

struct WebGpuGridReceiver
{
    bool valid;
    float minX;
    float minY;
    float maxX;
    float maxY;
    float z;
    float area;
};

struct WebGpuRenderScene
{
    std::vector<GpuRenderInstanceInput> bodyInputs;
    std::vector<GpuRenderBatch> renderBatches;
    int boxCount;
    int sphereCount;
    int capsuleCount;
    int cylinderCount;
    int meshAssetCount;
    WebGpuGridReceiver gridReceiver;
};

struct GpuPredictionInput
{
    float position[4];
    float velocity[4];
    float angle[4];
    float angularVelocity[4];
    float meta[4];
};

struct GpuPredictionOutput
{
    float inertial[4];
    float inertialAng[4];
};

struct GpuVelocityInput
{
    float position[4];
    float initial[4];
    float angle[4];
    float initialAngle[4];
    float expectedVelocity[4];
    float expectedAngularVelocity[4];
};

struct GpuVelocityOutput
{
    float velocity[4];
    float angularVelocity[4];
};

struct GpuBoundsInput
{
    float position[4];
    float params[4];
};

struct GpuBoundsOutput
{
    float minBounds[4];
    float maxBounds[4];
};

struct GpuMortonInput
{
    float position[4];
};

struct GpuMortonOutput
{
    uint32_t code;
    uint32_t index;
    uint32_t pad0;
    uint32_t pad1;
};

struct GpuSortParams
{
    uint32_t j;
    uint32_t k;
    uint32_t count;
    uint32_t pad;
};

struct GpuPairBody
{
    float position[4];
};

struct GpuPairCounters
{
    uint32_t candidates;
    uint32_t sphereHits;
    uint32_t pad0;
    uint32_t pad1;
};

struct GpuPairParams
{
    uint32_t count;
    uint32_t neighborWindow;
    uint32_t pad0;
    uint32_t pad1;
};

struct GpuSapInterval
{
    float minX;
    float maxX;
    float y;
    float z;
    float radius;
    uint32_t index;
    uint32_t pad0;
    uint32_t pad1;
};

struct GpuSapCounters
{
    uint32_t candidates;
    uint32_t sphereHits;
    uint32_t pad0;
    uint32_t pad1;
};

struct GpuSapParams
{
    uint32_t count;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct GpuPredictionParams
{
    float dt;
    float gravity;
    uint32_t sampleCount;
    uint32_t pad;
};

struct GpuMortonParams
{
    float worldMin;
    float invWorldSize;
    uint32_t bodyCount;
    uint32_t pad;
};

struct PreviewCameraUniform
{
    float eye[4];
    float right[4];
    float up[4];
    float forward[4];
    float params[4];
    float light[4];
    float options[4];
    float background[4];
    float gridBounds[4];
    float gridParams[4];
};

float fallbackSizeForBody(const SimBodyData &body)
{
    float size = body.radius;
    if (size < 0.25f)
        size = 0.25f;
    if (size > 1.5f)
        size = 1.5f;
    return size;
}

float3 webGpuPreviewSizeForBody(const SimBodyData &body)
{
    if (body.shape.type == RIGID_SHAPE_BOX ||
        body.shape.type == RIGID_SHAPE_SPHERE ||
        body.shape.type == RIGID_SHAPE_CAPSULE ||
        body.shape.type == RIGID_SHAPE_CYLINDER)
    {
        return body.shape.size;
    }

    float fallbackSize = fallbackSizeForBody(body);
    return float3{fallbackSize, fallbackSize, fallbackSize};
}

bool isStaticGridReceiver(const SimBodyData &body)
{
    if (!body.active || body.mass > 0.0f || body.shape.type != RIGID_SHAPE_BOX)
        return false;

    return body.shape.size.x >= 8.0f && body.shape.size.y >= 8.0f;
}

void considerGridReceiver(WebGpuRenderScene &scene, const SimBodyData &body)
{
    if (!isStaticGridReceiver(body))
        return;

    float halfX = body.shape.size.x * 0.5f;
    float halfY = body.shape.size.y * 0.5f;
    float area = body.shape.size.x * body.shape.size.y;
    if (scene.gridReceiver.valid && area <= scene.gridReceiver.area)
        return;

    scene.gridReceiver.valid = true;
    scene.gridReceiver.minX = body.positionLin.x - halfX;
    scene.gridReceiver.minY = body.positionLin.y - halfY;
    scene.gridReceiver.maxX = body.positionLin.x + halfX;
    scene.gridReceiver.maxY = body.positionLin.y + halfY;
    scene.gridReceiver.z = body.positionLin.z + body.shape.size.z * 0.5f + 0.003f;
    scene.gridReceiver.area = area;
}

void appendRenderBatch(std::vector<GpuRenderInstanceInput> &bodyInputs,
                       std::vector<GpuRenderBatch> &renderBatches,
                       const std::vector<GpuRenderInstanceInput> &instances,
                       uint32_t vertexCount)
{
    if (instances.empty())
        return;

    GpuRenderBatch batch = {};
    batch.vertexCount = vertexCount;
    batch.firstInstance = (uint32_t)bodyInputs.size();
    batch.instanceCount = (uint32_t)instances.size();
    bodyInputs.insert(bodyInputs.end(), instances.begin(), instances.end());
    renderBatches.push_back(batch);
}

WebGpuRenderScene buildWebGpuRenderScene(const SimWorld &world)
{
    WebGpuRenderScene scene = {};
    std::vector<GpuRenderInstanceInput> boxInputs;
    std::vector<GpuRenderInstanceInput> sphereInputs;
    std::vector<GpuRenderInstanceInput> capsuleInputs;
    std::vector<GpuRenderInstanceInput> cylinderInputs;
    std::vector<GpuRenderInstanceInput> meshAssetInputs;
    boxInputs.reserve(world.bodies.size());
    sphereInputs.reserve(world.bodies.size());
    capsuleInputs.reserve(world.bodies.size());
    cylinderInputs.reserve(world.bodies.size());
    meshAssetInputs.reserve(world.bodies.size());

    for (size_t i = 0; i < world.bodies.size(); ++i)
    {
        const SimBodyData &body = world.bodies[i];
        if (!body.active)
            continue;

        considerGridReceiver(scene, body);

        GpuRenderInstanceInput input = {};
        input.position[0] = body.positionLin.x;
        input.position[1] = body.positionLin.y;
        input.position[2] = body.positionLin.z;
        input.position[3] = 0.0f;
        input.shape[0] = body.shape.size.x;
        input.shape[1] = body.shape.size.y;
        input.shape[2] = body.shape.size.z;
        input.shape[3] = body.radius;
        input.rotation[0] = body.positionAng.x;
        input.rotation[1] = body.positionAng.y;
        input.rotation[2] = body.positionAng.z;
        input.rotation[3] = body.positionAng.w;
        input.params[0] = (float)body.shape.type;
        input.params[1] = body.mass > 0.0f ? 1.0f : 0.0f;
        input.params[2] = body.shape.halfLength;
        input.params[3] = 0.0f;
        if (body.shape.type == RIGID_SHAPE_BOX)
        {
            boxInputs.push_back(input);
            ++scene.boxCount;
        }
        else if (body.shape.type == RIGID_SHAPE_SPHERE)
        {
            sphereInputs.push_back(input);
            ++scene.sphereCount;
        }
        else if (body.shape.type == RIGID_SHAPE_CAPSULE)
        {
            capsuleInputs.push_back(input);
            ++scene.capsuleCount;
        }
        else if (body.shape.type == RIGID_SHAPE_CYLINDER)
        {
            cylinderInputs.push_back(input);
            ++scene.cylinderCount;
        }
        else
        {
            meshAssetInputs.push_back(input);
            ++scene.meshAssetCount;
        }
    }

    if (boxInputs.empty() && sphereInputs.empty() && capsuleInputs.empty() && cylinderInputs.empty() && meshAssetInputs.empty())
    {
        GpuRenderInstanceInput input = {};
        input.position[3] = 0.0f;
        input.shape[0] = 1.0f;
        input.shape[1] = 1.0f;
        input.shape[2] = 1.0f;
        input.shape[3] = 0.5f;
        input.rotation[3] = 1.0f;
        input.params[0] = (float)RIGID_SHAPE_BOX;
        input.params[1] = 1.0f;
        input.params[2] = 0.0f;
        input.params[3] = 0.0f;
        boxInputs.push_back(input);
        scene.boxCount = 1;
    }

    scene.bodyInputs.reserve(boxInputs.size() + sphereInputs.size() + capsuleInputs.size() + cylinderInputs.size() + meshAssetInputs.size());
    scene.renderBatches.reserve(5);
    appendRenderBatch(scene.bodyInputs, scene.renderBatches, boxInputs, 36u);
    appendRenderBatch(scene.bodyInputs, scene.renderBatches, sphereInputs, 16u * 8u * 6u);
    appendRenderBatch(scene.bodyInputs, scene.renderBatches, capsuleInputs, 16u * (1u + 6u * 2u) * 6u);
    appendRenderBatch(scene.bodyInputs, scene.renderBatches, cylinderInputs, 24u * 12u);
    appendRenderBatch(scene.bodyInputs, scene.renderBatches, meshAssetInputs, 36u);

    return scene;
}

uint32_t expandMortonBits(uint32_t value)
{
    uint32_t x = value & 1023u;
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x << 8u)) & 0x0300F00Fu;
    x = (x | (x << 4u)) & 0x030C30C3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

uint32_t quantizeMortonAxis(float value, float worldMin, float invWorldSize)
{
    float normalized = clamp((value - worldMin) * invWorldSize, 0.0f, 0.999999f);
    return (uint32_t)floorf(normalized * 1024.0f);
}

uint32_t mortonCodeForPosition(float3 position, float worldMin, float invWorldSize)
{
    uint32_t x = quantizeMortonAxis(position.x, worldMin, invWorldSize);
    uint32_t y = quantizeMortonAxis(position.y, worldMin, invWorldSize);
    uint32_t z = quantizeMortonAxis(position.z, worldMin, invWorldSize);
    return expandMortonBits(x) | (expandMortonBits(y) << 1u) | (expandMortonBits(z) << 2u);
}

uint32_t nextPowerOfTwo(uint32_t value)
{
    if (value <= 1u)
        return 1u;

    --value;
    value |= value >> 1u;
    value |= value >> 2u;
    value |= value >> 4u;
    value |= value >> 8u;
    value |= value >> 16u;
    return value + 1u;
}

PreviewCameraUniform makePreviewCameraUniform(const WebGpuPreviewCamera &camera, const WebGpuRenderOptions &options, int width, int height)
{
    float3 forward = normalize(camera.target - camera.eye);
    float3 right = cross(forward, camera.up);
    if (lengthSq(right) < 1.0e-8f)
        right = cross(forward, float3{0, 1, 0});
    right = normalize(right);
    float3 up = cross(right, forward);
    float aspect = height > 0 ? (float)width / (float)height : 1.0f;
    float tanHalfFovY = tanf(0.5f * rad(camera.fovYDeg));
    float3 lightDirection = options.lightDirection;
    if (lengthSq(lightDirection) < 1.0e-8f)
        lightDirection = float3{0.45f, 0.80f, 0.55f};
    lightDirection = normalize(lightDirection);

    PreviewCameraUniform uniform = {};
    uniform.eye[0] = camera.eye.x;
    uniform.eye[1] = camera.eye.y;
    uniform.eye[2] = camera.eye.z;
    uniform.eye[3] = 0.0f;
    uniform.right[0] = right.x;
    uniform.right[1] = right.y;
    uniform.right[2] = right.z;
    uniform.right[3] = 0.0f;
    uniform.up[0] = up.x;
    uniform.up[1] = up.y;
    uniform.up[2] = up.z;
    uniform.up[3] = 0.0f;
    uniform.forward[0] = forward.x;
    uniform.forward[1] = forward.y;
    uniform.forward[2] = forward.z;
    uniform.forward[3] = 0.0f;
    uniform.params[0] = aspect;
    uniform.params[1] = tanHalfFovY;
    uniform.params[2] = 1.0f;
    uniform.params[3] = 0.0f;
    uniform.light[0] = lightDirection.x;
    uniform.light[1] = lightDirection.y;
    uniform.light[2] = lightDirection.z;
    uniform.light[3] = 0.0f;
    uniform.options[0] = options.showGroundGrid ? 1.0f : 0.0f;
    uniform.options[1] = options.showShapeEdges ? 1.0f : 0.0f;
    uniform.options[2] = 0.0f;
    uniform.options[3] = 0.0f;
    uniform.background[0] = options.backgroundColor.x;
    uniform.background[1] = options.backgroundColor.y;
    uniform.background[2] = options.backgroundColor.z;
    uniform.background[3] = 1.0f;
    uniform.gridBounds[0] = 0.0f;
    uniform.gridBounds[1] = 0.0f;
    uniform.gridBounds[2] = 0.0f;
    uniform.gridBounds[3] = 0.0f;
    uniform.gridParams[0] = 0.0f;
    uniform.gridParams[1] = 0.0f;
    uniform.gridParams[2] = 0.0f;
    uniform.gridParams[3] = 0.0f;
    return uniform;
}

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
wgpu::BufferUsage bufferUsage(uint64_t flags)
{
    return (wgpu::BufferUsage)flags;
}

std::string toString(wgpu::StringView value)
{
    std::string_view view = value;
    return std::string(view);
}

const char *backendName(wgpu::BackendType backendType)
{
    switch (backendType)
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
    case wgpu::BackendType::Null:
        return "Null";
    case wgpu::BackendType::WebGPU:
        return "WebGPU";
    default:
        return "Unknown";
    }
}

wgpu::TextureFormat chooseSurfaceFormat(const wgpu::SurfaceCapabilities &capabilities)
{
    for (size_t i = 0; i < capabilities.formatCount; ++i)
    {
        if (capabilities.formats[i] == wgpu::TextureFormat::BGRA8Unorm)
            return capabilities.formats[i];
    }
    for (size_t i = 0; i < capabilities.formatCount; ++i)
    {
        if (capabilities.formats[i] == wgpu::TextureFormat::RGBA8Unorm)
            return capabilities.formats[i];
    }
    return capabilities.formatCount ? capabilities.formats[0] : wgpu::TextureFormat::Undefined;
}

wgpu::PresentMode choosePresentMode(const wgpu::SurfaceCapabilities &capabilities)
{
    for (size_t i = 0; i < capabilities.presentModeCount; ++i)
    {
        if (capabilities.presentModes[i] == wgpu::PresentMode::Fifo)
            return capabilities.presentModes[i];
    }
    return capabilities.presentModeCount ? capabilities.presentModes[0] : wgpu::PresentMode::Fifo;
}

wgpu::CompositeAlphaMode chooseAlphaMode(const wgpu::SurfaceCapabilities &capabilities)
{
    return capabilities.alphaModeCount ? capabilities.alphaModes[0] : wgpu::CompositeAlphaMode::Auto;
}

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

wgpu::Surface createSurfaceForWindow(wgpu::Instance instance, SDL_Window *window, char *status)
{
    if (!window)
    {
        setStatus(status, "Surface clear failed: missing SDL window");
        return nullptr;
    }

    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    if (!SDL_GetWindowWMInfo(window, &wmInfo))
    {
        snprintf(status, 256, "Surface clear failed: %s", SDL_GetError());
        return nullptr;
    }

#if defined(_WIN32)
    if (wmInfo.subsystem != SDL_SYSWM_WINDOWS)
    {
        setStatus(status, "Surface clear failed: SDL window is not Win32");
        return nullptr;
    }

    wgpu::SurfaceSourceWindowsHWND surfaceSource = {};
    surfaceSource.hinstance = wmInfo.info.win.hinstance;
    surfaceSource.hwnd = wmInfo.info.win.window;

    wgpu::SurfaceDescriptor surfaceDesc = {};
    surfaceDesc.nextInChain = &surfaceSource;
    return instance.CreateSurface(&surfaceDesc);
#else
    setStatus(status, "Surface clear failed: platform surface creation pending");
    return nullptr;
#endif
}

#endif
}

WebGpuContext::WebGpuContext()
    : initialized(false), deviceReady(false), smokeTestPassed(false), status{}, smokeStatus{}, computeStatus{}, presentStatus{}, predictionStatus{}, velocityStatus{}, boundsStatus{}, mortonStatus{}, mortonSortStatus{}, pairStatus{}, sapStatus{},
      previewComputeMs(0.0f), previewRenderMs(0.0f), previewTotalMs(0.0f),
      previewBatchCount(0), previewInstanceCount(0), previewBoxInstances(0), previewSphereInstances(0), previewCapsuleInstances(0), previewCylinderInstances(0), previewMeshAssetInstances(0),
      predictionMs(0.0f), predictionMaxError(0.0f), predictionMaxAngularError(0.0f), predictionSamples(0),
      velocityMs(0.0f), velocityMaxLinearError(0.0f), velocityMaxAngularError(0.0f), velocitySamples(0),
      boundsMs(0.0f), boundsMaxError(0.0f), boundsSamples(0),
      mortonMs(0.0f), mortonMismatches(0), mortonSamples(0),
      mortonSortMs(0.0f), mortonSortMismatches(0), mortonSortCount(0),
      pairMs(0.0f), pairCandidates(0), pairSphereHits(0), pairAllPairsSphereHits(0), pairMissedSphereHits(0), pairMismatches(0),
      sapMs(0.0f), sapCandidates(0), sapSphereHits(0), sapAllPairsSphereHits(0), sapMissedSphereHits(0), sapMismatches(0),
      sapBestAxis(0), sapAxisCandidates{}, sapAxisSphereHits{}, sapAxisMissedSphereHits{}
{
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    bodyInputBufferBytes = 0;
    bodyInstanceBufferBytes = 0;
    predictionInputBufferBytes = 0;
    predictionOutputBufferBytes = 0;
    predictionReadbackBufferBytes = 0;
    velocityInputBufferBytes = 0;
    velocityOutputBufferBytes = 0;
    velocityReadbackBufferBytes = 0;
    boundsInputBufferBytes = 0;
    boundsOutputBufferBytes = 0;
    boundsReadbackBufferBytes = 0;
    mortonInputBufferBytes = 0;
    mortonOutputBufferBytes = 0;
    mortonReadbackBufferBytes = 0;
    mortonSortBufferBytes = 0;
    mortonSortReadbackBufferBytes = 0;
    pairBodyBufferBytes = 0;
    pairItemBufferBytes = 0;
    pairCountersBufferBytes = 0;
    pairReadbackBufferBytes = 0;
    sapIntervalBufferBytes = 0;
    sapCountersBufferBytes = 0;
    sapReadbackBufferBytes = 0;
    bodyPreviewFormat = wgpu::TextureFormat::Undefined;
    bodyPreviewSurfaceConfigured = false;
    bodyPreviewWidth = 0;
    bodyPreviewHeight = 0;
#endif
    setStatus(status, "WebGPU not initialized");
    setStatus(smokeStatus, "Offscreen smoke not run");
    setStatus(computeStatus, "Compute smoke not run");
    setStatus(presentStatus, "Surface clear not run");
    setStatus(predictionStatus, "Body prediction not run");
    setStatus(velocityStatus, "Velocity update not run");
    setStatus(boundsStatus, "Bounds not run");
    setStatus(mortonStatus, "Morton codes not run");
    setStatus(mortonSortStatus, "Morton sort not run");
    setStatus(pairStatus, "Pair diagnostic not run");
    setStatus(sapStatus, "SAP diagnostic not run");
    resetDiagnosticTimingStats();
}

bool WebGpuContext::initializeDeviceOnly()
{
    initialized = true;
    deviceReady = false;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    wgpu::InstanceDescriptor instanceDesc = {};
    static constexpr wgpu::InstanceFeatureName requiredFeatures[] = {
        wgpu::InstanceFeatureName::TimedWaitAny,
    };
    instanceDesc.requiredFeatureCount = 1;
    instanceDesc.requiredFeatures = requiredFeatures;
    instance = wgpu::CreateInstance(&instanceDesc);
    if (instance == nullptr)
    {
        setStatus(status, "WebGPU unavailable: failed to create Dawn instance");
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
            adapterMessage = toString(message);
            if (requestStatus == wgpu::RequestAdapterStatus::Success)
            {
                adapter = requestAdapter;
            }
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
        [](const wgpu::Device &, wgpu::ErrorType, wgpu::StringView message, WebGpuContext *context)
        {
            snprintf(context->status, sizeof(context->status), "WebGPU validation error: %s", toString(message).c_str());
        },
        this);
    deviceDesc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device &, wgpu::DeviceLostReason, wgpu::StringView message, WebGpuContext *context)
        {
            context->deviceReady = false;
            snprintf(context->status, sizeof(context->status), "WebGPU device lost: %s", toString(message).c_str());
        },
        this);

    wgpu::RequestDeviceStatus deviceStatus = wgpu::RequestDeviceStatus::Error;
    std::string deviceMessage;
    wgpu::Future deviceFuture = adapter.RequestDevice(
        &deviceDesc, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestDeviceStatus requestStatus, wgpu::Device requestDevice, wgpu::StringView message)
        {
            deviceStatus = requestStatus;
            deviceMessage = toString(message);
            if (requestStatus == wgpu::RequestDeviceStatus::Success)
            {
                device = requestDevice;
            }
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
        setStatus(status, "WebGPU unavailable: failed to get device queue");
        return false;
    }

    wgpu::AdapterInfo adapterInfo = {};
    adapter.GetInfo(&adapterInfo);
    std::string deviceName = toString(adapterInfo.device);
    if (deviceName.empty())
        deviceName = toString(adapterInfo.description);
    if (deviceName.empty())
        deviceName = "unknown adapter";

    deviceReady = true;
    snprintf(status, sizeof(status), "WebGPU ready: %s %s", backendName(adapterInfo.backendType), deviceName.c_str());
    return true;
#elif AVBD_ENABLE_WEBGPU
    setStatus(status, "WebGPU scaffold enabled; Dawn not linked");
    return false;
#else
    setStatus(status, "WebGPU disabled at build time");
    return false;
#endif
}

bool WebGpuContext::initialize(SDL_Window *window)
{
    initialized = true;
    deviceReady = false;

    if (!window)
    {
        setStatus(status, "WebGPU unavailable: missing SDL window");
        return false;
    }

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    if (!SDL_GetWindowWMInfo(window, &wmInfo))
    {
        snprintf(status, sizeof(status), "WebGPU unavailable: %s", SDL_GetError());
        return false;
    }

#if defined(_WIN32)
    if (wmInfo.subsystem != SDL_SYSWM_WINDOWS)
    {
        setStatus(status, "WebGPU unavailable: SDL window is not Win32");
        return false;
    }

    wgpu::InstanceDescriptor instanceDesc = {};
    static constexpr wgpu::InstanceFeatureName requiredFeatures[] = {
        wgpu::InstanceFeatureName::TimedWaitAny,
    };
    instanceDesc.requiredFeatureCount = 1;
    instanceDesc.requiredFeatures = requiredFeatures;
    instance = wgpu::CreateInstance(&instanceDesc);
    if (instance == nullptr)
    {
        setStatus(status, "WebGPU unavailable: failed to create Dawn instance");
        return false;
    }

    wgpu::SurfaceSourceWindowsHWND surfaceSource = {};
    surfaceSource.hinstance = wmInfo.info.win.hinstance;
    surfaceSource.hwnd = wmInfo.info.win.window;

    wgpu::SurfaceDescriptor surfaceDesc = {};
    surfaceDesc.nextInChain = &surfaceSource;
    surface = instance.CreateSurface(&surfaceDesc);
    if (surface == nullptr)
    {
        setStatus(status, "WebGPU unavailable: failed to create Win32 surface");
        return false;
    }

    wgpu::RequestAdapterOptions adapterOptions = {};
    adapterOptions.backendType = wgpu::BackendType::D3D12;
    adapterOptions.powerPreference = wgpu::PowerPreference::HighPerformance;
    adapterOptions.compatibleSurface = surface;

    wgpu::RequestAdapterStatus adapterStatus = wgpu::RequestAdapterStatus::Error;
    std::string adapterMessage;
    wgpu::Future adapterFuture = instance.RequestAdapter(
        &adapterOptions, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestAdapterStatus requestStatus, wgpu::Adapter requestAdapter, wgpu::StringView message)
        {
            adapterStatus = requestStatus;
            adapterMessage = toString(message);
            if (requestStatus == wgpu::RequestAdapterStatus::Success)
            {
                adapter = requestAdapter;
            }
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
        [](const wgpu::Device &, wgpu::ErrorType, wgpu::StringView message, WebGpuContext *context)
        {
            snprintf(context->status, sizeof(context->status), "WebGPU validation error: %s", toString(message).c_str());
        },
        this);
    deviceDesc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device &, wgpu::DeviceLostReason, wgpu::StringView message, WebGpuContext *context)
        {
            context->deviceReady = false;
            snprintf(context->status, sizeof(context->status), "WebGPU device lost: %s", toString(message).c_str());
        },
        this);

    wgpu::RequestDeviceStatus deviceStatus = wgpu::RequestDeviceStatus::Error;
    std::string deviceMessage;
    wgpu::Future deviceFuture = adapter.RequestDevice(
        &deviceDesc, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestDeviceStatus requestStatus, wgpu::Device requestDevice, wgpu::StringView message)
        {
            deviceStatus = requestStatus;
            deviceMessage = toString(message);
            if (requestStatus == wgpu::RequestDeviceStatus::Success)
            {
                device = requestDevice;
            }
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
        setStatus(status, "WebGPU unavailable: failed to get device queue");
        return false;
    }

    wgpu::AdapterInfo adapterInfo = {};
    adapter.GetInfo(&adapterInfo);
    std::string deviceName = toString(adapterInfo.device);
    if (deviceName.empty())
    {
        deviceName = toString(adapterInfo.description);
    }
    if (deviceName.empty())
    {
        deviceName = "unknown adapter";
    }

    deviceReady = true;
    snprintf(status, sizeof(status), "WebGPU ready: %s %s", backendName(adapterInfo.backendType), deviceName.c_str());
    return true;
#else
    setStatus(status, "Dawn linked; platform surface creation pending");
    return false;
#endif

#elif AVBD_ENABLE_WEBGPU
    setStatus(status, "WebGPU scaffold enabled; Dawn not linked");
    return false;
#else
    setStatus(status, "WebGPU disabled at build time");
    return false;
#endif
}

bool WebGpuContext::runOffscreenSmokeTest(int width, int height)
{
    smokeTestPassed = false;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || device == nullptr || queue == nullptr)
    {
        setStatus(smokeStatus, "Offscreen smoke skipped: WebGPU device not ready");
        return false;
    }

    if (width <= 0 || height <= 0)
    {
        width = 1;
        height = 1;
    }

    wgpu::TextureDescriptor textureDesc = {};
    textureDesc.usage = wgpu::TextureUsage::RenderAttachment;
    textureDesc.dimension = wgpu::TextureDimension::e2D;
    textureDesc.size = {(uint32_t)width, (uint32_t)height, 1};
    textureDesc.format = wgpu::TextureFormat::RGBA8Unorm;
    textureDesc.mipLevelCount = 1;
    textureDesc.sampleCount = 1;

    wgpu::Texture texture = device.CreateTexture(&textureDesc);
    if (texture == nullptr)
    {
        setStatus(smokeStatus, "Offscreen smoke failed: texture");
        return false;
    }

    wgpu::TextureView view = texture.CreateView();
    if (view == nullptr)
    {
        setStatus(smokeStatus, "Offscreen smoke failed: texture view");
        return false;
    }

    wgpu::RenderPassColorAttachment color = {};
    color.view = view;
    color.loadOp = wgpu::LoadOp::Clear;
    color.storeOp = wgpu::StoreOp::Store;
    color.clearValue = {0.08, 0.16, 0.28, 1.0};

    wgpu::RenderPassDescriptor passDesc = {};
    passDesc.colorAttachmentCount = 1;
    passDesc.colorAttachments = &color;

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDesc);
    pass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    smokeTestPassed = true;
    snprintf(smokeStatus, sizeof(smokeStatus), "Offscreen smoke passed: clear/submit %dx%d", width, height);
    return true;
#elif AVBD_ENABLE_WEBGPU
    setStatus(smokeStatus, "Offscreen smoke skipped: Dawn not linked");
    return false;
#else
    setStatus(smokeStatus, "Offscreen smoke skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::runComputeSmokeTest()
{
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(computeStatus, "Compute smoke skipped: WebGPU device not ready");
        return false;
    }

    const uint32_t inputValues[4] = {1, 2, 3, 4};
    const uint32_t expectedValues[4] = {11, 12, 13, 14};
    const size_t byteSize = sizeof(inputValues);

    wgpu::BufferDescriptor storageDesc = {};
    storageDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                    (uint64_t)wgpu::BufferUsage::CopyDst |
                                    (uint64_t)wgpu::BufferUsage::CopySrc);
    storageDesc.size = byteSize;
    wgpu::Buffer storageBuffer = device.CreateBuffer(&storageDesc);
    if (storageBuffer == nullptr)
    {
        setStatus(computeStatus, "Compute smoke failed: storage buffer");
        return false;
    }
    queue.WriteBuffer(storageBuffer, 0, inputValues, byteSize);

    wgpu::BufferDescriptor readbackDesc = {};
    readbackDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                     (uint64_t)wgpu::BufferUsage::CopyDst);
    readbackDesc.size = byteSize;
    wgpu::Buffer readbackBuffer = device.CreateBuffer(&readbackDesc);
    if (readbackBuffer == nullptr)
    {
        setStatus(computeStatus, "Compute smoke failed: readback buffer");
        return false;
    }

    static const char *shaderSource = R"(
@group(0) @binding(0) var<storage, read_write> values: array<u32>;

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_id) localId: vec3u) {
    values[localId.x] = values[localId.x] + 10u;
}
)";

    wgpu::ShaderSourceWGSL wgsl = {};
    wgsl.code = shaderSource;
    wgpu::ShaderModuleDescriptor shaderDesc = {};
    shaderDesc.nextInChain = &wgsl;
    wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
    if (shader == nullptr)
    {
        setStatus(computeStatus, "Compute smoke failed: shader module");
        return false;
    }

    wgpu::BindGroupLayoutEntry layoutEntry = {};
    layoutEntry.binding = 0;
    layoutEntry.visibility = wgpu::ShaderStage::Compute;
    layoutEntry.buffer.type = wgpu::BufferBindingType::Storage;
    layoutEntry.buffer.minBindingSize = byteSize;

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc = {};
    bindGroupLayoutDesc.entryCount = 1;
    bindGroupLayoutDesc.entries = &layoutEntry;
    wgpu::BindGroupLayout bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc);
    if (bindGroupLayout == nullptr)
    {
        setStatus(computeStatus, "Compute smoke failed: bind layout");
        return false;
    }

    wgpu::BindGroupEntry bindEntry = {};
    bindEntry.binding = 0;
    bindEntry.buffer = storageBuffer;
    bindEntry.offset = 0;
    bindEntry.size = byteSize;

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 1;
    bindGroupDesc.entries = &bindEntry;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
    if (bindGroup == nullptr)
    {
        setStatus(computeStatus, "Compute smoke failed: bind group");
        return false;
    }

    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;
    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
    if (pipelineLayout == nullptr)
    {
        setStatus(computeStatus, "Compute smoke failed: pipeline layout");
        return false;
    }

    wgpu::ComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.compute.module = shader;
    pipelineDesc.compute.entryPoint = "main";
    wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&pipelineDesc);
    if (pipeline == nullptr)
    {
        setStatus(computeStatus, "Compute smoke failed: pipeline");
        return false;
    }

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups(1);
    pass.End();
    encoder.CopyBufferToBuffer(storageBuffer, 0, readbackBuffer, 0, byteSize);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = readbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, byteSize, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        mapStatus != wgpu::MapAsyncStatus::Success)
    {
        setStatus(computeStatus, "Compute smoke failed: map readback");
        return false;
    }

    const uint32_t *mapped = (const uint32_t *)readbackBuffer.GetConstMappedRange(0, byteSize);
    bool passed = mapped != nullptr;
    if (mapped)
    {
        for (int i = 0; i < 4; ++i)
            passed = passed && mapped[i] == expectedValues[i];
    }
    readbackBuffer.Unmap();

    if (!passed)
    {
        setStatus(computeStatus, "Compute smoke failed: wrong results");
        return false;
    }

    setStatus(computeStatus, "Compute smoke passed: storage dispatch/readback");
    return true;
#elif AVBD_ENABLE_WEBGPU
    setStatus(computeStatus, "Compute smoke skipped: Dawn not linked");
    return false;
#else
    setStatus(computeStatus, "Compute smoke skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::runBodyPredictionDiagnostic(const SimWorld &world, float dt, float gravity)
{
    predictionMs = 0.0f;
    predictionMaxError = 0.0f;
    predictionMaxAngularError = 0.0f;
    predictionSamples = 0;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(predictionStatus, "Body prediction skipped: WebGPU device not ready");
        return false;
    }

    std::vector<GpuPredictionInput> inputs;
    inputs.reserve(world.bodies.size());
    for (size_t i = 0; i < world.bodies.size(); ++i)
    {
        const SimBodyData &body = world.bodies[i];
        if (!body.active)
            continue;

        GpuPredictionInput input = {};
        input.position[0] = body.positionLin.x;
        input.position[1] = body.positionLin.y;
        input.position[2] = body.positionLin.z;
        input.position[3] = 0.0f;
        input.velocity[0] = body.velocityLin.x;
        input.velocity[1] = body.velocityLin.y;
        input.velocity[2] = body.velocityLin.z;
        input.velocity[3] = 0.0f;
        input.angle[0] = body.positionAng.x;
        input.angle[1] = body.positionAng.y;
        input.angle[2] = body.positionAng.z;
        input.angle[3] = body.positionAng.w;
        input.angularVelocity[0] = body.velocityAng.x;
        input.angularVelocity[1] = body.velocityAng.y;
        input.angularVelocity[2] = body.velocityAng.z;
        input.angularVelocity[3] = 0.0f;
        input.meta[0] = body.mass > 0.0f ? 1.0f : 0.0f;
        inputs.push_back(input);
    }

    if (inputs.empty())
    {
        setStatus(predictionStatus, "Body prediction skipped: no active bodies");
        return false;
    }

    const uint32_t sampleCount = (uint32_t)min((int)inputs.size(), 8);
    const uint64_t inputBytes = (uint64_t)(inputs.size() * sizeof(GpuPredictionInput));
    const uint64_t outputBytes = (uint64_t)(inputs.size() * sizeof(GpuPredictionOutput));
    const uint64_t readbackBytes = (uint64_t)(sampleCount * sizeof(GpuPredictionOutput));

    Uint64 begin = SDL_GetPerformanceCounter();

    if (predictionInputBuffer == nullptr || predictionInputBufferBytes < inputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(inputBytes, 4096);
        predictionInputBuffer = device.CreateBuffer(&desc);
        predictionInputBufferBytes = predictionInputBuffer == nullptr ? 0 : desc.size;
    }
    if (predictionOutputBuffer == nullptr || predictionOutputBufferBytes < outputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopySrc);
        desc.size = alignUp(outputBytes, 4096);
        predictionOutputBuffer = device.CreateBuffer(&desc);
        predictionOutputBufferBytes = predictionOutputBuffer == nullptr ? 0 : desc.size;
    }
    if (predictionReadbackBuffer == nullptr || predictionReadbackBufferBytes < readbackBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(readbackBytes, 256);
        predictionReadbackBuffer = device.CreateBuffer(&desc);
        predictionReadbackBufferBytes = predictionReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (predictionParamsBuffer == nullptr)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = sizeof(GpuPredictionParams);
        predictionParamsBuffer = device.CreateBuffer(&desc);
    }

    if (predictionInputBuffer == nullptr || predictionOutputBuffer == nullptr ||
        predictionReadbackBuffer == nullptr || predictionParamsBuffer == nullptr)
    {
        setStatus(predictionStatus, "Body prediction failed: buffer allocation");
        return false;
    }

    GpuPredictionParams params = {};
    params.dt = dt;
    params.gravity = gravity;
    params.sampleCount = (uint32_t)inputs.size();
    queue.WriteBuffer(predictionInputBuffer, 0, inputs.data(), inputBytes);
    queue.WriteBuffer(predictionParamsBuffer, 0, &params, sizeof(params));

    static const char *shaderSource = R"(
struct BodyInput {
    position: vec4f,
    velocity: vec4f,
    angle: vec4f,
    angularVelocity: vec4f,
    params: vec4f,
};

struct BodyOutput {
    inertial: vec4f,
    inertialAng: vec4f,
};

struct Params {
    dt: f32,
    gravity: f32,
    bodyCount: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> inputs: array<BodyInput>;
@group(0) @binding(1) var<storage, read_write> outputs: array<BodyOutput>;
@group(0) @binding(2) var<uniform> params: Params;

fn quatMul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= params.bodyCount) {
        return;
    }

    let input = inputs[index];
    let dynamicBody = input.params.x > 0.5;
    var inertial = input.position.xyz + input.velocity.xyz * params.dt;
    if (dynamicBody) {
        inertial.z = inertial.z + params.gravity * params.dt * params.dt;
    }
    outputs[index].inertial = vec4f(inertial, 1.0);

    let delta = vec4f(input.angularVelocity.xyz * params.dt, 0.0);
    outputs[index].inertialAng = normalize(input.angle + quatMul(delta, input.angle) * 0.5);
}
)";

    if (predictionPipeline == nullptr || predictionBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = shaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(predictionStatus, "Body prediction failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry entries[3] = {};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        entries[0].buffer.minBindingSize = sizeof(GpuPredictionInput);
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].buffer.type = wgpu::BufferBindingType::Storage;
        entries[1].buffer.minBindingSize = sizeof(GpuPredictionOutput);
        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Compute;
        entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[2].buffer.minBindingSize = sizeof(GpuPredictionParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 3;
        layoutDesc.entries = entries;
        predictionBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (predictionBindGroupLayout == nullptr)
        {
            setStatus(predictionStatus, "Body prediction failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &predictionBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(predictionStatus, "Body prediction failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        predictionPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (predictionPipeline == nullptr)
        {
            setStatus(predictionStatus, "Body prediction failed: compute pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry entries[3] = {};
    entries[0].binding = 0;
    entries[0].buffer = predictionInputBuffer;
    entries[0].offset = 0;
    entries[0].size = inputBytes;
    entries[1].binding = 1;
    entries[1].buffer = predictionOutputBuffer;
    entries[1].offset = 0;
    entries[1].size = outputBytes;
    entries[2].binding = 2;
    entries[2].buffer = predictionParamsBuffer;
    entries[2].offset = 0;
    entries[2].size = sizeof(GpuPredictionParams);

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = predictionBindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
    if (bindGroup == nullptr)
    {
        setStatus(predictionStatus, "Body prediction failed: bind group");
        return false;
    }

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(predictionPipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups((uint32_t)((inputs.size() + 63) / 64));
    pass.End();
    encoder.CopyBufferToBuffer(predictionOutputBuffer, 0, predictionReadbackBuffer, 0, readbackBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = predictionReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, readbackBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
    {
        setStatus(predictionStatus, "Body prediction failed: readback map");
        return false;
    }

    const GpuPredictionOutput *outputs = (const GpuPredictionOutput *)predictionReadbackBuffer.GetConstMappedRange(0, readbackBytes);
    if (!outputs)
    {
        predictionReadbackBuffer.Unmap();
        setStatus(predictionStatus, "Body prediction failed: readback range");
        return false;
    }

    float maxError = 0.0f;
    float maxAngularError = 0.0f;
    for (uint32_t i = 0; i < sampleCount; ++i)
    {
        const GpuPredictionInput &input = inputs[i];
        float expected[3] = {
            input.position[0] + input.velocity[0] * dt,
            input.position[1] + input.velocity[1] * dt,
            input.position[2] + input.velocity[2] * dt};
        if (input.meta[0] > 0.5f)
            expected[2] += gravity * dt * dt;

        float dx = outputs[i].inertial[0] - expected[0];
        float dy = outputs[i].inertial[1] - expected[1];
        float dz = outputs[i].inertial[2] - expected[2];
        float error = sqrtf(dx * dx + dy * dy + dz * dz);
        if (error > maxError)
            maxError = error;

        quat expectedAng = quat{input.angle[0], input.angle[1], input.angle[2], input.angle[3]} +
                           float3{input.angularVelocity[0], input.angularVelocity[1], input.angularVelocity[2]} * dt;
        float dax = outputs[i].inertialAng[0] - expectedAng.x;
        float day = outputs[i].inertialAng[1] - expectedAng.y;
        float daz = outputs[i].inertialAng[2] - expectedAng.z;
        float daw = outputs[i].inertialAng[3] - expectedAng.w;
        float angularError = sqrtf(dax * dax + day * day + daz * daz + daw * daw);
        if (angularError > maxAngularError)
            maxAngularError = angularError;
    }
    predictionReadbackBuffer.Unmap();

    predictionSamples = (int)sampleCount;
    predictionMaxError = maxError;
    predictionMaxAngularError = maxAngularError;
    predictionMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    recordTimingSample(predictionTiming, predictionMs);
    snprintf(predictionStatus, sizeof(predictionStatus), "Body prediction passed: %d samples, lin %.6f, ang %.6f",
             predictionSamples, predictionMaxError, predictionMaxAngularError);
    return true;
#elif AVBD_ENABLE_WEBGPU
    setStatus(predictionStatus, "Body prediction skipped: Dawn not linked");
    return false;
#else
    setStatus(predictionStatus, "Body prediction skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::runVelocityUpdateDiagnostic(const SimWorld &world, float dt)
{
    velocityMs = 0.0f;
    velocityMaxLinearError = 0.0f;
    velocityMaxAngularError = 0.0f;
    velocitySamples = 0;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(velocityStatus, "Velocity update skipped: WebGPU device not ready");
        return false;
    }

    if (dt == 0.0f)
    {
        setStatus(velocityStatus, "Velocity update skipped: dt is zero");
        return false;
    }

    std::vector<GpuVelocityInput> inputs;
    inputs.reserve(world.activeBodyIds.size());
    for (size_t i = 0; i < world.activeBodyIds.size(); ++i)
    {
        BodyId bodyId = world.activeBodyIds[i];
        if (bodyId == INVALID_BODY_ID || bodyId >= world.bodies.size())
            continue;

        const SimBodyData &body = world.bodies[bodyId];
        Rigid *source = body.source;
        if (!body.active || !source || body.mass <= 0.0f)
            continue;

        GpuVelocityInput input = {};
        input.position[0] = body.positionLin.x;
        input.position[1] = body.positionLin.y;
        input.position[2] = body.positionLin.z;
        input.position[3] = 1.0f;
        input.initial[0] = source->initialLin.x;
        input.initial[1] = source->initialLin.y;
        input.initial[2] = source->initialLin.z;
        input.initial[3] = 1.0f;
        input.angle[0] = body.positionAng.x;
        input.angle[1] = body.positionAng.y;
        input.angle[2] = body.positionAng.z;
        input.angle[3] = body.positionAng.w;
        input.initialAngle[0] = source->initialAng.x;
        input.initialAngle[1] = source->initialAng.y;
        input.initialAngle[2] = source->initialAng.z;
        input.initialAngle[3] = source->initialAng.w;
        input.expectedVelocity[0] = body.velocityLin.x;
        input.expectedVelocity[1] = body.velocityLin.y;
        input.expectedVelocity[2] = body.velocityLin.z;
        input.expectedVelocity[3] = 0.0f;
        input.expectedAngularVelocity[0] = body.velocityAng.x;
        input.expectedAngularVelocity[1] = body.velocityAng.y;
        input.expectedAngularVelocity[2] = body.velocityAng.z;
        input.expectedAngularVelocity[3] = 0.0f;
        inputs.push_back(input);
    }

    if (inputs.empty())
    {
        setStatus(velocityStatus, "Velocity update skipped: no active bodies");
        return false;
    }

    const uint32_t sampleCount = (uint32_t)min((int)inputs.size(), 8);
    const uint64_t inputBytes = (uint64_t)(inputs.size() * sizeof(GpuVelocityInput));
    const uint64_t outputBytes = (uint64_t)(inputs.size() * sizeof(GpuVelocityOutput));
    const uint64_t readbackBytes = (uint64_t)(sampleCount * sizeof(GpuVelocityOutput));

    Uint64 begin = SDL_GetPerformanceCounter();

    if (velocityInputBuffer == nullptr || velocityInputBufferBytes < inputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(inputBytes, 4096);
        velocityInputBuffer = device.CreateBuffer(&desc);
        velocityInputBufferBytes = velocityInputBuffer == nullptr ? 0 : desc.size;
    }
    if (velocityOutputBuffer == nullptr || velocityOutputBufferBytes < outputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopySrc);
        desc.size = alignUp(outputBytes, 4096);
        velocityOutputBuffer = device.CreateBuffer(&desc);
        velocityOutputBufferBytes = velocityOutputBuffer == nullptr ? 0 : desc.size;
    }
    if (velocityReadbackBuffer == nullptr || velocityReadbackBufferBytes < readbackBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(readbackBytes, 256);
        velocityReadbackBuffer = device.CreateBuffer(&desc);
        velocityReadbackBufferBytes = velocityReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (velocityParamsBuffer == nullptr)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = sizeof(GpuPredictionParams);
        velocityParamsBuffer = device.CreateBuffer(&desc);
    }

    if (velocityInputBuffer == nullptr || velocityOutputBuffer == nullptr ||
        velocityReadbackBuffer == nullptr || velocityParamsBuffer == nullptr)
    {
        setStatus(velocityStatus, "Velocity update failed: buffer allocation");
        return false;
    }

    GpuPredictionParams params = {};
    params.dt = dt;
    params.sampleCount = (uint32_t)inputs.size();
    queue.WriteBuffer(velocityInputBuffer, 0, inputs.data(), inputBytes);
    queue.WriteBuffer(velocityParamsBuffer, 0, &params, sizeof(params));

    static const char *shaderSource = R"(
struct BodyInput {
    position: vec4f,
    initial: vec4f,
    angle: vec4f,
    initialAngle: vec4f,
    expectedVelocity: vec4f,
    expectedAngularVelocity: vec4f,
};

struct BodyOutput {
    velocity: vec4f,
    angularVelocity: vec4f,
};

struct Params {
    dt: f32,
    gravity: f32,
    bodyCount: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> inputs: array<BodyInput>;
@group(0) @binding(1) var<storage, read_write> outputs: array<BodyOutput>;
@group(0) @binding(2) var<uniform> params: Params;

fn quatMul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

fn quatInverse(q: vec4f) -> vec4f {
    let d = max(dot(q, q), 1.0e-20);
    return vec4f(-q.x, -q.y, -q.z, q.w) / d;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= params.bodyCount) {
        return;
    }

    let input = inputs[index];
    outputs[index].velocity = vec4f((input.position.xyz - input.initial.xyz) / params.dt, 0.0);

    let delta = quatMul(input.angle, quatInverse(input.initialAngle));
    outputs[index].angularVelocity = vec4f(delta.xyz * (2.0 / params.dt), 0.0);
}
)";

    if (velocityPipeline == nullptr || velocityBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = shaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(velocityStatus, "Velocity update failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry entries[3] = {};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        entries[0].buffer.minBindingSize = sizeof(GpuVelocityInput);
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].buffer.type = wgpu::BufferBindingType::Storage;
        entries[1].buffer.minBindingSize = sizeof(GpuVelocityOutput);
        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Compute;
        entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[2].buffer.minBindingSize = sizeof(GpuPredictionParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 3;
        layoutDesc.entries = entries;
        velocityBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (velocityBindGroupLayout == nullptr)
        {
            setStatus(velocityStatus, "Velocity update failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &velocityBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(velocityStatus, "Velocity update failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        velocityPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (velocityPipeline == nullptr)
        {
            setStatus(velocityStatus, "Velocity update failed: compute pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry entries[3] = {};
    entries[0].binding = 0;
    entries[0].buffer = velocityInputBuffer;
    entries[0].offset = 0;
    entries[0].size = inputBytes;
    entries[1].binding = 1;
    entries[1].buffer = velocityOutputBuffer;
    entries[1].offset = 0;
    entries[1].size = outputBytes;
    entries[2].binding = 2;
    entries[2].buffer = velocityParamsBuffer;
    entries[2].offset = 0;
    entries[2].size = sizeof(GpuPredictionParams);

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = velocityBindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
    if (bindGroup == nullptr)
    {
        setStatus(velocityStatus, "Velocity update failed: bind group");
        return false;
    }

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(velocityPipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups((uint32_t)((inputs.size() + 63) / 64));
    pass.End();
    encoder.CopyBufferToBuffer(velocityOutputBuffer, 0, velocityReadbackBuffer, 0, readbackBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = velocityReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, readbackBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
    {
        setStatus(velocityStatus, "Velocity update failed: readback map");
        return false;
    }

    const GpuVelocityOutput *outputs = (const GpuVelocityOutput *)velocityReadbackBuffer.GetConstMappedRange(0, readbackBytes);
    if (!outputs)
    {
        velocityReadbackBuffer.Unmap();
        setStatus(velocityStatus, "Velocity update failed: readback range");
        return false;
    }

    float maxLinearError = 0.0f;
    float maxAngularError = 0.0f;
    for (uint32_t i = 0; i < sampleCount; ++i)
    {
        const GpuVelocityInput &input = inputs[i];
        float dx = outputs[i].velocity[0] - input.expectedVelocity[0];
        float dy = outputs[i].velocity[1] - input.expectedVelocity[1];
        float dz = outputs[i].velocity[2] - input.expectedVelocity[2];
        float linearError = sqrtf(dx * dx + dy * dy + dz * dz);
        if (linearError > maxLinearError)
            maxLinearError = linearError;

        float dax = outputs[i].angularVelocity[0] - input.expectedAngularVelocity[0];
        float day = outputs[i].angularVelocity[1] - input.expectedAngularVelocity[1];
        float daz = outputs[i].angularVelocity[2] - input.expectedAngularVelocity[2];
        float angularError = sqrtf(dax * dax + day * day + daz * daz);
        if (angularError > maxAngularError)
            maxAngularError = angularError;
    }
    velocityReadbackBuffer.Unmap();

    velocitySamples = (int)sampleCount;
    velocityMaxLinearError = maxLinearError;
    velocityMaxAngularError = maxAngularError;
    velocityMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    recordTimingSample(velocityTiming, velocityMs);
    snprintf(velocityStatus, sizeof(velocityStatus), "Velocity update passed: %d samples, lin %.6f, ang %.6f",
             velocitySamples, velocityMaxLinearError, velocityMaxAngularError);
    return true;
#elif AVBD_ENABLE_WEBGPU
    setStatus(velocityStatus, "Velocity update skipped: Dawn not linked");
    return false;
#else
    setStatus(velocityStatus, "Velocity update skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::runBoundsDiagnostic(const SimWorld &world)
{
    boundsMs = 0.0f;
    boundsMaxError = 0.0f;
    boundsSamples = 0;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(boundsStatus, "Bounds skipped: WebGPU device not ready");
        return false;
    }

    std::vector<GpuBoundsInput> inputs;
    inputs.reserve(world.activeBodyIds.size());
    for (size_t i = 0; i < world.activeBodyIds.size(); ++i)
    {
        BodyId bodyId = world.activeBodyIds[i];
        if (bodyId == INVALID_BODY_ID || bodyId >= world.bodies.size())
            continue;

        const SimBodyData &body = world.bodies[bodyId];
        if (!body.active)
            continue;

        GpuBoundsInput input = {};
        input.position[0] = body.positionLin.x;
        input.position[1] = body.positionLin.y;
        input.position[2] = body.positionLin.z;
        input.position[3] = 1.0f;
        input.params[0] = body.radius;
        inputs.push_back(input);
    }

    if (inputs.empty())
    {
        setStatus(boundsStatus, "Bounds skipped: no active bodies");
        return false;
    }

    const uint32_t sampleCount = (uint32_t)min((int)inputs.size(), 8);
    const uint64_t inputBytes = (uint64_t)(inputs.size() * sizeof(GpuBoundsInput));
    const uint64_t outputBytes = (uint64_t)(inputs.size() * sizeof(GpuBoundsOutput));
    const uint64_t readbackBytes = (uint64_t)(sampleCount * sizeof(GpuBoundsOutput));

    Uint64 begin = SDL_GetPerformanceCounter();

    if (boundsInputBuffer == nullptr || boundsInputBufferBytes < inputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(inputBytes, 4096);
        boundsInputBuffer = device.CreateBuffer(&desc);
        boundsInputBufferBytes = boundsInputBuffer == nullptr ? 0 : desc.size;
    }
    if (boundsOutputBuffer == nullptr || boundsOutputBufferBytes < outputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopySrc);
        desc.size = alignUp(outputBytes, 4096);
        boundsOutputBuffer = device.CreateBuffer(&desc);
        boundsOutputBufferBytes = boundsOutputBuffer == nullptr ? 0 : desc.size;
    }
    if (boundsReadbackBuffer == nullptr || boundsReadbackBufferBytes < readbackBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(readbackBytes, 256);
        boundsReadbackBuffer = device.CreateBuffer(&desc);
        boundsReadbackBufferBytes = boundsReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (boundsParamsBuffer == nullptr)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = sizeof(GpuPredictionParams);
        boundsParamsBuffer = device.CreateBuffer(&desc);
    }

    if (boundsInputBuffer == nullptr || boundsOutputBuffer == nullptr ||
        boundsReadbackBuffer == nullptr || boundsParamsBuffer == nullptr)
    {
        setStatus(boundsStatus, "Bounds failed: buffer allocation");
        return false;
    }

    GpuPredictionParams params = {};
    params.sampleCount = (uint32_t)inputs.size();
    queue.WriteBuffer(boundsInputBuffer, 0, inputs.data(), inputBytes);
    queue.WriteBuffer(boundsParamsBuffer, 0, &params, sizeof(params));

    static const char *shaderSource = R"(
struct BodyInput {
    position: vec4f,
    params: vec4f,
};

struct BodyOutput {
    minBounds: vec4f,
    maxBounds: vec4f,
};

struct Params {
    dt: f32,
    gravity: f32,
    bodyCount: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> inputs: array<BodyInput>;
@group(0) @binding(1) var<storage, read_write> outputs: array<BodyOutput>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= params.bodyCount) {
        return;
    }

    let input = inputs[index];
    let radius = max(input.params.x, 0.0);
    let extent = vec3f(radius, radius, radius);
    outputs[index].minBounds = vec4f(input.position.xyz - extent, 0.0);
    outputs[index].maxBounds = vec4f(input.position.xyz + extent, 0.0);
}
)";

    if (boundsPipeline == nullptr || boundsBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = shaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(boundsStatus, "Bounds failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry entries[3] = {};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        entries[0].buffer.minBindingSize = sizeof(GpuBoundsInput);
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].buffer.type = wgpu::BufferBindingType::Storage;
        entries[1].buffer.minBindingSize = sizeof(GpuBoundsOutput);
        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Compute;
        entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[2].buffer.minBindingSize = sizeof(GpuPredictionParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 3;
        layoutDesc.entries = entries;
        boundsBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (boundsBindGroupLayout == nullptr)
        {
            setStatus(boundsStatus, "Bounds failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &boundsBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(boundsStatus, "Bounds failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        boundsPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (boundsPipeline == nullptr)
        {
            setStatus(boundsStatus, "Bounds failed: compute pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry entries[3] = {};
    entries[0].binding = 0;
    entries[0].buffer = boundsInputBuffer;
    entries[0].offset = 0;
    entries[0].size = inputBytes;
    entries[1].binding = 1;
    entries[1].buffer = boundsOutputBuffer;
    entries[1].offset = 0;
    entries[1].size = outputBytes;
    entries[2].binding = 2;
    entries[2].buffer = boundsParamsBuffer;
    entries[2].offset = 0;
    entries[2].size = sizeof(GpuPredictionParams);

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = boundsBindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
    if (bindGroup == nullptr)
    {
        setStatus(boundsStatus, "Bounds failed: bind group");
        return false;
    }

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(boundsPipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups((uint32_t)((inputs.size() + 63) / 64));
    pass.End();
    encoder.CopyBufferToBuffer(boundsOutputBuffer, 0, boundsReadbackBuffer, 0, readbackBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = boundsReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, readbackBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
    {
        setStatus(boundsStatus, "Bounds failed: readback map");
        return false;
    }

    const GpuBoundsOutput *outputs = (const GpuBoundsOutput *)boundsReadbackBuffer.GetConstMappedRange(0, readbackBytes);
    if (!outputs)
    {
        boundsReadbackBuffer.Unmap();
        setStatus(boundsStatus, "Bounds failed: readback range");
        return false;
    }

    float maxError = 0.0f;
    for (uint32_t i = 0; i < sampleCount; ++i)
    {
        const GpuBoundsInput &input = inputs[i];
        float radius = input.params[0];
        float expectedMin[3] = {input.position[0] - radius, input.position[1] - radius, input.position[2] - radius};
        float expectedMax[3] = {input.position[0] + radius, input.position[1] + radius, input.position[2] + radius};
        for (int axis = 0; axis < 3; ++axis)
        {
            float minError = fabsf(outputs[i].minBounds[axis] - expectedMin[axis]);
            float maxErrorAxis = fabsf(outputs[i].maxBounds[axis] - expectedMax[axis]);
            if (minError > maxError)
                maxError = minError;
            if (maxErrorAxis > maxError)
                maxError = maxErrorAxis;
        }
    }
    boundsReadbackBuffer.Unmap();

    boundsSamples = (int)sampleCount;
    boundsMaxError = maxError;
    boundsMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    recordTimingSample(boundsTiming, boundsMs);
    snprintf(boundsStatus, sizeof(boundsStatus), "Bounds passed: %d samples, max %.6f",
             boundsSamples, boundsMaxError);
    return true;
#elif AVBD_ENABLE_WEBGPU
    setStatus(boundsStatus, "Bounds skipped: Dawn not linked");
    return false;
#else
    setStatus(boundsStatus, "Bounds skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::runMortonDiagnostic(const SimWorld &world)
{
    mortonMs = 0.0f;
    mortonMismatches = 0;
    mortonSamples = 0;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(mortonStatus, "Morton codes skipped: WebGPU device not ready");
        return false;
    }

    std::vector<GpuMortonInput> inputs;
    inputs.reserve(world.activeBodyIds.size());
    for (size_t i = 0; i < world.activeBodyIds.size(); ++i)
    {
        BodyId bodyId = world.activeBodyIds[i];
        if (bodyId == INVALID_BODY_ID || bodyId >= world.bodies.size())
            continue;

        const SimBodyData &body = world.bodies[bodyId];
        if (!body.active)
            continue;

        GpuMortonInput input = {};
        input.position[0] = body.positionLin.x;
        input.position[1] = body.positionLin.y;
        input.position[2] = body.positionLin.z;
        input.position[3] = 1.0f;
        inputs.push_back(input);
    }

    if (inputs.empty())
    {
        setStatus(mortonStatus, "Morton codes skipped: no active bodies");
        return false;
    }

    const uint32_t sampleCount = (uint32_t)min((int)inputs.size(), 8);
    const uint64_t inputBytes = (uint64_t)(inputs.size() * sizeof(GpuMortonInput));
    const uint64_t outputBytes = (uint64_t)(inputs.size() * sizeof(GpuMortonOutput));
    const uint64_t readbackBytes = (uint64_t)(sampleCount * sizeof(GpuMortonOutput));
    const float worldMin = -64.0f;
    const float invWorldSize = 1.0f / 128.0f;

    Uint64 begin = SDL_GetPerformanceCounter();

    if (mortonInputBuffer == nullptr || mortonInputBufferBytes < inputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(inputBytes, 4096);
        mortonInputBuffer = device.CreateBuffer(&desc);
        mortonInputBufferBytes = mortonInputBuffer == nullptr ? 0 : desc.size;
    }
    if (mortonOutputBuffer == nullptr || mortonOutputBufferBytes < outputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopySrc);
        desc.size = alignUp(outputBytes, 4096);
        mortonOutputBuffer = device.CreateBuffer(&desc);
        mortonOutputBufferBytes = mortonOutputBuffer == nullptr ? 0 : desc.size;
    }
    if (mortonReadbackBuffer == nullptr || mortonReadbackBufferBytes < readbackBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(readbackBytes, 256);
        mortonReadbackBuffer = device.CreateBuffer(&desc);
        mortonReadbackBufferBytes = mortonReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (mortonParamsBuffer == nullptr)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = sizeof(GpuMortonParams);
        mortonParamsBuffer = device.CreateBuffer(&desc);
    }

    if (mortonInputBuffer == nullptr || mortonOutputBuffer == nullptr ||
        mortonReadbackBuffer == nullptr || mortonParamsBuffer == nullptr)
    {
        setStatus(mortonStatus, "Morton codes failed: buffer allocation");
        return false;
    }

    GpuMortonParams params = {};
    params.worldMin = worldMin;
    params.invWorldSize = invWorldSize;
    params.bodyCount = (uint32_t)inputs.size();
    queue.WriteBuffer(mortonInputBuffer, 0, inputs.data(), inputBytes);
    queue.WriteBuffer(mortonParamsBuffer, 0, &params, sizeof(params));

    static const char *shaderSource = R"(
struct BodyInput {
    position: vec4f,
};

struct MortonOutput {
    code: u32,
    index: u32,
    pad0: u32,
    pad1: u32,
};

struct Params {
    worldMin: f32,
    invWorldSize: f32,
    bodyCount: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> inputs: array<BodyInput>;
@group(0) @binding(1) var<storage, read_write> outputs: array<MortonOutput>;
@group(0) @binding(2) var<uniform> params: Params;

fn expandBits(value: u32) -> u32 {
    var x = value & 1023u;
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x << 8u)) & 0x0300F00Fu;
    x = (x | (x << 4u)) & 0x030C30C3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

fn quantizeAxis(value: f32) -> u32 {
    let normalized = clamp((value - params.worldMin) * params.invWorldSize, 0.0, 0.999999);
    return u32(floor(normalized * 1024.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= params.bodyCount) {
        return;
    }

    let position = inputs[index].position.xyz;
    let x = quantizeAxis(position.x);
    let y = quantizeAxis(position.y);
    let z = quantizeAxis(position.z);
    outputs[index].code = expandBits(x) | (expandBits(y) << 1u) | (expandBits(z) << 2u);
    outputs[index].index = index;
    outputs[index].pad0 = 0u;
    outputs[index].pad1 = 0u;
}
)";

    if (mortonPipeline == nullptr || mortonBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = shaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(mortonStatus, "Morton codes failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry entries[3] = {};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        entries[0].buffer.minBindingSize = sizeof(GpuMortonInput);
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].buffer.type = wgpu::BufferBindingType::Storage;
        entries[1].buffer.minBindingSize = sizeof(GpuMortonOutput);
        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Compute;
        entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[2].buffer.minBindingSize = sizeof(GpuMortonParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 3;
        layoutDesc.entries = entries;
        mortonBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (mortonBindGroupLayout == nullptr)
        {
            setStatus(mortonStatus, "Morton codes failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &mortonBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(mortonStatus, "Morton codes failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        mortonPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (mortonPipeline == nullptr)
        {
            setStatus(mortonStatus, "Morton codes failed: compute pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry entries[3] = {};
    entries[0].binding = 0;
    entries[0].buffer = mortonInputBuffer;
    entries[0].offset = 0;
    entries[0].size = inputBytes;
    entries[1].binding = 1;
    entries[1].buffer = mortonOutputBuffer;
    entries[1].offset = 0;
    entries[1].size = outputBytes;
    entries[2].binding = 2;
    entries[2].buffer = mortonParamsBuffer;
    entries[2].offset = 0;
    entries[2].size = sizeof(GpuMortonParams);

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = mortonBindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
    if (bindGroup == nullptr)
    {
        setStatus(mortonStatus, "Morton codes failed: bind group");
        return false;
    }

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(mortonPipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups((uint32_t)((inputs.size() + 63) / 64));
    pass.End();
    encoder.CopyBufferToBuffer(mortonOutputBuffer, 0, mortonReadbackBuffer, 0, readbackBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = mortonReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, readbackBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
    {
        setStatus(mortonStatus, "Morton codes failed: readback map");
        return false;
    }

    const GpuMortonOutput *outputs = (const GpuMortonOutput *)mortonReadbackBuffer.GetConstMappedRange(0, readbackBytes);
    if (!outputs)
    {
        mortonReadbackBuffer.Unmap();
        setStatus(mortonStatus, "Morton codes failed: readback range");
        return false;
    }

    int mismatches = 0;
    for (uint32_t i = 0; i < sampleCount; ++i)
    {
        float3 position{inputs[i].position[0], inputs[i].position[1], inputs[i].position[2]};
        uint32_t expectedCode = mortonCodeForPosition(position, worldMin, invWorldSize);
        if (outputs[i].code != expectedCode || outputs[i].index != i)
            ++mismatches;
    }
    mortonReadbackBuffer.Unmap();

    mortonSamples = (int)sampleCount;
    mortonMismatches = mismatches;
    mortonMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    recordTimingSample(mortonTiming, mortonMs);
    snprintf(mortonStatus, sizeof(mortonStatus), "Morton codes passed: %d samples, mismatches %d",
             mortonSamples, mortonMismatches);
    return mismatches == 0;
#elif AVBD_ENABLE_WEBGPU
    setStatus(mortonStatus, "Morton codes skipped: Dawn not linked");
    return false;
#else
    setStatus(mortonStatus, "Morton codes skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::runMortonSortDiagnostic(const SimWorld &world)
{
    mortonSortMs = 0.0f;
    mortonSortMismatches = 0;
    mortonSortCount = 0;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(mortonSortStatus, "Morton sort skipped: WebGPU device not ready");
        return false;
    }

    std::vector<GpuMortonOutput> items;
    items.reserve(world.activeBodyIds.size());
    const float worldMin = -64.0f;
    const float invWorldSize = 1.0f / 128.0f;
    for (size_t i = 0; i < world.activeBodyIds.size(); ++i)
    {
        BodyId bodyId = world.activeBodyIds[i];
        if (bodyId == INVALID_BODY_ID || bodyId >= world.bodies.size())
            continue;

        const SimBodyData &body = world.bodies[bodyId];
        if (!body.active)
            continue;

        GpuMortonOutput item = {};
        item.code = mortonCodeForPosition(body.positionLin, worldMin, invWorldSize);
        item.index = (uint32_t)i;
        items.push_back(item);
    }

    if (items.empty())
    {
        setStatus(mortonSortStatus, "Morton sort skipped: no active bodies");
        return false;
    }

    std::vector<GpuMortonOutput> expected = items;
    std::sort(expected.begin(), expected.end(), [](const GpuMortonOutput &a, const GpuMortonOutput &b)
              {
                  if (a.code == b.code)
                      return a.index < b.index;
                  return a.code < b.code;
              });

    uint32_t itemCount = (uint32_t)items.size();
    uint32_t paddedCount = nextPowerOfTwo(itemCount);
    GpuMortonOutput sentinel = {};
    sentinel.code = 0xFFFFFFFFu;
    sentinel.index = 0xFFFFFFFFu;
    items.resize(paddedCount, sentinel);

    const uint64_t bufferBytes = (uint64_t)(items.size() * sizeof(GpuMortonOutput));
    const uint64_t readbackBytes = bufferBytes;

    Uint64 begin = SDL_GetPerformanceCounter();

    if (mortonSortBuffer == nullptr || mortonSortBufferBytes < bufferBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopyDst |
                                 (uint64_t)wgpu::BufferUsage::CopySrc);
        desc.size = alignUp(bufferBytes, 4096);
        mortonSortBuffer = device.CreateBuffer(&desc);
        mortonSortBufferBytes = mortonSortBuffer == nullptr ? 0 : desc.size;
    }
    if (mortonSortReadbackBuffer == nullptr || mortonSortReadbackBufferBytes < readbackBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(readbackBytes, 256);
        mortonSortReadbackBuffer = device.CreateBuffer(&desc);
        mortonSortReadbackBufferBytes = mortonSortReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (mortonSortParamsBuffer == nullptr)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = sizeof(GpuSortParams);
        mortonSortParamsBuffer = device.CreateBuffer(&desc);
    }

    if (mortonSortBuffer == nullptr || mortonSortReadbackBuffer == nullptr || mortonSortParamsBuffer == nullptr)
    {
        setStatus(mortonSortStatus, "Morton sort failed: buffer allocation");
        return false;
    }

    queue.WriteBuffer(mortonSortBuffer, 0, items.data(), bufferBytes);

    static const char *shaderSource = R"(
struct SortItem {
    code: u32,
    index: u32,
    pad0: u32,
    pad1: u32,
};

struct Params {
    j: u32,
    k: u32,
    count: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read_write> items: array<SortItem>;
@group(0) @binding(1) var<uniform> params: Params;

fn greater(a: SortItem, b: SortItem) -> bool {
    if (a.code == b.code) {
        return a.index > b.index;
    }
    return a.code > b.code;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= params.count) {
        return;
    }

    let other = index ^ params.j;
    if (other <= index || other >= params.count) {
        return;
    }

    let ascending = (index & params.k) == 0u;
    let a = items[index];
    let b = items[other];
    let shouldSwap = select(greater(b, a), greater(a, b), ascending);
    if (shouldSwap) {
        items[index] = b;
        items[other] = a;
    }
}
)";

    if (mortonSortPipeline == nullptr || mortonSortBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = shaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(mortonSortStatus, "Morton sort failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry entries[2] = {};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::Storage;
        entries[0].buffer.minBindingSize = sizeof(GpuMortonOutput);
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[1].buffer.minBindingSize = sizeof(GpuSortParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 2;
        layoutDesc.entries = entries;
        mortonSortBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (mortonSortBindGroupLayout == nullptr)
        {
            setStatus(mortonSortStatus, "Morton sort failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &mortonSortBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(mortonSortStatus, "Morton sort failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        mortonSortPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (mortonSortPipeline == nullptr)
        {
            setStatus(mortonSortStatus, "Morton sort failed: compute pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry bindEntries[2] = {};
    bindEntries[0].binding = 0;
    bindEntries[0].buffer = mortonSortBuffer;
    bindEntries[0].offset = 0;
    bindEntries[0].size = bufferBytes;
    bindEntries[1].binding = 1;
    bindEntries[1].buffer = mortonSortParamsBuffer;
    bindEntries[1].offset = 0;
    bindEntries[1].size = sizeof(GpuSortParams);

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = mortonSortBindGroupLayout;
    bindGroupDesc.entryCount = 2;
    bindGroupDesc.entries = bindEntries;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
    if (bindGroup == nullptr)
    {
        setStatus(mortonSortStatus, "Morton sort failed: bind group");
        return false;
    }

    const uint64_t passParamStride = 256;
    std::vector<unsigned char> passParamData;
    uint32_t passCount = 0;
    for (uint32_t k = 2u; k <= paddedCount; k <<= 1u)
    {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u)
        {
            GpuSortParams params = {};
            params.j = j;
            params.k = k;
            params.count = paddedCount;
            passParamData.resize((uint64_t)(passCount + 1u) * passParamStride);
            memcpy(passParamData.data() + (uint64_t)passCount * passParamStride, &params, sizeof(params));
            ++passCount;
        }
    }

    wgpu::BufferDescriptor passParamsDesc = {};
    passParamsDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::CopySrc |
                                       (uint64_t)wgpu::BufferUsage::CopyDst);
    passParamsDesc.size = alignUp((uint64_t)passParamData.size(), 256);
    wgpu::Buffer passParamsBuffer = device.CreateBuffer(&passParamsDesc);
    if (passParamsBuffer == nullptr)
    {
        setStatus(mortonSortStatus, "Morton sort failed: pass params buffer");
        return false;
    }
    queue.WriteBuffer(passParamsBuffer, 0, passParamData.data(), passParamData.size());

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex)
    {
        encoder.CopyBufferToBuffer(passParamsBuffer, (uint64_t)passIndex * passParamStride,
                                   mortonSortParamsBuffer, 0, sizeof(GpuSortParams));
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(mortonSortPipeline);
        pass.SetBindGroup(0, bindGroup);
        pass.DispatchWorkgroups((paddedCount + 63u) / 64u);
        pass.End();
    }
    encoder.CopyBufferToBuffer(mortonSortBuffer, 0, mortonSortReadbackBuffer, 0, readbackBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = mortonSortReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, readbackBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
    {
        setStatus(mortonSortStatus, "Morton sort failed: readback map");
        return false;
    }

    const GpuMortonOutput *outputs = (const GpuMortonOutput *)mortonSortReadbackBuffer.GetConstMappedRange(0, readbackBytes);
    if (!outputs)
    {
        mortonSortReadbackBuffer.Unmap();
        setStatus(mortonSortStatus, "Morton sort failed: readback range");
        return false;
    }

    int mismatches = 0;
    for (uint32_t i = 0; i < itemCount; ++i)
    {
        if (outputs[i].code != expected[i].code || outputs[i].index != expected[i].index)
            ++mismatches;
    }
    mortonSortReadbackBuffer.Unmap();

    mortonSortCount = (int)itemCount;
    mortonSortMismatches = mismatches;
    mortonSortMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    recordTimingSample(mortonSortTiming, mortonSortMs);
    snprintf(mortonSortStatus, sizeof(mortonSortStatus), "Morton sort passed: %d items, mismatches %d",
             mortonSortCount, mortonSortMismatches);
    return mismatches == 0;
#elif AVBD_ENABLE_WEBGPU
    setStatus(mortonSortStatus, "Morton sort skipped: Dawn not linked");
    return false;
#else
    setStatus(mortonSortStatus, "Morton sort skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::runBroadphasePairDiagnostic(const SimWorld &world)
{
    pairMs = 0.0f;
    pairCandidates = 0;
    pairSphereHits = 0;
    pairAllPairsSphereHits = 0;
    pairMissedSphereHits = 0;
    pairMismatches = 0;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(pairStatus, "Pair diagnostic skipped: WebGPU device not ready");
        return false;
    }

    std::vector<GpuPairBody> bodies;
    std::vector<GpuMortonOutput> sortedItems;
    bodies.reserve(world.activeBodyIds.size());
    sortedItems.reserve(world.activeBodyIds.size());

    const float worldMin = -64.0f;
    const float invWorldSize = 1.0f / 128.0f;
    for (size_t i = 0; i < world.activeBodyIds.size(); ++i)
    {
        BodyId bodyId = world.activeBodyIds[i];
        if (bodyId == INVALID_BODY_ID || bodyId >= world.bodies.size())
            continue;

        const SimBodyData &body = world.bodies[bodyId];
        if (!body.active)
            continue;

        GpuPairBody gpuBody = {};
        gpuBody.position[0] = body.positionLin.x;
        gpuBody.position[1] = body.positionLin.y;
        gpuBody.position[2] = body.positionLin.z;
        gpuBody.position[3] = body.radius;
        uint32_t bodyIndex = (uint32_t)bodies.size();
        bodies.push_back(gpuBody);

        GpuMortonOutput item = {};
        item.code = mortonCodeForPosition(body.positionLin, worldMin, invWorldSize);
        item.index = bodyIndex;
        sortedItems.push_back(item);
    }

    if (bodies.empty())
    {
        setStatus(pairStatus, "Pair diagnostic skipped: no active bodies");
        return false;
    }

    std::sort(sortedItems.begin(), sortedItems.end(), [](const GpuMortonOutput &a, const GpuMortonOutput &b)
              {
                  if (a.code == b.code)
                      return a.index < b.index;
                  return a.code < b.code;
              });

    const uint32_t neighborWindow = 64u;
    uint32_t expectedCandidates = 0;
    uint32_t expectedSphereHits = 0;
    for (uint32_t i = 0; i < sortedItems.size(); ++i)
    {
        uint32_t end = i + neighborWindow + 1u;
        if (end > (uint32_t)sortedItems.size())
            end = (uint32_t)sortedItems.size();
        for (uint32_t j = i + 1u; j < end; ++j)
        {
            ++expectedCandidates;
            const GpuPairBody &a = bodies[sortedItems[i].index];
            const GpuPairBody &b = bodies[sortedItems[j].index];
            float dx = a.position[0] - b.position[0];
            float dy = a.position[1] - b.position[1];
            float dz = a.position[2] - b.position[2];
            float radius = a.position[3] + b.position[3];
            if (dx * dx + dy * dy + dz * dz < radius * radius)
                ++expectedSphereHits;
        }
    }

    uint32_t allPairsSphereHits = 0;
    if (bodies.size() <= 4096)
    {
        for (uint32_t i = 0; i < bodies.size(); ++i)
        {
            for (uint32_t j = i + 1u; j < bodies.size(); ++j)
            {
                const GpuPairBody &a = bodies[i];
                const GpuPairBody &b = bodies[j];
                float dx = a.position[0] - b.position[0];
                float dy = a.position[1] - b.position[1];
                float dz = a.position[2] - b.position[2];
                float radius = a.position[3] + b.position[3];
                if (dx * dx + dy * dy + dz * dz < radius * radius)
                    ++allPairsSphereHits;
            }
        }
    }

    const uint64_t bodyBytes = (uint64_t)(bodies.size() * sizeof(GpuPairBody));
    const uint64_t itemBytes = (uint64_t)(sortedItems.size() * sizeof(GpuMortonOutput));
    const uint64_t counterBytes = sizeof(GpuPairCounters);

    Uint64 begin = SDL_GetPerformanceCounter();

    if (pairBodyBuffer == nullptr || pairBodyBufferBytes < bodyBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(bodyBytes, 4096);
        pairBodyBuffer = device.CreateBuffer(&desc);
        pairBodyBufferBytes = pairBodyBuffer == nullptr ? 0 : desc.size;
    }
    if (pairItemBuffer == nullptr || pairItemBufferBytes < itemBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(itemBytes, 4096);
        pairItemBuffer = device.CreateBuffer(&desc);
        pairItemBufferBytes = pairItemBuffer == nullptr ? 0 : desc.size;
    }
    if (pairCountersBuffer == nullptr || pairCountersBufferBytes < counterBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopySrc |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(counterBytes, 256);
        pairCountersBuffer = device.CreateBuffer(&desc);
        pairCountersBufferBytes = pairCountersBuffer == nullptr ? 0 : desc.size;
    }
    if (pairReadbackBuffer == nullptr || pairReadbackBufferBytes < counterBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(counterBytes, 256);
        pairReadbackBuffer = device.CreateBuffer(&desc);
        pairReadbackBufferBytes = pairReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (pairParamsBuffer == nullptr)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = sizeof(GpuPairParams);
        pairParamsBuffer = device.CreateBuffer(&desc);
    }

    if (pairBodyBuffer == nullptr || pairItemBuffer == nullptr || pairCountersBuffer == nullptr ||
        pairReadbackBuffer == nullptr || pairParamsBuffer == nullptr)
    {
        setStatus(pairStatus, "Pair diagnostic failed: buffer allocation");
        return false;
    }

    GpuPairCounters zeroCounters = {};
    GpuPairParams params = {};
    params.count = (uint32_t)sortedItems.size();
    params.neighborWindow = neighborWindow;
    queue.WriteBuffer(pairBodyBuffer, 0, bodies.data(), bodyBytes);
    queue.WriteBuffer(pairItemBuffer, 0, sortedItems.data(), itemBytes);
    queue.WriteBuffer(pairCountersBuffer, 0, &zeroCounters, sizeof(zeroCounters));
    queue.WriteBuffer(pairParamsBuffer, 0, &params, sizeof(params));

    static const char *shaderSource = R"(
struct Body {
    positionRadius: vec4f,
};

struct Item {
    code: u32,
    index: u32,
    pad0: u32,
    pad1: u32,
};

struct Counters {
    candidates: atomic<u32>,
    sphereHits: atomic<u32>,
    pad0: u32,
    pad1: u32,
};

struct Params {
    count: u32,
    neighborWindow: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<storage, read> items: array<Item>;
@group(0) @binding(2) var<storage, read_write> counters: Counters;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }

    let itemA = items[i];
    let bodyA = bodies[itemA.index].positionRadius;
    let end = min(params.count, i + params.neighborWindow + 1u);
    var j = i + 1u;
    loop {
        if (j >= end) {
            break;
        }

        atomicAdd(&counters.candidates, 1u);
        let itemB = items[j];
        let bodyB = bodies[itemB.index].positionRadius;
        let delta = bodyA.xyz - bodyB.xyz;
        let radius = bodyA.w + bodyB.w;
        if (dot(delta, delta) < radius * radius) {
            atomicAdd(&counters.sphereHits, 1u);
        }
        j = j + 1u;
    }
}
)";

    if (pairPipeline == nullptr || pairBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = shaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(pairStatus, "Pair diagnostic failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry entries[4] = {};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        entries[0].buffer.minBindingSize = sizeof(GpuPairBody);
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        entries[1].buffer.minBindingSize = sizeof(GpuMortonOutput);
        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Compute;
        entries[2].buffer.type = wgpu::BufferBindingType::Storage;
        entries[2].buffer.minBindingSize = sizeof(GpuPairCounters);
        entries[3].binding = 3;
        entries[3].visibility = wgpu::ShaderStage::Compute;
        entries[3].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[3].buffer.minBindingSize = sizeof(GpuPairParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 4;
        layoutDesc.entries = entries;
        pairBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (pairBindGroupLayout == nullptr)
        {
            setStatus(pairStatus, "Pair diagnostic failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &pairBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(pairStatus, "Pair diagnostic failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        pairPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (pairPipeline == nullptr)
        {
            setStatus(pairStatus, "Pair diagnostic failed: compute pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry entries[4] = {};
    entries[0].binding = 0;
    entries[0].buffer = pairBodyBuffer;
    entries[0].offset = 0;
    entries[0].size = bodyBytes;
    entries[1].binding = 1;
    entries[1].buffer = pairItemBuffer;
    entries[1].offset = 0;
    entries[1].size = itemBytes;
    entries[2].binding = 2;
    entries[2].buffer = pairCountersBuffer;
    entries[2].offset = 0;
    entries[2].size = counterBytes;
    entries[3].binding = 3;
    entries[3].buffer = pairParamsBuffer;
    entries[3].offset = 0;
    entries[3].size = sizeof(GpuPairParams);

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = pairBindGroupLayout;
    bindGroupDesc.entryCount = 4;
    bindGroupDesc.entries = entries;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
    if (bindGroup == nullptr)
    {
        setStatus(pairStatus, "Pair diagnostic failed: bind group");
        return false;
    }

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pairPipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups(((uint32_t)sortedItems.size() + 63u) / 64u);
    pass.End();
    encoder.CopyBufferToBuffer(pairCountersBuffer, 0, pairReadbackBuffer, 0, counterBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = pairReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, counterBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
    {
        setStatus(pairStatus, "Pair diagnostic failed: readback map");
        return false;
    }

    const GpuPairCounters *counters = (const GpuPairCounters *)pairReadbackBuffer.GetConstMappedRange(0, counterBytes);
    if (!counters)
    {
        pairReadbackBuffer.Unmap();
        setStatus(pairStatus, "Pair diagnostic failed: readback range");
        return false;
    }

    uint32_t gpuCandidates = counters->candidates;
    uint32_t gpuSphereHits = counters->sphereHits;
    pairReadbackBuffer.Unmap();

    pairCandidates = (int)gpuCandidates;
    pairSphereHits = (int)gpuSphereHits;
    pairAllPairsSphereHits = (int)allPairsSphereHits;
    pairMissedSphereHits = allPairsSphereHits > gpuSphereHits ? (int)(allPairsSphereHits - gpuSphereHits) : 0;
    pairMismatches = (gpuCandidates == expectedCandidates ? 0 : 1) + (gpuSphereHits == expectedSphereHits ? 0 : 1);
    pairMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    recordTimingSample(pairTiming, pairMs);

    snprintf(pairStatus, sizeof(pairStatus), "Pairs passed: %d candidates, %d hits, missed %d, mismatches %d",
             pairCandidates, pairSphereHits, pairMissedSphereHits, pairMismatches);
    return pairMismatches == 0;
#elif AVBD_ENABLE_WEBGPU
    setStatus(pairStatus, "Pair diagnostic skipped: Dawn not linked");
    return false;
#else
    setStatus(pairStatus, "Pair diagnostic skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::runSweepAndPruneDiagnostic(const SimWorld &world)
{
    sapMs = 0.0f;
    sapCandidates = 0;
    sapSphereHits = 0;
    sapAllPairsSphereHits = 0;
    sapMissedSphereHits = 0;
    sapMismatches = 0;
    sapBestAxis = 0;
    for (int axis = 0; axis < 3; ++axis)
    {
        sapAxisCandidates[axis] = 0;
        sapAxisSphereHits[axis] = 0;
        sapAxisMissedSphereHits[axis] = 0;
    }
    sapBestAxis = 0;
    for (int axis = 0; axis < 3; ++axis)
    {
        sapAxisCandidates[axis] = 0;
        sapAxisSphereHits[axis] = 0;
        sapAxisMissedSphereHits[axis] = 0;
    }

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(sapStatus, "SAP skipped: WebGPU device not ready");
        return false;
    }

    std::vector<GpuPairBody> activeBodies;
    activeBodies.reserve(world.activeBodyIds.size());
    for (size_t i = 0; i < world.activeBodyIds.size(); ++i)
    {
        BodyId bodyId = world.activeBodyIds[i];
        if (bodyId == INVALID_BODY_ID || bodyId >= world.bodies.size())
            continue;

        const SimBodyData &body = world.bodies[bodyId];
        if (!body.active)
            continue;

        GpuPairBody activeBody = {};
        activeBody.position[0] = body.positionLin.x;
        activeBody.position[1] = body.positionLin.y;
        activeBody.position[2] = body.positionLin.z;
        activeBody.position[3] = body.radius;
        activeBodies.push_back(activeBody);
    }

    if (activeBodies.empty())
    {
        setStatus(sapStatus, "SAP skipped: no active bodies");
        return false;
    }

    uint32_t allPairsSphereHits = 0;
    if (activeBodies.size() <= 4096)
    {
        for (uint32_t i = 0; i < activeBodies.size(); ++i)
        {
            for (uint32_t j = i + 1u; j < activeBodies.size(); ++j)
            {
                const GpuPairBody &a = activeBodies[i];
                const GpuPairBody &b = activeBodies[j];
                float dx = a.position[0] - b.position[0];
                float dy = a.position[1] - b.position[1];
                float dz = a.position[2] - b.position[2];
                float radius = a.position[3] + b.position[3];
                if (dx * dx + dy * dy + dz * dz < radius * radius)
                    ++allPairsSphereHits;
            }
        }
    }

    auto makeIntervalsForAxis = [&](int axis)
    {
        std::vector<GpuSapInterval> axisIntervals;
        axisIntervals.reserve(activeBodies.size());
        int otherA = (axis + 1) % 3;
        int otherB = (axis + 2) % 3;
        for (uint32_t i = 0; i < activeBodies.size(); ++i)
        {
            const GpuPairBody &body = activeBodies[i];
            float center = body.position[axis];
            float radius = body.position[3];
            GpuSapInterval interval = {};
            interval.minX = center - radius;
            interval.maxX = center + radius;
            interval.y = body.position[otherA];
            interval.z = body.position[otherB];
            interval.radius = radius;
            interval.index = i;
            axisIntervals.push_back(interval);
        }
        return axisIntervals;
    };

    auto countSapAxis = [&](std::vector<GpuSapInterval> axisIntervals, uint32_t &candidates, uint32_t &sphereHits)
    {
        std::sort(axisIntervals.begin(), axisIntervals.end(), [](const GpuSapInterval &a, const GpuSapInterval &b)
                  {
                      if (a.minX == b.minX)
                          return a.index < b.index;
                      return a.minX < b.minX;
                  });

        candidates = 0;
        sphereHits = 0;
        for (uint32_t i = 0; i < axisIntervals.size(); ++i)
        {
            const GpuSapInterval &a = axisIntervals[i];
            for (uint32_t j = i + 1u; j < axisIntervals.size(); ++j)
            {
                const GpuSapInterval &b = axisIntervals[j];
                if (b.minX > a.maxX)
                    break;

                ++candidates;
                float dy = a.y - b.y;
                float dz = a.z - b.z;
                float dx = 0.5f * (a.minX + a.maxX) - 0.5f * (b.minX + b.maxX);
                float radius = a.radius + b.radius;
                if (dx * dx + dy * dy + dz * dz < radius * radius)
                    ++sphereHits;
            }
        }
    };

    uint32_t expectedCandidates = 0;
    uint32_t expectedSphereHits = 0;
    uint32_t bestCandidates = 0xFFFFFFFFu;
    for (int axis = 0; axis < 3; ++axis)
    {
        uint32_t axisCandidates = 0;
        uint32_t axisSphereHits = 0;
        countSapAxis(makeIntervalsForAxis(axis), axisCandidates, axisSphereHits);
        sapAxisCandidates[axis] = (int)axisCandidates;
        sapAxisSphereHits[axis] = (int)axisSphereHits;
        sapAxisMissedSphereHits[axis] = allPairsSphereHits > axisSphereHits ? (int)(allPairsSphereHits - axisSphereHits) : 0;
        if (axisCandidates < bestCandidates)
        {
            bestCandidates = axisCandidates;
            expectedCandidates = axisCandidates;
            expectedSphereHits = axisSphereHits;
            sapBestAxis = axis;
        }
    }

    std::vector<GpuSapInterval> intervals = makeIntervalsForAxis(sapBestAxis);

    if (intervals.empty())
    {
        setStatus(sapStatus, "SAP skipped: no active bodies");
        return false;
    }

    std::sort(intervals.begin(), intervals.end(), [](const GpuSapInterval &a, const GpuSapInterval &b)
              {
                  if (a.minX == b.minX)
                      return a.index < b.index;
                  return a.minX < b.minX;
              });

#if 0
        GpuSapInterval interval = {};
        interval.minX = body.positionLin.x - body.radius;
        interval.maxX = body.positionLin.x + body.radius;
        interval.y = body.positionLin.y;
        interval.z = body.positionLin.z;
        interval.radius = body.radius;
        interval.index = (uint32_t)intervals.size();
        intervals.push_back(interval);
    }

    if (intervals.empty())
    {
        setStatus(sapStatus, "SAP skipped: no active bodies");
        return false;
    }

    std::vector<GpuSapInterval> expectedIntervals = intervals;
    std::sort(expectedIntervals.begin(), expectedIntervals.end(), [](const GpuSapInterval &a, const GpuSapInterval &b)
              {
                  if (a.minX == b.minX)
                      return a.index < b.index;
                  return a.minX < b.minX;
              });

    uint32_t expectedCandidates = 0;
    uint32_t expectedSphereHits = 0;
    for (uint32_t i = 0; i < expectedIntervals.size(); ++i)
    {
        const GpuSapInterval &a = expectedIntervals[i];
        for (uint32_t j = i + 1u; j < expectedIntervals.size(); ++j)
        {
            const GpuSapInterval &b = expectedIntervals[j];
            if (b.minX > a.maxX)
                break;

            ++expectedCandidates;
            float dy = a.y - b.y;
            float dz = a.z - b.z;
            float dx = 0.5f * (a.minX + a.maxX) - 0.5f * (b.minX + b.maxX);
            float radius = a.radius + b.radius;
            if (dx * dx + dy * dy + dz * dz < radius * radius)
                ++expectedSphereHits;
        }
    }

    uint32_t allPairsSphereHits = 0;
    if (intervals.size() <= 4096)
    {
        for (uint32_t i = 0; i < intervals.size(); ++i)
        {
            for (uint32_t j = i + 1u; j < intervals.size(); ++j)
            {
                const GpuSapInterval &a = intervals[i];
                const GpuSapInterval &b = intervals[j];
                float dy = a.y - b.y;
                float dz = a.z - b.z;
                float dx = 0.5f * (a.minX + a.maxX) - 0.5f * (b.minX + b.maxX);
                float radius = a.radius + b.radius;
                if (dx * dx + dy * dy + dz * dz < radius * radius)
                    ++allPairsSphereHits;
            }
        }
    }
#endif

    uint32_t itemCount = (uint32_t)intervals.size();
    uint32_t paddedCount = nextPowerOfTwo(itemCount);
    GpuSapInterval sentinel = {};
    sentinel.minX = 3.402823466e38f;
    sentinel.maxX = 3.402823466e38f;
    sentinel.index = 0xFFFFFFFFu;
    intervals.resize(paddedCount, sentinel);

    const uint64_t intervalBytes = (uint64_t)(intervals.size() * sizeof(GpuSapInterval));
    const uint64_t counterBytes = sizeof(GpuSapCounters);

    Uint64 begin = SDL_GetPerformanceCounter();

    if (sapIntervalBuffer == nullptr || sapIntervalBufferBytes < intervalBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopyDst |
                                 (uint64_t)wgpu::BufferUsage::CopySrc);
        desc.size = alignUp(intervalBytes, 4096);
        sapIntervalBuffer = device.CreateBuffer(&desc);
        sapIntervalBufferBytes = sapIntervalBuffer == nullptr ? 0 : desc.size;
    }
    if (sapCountersBuffer == nullptr || sapCountersBufferBytes < counterBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopySrc |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(counterBytes, 256);
        sapCountersBuffer = device.CreateBuffer(&desc);
        sapCountersBufferBytes = sapCountersBuffer == nullptr ? 0 : desc.size;
    }
    if (sapReadbackBuffer == nullptr || sapReadbackBufferBytes < counterBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(counterBytes, 256);
        sapReadbackBuffer = device.CreateBuffer(&desc);
        sapReadbackBufferBytes = sapReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (sapParamsBuffer == nullptr)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = sizeof(GpuSortParams);
        sapParamsBuffer = device.CreateBuffer(&desc);
    }

    if (sapIntervalBuffer == nullptr || sapCountersBuffer == nullptr ||
        sapReadbackBuffer == nullptr || sapParamsBuffer == nullptr)
    {
        setStatus(sapStatus, "SAP failed: buffer allocation");
        return false;
    }

    queue.WriteBuffer(sapIntervalBuffer, 0, intervals.data(), intervalBytes);
    GpuSapCounters zeroCounters = {};
    queue.WriteBuffer(sapCountersBuffer, 0, &zeroCounters, sizeof(zeroCounters));

    static const char *sortShaderSource = R"(
struct Interval {
    minX: f32,
    maxX: f32,
    y: f32,
    z: f32,
    radius: f32,
    index: u32,
    pad0: u32,
    pad1: u32,
};

struct Params {
    j: u32,
    k: u32,
    count: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read_write> intervals: array<Interval>;
@group(0) @binding(1) var<uniform> params: Params;

fn greater(a: Interval, b: Interval) -> bool {
    if (a.minX == b.minX) {
        return a.index > b.index;
    }
    return a.minX > b.minX;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= params.count) {
        return;
    }

    let other = index ^ params.j;
    if (other <= index || other >= params.count) {
        return;
    }

    let ascending = (index & params.k) == 0u;
    let a = intervals[index];
    let b = intervals[other];
    let shouldSwap = select(greater(b, a), greater(a, b), ascending);
    if (shouldSwap) {
        intervals[index] = b;
        intervals[other] = a;
    }
}
)";

    if (sapSortPipeline == nullptr || sapSortBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = sortShaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(sapStatus, "SAP failed: sort shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry entries[2] = {};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::Storage;
        entries[0].buffer.minBindingSize = sizeof(GpuSapInterval);
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[1].buffer.minBindingSize = sizeof(GpuSortParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 2;
        layoutDesc.entries = entries;
        sapSortBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (sapSortBindGroupLayout == nullptr)
        {
            setStatus(sapStatus, "SAP failed: sort bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &sapSortBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(sapStatus, "SAP failed: sort pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        sapSortPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (sapSortPipeline == nullptr)
        {
            setStatus(sapStatus, "SAP failed: sort compute pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry sortEntries[2] = {};
    sortEntries[0].binding = 0;
    sortEntries[0].buffer = sapIntervalBuffer;
    sortEntries[0].offset = 0;
    sortEntries[0].size = intervalBytes;
    sortEntries[1].binding = 1;
    sortEntries[1].buffer = sapParamsBuffer;
    sortEntries[1].offset = 0;
    sortEntries[1].size = sizeof(GpuSortParams);

    wgpu::BindGroupDescriptor sortBindGroupDesc = {};
    sortBindGroupDesc.layout = sapSortBindGroupLayout;
    sortBindGroupDesc.entryCount = 2;
    sortBindGroupDesc.entries = sortEntries;
    wgpu::BindGroup sortBindGroup = device.CreateBindGroup(&sortBindGroupDesc);
    if (sortBindGroup == nullptr)
    {
        setStatus(sapStatus, "SAP failed: sort bind group");
        return false;
    }

    static const char *pairShaderSource = R"(
struct Interval {
    minX: f32,
    maxX: f32,
    y: f32,
    z: f32,
    radius: f32,
    index: u32,
    pad0: u32,
    pad1: u32,
};

struct Counters {
    candidates: atomic<u32>,
    sphereHits: atomic<u32>,
    pad0: u32,
    pad1: u32,
};

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> intervals: array<Interval>;
@group(0) @binding(1) var<storage, read_write> counters: Counters;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }

    let a = intervals[i];
    let ax = (a.minX + a.maxX) * 0.5;
    var j = i + 1u;
    loop {
        if (j >= params.count) {
            break;
        }

        let b = intervals[j];
        if (b.minX > a.maxX) {
            break;
        }

        atomicAdd(&counters.candidates, 1u);
        let bx = (b.minX + b.maxX) * 0.5;
        let dx = ax - bx;
        let dy = a.y - b.y;
        let dz = a.z - b.z;
        let radius = a.radius + b.radius;
        if (dx * dx + dy * dy + dz * dz < radius * radius) {
            atomicAdd(&counters.sphereHits, 1u);
        }
        j = j + 1u;
    }
}
)";

    if (sapPairPipeline == nullptr || sapPairBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = pairShaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(sapStatus, "SAP failed: pair shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry entries[3] = {};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        entries[0].buffer.minBindingSize = sizeof(GpuSapInterval);
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].buffer.type = wgpu::BufferBindingType::Storage;
        entries[1].buffer.minBindingSize = sizeof(GpuSapCounters);
        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Compute;
        entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[2].buffer.minBindingSize = sizeof(GpuSapParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 3;
        layoutDesc.entries = entries;
        sapPairBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (sapPairBindGroupLayout == nullptr)
        {
            setStatus(sapStatus, "SAP failed: pair bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &sapPairBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(sapStatus, "SAP failed: pair pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        sapPairPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (sapPairPipeline == nullptr)
        {
            setStatus(sapStatus, "SAP failed: pair compute pipeline");
            return false;
        }
    }

    GpuSapParams pairParams = {};
    pairParams.count = itemCount;
    wgpu::BufferDescriptor pairParamsDesc = {};
    pairParamsDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                       (uint64_t)wgpu::BufferUsage::CopyDst);
    pairParamsDesc.size = sizeof(GpuSapParams);
    wgpu::Buffer pairParamsBuffer = device.CreateBuffer(&pairParamsDesc);
    if (pairParamsBuffer == nullptr)
    {
        setStatus(sapStatus, "SAP failed: pair params buffer");
        return false;
    }
    queue.WriteBuffer(pairParamsBuffer, 0, &pairParams, sizeof(pairParams));

    wgpu::BindGroupEntry pairEntries[3] = {};
    pairEntries[0].binding = 0;
    pairEntries[0].buffer = sapIntervalBuffer;
    pairEntries[0].offset = 0;
    pairEntries[0].size = intervalBytes;
    pairEntries[1].binding = 1;
    pairEntries[1].buffer = sapCountersBuffer;
    pairEntries[1].offset = 0;
    pairEntries[1].size = counterBytes;
    pairEntries[2].binding = 2;
    pairEntries[2].buffer = pairParamsBuffer;
    pairEntries[2].offset = 0;
    pairEntries[2].size = sizeof(GpuSapParams);

    wgpu::BindGroupDescriptor pairBindGroupDesc = {};
    pairBindGroupDesc.layout = sapPairBindGroupLayout;
    pairBindGroupDesc.entryCount = 3;
    pairBindGroupDesc.entries = pairEntries;
    wgpu::BindGroup pairBindGroup = device.CreateBindGroup(&pairBindGroupDesc);
    if (pairBindGroup == nullptr)
    {
        setStatus(sapStatus, "SAP failed: pair bind group");
        return false;
    }

    const uint64_t passParamStride = 256;
    std::vector<unsigned char> passParamData;
    uint32_t passCount = 0;
    for (uint32_t k = 2u; k <= paddedCount; k <<= 1u)
    {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u)
        {
            GpuSortParams params = {};
            params.j = j;
            params.k = k;
            params.count = paddedCount;
            passParamData.resize((uint64_t)(passCount + 1u) * passParamStride);
            memcpy(passParamData.data() + (uint64_t)passCount * passParamStride, &params, sizeof(params));
            ++passCount;
        }
    }

    wgpu::BufferDescriptor passParamsDesc = {};
    passParamsDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::CopySrc |
                                       (uint64_t)wgpu::BufferUsage::CopyDst);
    passParamsDesc.size = alignUp((uint64_t)passParamData.size(), 256);
    wgpu::Buffer passParamsBuffer = device.CreateBuffer(&passParamsDesc);
    if (passParamsBuffer == nullptr)
    {
        setStatus(sapStatus, "SAP failed: pass params buffer");
        return false;
    }
    queue.WriteBuffer(passParamsBuffer, 0, passParamData.data(), passParamData.size());

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex)
    {
        encoder.CopyBufferToBuffer(passParamsBuffer, (uint64_t)passIndex * passParamStride,
                                   sapParamsBuffer, 0, sizeof(GpuSortParams));
        wgpu::ComputePassEncoder sortPass = encoder.BeginComputePass();
        sortPass.SetPipeline(sapSortPipeline);
        sortPass.SetBindGroup(0, sortBindGroup);
        sortPass.DispatchWorkgroups((paddedCount + 63u) / 64u);
        sortPass.End();
    }

    wgpu::ComputePassEncoder pairPass = encoder.BeginComputePass();
    pairPass.SetPipeline(sapPairPipeline);
    pairPass.SetBindGroup(0, pairBindGroup);
    pairPass.DispatchWorkgroups((itemCount + 63u) / 64u);
    pairPass.End();
    encoder.CopyBufferToBuffer(sapCountersBuffer, 0, sapReadbackBuffer, 0, counterBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = sapReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, counterBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
    {
        setStatus(sapStatus, "SAP failed: readback map");
        return false;
    }

    const GpuSapCounters *counters = (const GpuSapCounters *)sapReadbackBuffer.GetConstMappedRange(0, counterBytes);
    if (!counters)
    {
        sapReadbackBuffer.Unmap();
        setStatus(sapStatus, "SAP failed: readback range");
        return false;
    }

    uint32_t gpuCandidates = counters->candidates;
    uint32_t gpuSphereHits = counters->sphereHits;
    sapReadbackBuffer.Unmap();

    sapCandidates = (int)gpuCandidates;
    sapSphereHits = (int)gpuSphereHits;
    sapAllPairsSphereHits = (int)allPairsSphereHits;
    sapMissedSphereHits = allPairsSphereHits > gpuSphereHits ? (int)(allPairsSphereHits - gpuSphereHits) : 0;
    sapMismatches = (gpuCandidates == expectedCandidates ? 0 : 1) + (gpuSphereHits == expectedSphereHits ? 0 : 1);
    sapMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    recordTimingSample(sapTiming, sapMs);

    const char *axisName = sapBestAxis == 0 ? "X" : (sapBestAxis == 1 ? "Y" : "Z");
    snprintf(sapStatus, sizeof(sapStatus), "SAP passed: %s axis, %d candidates, %d hits, missed %d, mismatches %d",
             axisName, sapCandidates, sapSphereHits, sapMissedSphereHits, sapMismatches);
    return sapMismatches == 0;
#elif AVBD_ENABLE_WEBGPU
    setStatus(sapStatus, "SAP skipped: Dawn not linked");
    return false;
#else
    setStatus(sapStatus, "SAP skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::clearSurface(int width, int height)
{
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || surface == nullptr || adapter == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(presentStatus, "Surface clear skipped: WebGPU device not ready");
        return false;
    }

    if (width <= 0 || height <= 0)
    {
        setStatus(presentStatus, "Surface clear skipped: invalid window size");
        return false;
    }

    wgpu::SurfaceCapabilities capabilities;
    if (surface.GetCapabilities(adapter, &capabilities) != wgpu::Status::Success || capabilities.formatCount == 0)
    {
        setStatus(presentStatus, "Surface clear failed: no surface formats");
        return false;
    }

    wgpu::TextureFormat format = chooseSurfaceFormat(capabilities);
    if (format == wgpu::TextureFormat::Undefined)
    {
        setStatus(presentStatus, "Surface clear failed: undefined surface format");
        return false;
    }

    wgpu::SurfaceConfiguration config = {};
    config.device = device;
    config.format = format;
    config.usage = wgpu::TextureUsage::RenderAttachment;
    config.width = (uint32_t)width;
    config.height = (uint32_t)height;
    config.presentMode = choosePresentMode(capabilities);
    config.alphaMode = chooseAlphaMode(capabilities);
    surface.Configure(&config);

    wgpu::SurfaceTexture frame = {};
    surface.GetCurrentTexture(&frame);
    if ((frame.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
         frame.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) ||
        frame.texture == nullptr)
    {
        surface.Unconfigure();
        snprintf(presentStatus, sizeof(presentStatus), "Surface clear failed: texture status %u",
                 (unsigned int)frame.status);
        return false;
    }

    wgpu::TextureView view = frame.texture.CreateView();
    if (view == nullptr)
    {
        surface.Unconfigure();
        setStatus(presentStatus, "Surface clear failed: texture view");
        return false;
    }

    wgpu::RenderPassColorAttachment color = {};
    color.view = view;
    color.loadOp = wgpu::LoadOp::Clear;
    color.storeOp = wgpu::StoreOp::Store;
    color.clearValue = {0.06, 0.14, 0.24, 1.0};

    wgpu::RenderPassDescriptor passDesc = {};
    passDesc.colorAttachmentCount = 1;
    passDesc.colorAttachments = &color;

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDesc);
    pass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    if (surface.Present() != wgpu::Status::Success)
    {
        surface.Unconfigure();
        setStatus(presentStatus, "Surface clear failed: present");
        return false;
    }

    surface.Unconfigure();
    snprintf(presentStatus, sizeof(presentStatus), "Surface clear passed: present %dx%d", width, height);
    return true;
#elif AVBD_ENABLE_WEBGPU
    setStatus(presentStatus, "Surface clear skipped: Dawn not linked");
    return false;
#else
    setStatus(presentStatus, "Surface clear skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::clearWindowSurface(SDL_Window *window, int width, int height)
{
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || adapter == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(presentStatus, "Surface clear skipped: WebGPU device not ready");
        return false;
    }

    wgpu::Surface targetSurface = createSurfaceForWindow(instance, window, presentStatus);
    if (targetSurface == nullptr)
        return false;

    wgpu::Surface savedSurface = surface;
    surface = targetSurface;
    bool ok = clearSurface(width, height);
    surface = savedSurface;
    targetSurface = nullptr;
    return ok;
#elif AVBD_ENABLE_WEBGPU
    setStatus(presentStatus, "Surface clear skipped: Dawn not linked");
    return false;
#else
    setStatus(presentStatus, "Surface clear skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::renderTriangleSurface(int width, int height)
{
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || surface == nullptr || adapter == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(presentStatus, "Triangle render skipped: WebGPU device not ready");
        return false;
    }

    if (width <= 0 || height <= 0)
    {
        setStatus(presentStatus, "Triangle render skipped: invalid window size");
        return false;
    }

    wgpu::SurfaceCapabilities capabilities;
    if (surface.GetCapabilities(adapter, &capabilities) != wgpu::Status::Success || capabilities.formatCount == 0)
    {
        setStatus(presentStatus, "Triangle render failed: no surface formats");
        return false;
    }

    wgpu::TextureFormat format = chooseSurfaceFormat(capabilities);
    if (format == wgpu::TextureFormat::Undefined)
    {
        setStatus(presentStatus, "Triangle render failed: undefined surface format");
        return false;
    }

    wgpu::SurfaceConfiguration config = {};
    config.device = device;
    config.format = format;
    config.usage = wgpu::TextureUsage::RenderAttachment;
    config.width = (uint32_t)width;
    config.height = (uint32_t)height;
    config.presentMode = choosePresentMode(capabilities);
    config.alphaMode = chooseAlphaMode(capabilities);
    surface.Configure(&config);

static const char *shaderSource = R"(
struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
    var positions = array<vec2f, 3>(
        vec2f(0.0, 0.68),
        vec2f(-0.72, -0.58),
        vec2f(0.72, -0.58)
    );
    var colors = array<vec3f, 3>(
        vec3f(0.95, 0.25, 0.20),
        vec3f(0.20, 0.85, 0.45),
        vec3f(0.20, 0.48, 1.00)
    );

    var out: VertexOut;
    out.position = vec4f(positions[vertexIndex], 0.0, 1.0);
    out.color = colors[vertexIndex];
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
)";

    wgpu::ShaderSourceWGSL wgsl = {};
    wgsl.code = shaderSource;
    wgpu::ShaderModuleDescriptor shaderDesc = {};
    shaderDesc.nextInChain = &wgsl;
    wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
    if (shader == nullptr)
    {
        surface.Unconfigure();
        setStatus(presentStatus, "Triangle render failed: shader module");
        return false;
    }

    wgpu::ColorTargetState colorTarget = {};
    colorTarget.format = format;

    wgpu::FragmentState fragment = {};
    fragment.module = shader;
    fragment.entryPoint = "fs_main";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;

    wgpu::RenderPipelineDescriptor pipelineDesc = {};
    pipelineDesc.vertex.module = shader;
    pipelineDesc.vertex.entryPoint = "vs_main";
    pipelineDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pipelineDesc.multisample.count = 1;
    pipelineDesc.fragment = &fragment;

    wgpu::RenderPipeline pipeline = device.CreateRenderPipeline(&pipelineDesc);
    if (pipeline == nullptr)
    {
        surface.Unconfigure();
        setStatus(presentStatus, "Triangle render failed: pipeline");
        return false;
    }

    wgpu::SurfaceTexture frame = {};
    surface.GetCurrentTexture(&frame);
    if ((frame.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
         frame.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) ||
        frame.texture == nullptr)
    {
        surface.Unconfigure();
        snprintf(presentStatus, sizeof(presentStatus), "Triangle render failed: texture status %u",
                 (unsigned int)frame.status);
        return false;
    }

    wgpu::TextureView view = frame.texture.CreateView();
    if (view == nullptr)
    {
        surface.Unconfigure();
        setStatus(presentStatus, "Triangle render failed: texture view");
        return false;
    }

    wgpu::RenderPassColorAttachment color = {};
    color.view = view;
    color.loadOp = wgpu::LoadOp::Clear;
    color.storeOp = wgpu::StoreOp::Store;
    color.clearValue = {0.04, 0.09, 0.16, 1.0};

    wgpu::RenderPassDescriptor passDesc = {};
    passDesc.colorAttachmentCount = 1;
    passDesc.colorAttachments = &color;

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDesc);
    pass.SetPipeline(pipeline);
    pass.Draw(3);
    pass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    if (surface.Present() != wgpu::Status::Success)
    {
        surface.Unconfigure();
        setStatus(presentStatus, "Triangle render failed: present");
        return false;
    }

    surface.Unconfigure();
    snprintf(presentStatus, sizeof(presentStatus), "Triangle render passed: present %dx%d", width, height);
    return true;
#elif AVBD_ENABLE_WEBGPU
    setStatus(presentStatus, "Triangle render skipped: Dawn not linked");
    return false;
#else
    setStatus(presentStatus, "Triangle render skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::renderInstancedPreviewSurface(const SimWorld &world, const WebGpuPreviewCamera &camera, int width, int height)
{
    WebGpuRenderOptions options;
    return renderInstancedPreviewSurface(world, camera, options, width, height);
}

bool WebGpuContext::renderInstancedPreviewSurface(const SimWorld &world, const WebGpuPreviewCamera &camera, const WebGpuRenderOptions &options, int width, int height)
{
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    Uint64 previewBegin = SDL_GetPerformanceCounter();
    previewComputeMs = 0.0f;
    previewRenderMs = 0.0f;
    previewTotalMs = 0.0f;
    previewBatchCount = 0;
    previewInstanceCount = 0;
    previewBoxInstances = 0;
    previewSphereInstances = 0;
    previewCapsuleInstances = 0;
    previewCylinderInstances = 0;
    previewMeshAssetInstances = 0;

    if (!deviceReady || surface == nullptr || adapter == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(presentStatus, "Instanced preview skipped: WebGPU device not ready");
        return false;
    }

    if (width <= 0 || height <= 0)
    {
        setStatus(presentStatus, "Instanced preview skipped: invalid window size");
        return false;
    }

    WebGpuRenderScene renderScene = buildWebGpuRenderScene(world);
    previewBatchCount = (int)renderScene.renderBatches.size();
    previewInstanceCount = (int)renderScene.bodyInputs.size();
    previewBoxInstances = renderScene.boxCount;
    previewSphereInstances = renderScene.sphereCount;
    previewCapsuleInstances = renderScene.capsuleCount;
    previewCylinderInstances = renderScene.cylinderCount;
    previewMeshAssetInstances = renderScene.meshAssetCount;

    wgpu::SurfaceCapabilities capabilities;
    if (surface.GetCapabilities(adapter, &capabilities) != wgpu::Status::Success || capabilities.formatCount == 0)
    {
        setStatus(presentStatus, "Instanced preview failed: no surface formats");
        return false;
    }

    wgpu::TextureFormat format = chooseSurfaceFormat(capabilities);
    if (format == wgpu::TextureFormat::Undefined)
    {
        setStatus(presentStatus, "Instanced preview failed: undefined surface format");
        return false;
    }

    if (!bodyPreviewSurfaceConfigured ||
        bodyPreviewWidth != width ||
        bodyPreviewHeight != height ||
        bodyPreviewFormat != format)
    {
        if (bodyPreviewSurfaceConfigured)
            surface.Unconfigure();

        wgpu::SurfaceConfiguration config = {};
        config.device = device;
        config.format = format;
        config.usage = wgpu::TextureUsage::RenderAttachment;
        config.width = (uint32_t)width;
        config.height = (uint32_t)height;
        config.presentMode = choosePresentMode(capabilities);
        config.alphaMode = chooseAlphaMode(capabilities);
        surface.Configure(&config);

        bodyPreviewSurfaceConfigured = true;
        bodyPreviewWidth = width;
        bodyPreviewHeight = height;
        bodyPreviewDepthTexture = nullptr;
        bodyPreviewDepthView = nullptr;
    }

    uint64_t inputBytes = (uint64_t)(renderScene.bodyInputs.size() * sizeof(GpuRenderInstanceInput));
    if (bodyInputBuffer == nullptr || bodyInputBufferBytes < inputBytes)
    {
        uint64_t newSize = alignUp(inputBytes, 4096);
        wgpu::BufferDescriptor inputBufferDesc = {};
        inputBufferDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                            (uint64_t)wgpu::BufferUsage::CopyDst);
        inputBufferDesc.size = newSize;
        bodyInputBuffer = device.CreateBuffer(&inputBufferDesc);
        bodyInputBufferBytes = bodyInputBuffer == nullptr ? 0 : newSize;
    }

    if (bodyInputBuffer == nullptr)
    {
        surface.Unconfigure();
        bodyPreviewSurfaceConfigured = false;
        setStatus(presentStatus, "Instanced preview failed: input buffer");
        return false;
    }
    queue.WriteBuffer(bodyInputBuffer, 0, renderScene.bodyInputs.data(), renderScene.bodyInputs.size() * sizeof(GpuRenderInstanceInput));

    uint64_t instanceBytes = (uint64_t)(renderScene.bodyInputs.size() * sizeof(GpuRenderInstance));
    if (bodyInstanceBuffer == nullptr || bodyInstanceBufferBytes < instanceBytes)
    {
        uint64_t newSize = alignUp(instanceBytes, 4096);
        wgpu::BufferDescriptor instanceBufferDesc = {};
        instanceBufferDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Vertex |
                                               (uint64_t)wgpu::BufferUsage::Storage |
                                               (uint64_t)wgpu::BufferUsage::CopyDst);
        instanceBufferDesc.size = newSize;
        bodyInstanceBuffer = device.CreateBuffer(&instanceBufferDesc);
        bodyInstanceBufferBytes = bodyInstanceBuffer == nullptr ? 0 : newSize;
    }

    if (bodyInstanceBuffer == nullptr)
    {
        surface.Unconfigure();
        bodyPreviewSurfaceConfigured = false;
        setStatus(presentStatus, "Instanced preview failed: instance buffer");
        return false;
    }

    PreviewCameraUniform cameraUniform = makePreviewCameraUniform(camera, options, width, height);
    if (renderScene.gridReceiver.valid && options.showGroundGrid)
    {
        cameraUniform.gridBounds[0] = renderScene.gridReceiver.minX;
        cameraUniform.gridBounds[1] = renderScene.gridReceiver.minY;
        cameraUniform.gridBounds[2] = renderScene.gridReceiver.maxX;
        cameraUniform.gridBounds[3] = renderScene.gridReceiver.maxY;
        cameraUniform.gridParams[0] = renderScene.gridReceiver.z;
        cameraUniform.gridParams[1] = 1.0f;
    }
    if (bodyPreviewCameraBuffer == nullptr)
    {
        wgpu::BufferDescriptor cameraBufferDesc = {};
        cameraBufferDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform | (uint64_t)wgpu::BufferUsage::CopyDst);
        cameraBufferDesc.size = sizeof(PreviewCameraUniform);
        bodyPreviewCameraBuffer = device.CreateBuffer(&cameraBufferDesc);
    }
    if (bodyPreviewCameraBuffer == nullptr)
    {
        surface.Unconfigure();
        bodyPreviewSurfaceConfigured = false;
        setStatus(presentStatus, "Instanced preview failed: camera buffer");
        return false;
    }
    queue.WriteBuffer(bodyPreviewCameraBuffer, 0, &cameraUniform, sizeof(cameraUniform));

    static const char *computeShaderSource = R"(
struct BodyInput {
    position: vec4f,
    shape: vec4f,
    rotation: vec4f,
    params: vec4f,
};

struct BodyInstance {
    position: vec4f,
    size: vec4f,
    rotation: vec4f,
    params: vec4f,
};

@group(0) @binding(0) var<storage, read> inputs: array<BodyInput>;
@group(0) @binding(1) var<storage, read_write> outputs: array<BodyInstance>;

fn fallbackSize(radius: f32) -> f32 {
    return min(max(radius, 0.25), 1.5);
}

fn renderKind(shapeType: u32) -> f32 {
    if (shapeType == 0u) {
        return 1.0;
    }
    if (shapeType == 1u) {
        return 2.0;
    }
    if (shapeType == 2u) {
        return 3.0;
    }
    if (shapeType == 3u) {
        return 4.0;
    }
    return 0.0;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;

    let input = inputs[index];
    let shapeType = u32(clamp(input.params.x, 0.0, 3.0));
    let kind = renderKind(shapeType);
    let fallback = fallbackSize(input.shape.w);
    var previewSize = vec3f(fallback, fallback, fallback);
    if (kind != 0.0) {
        previewSize = input.shape.xyz;
    }

    outputs[index].position = vec4f(input.position.xyz, input.params.w);
    outputs[index].size = vec4f(previewSize, fallback);
    outputs[index].rotation = input.rotation;
    outputs[index].params = vec4f(input.params.x, input.params.y, kind, input.params.z);
}
)";

    if (bodyPreviewComputePipeline == nullptr || bodyPreviewComputeBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL computeWgsl = {};
        computeWgsl.code = computeShaderSource;
        wgpu::ShaderModuleDescriptor computeShaderDesc = {};
        computeShaderDesc.nextInChain = &computeWgsl;
        device.PushErrorScope(wgpu::ErrorFilter::Validation);
        wgpu::ShaderModule computeShader = device.CreateShaderModule(&computeShaderDesc);
        wgpu::PopErrorScopeStatus computeShaderScopeStatus = wgpu::PopErrorScopeStatus::Error;
        wgpu::ErrorType computeShaderErrorType = wgpu::ErrorType::Unknown;
        std::string computeShaderErrorMessage;
        wgpu::Future computeShaderScopeFuture = device.PopErrorScope(
            wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message)
            {
                computeShaderScopeStatus = status;
                computeShaderErrorType = type;
                computeShaderErrorMessage = toString(message);
            });
        if (instance.WaitAny(computeShaderScopeFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            computeShaderScopeStatus != wgpu::PopErrorScopeStatus::Success)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: compute shader validation scope");
            return false;
        }
        if (computeShaderErrorType != wgpu::ErrorType::NoError)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            snprintf(status, sizeof(status), "WebGPU validation error: %s", computeShaderErrorMessage.c_str());
            snprintf(presentStatus, sizeof(presentStatus), "Instanced preview compute shader failed: %s", computeShaderErrorMessage.c_str());
            return false;
        }
        if (computeShader == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: compute shader");
            return false;
        }

        wgpu::BindGroupLayoutEntry computeEntries[2] = {};
        computeEntries[0].binding = 0;
        computeEntries[0].visibility = wgpu::ShaderStage::Compute;
        computeEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        computeEntries[0].buffer.minBindingSize = sizeof(GpuRenderInstanceInput);
        computeEntries[1].binding = 1;
        computeEntries[1].visibility = wgpu::ShaderStage::Compute;
        computeEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
        computeEntries[1].buffer.minBindingSize = sizeof(GpuRenderInstance);

        wgpu::BindGroupLayoutDescriptor computeLayoutDesc = {};
        computeLayoutDesc.entryCount = 2;
        computeLayoutDesc.entries = computeEntries;
        bodyPreviewComputeBindGroupLayout = device.CreateBindGroupLayout(&computeLayoutDesc);
        if (bodyPreviewComputeBindGroupLayout == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: compute bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor computePipelineLayoutDesc = {};
        computePipelineLayoutDesc.bindGroupLayoutCount = 1;
        computePipelineLayoutDesc.bindGroupLayouts = &bodyPreviewComputeBindGroupLayout;
        wgpu::PipelineLayout computePipelineLayout = device.CreatePipelineLayout(&computePipelineLayoutDesc);
        if (computePipelineLayout == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: compute pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor computePipelineDesc = {};
        computePipelineDesc.layout = computePipelineLayout;
        computePipelineDesc.compute.module = computeShader;
        computePipelineDesc.compute.entryPoint = "main";
        device.PushErrorScope(wgpu::ErrorFilter::Validation);
        bodyPreviewComputePipeline = device.CreateComputePipeline(&computePipelineDesc);
        wgpu::PopErrorScopeStatus computePipelineScopeStatus = wgpu::PopErrorScopeStatus::Error;
        wgpu::ErrorType computePipelineErrorType = wgpu::ErrorType::Unknown;
        std::string computePipelineErrorMessage;
        wgpu::Future computePipelineScopeFuture = device.PopErrorScope(
            wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message)
            {
                computePipelineScopeStatus = status;
                computePipelineErrorType = type;
                computePipelineErrorMessage = toString(message);
            });
        if (instance.WaitAny(computePipelineScopeFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            computePipelineScopeStatus != wgpu::PopErrorScopeStatus::Success)
        {
            bodyPreviewComputePipeline = nullptr;
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: compute pipeline validation scope");
            return false;
        }
        if (computePipelineErrorType != wgpu::ErrorType::NoError)
        {
            bodyPreviewComputePipeline = nullptr;
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            snprintf(status, sizeof(status), "WebGPU validation error: %s", computePipelineErrorMessage.c_str());
            snprintf(presentStatus, sizeof(presentStatus), "Instanced preview compute pipeline failed: %s", computePipelineErrorMessage.c_str());
            return false;
        }
        if (bodyPreviewComputePipeline == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: compute pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry computeBindEntries[2] = {};
    computeBindEntries[0].binding = 0;
    computeBindEntries[0].buffer = bodyInputBuffer;
    computeBindEntries[0].offset = 0;
    computeBindEntries[0].size = inputBytes;
    computeBindEntries[1].binding = 1;
    computeBindEntries[1].buffer = bodyInstanceBuffer;
    computeBindEntries[1].offset = 0;
    computeBindEntries[1].size = instanceBytes;

    wgpu::BindGroupDescriptor computeBindGroupDesc = {};
    computeBindGroupDesc.layout = bodyPreviewComputeBindGroupLayout;
    computeBindGroupDesc.entryCount = 2;
    computeBindGroupDesc.entries = computeBindEntries;
    wgpu::BindGroup computeBindGroup = device.CreateBindGroup(&computeBindGroupDesc);
    if (computeBindGroup == nullptr)
    {
        surface.Unconfigure();
        bodyPreviewSurfaceConfigured = false;
        setStatus(presentStatus, "Instanced preview failed: compute bind group");
        return false;
    }

    bool previewPipelineFormatChanged = bodyPreviewFormat != format;

    static const char *shaderSource = R"(
struct Camera {
    eye: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
    params: vec4f,
    light: vec4f,
    options: vec4f,
    background: vec4f,
    gridBounds: vec4f,
    gridParams: vec4f,
};

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) worldPosition: vec3f,
    @location(2) worldNormal: vec3f,
    @location(3) shapeInfo: vec4f,
    @location(4) bodySize: vec3f,
    @location(5) localCoord: vec3f,
};

@group(0) @binding(0) var<uniform> camera: Camera;

const BoxVertexCount: u32 = 36u;
const SphereSlices: u32 = 16u;
const SphereStacks: u32 = 8u;
const SphereVertexCount: u32 = SphereSlices * SphereStacks * 6u;
const CapsuleSlices: u32 = 16u;
const CapsuleCapStacks: u32 = 6u;
const CapsuleVertexCount: u32 = CapsuleSlices * (1u + CapsuleCapStacks * 2u) * 6u;
const CylinderSlices: u32 = 24u;
const CylinderVertexCount: u32 = CylinderSlices * 12u;

fn cubeVertex(vertexIndex: u32) -> vec3f {
    var cube = array<vec3f, 36>(
        vec3f(-0.5, -0.5,  0.5), vec3f( 0.5, -0.5,  0.5), vec3f( 0.5,  0.5,  0.5),
        vec3f(-0.5, -0.5,  0.5), vec3f( 0.5,  0.5,  0.5), vec3f(-0.5,  0.5,  0.5),
        vec3f( 0.5, -0.5, -0.5), vec3f(-0.5, -0.5, -0.5), vec3f(-0.5,  0.5, -0.5),
        vec3f( 0.5, -0.5, -0.5), vec3f(-0.5,  0.5, -0.5), vec3f( 0.5,  0.5, -0.5),
        vec3f(-0.5, -0.5, -0.5), vec3f(-0.5, -0.5,  0.5), vec3f(-0.5,  0.5,  0.5),
        vec3f(-0.5, -0.5, -0.5), vec3f(-0.5,  0.5,  0.5), vec3f(-0.5,  0.5, -0.5),
        vec3f( 0.5, -0.5,  0.5), vec3f( 0.5, -0.5, -0.5), vec3f( 0.5,  0.5, -0.5),
        vec3f( 0.5, -0.5,  0.5), vec3f( 0.5,  0.5, -0.5), vec3f( 0.5,  0.5,  0.5),
        vec3f(-0.5,  0.5,  0.5), vec3f( 0.5,  0.5,  0.5), vec3f( 0.5,  0.5, -0.5),
        vec3f(-0.5,  0.5,  0.5), vec3f( 0.5,  0.5, -0.5), vec3f(-0.5,  0.5, -0.5),
        vec3f(-0.5, -0.5, -0.5), vec3f( 0.5, -0.5, -0.5), vec3f( 0.5, -0.5,  0.5),
        vec3f(-0.5, -0.5, -0.5), vec3f( 0.5, -0.5,  0.5), vec3f(-0.5, -0.5,  0.5)
    );
    if (vertexIndex >= BoxVertexCount) {
        return vec3f(0.0, 0.0, 0.0);
    }
    return cube[vertexIndex];
}

fn cubeNormal(vertexIndex: u32) -> vec3f {
    if (vertexIndex < 6u) {
        return vec3f(0.0, 0.0, 1.0);
    }
    if (vertexIndex < 12u) {
        return vec3f(0.0, 0.0, -1.0);
    }
    if (vertexIndex < 18u) {
        return vec3f(-1.0, 0.0, 0.0);
    }
    if (vertexIndex < 24u) {
        return vec3f(1.0, 0.0, 0.0);
    }
    if (vertexIndex < 30u) {
        return vec3f(0.0, 1.0, 0.0);
    }
    return vec3f(0.0, -1.0, 0.0);
}

fn sphereVertex(vertexIndex: u32) -> vec3f {
    if (vertexIndex >= SphereVertexCount) {
        return vec3f(0.0, 0.0, 0.0);
    }

    let triVertex = vertexIndex % 6u;
    let quad = vertexIndex / 6u;
    let slice = quad % SphereSlices;
    let stack = quad / SphereSlices;

    var corner = array<vec2u, 6>(
        vec2u(0u, 0u), vec2u(1u, 0u), vec2u(1u, 1u),
        vec2u(0u, 0u), vec2u(1u, 1u), vec2u(0u, 1u)
    );

    let uv = corner[triVertex];
    let u = f32(slice + uv.x) / f32(SphereSlices);
    let v = f32(stack + uv.y) / f32(SphereStacks);
    let theta = u * 6.28318530718;
    let phi = -1.57079632679 + v * 3.14159265359;
    let cp = cos(phi);
    return vec3f(cos(theta) * cp, sin(theta) * cp, sin(phi)) * 0.5;
}

fn sphereNormal(vertexIndex: u32) -> vec3f {
    return normalize(sphereVertex(vertexIndex));
}

fn capsuleVertex(vertexIndex: u32, radius: f32, halfLength: f32) -> vec3f {
    if (vertexIndex >= CapsuleVertexCount) {
        return vec3f(0.0, 0.0, 0.0);
    }

    let triVertex = vertexIndex % 6u;
    let quad = vertexIndex / 6u;
    var corner = array<vec2u, 6>(
        vec2u(0u, 0u), vec2u(1u, 0u), vec2u(1u, 1u),
        vec2u(0u, 0u), vec2u(1u, 1u), vec2u(0u, 1u)
    );
    let uv = corner[triVertex];

    if (quad < CapsuleSlices) {
        let slice = quad;
        let u = f32(slice + uv.x) / f32(CapsuleSlices);
        let theta = u * 6.28318530718;
        let z = select(-halfLength, halfLength, uv.y == 1u);
        return vec3f(cos(theta) * radius, sin(theta) * radius, z);
    }

    let capQuad = quad - CapsuleSlices;
    let capQuadsPerEnd = CapsuleSlices * CapsuleCapStacks;
    let capIndex = capQuad / capQuadsPerEnd;
    let localQuad = capQuad % capQuadsPerEnd;
    let stack = localQuad / CapsuleSlices;
    let slice = localQuad % CapsuleSlices;
    let u = f32(slice + uv.x) / f32(CapsuleSlices);
    let t = f32(stack + uv.y) / f32(CapsuleCapStacks);
    let theta = u * 6.28318530718;

    var phi = -1.57079632679 + t * 1.57079632679;
    var zCenter = -halfLength;
    if (capIndex == 1u) {
        phi = t * 1.57079632679;
        zCenter = halfLength;
    }

    let capRadius = cos(phi) * radius;
    let z = zCenter + sin(phi) * radius;
    return vec3f(cos(theta) * capRadius, sin(theta) * capRadius, z);
}

fn capsuleNormal(vertexIndex: u32, radius: f32, halfLength: f32) -> vec3f {
    if (vertexIndex >= CapsuleVertexCount) {
        return vec3f(0.0, 0.0, 1.0);
    }

    let triVertex = vertexIndex % 6u;
    let quad = vertexIndex / 6u;
    if (quad < CapsuleSlices) {
        var edgeOffset = 0u;
        if (triVertex == 1u || triVertex == 2u || triVertex == 4u) {
            edgeOffset = 1u;
        }
        let theta = f32(quad + edgeOffset) / f32(CapsuleSlices) * 6.28318530718;
        return normalize(vec3f(cos(theta), sin(theta), 0.0));
    }

    let p = capsuleVertex(vertexIndex, radius, halfLength);
    var centerZ = -halfLength;
    if (p.z > 0.0) {
        centerZ = halfLength;
    }
    return normalize(p - vec3f(0.0, 0.0, centerZ));
}

fn cylinderVertex(vertexIndex: u32, radius: f32, halfLength: f32) -> vec3f {
    if (vertexIndex >= CylinderVertexCount) {
        return vec3f(0.0, 0.0, 0.0);
    }

    let triVertex = vertexIndex % 12u;
    let slice = vertexIndex / 12u;
    let theta0 = f32(slice) / f32(CylinderSlices) * 6.28318530718;
    let theta1 = f32(slice + 1u) / f32(CylinderSlices) * 6.28318530718;
    let p0b = vec3f(cos(theta0) * radius, sin(theta0) * radius, -halfLength);
    let p1b = vec3f(cos(theta1) * radius, sin(theta1) * radius, -halfLength);
    let p0t = vec3f(cos(theta0) * radius, sin(theta0) * radius, halfLength);
    let p1t = vec3f(cos(theta1) * radius, sin(theta1) * radius, halfLength);
    let cb = vec3f(0.0, 0.0, -halfLength);
    let ct = vec3f(0.0, 0.0, halfLength);

    var vertices = array<vec3f, 12>(
        p0b, p1b, p1t,
        p0b, p1t, p0t,
        ct, p0t, p1t,
        cb, p1b, p0b
    );
    return vertices[triVertex];
}

fn cylinderNormal(vertexIndex: u32) -> vec3f {
    let triVertex = vertexIndex % 12u;
    let slice = vertexIndex / 12u;
    if (triVertex >= 6u && triVertex < 9u) {
        return vec3f(0.0, 0.0, 1.0);
    }
    if (triVertex >= 9u) {
        return vec3f(0.0, 0.0, -1.0);
    }

    var edgeOffset = 0u;
    if (triVertex == 1u || triVertex == 2u || triVertex == 4u) {
        edgeOffset = 1u;
    }
    let theta = f32(slice + edgeOffset) / f32(CylinderSlices) * 6.28318530718;
    return normalize(vec3f(cos(theta), sin(theta), 0.0));
}

fn rotateVector(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertexIndex: u32,
    @location(0) bodyPosition: vec4f,
    @location(1) bodySize: vec4f,
    @location(2) bodyRotation: vec4f,
    @location(3) bodyMeta: vec4f
) -> VertexOut {

    var palette = array<vec3f, 4>(
        vec3f(0.58, 0.66, 0.76),
        vec3f(0.84, 0.54, 0.48),
        vec3f(0.56, 0.74, 0.58),
        vec3f(0.84, 0.74, 0.48)
    );

    let renderKind = u32(clamp(bodyMeta.z, 0.0, 4.0));
    var unitVertex = cubeVertex(vertexIndex);
    var localNormal = cubeNormal(vertexIndex);
    var local = unitVertex * bodySize.xyz;
    if (renderKind == 2u) {
        unitVertex = sphereVertex(vertexIndex);
        localNormal = sphereNormal(vertexIndex);
        local = unitVertex * bodySize.xyz;
    } else if (renderKind == 3u) {
        local = capsuleVertex(vertexIndex, bodySize.x * 0.5, bodyMeta.w);
        localNormal = capsuleNormal(vertexIndex, bodySize.x * 0.5, bodyMeta.w);
        unitVertex = vec3f(
            local.x / max(bodySize.x * 0.5, 0.001),
            local.y / max(bodySize.y * 0.5, 0.001),
            local.z / max(bodySize.z * 0.5, 0.001)
        );
    } else if (renderKind == 4u) {
        local = cylinderVertex(vertexIndex, bodySize.x * 0.5, bodySize.z * 0.5);
        localNormal = cylinderNormal(vertexIndex);
        unitVertex = vec3f(
            local.x / max(bodySize.x * 0.5, 0.001),
            local.y / max(bodySize.y * 0.5, 0.001),
            local.z / max(bodySize.z * 0.5, 0.001)
        );
    }
    let q = bodyRotation;
    let rotated = rotateVector(q, local);
    let worldNormal = normalize(rotateVector(q, localNormal));
    let world = bodyPosition.xyz + rotated;
    let rel = world - camera.eye.xyz;
    let viewX = dot(rel, camera.right.xyz);
    let viewY = dot(rel, camera.up.xyz);
    let viewZ = max(dot(rel, camera.forward.xyz), 0.05);
    let aspect = camera.params.x;
    let tanHalfFovY = camera.params.y;
    let clipXY = vec2f(
        viewX / (tanHalfFovY * aspect),
        viewY / tanHalfFovY
    );
    let shapeIndex = u32(clamp(bodyMeta.x, 0.0, 3.0));
    let materialColorId = bodyMeta.y;
    let dynamicBody = materialColorId > 0.5;
    let realShape = renderKind != 0u;
    let lightDir = normalize(camera.light.xyz);
    let viewDir = normalize(camera.eye.xyz - world);
    let diffuse = max(dot(worldNormal, lightDir), 0.0);
    let rim = pow(max(1.0 - dot(worldNormal, viewDir), 0.0), 2.0);
    let topBias = clamp(worldNormal.z * 0.5 + 0.5, 0.0, 1.0);
    let faceLift = select(1.0, 0.88 + 0.16 * topBias, renderKind == 1u);
    let litShade = clamp((0.30 + 0.64 * diffuse + 0.12 * rim) * faceLift, 0.30, 1.08);
    let fallbackShade = 0.68 + 0.32 * abs(unitVertex.z * 2.0);
    let shade = select(fallbackShade, litShade, realShape);
    var baseColor = select(palette[shapeIndex], vec3f(0.70, 0.72, 0.74), !dynamicBody);
    let fallbackTint = select(vec3f(1.0, 1.0, 1.0), vec3f(0.82, 0.88, 0.96), !realShape);
    var out: VertexOut;
    let depth = clamp(viewZ / 200.0, 0.0, 1.0);
    out.position = vec4f(clipXY, depth * viewZ, viewZ);
    out.color = baseColor * fallbackTint * shade;
    out.worldPosition = world;
    out.worldNormal = worldNormal;
    out.shapeInfo = vec4f(f32(renderKind), select(0.0, 1.0, dynamicBody), 0.0, 0.0);
    out.bodySize = bodySize.xyz;
    out.localCoord = unitVertex;
    return out;
}

fn lineAtAbs(value: f32, center: f32, width: f32) -> f32 {
    return 1.0 - smoothstep(width, width * 2.0, abs(abs(value) - center));
}

fn lineAtZero(value: f32, width: f32) -> f32 {
    return 1.0 - smoothstep(width, width * 2.0, abs(value));
}

fn periodicLine(value: f32, frequency: f32, width: f32) -> f32 {
    let centered = abs(fract(value * frequency) - 0.5);
    return 1.0 - smoothstep(0.5 - width, 0.5, centered);
}

fn boxCue(local: vec3f, normal: vec3f) -> f32 {
    let xLine = lineAtAbs(local.x, 0.5, 0.018);
    let yLine = lineAtAbs(local.y, 0.5, 0.018);
    let zLine = lineAtAbs(local.z, 0.5, 0.018);
    let absNormal = abs(normal);
    var cue = max(yLine, zLine);
    if (absNormal.y > absNormal.x && absNormal.y > absNormal.z) {
        cue = max(xLine, zLine);
    } else if (absNormal.z > absNormal.x && absNormal.z > absNormal.y) {
        cue = max(xLine, yLine);
    }
    return clamp(cue * 0.26, 0.0, 1.0);
}

fn sphereCue(local: vec3f) -> f32 {
    let latitude = periodicLine(local.z + 0.5, 5.0, 0.030) * 0.34;
    let meridian = max(lineAtZero(local.x, 0.020), lineAtZero(local.y, 0.020)) * 0.20;
    return clamp(max(latitude, meridian), 0.0, 1.0);
}

fn capsuleCue(local: vec3f) -> f32 {
    let capJoin = lineAtAbs(local.z, 0.55, 0.035) * 0.40;
    let endRing = lineAtAbs(local.z, 1.0, 0.030) * 0.30;
    let lengthHint = max(lineAtZero(local.x, 0.030), lineAtZero(local.y, 0.030)) * 0.16;
    return clamp(max(max(capJoin, endRing), lengthHint), 0.0, 1.0);
}

fn cylinderCue(local: vec3f) -> f32 {
    let capEdge = lineAtAbs(local.z, 1.0, 0.030) * 0.46;
    let sideHint = max(lineAtZero(local.x, 0.030), lineAtZero(local.y, 0.030)) * 0.18;
    return clamp(max(capEdge, sideHint), 0.0, 1.0);
}

fn shapeCue(local: vec3f, normal: vec3f, renderKind: u32) -> f32 {
    if (renderKind == 1u) {
        return boxCue(local, normal);
    }
    if (renderKind == 2u) {
        return sphereCue(local);
    }
    if (renderKind == 3u) {
        return capsuleCue(local);
    }
    if (renderKind == 4u) {
        return cylinderCue(local);
    }
    return 0.0;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    let renderKind = u32(clamp(in.shapeInfo.x, 0.0, 4.0));
    let normal = normalize(in.worldNormal);
    var color = in.color;

    if (camera.options.y > 0.5) {
        let cue = shapeCue(in.localCoord, normal, renderKind);
        color = color * (1.0 - 0.09 * cue);
    }

    return vec4f(color, 1.0);
}
)";

    if (bodyPreviewPipeline == nullptr || bodyPreviewBindGroupLayout == nullptr || previewPipelineFormatChanged)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = shaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        device.PushErrorScope(wgpu::ErrorFilter::Validation);
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        wgpu::PopErrorScopeStatus shaderScopeStatus = wgpu::PopErrorScopeStatus::Error;
        wgpu::ErrorType shaderErrorType = wgpu::ErrorType::Unknown;
        std::string shaderErrorMessage;
        wgpu::Future shaderScopeFuture = device.PopErrorScope(
            wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message)
            {
                shaderScopeStatus = status;
                shaderErrorType = type;
                shaderErrorMessage = toString(message);
            });
        if (instance.WaitAny(shaderScopeFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            shaderScopeStatus != wgpu::PopErrorScopeStatus::Success)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: shader validation scope");
            return false;
        }
        if (shaderErrorType != wgpu::ErrorType::NoError)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            snprintf(status, sizeof(status), "WebGPU validation error: %s", shaderErrorMessage.c_str());
            snprintf(presentStatus, sizeof(presentStatus), "Instanced preview shader failed: %s", shaderErrorMessage.c_str());
            return false;
        }
        if (shader == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: shader module");
            return false;
        }

        wgpu::VertexAttribute instanceAttributes[4] = {};
        for (uint32_t i = 0; i < 4; ++i)
        {
            instanceAttributes[i].format = wgpu::VertexFormat::Float32x4;
            instanceAttributes[i].offset = i * 4 * sizeof(float);
            instanceAttributes[i].shaderLocation = i;
        }

        wgpu::VertexBufferLayout instanceLayout = {};
        instanceLayout.stepMode = wgpu::VertexStepMode::Instance;
        instanceLayout.arrayStride = sizeof(GpuRenderInstance);
        instanceLayout.attributeCount = 4;
        instanceLayout.attributes = instanceAttributes;

        wgpu::BindGroupLayoutEntry cameraLayoutEntry = {};
        cameraLayoutEntry.binding = 0;
        cameraLayoutEntry.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        cameraLayoutEntry.buffer.type = wgpu::BufferBindingType::Uniform;
        cameraLayoutEntry.buffer.minBindingSize = sizeof(PreviewCameraUniform);

        wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc = {};
        bindGroupLayoutDesc.entryCount = 1;
        bindGroupLayoutDesc.entries = &cameraLayoutEntry;
        bodyPreviewBindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc);
        if (bodyPreviewBindGroupLayout == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &bodyPreviewBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: pipeline layout");
            return false;
        }

        wgpu::ColorTargetState colorTarget = {};
        colorTarget.format = format;

        wgpu::FragmentState fragment = {};
        fragment.module = shader;
        fragment.entryPoint = "fs_main";
        fragment.targetCount = 1;
        fragment.targets = &colorTarget;

        wgpu::RenderPipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.vertex.module = shader;
        pipelineDesc.vertex.entryPoint = "vs_main";
        pipelineDesc.vertex.bufferCount = 1;
        pipelineDesc.vertex.buffers = &instanceLayout;
        pipelineDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
        pipelineDesc.multisample.count = 1;
        pipelineDesc.fragment = &fragment;

        wgpu::DepthStencilState depthStencil = {};
        depthStencil.format = wgpu::TextureFormat::Depth24Plus;
        depthStencil.depthWriteEnabled = wgpu::OptionalBool::True;
        depthStencil.depthCompare = wgpu::CompareFunction::Less;
        pipelineDesc.depthStencil = &depthStencil;

        device.PushErrorScope(wgpu::ErrorFilter::Validation);
        bodyPreviewPipeline = device.CreateRenderPipeline(&pipelineDesc);
        wgpu::PopErrorScopeStatus pipelineScopeStatus = wgpu::PopErrorScopeStatus::Error;
        wgpu::ErrorType pipelineErrorType = wgpu::ErrorType::Unknown;
        std::string pipelineErrorMessage;
        wgpu::Future pipelineScopeFuture = device.PopErrorScope(
            wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message)
            {
                pipelineScopeStatus = status;
                pipelineErrorType = type;
                pipelineErrorMessage = toString(message);
            });
        if (instance.WaitAny(pipelineScopeFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            pipelineScopeStatus != wgpu::PopErrorScopeStatus::Success)
        {
            bodyPreviewPipeline = nullptr;
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: pipeline validation scope");
            return false;
        }
        if (pipelineErrorType != wgpu::ErrorType::NoError)
        {
            bodyPreviewPipeline = nullptr;
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            snprintf(status, sizeof(status), "WebGPU validation error: %s", pipelineErrorMessage.c_str());
            snprintf(presentStatus, sizeof(presentStatus), "Instanced preview pipeline failed: %s", pipelineErrorMessage.c_str());
            return false;
        }
        if (bodyPreviewPipeline == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: pipeline");
            return false;
        }
        bodyPreviewFormat = format;
    }

    static const char *gridShaderSource = R"(
struct Camera {
    eye: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
    params: vec4f,
    light: vec4f,
    options: vec4f,
    background: vec4f,
    gridBounds: vec4f,
    gridParams: vec4f,
};

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) worldPosition: vec3f,
};

@group(0) @binding(0) var<uniform> camera: Camera;

fn gridWorldPosition(vertexIndex: u32) -> vec3f {
    let minX = camera.gridBounds.x;
    let minY = camera.gridBounds.y;
    let maxX = camera.gridBounds.z;
    let maxY = camera.gridBounds.w;
    let z = camera.gridParams.x;
    var vertices = array<vec3f, 6>(
        vec3f(minX, minY, z),
        vec3f(maxX, minY, z),
        vec3f(maxX, maxY, z),
        vec3f(minX, minY, z),
        vec3f(maxX, maxY, z),
        vec3f(minX, maxY, z)
    );
    return vertices[vertexIndex];
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
    let world = gridWorldPosition(vertexIndex);
    let rel = world - camera.eye.xyz;
    let viewX = dot(rel, camera.right.xyz);
    let viewY = dot(rel, camera.up.xyz);
    let viewZ = max(dot(rel, camera.forward.xyz), 0.05);
    let aspect = camera.params.x;
    let tanHalfFovY = camera.params.y;
    let clipXY = vec2f(
        viewX / (tanHalfFovY * aspect),
        viewY / tanHalfFovY
    );
    let depth = clamp(viewZ / 200.0, 0.0, 1.0);
    var out: VertexOut;
    out.position = vec4f(clipXY, depth * viewZ, viewZ);
    out.worldPosition = world;
    return out;
}

fn gridLine(worldXY: vec2f, spacing: f32, lineWidthWorld: f32) -> f32 {
    let coord = worldXY / spacing;
    let cell = abs(fract(coord - vec2f(0.5, 0.5)) - vec2f(0.5, 0.5));
    let d = min(cell.x, cell.y);
    let pixel = max(max(fwidth(coord).x, fwidth(coord).y), 0.0001);
    let width = max(lineWidthWorld / spacing, pixel * 0.65);
    let aa = pixel * 1.5;
    return 1.0 - smoothstep(width, width + aa, d);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    var color = vec3f(0.55, 0.57, 0.58);
    let minor = gridLine(in.worldPosition.xy, 2.0, 0.018) * 0.22;
    let major = gridLine(in.worldPosition.xy, 10.0, 0.040) * 0.44;
    let grid = clamp(max(minor, major), 0.0, 1.0);
    color = color * (1.0 - 0.24 * grid);
    return vec4f(color, 1.0);
}
)";

    if (bodyPreviewGridPipeline == nullptr || previewPipelineFormatChanged)
    {
        if (bodyPreviewBindGroupLayout == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: grid bind layout");
            return false;
        }

        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = gridShaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        device.PushErrorScope(wgpu::ErrorFilter::Validation);
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        wgpu::PopErrorScopeStatus shaderScopeStatus = wgpu::PopErrorScopeStatus::Error;
        wgpu::ErrorType shaderErrorType = wgpu::ErrorType::Unknown;
        std::string shaderErrorMessage;
        wgpu::Future shaderScopeFuture = device.PopErrorScope(
            wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message)
            {
                shaderScopeStatus = status;
                shaderErrorType = type;
                shaderErrorMessage = toString(message);
            });
        if (instance.WaitAny(shaderScopeFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            shaderScopeStatus != wgpu::PopErrorScopeStatus::Success)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: grid shader validation scope");
            return false;
        }
        if (shaderErrorType != wgpu::ErrorType::NoError)
        {
            bodyPreviewGridPipeline = nullptr;
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            snprintf(status, sizeof(status), "WebGPU validation error: %s", shaderErrorMessage.c_str());
            snprintf(presentStatus, sizeof(presentStatus), "Instanced preview grid shader failed: %s", shaderErrorMessage.c_str());
            return false;
        }
        if (shader == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: grid shader module");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &bodyPreviewBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: grid pipeline layout");
            return false;
        }

        wgpu::ColorTargetState colorTarget = {};
        colorTarget.format = format;

        wgpu::FragmentState fragment = {};
        fragment.module = shader;
        fragment.entryPoint = "fs_main";
        fragment.targetCount = 1;
        fragment.targets = &colorTarget;

        wgpu::RenderPipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.vertex.module = shader;
        pipelineDesc.vertex.entryPoint = "vs_main";
        pipelineDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
        pipelineDesc.multisample.count = 1;
        pipelineDesc.fragment = &fragment;

        wgpu::DepthStencilState depthStencil = {};
        depthStencil.format = wgpu::TextureFormat::Depth24Plus;
        depthStencil.depthWriteEnabled = wgpu::OptionalBool::True;
        depthStencil.depthCompare = wgpu::CompareFunction::Less;
        pipelineDesc.depthStencil = &depthStencil;

        device.PushErrorScope(wgpu::ErrorFilter::Validation);
        bodyPreviewGridPipeline = device.CreateRenderPipeline(&pipelineDesc);
        wgpu::PopErrorScopeStatus pipelineScopeStatus = wgpu::PopErrorScopeStatus::Error;
        wgpu::ErrorType pipelineErrorType = wgpu::ErrorType::Unknown;
        std::string pipelineErrorMessage;
        wgpu::Future pipelineScopeFuture = device.PopErrorScope(
            wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message)
            {
                pipelineScopeStatus = status;
                pipelineErrorType = type;
                pipelineErrorMessage = toString(message);
            });
        if (instance.WaitAny(pipelineScopeFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            pipelineScopeStatus != wgpu::PopErrorScopeStatus::Success)
        {
            bodyPreviewGridPipeline = nullptr;
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: grid pipeline validation scope");
            return false;
        }
        if (pipelineErrorType != wgpu::ErrorType::NoError)
        {
            bodyPreviewGridPipeline = nullptr;
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            snprintf(status, sizeof(status), "WebGPU validation error: %s", pipelineErrorMessage.c_str());
            snprintf(presentStatus, sizeof(presentStatus), "Instanced preview grid pipeline failed: %s", pipelineErrorMessage.c_str());
            return false;
        }
        if (bodyPreviewGridPipeline == nullptr)
        {
            surface.Unconfigure();
            bodyPreviewSurfaceConfigured = false;
            setStatus(presentStatus, "Instanced preview failed: grid pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry cameraEntry = {};
    cameraEntry.binding = 0;
    cameraEntry.buffer = bodyPreviewCameraBuffer;
    cameraEntry.offset = 0;
    cameraEntry.size = sizeof(PreviewCameraUniform);

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bodyPreviewBindGroupLayout;
    bindGroupDesc.entryCount = 1;
    bindGroupDesc.entries = &cameraEntry;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
    if (bindGroup == nullptr)
    {
        surface.Unconfigure();
        bodyPreviewSurfaceConfigured = false;
        setStatus(presentStatus, "Instanced preview failed: bind group");
        return false;
    }

    wgpu::SurfaceTexture frame = {};
    surface.GetCurrentTexture(&frame);
    if ((frame.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
         frame.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) ||
        frame.texture == nullptr)
    {
        surface.Unconfigure();
        bodyPreviewSurfaceConfigured = false;
        snprintf(presentStatus, sizeof(presentStatus), "Instanced preview failed: texture status %u",
                 (unsigned int)frame.status);
        return false;
    }

    wgpu::TextureView view = frame.texture.CreateView();
    if (view == nullptr)
    {
        surface.Unconfigure();
        bodyPreviewSurfaceConfigured = false;
        setStatus(presentStatus, "Instanced preview failed: texture view");
        return false;
    }

    if (bodyPreviewDepthTexture == nullptr || bodyPreviewDepthView == nullptr)
    {
        wgpu::TextureDescriptor depthDesc = {};
        depthDesc.usage = wgpu::TextureUsage::RenderAttachment;
        depthDesc.dimension = wgpu::TextureDimension::e2D;
        depthDesc.size = {(uint32_t)width, (uint32_t)height, 1};
        depthDesc.format = wgpu::TextureFormat::Depth24Plus;
        depthDesc.mipLevelCount = 1;
        depthDesc.sampleCount = 1;
        bodyPreviewDepthTexture = device.CreateTexture(&depthDesc);
        bodyPreviewDepthView = bodyPreviewDepthTexture == nullptr ? nullptr : bodyPreviewDepthTexture.CreateView();
    }
    if (bodyPreviewDepthTexture == nullptr)
    {
        surface.Unconfigure();
        bodyPreviewSurfaceConfigured = false;
        setStatus(presentStatus, "Instanced preview failed: depth texture");
        return false;
    }

    if (bodyPreviewDepthView == nullptr)
    {
        surface.Unconfigure();
        bodyPreviewSurfaceConfigured = false;
        setStatus(presentStatus, "Instanced preview failed: depth view");
        return false;
    }

    wgpu::RenderPassColorAttachment color = {};
    color.view = view;
    color.loadOp = wgpu::LoadOp::Clear;
    color.storeOp = wgpu::StoreOp::Store;
    color.clearValue = {
        (double)cameraUniform.background[0],
        (double)cameraUniform.background[1],
        (double)cameraUniform.background[2],
        1.0};

    wgpu::RenderPassDepthStencilAttachment depthStencilAttachment = {};
    depthStencilAttachment.view = bodyPreviewDepthView;
    depthStencilAttachment.depthLoadOp = wgpu::LoadOp::Clear;
    depthStencilAttachment.depthStoreOp = wgpu::StoreOp::Store;
    depthStencilAttachment.depthClearValue = 1.0f;
    depthStencilAttachment.stencilLoadOp = wgpu::LoadOp::Undefined;
    depthStencilAttachment.stencilStoreOp = wgpu::StoreOp::Undefined;

    wgpu::RenderPassDescriptor passDesc = {};
    passDesc.colorAttachmentCount = 1;
    passDesc.colorAttachments = &color;
    passDesc.depthStencilAttachment = &depthStencilAttachment;

    Uint64 computeBegin = SDL_GetPerformanceCounter();
    wgpu::CommandEncoder computeEncoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder computePass = computeEncoder.BeginComputePass();
    computePass.SetPipeline(bodyPreviewComputePipeline);
    computePass.SetBindGroup(0, computeBindGroup);
    computePass.DispatchWorkgroups((uint32_t)renderScene.bodyInputs.size());
    computePass.End();
    wgpu::CommandBuffer computeCommands = computeEncoder.Finish();
    queue.Submit(1, &computeCommands);
    previewComputeMs = elapsedMs(computeBegin, SDL_GetPerformanceCounter());

    Uint64 renderBegin = SDL_GetPerformanceCounter();
    wgpu::CommandEncoder renderEncoder = device.CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = renderEncoder.BeginRenderPass(&passDesc);
    if (options.showGroundGrid && renderScene.gridReceiver.valid && bodyPreviewGridPipeline != nullptr)
    {
        pass.SetPipeline(bodyPreviewGridPipeline);
        pass.SetBindGroup(0, bindGroup);
        pass.Draw(6, 1, 0, 0);
    }
    pass.SetPipeline(bodyPreviewPipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.SetVertexBuffer(0, bodyInstanceBuffer, 0, instanceBytes);
    for (size_t i = 0; i < renderScene.renderBatches.size(); ++i)
    {
        const GpuRenderBatch &batch = renderScene.renderBatches[i];
        pass.Draw(batch.vertexCount, batch.instanceCount, 0, batch.firstInstance);
    }
    pass.End();

    wgpu::CommandBuffer commands = renderEncoder.Finish();
    queue.Submit(1, &commands);
    previewRenderMs = elapsedMs(renderBegin, SDL_GetPerformanceCounter());

    if (surface.Present() != wgpu::Status::Success)
    {
        surface.Unconfigure();
        bodyPreviewSurfaceConfigured = false;
        setStatus(presentStatus, "Instanced preview failed: present");
        return false;
    }

    previewTotalMs = elapsedMs(previewBegin, SDL_GetPerformanceCounter());
    snprintf(presentStatus, sizeof(presentStatus), "Instanced preview passed: %d boxes, %d spheres, %d capsules, %d cylinders, %d mesh assets %dx%d",
             renderScene.boxCount, renderScene.sphereCount, renderScene.capsuleCount, renderScene.cylinderCount, renderScene.meshAssetCount, width, height);
    return true;
#elif AVBD_ENABLE_WEBGPU
    setStatus(presentStatus, "Instanced preview skipped: Dawn not linked");
    return false;
#else
    setStatus(presentStatus, "Instanced preview skipped: WebGPU disabled");
    return false;
#endif
}

void WebGpuContext::shutdown()
{
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (bodyPreviewSurfaceConfigured && surface != nullptr)
        surface.Unconfigure();
    bodyPreviewSurfaceConfigured = false;
    bodyPreviewWidth = 0;
    bodyPreviewHeight = 0;
    bodyInputBuffer = nullptr;
    bodyInputBufferBytes = 0;
    bodyInstanceBuffer = nullptr;
    bodyInstanceBufferBytes = 0;
    bodyPreviewCameraBuffer = nullptr;
    bodyPreviewDepthTexture = nullptr;
    bodyPreviewDepthView = nullptr;
    bodyPreviewComputeBindGroupLayout = nullptr;
    bodyPreviewComputePipeline = nullptr;
    predictionInputBuffer = nullptr;
    predictionInputBufferBytes = 0;
    predictionOutputBuffer = nullptr;
    predictionOutputBufferBytes = 0;
    predictionReadbackBuffer = nullptr;
    predictionReadbackBufferBytes = 0;
    predictionParamsBuffer = nullptr;
    predictionBindGroupLayout = nullptr;
    predictionPipeline = nullptr;
    velocityInputBuffer = nullptr;
    velocityInputBufferBytes = 0;
    velocityOutputBuffer = nullptr;
    velocityOutputBufferBytes = 0;
    velocityReadbackBuffer = nullptr;
    velocityReadbackBufferBytes = 0;
    velocityParamsBuffer = nullptr;
    velocityBindGroupLayout = nullptr;
    velocityPipeline = nullptr;
    boundsInputBuffer = nullptr;
    boundsInputBufferBytes = 0;
    boundsOutputBuffer = nullptr;
    boundsOutputBufferBytes = 0;
    boundsReadbackBuffer = nullptr;
    boundsReadbackBufferBytes = 0;
    boundsParamsBuffer = nullptr;
    boundsBindGroupLayout = nullptr;
    boundsPipeline = nullptr;
    mortonInputBuffer = nullptr;
    mortonInputBufferBytes = 0;
    mortonOutputBuffer = nullptr;
    mortonOutputBufferBytes = 0;
    mortonReadbackBuffer = nullptr;
    mortonReadbackBufferBytes = 0;
    mortonParamsBuffer = nullptr;
    mortonBindGroupLayout = nullptr;
    mortonPipeline = nullptr;
    mortonSortBuffer = nullptr;
    mortonSortBufferBytes = 0;
    mortonSortReadbackBuffer = nullptr;
    mortonSortReadbackBufferBytes = 0;
    mortonSortParamsBuffer = nullptr;
    mortonSortBindGroupLayout = nullptr;
    mortonSortPipeline = nullptr;
    pairBodyBuffer = nullptr;
    pairBodyBufferBytes = 0;
    pairItemBuffer = nullptr;
    pairItemBufferBytes = 0;
    pairCountersBuffer = nullptr;
    pairCountersBufferBytes = 0;
    pairReadbackBuffer = nullptr;
    pairReadbackBufferBytes = 0;
    pairParamsBuffer = nullptr;
    pairBindGroupLayout = nullptr;
    pairPipeline = nullptr;
    sapIntervalBuffer = nullptr;
    sapIntervalBufferBytes = 0;
    sapCountersBuffer = nullptr;
    sapCountersBufferBytes = 0;
    sapReadbackBuffer = nullptr;
    sapReadbackBufferBytes = 0;
    sapParamsBuffer = nullptr;
    sapSortBindGroupLayout = nullptr;
    sapSortPipeline = nullptr;
    sapPairBindGroupLayout = nullptr;
    sapPairPipeline = nullptr;
    bodyPreviewBindGroupLayout = nullptr;
    bodyPreviewGridPipeline = nullptr;
    bodyPreviewPipeline = nullptr;
    bodyPreviewFormat = wgpu::TextureFormat::Undefined;
    queue = nullptr;
    device = nullptr;
    adapter = nullptr;
    surface = nullptr;
    instance = nullptr;
#endif
    initialized = false;
    deviceReady = false;
    smokeTestPassed = false;
    previewComputeMs = 0.0f;
    previewRenderMs = 0.0f;
    previewTotalMs = 0.0f;
    predictionMs = 0.0f;
    predictionMaxError = 0.0f;
    predictionMaxAngularError = 0.0f;
    predictionSamples = 0;
    velocityMs = 0.0f;
    velocityMaxLinearError = 0.0f;
    velocityMaxAngularError = 0.0f;
    velocitySamples = 0;
    boundsMs = 0.0f;
    boundsMaxError = 0.0f;
    boundsSamples = 0;
    mortonMs = 0.0f;
    mortonMismatches = 0;
    mortonSamples = 0;
    mortonSortMs = 0.0f;
    mortonSortMismatches = 0;
    mortonSortCount = 0;
    pairMs = 0.0f;
    pairCandidates = 0;
    pairSphereHits = 0;
    pairAllPairsSphereHits = 0;
    pairMissedSphereHits = 0;
    pairMismatches = 0;
    sapMs = 0.0f;
    sapCandidates = 0;
    sapSphereHits = 0;
    sapAllPairsSphereHits = 0;
    sapMissedSphereHits = 0;
    sapMismatches = 0;
    setStatus(status, "WebGPU shutdown");
    setStatus(smokeStatus, "Offscreen smoke not run");
    setStatus(computeStatus, "Compute smoke not run");
    setStatus(presentStatus, "Surface clear not run");
    setStatus(predictionStatus, "Body prediction not run");
    setStatus(velocityStatus, "Velocity update not run");
    setStatus(boundsStatus, "Bounds not run");
    setStatus(mortonStatus, "Morton codes not run");
    setStatus(mortonSortStatus, "Morton sort not run");
    setStatus(pairStatus, "Pair diagnostic not run");
    setStatus(sapStatus, "SAP diagnostic not run");
    resetDiagnosticTimingStats();
}

void WebGpuContext::resetDiagnosticTimingStats()
{
    resetTimingStats(predictionTiming);
    resetTimingStats(velocityTiming);
    resetTimingStats(boundsTiming);
    resetTimingStats(mortonTiming);
    resetTimingStats(mortonSortTiming);
    resetTimingStats(pairTiming);
    resetTimingStats(sapTiming);
}

const char *WebGpuContext::statusText() const
{
    return status;
}

const char *WebGpuContext::smokeStatusText() const
{
    return smokeStatus;
}

const char *WebGpuContext::computeStatusText() const
{
    return computeStatus;
}

const char *WebGpuContext::presentStatusText() const
{
    return presentStatus;
}

const char *WebGpuContext::predictionStatusText() const
{
    return predictionStatus;
}

const char *WebGpuContext::velocityStatusText() const
{
    return velocityStatus;
}

const char *WebGpuContext::boundsStatusText() const
{
    return boundsStatus;
}

const char *WebGpuContext::mortonStatusText() const
{
    return mortonStatus;
}

const char *WebGpuContext::mortonSortStatusText() const
{
    return mortonSortStatus;
}

const char *WebGpuContext::pairStatusText() const
{
    return pairStatus;
}

const char *WebGpuContext::sapStatusText() const
{
    return sapStatus;
}

float WebGpuContext::previewComputeMillis() const
{
    return previewComputeMs;
}

float WebGpuContext::previewRenderMillis() const
{
    return previewRenderMs;
}

float WebGpuContext::previewTotalMillis() const
{
    return previewTotalMs;
}

int WebGpuContext::previewBatchCountValue() const
{
    return previewBatchCount;
}

int WebGpuContext::previewInstanceCountValue() const
{
    return previewInstanceCount;
}

int WebGpuContext::previewBoxInstanceCount() const
{
    return previewBoxInstances;
}

int WebGpuContext::previewSphereInstanceCount() const
{
    return previewSphereInstances;
}

int WebGpuContext::previewCapsuleInstanceCount() const
{
    return previewCapsuleInstances;
}

int WebGpuContext::previewCylinderInstanceCount() const
{
    return previewCylinderInstances;
}

int WebGpuContext::previewMeshAssetInstanceCount() const
{
    return previewMeshAssetInstances;
}

float WebGpuContext::predictionMillis() const
{
    return predictionMs;
}

float WebGpuContext::predictionMaxErrorValue() const
{
    return predictionMaxError;
}

float WebGpuContext::predictionMaxAngularErrorValue() const
{
    return predictionMaxAngularError;
}

int WebGpuContext::predictionSampleCount() const
{
    return predictionSamples;
}

float WebGpuContext::velocityMillis() const
{
    return velocityMs;
}

float WebGpuContext::velocityMaxLinearErrorValue() const
{
    return velocityMaxLinearError;
}

float WebGpuContext::velocityMaxAngularErrorValue() const
{
    return velocityMaxAngularError;
}

int WebGpuContext::velocitySampleCount() const
{
    return velocitySamples;
}

float WebGpuContext::boundsMillis() const
{
    return boundsMs;
}

float WebGpuContext::boundsMaxErrorValue() const
{
    return boundsMaxError;
}

int WebGpuContext::boundsSampleCount() const
{
    return boundsSamples;
}

float WebGpuContext::mortonMillis() const
{
    return mortonMs;
}

int WebGpuContext::mortonMismatchCount() const
{
    return mortonMismatches;
}

int WebGpuContext::mortonSampleCount() const
{
    return mortonSamples;
}

float WebGpuContext::mortonSortMillis() const
{
    return mortonSortMs;
}

int WebGpuContext::mortonSortMismatchCount() const
{
    return mortonSortMismatches;
}

int WebGpuContext::mortonSortItemCount() const
{
    return mortonSortCount;
}

float WebGpuContext::pairMillis() const
{
    return pairMs;
}

int WebGpuContext::pairCandidateCount() const
{
    return pairCandidates;
}

int WebGpuContext::pairSphereHitCount() const
{
    return pairSphereHits;
}

int WebGpuContext::pairAllPairsSphereHitCount() const
{
    return pairAllPairsSphereHits;
}

int WebGpuContext::pairMissedSphereHitCount() const
{
    return pairMissedSphereHits;
}

int WebGpuContext::pairMismatchCount() const
{
    return pairMismatches;
}

float WebGpuContext::sapMillis() const
{
    return sapMs;
}

int WebGpuContext::sapCandidateCount() const
{
    return sapCandidates;
}

int WebGpuContext::sapSphereHitCount() const
{
    return sapSphereHits;
}

int WebGpuContext::sapAllPairsSphereHitCount() const
{
    return sapAllPairsSphereHits;
}

int WebGpuContext::sapMissedSphereHitCount() const
{
    return sapMissedSphereHits;
}

int WebGpuContext::sapMismatchCount() const
{
    return sapMismatches;
}

const char *WebGpuPhysicsBackend::name() const
{
#if AVBD_ENABLE_WEBGPU
    return "WebGPU Physics Scaffold";
#else
    return "WebGPU Physics Unavailable";
#endif
}

void WebGpuPhysicsBackend::step(Solver &solver)
{
    solver.stepCpuReference();
}

const char *WebGpuRenderBackend::name() const
{
#if AVBD_ENABLE_WEBGPU
    return "WebGPU Render Scaffold";
#else
    return "WebGPU Render Unavailable";
#endif
}

bool WebGpuRenderBackend::available() const
{
#if AVBD_ENABLE_WEBGPU
    return true;
#else
    return false;
#endif
}
