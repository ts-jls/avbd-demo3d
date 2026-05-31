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

#pragma once

#include "solver.h"
#include "render_backend.h"

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
#include <dawn/webgpu_cpp.h>
#endif

struct SDL_Window;

struct WebGpuPreviewCamera
{
    float3 eye;
    float3 target;
    float3 up;
    float fovYDeg;
};

struct WebGpuRenderOptions
{
    bool showGroundGrid = true;
    bool showShapeEdges = true;
    float3 lightDirection = {0.45f, 0.80f, 0.55f};
    float3 backgroundColor = {0.035f, 0.055f, 0.085f};
};

struct WebGpuTimingStats
{
    float lastMs;
    float avgMs;
    float recentAvgMs;
    float minMs;
    float maxMs;
    int samples;
    int recentSamples;
    int recentIndex;
    float recentSumMs;
    float recentMs[64];
};

struct WebGpuContext
{
    bool initialized;
    bool deviceReady;
    bool smokeTestPassed;
    char status[1024];
    char smokeStatus[1024];
    char computeStatus[1024];
    char presentStatus[1024];
    char predictionStatus[1024];
    char velocityStatus[1024];
    char boundsStatus[1024];
    char mortonStatus[1024];
    char mortonSortStatus[1024];
    char pairStatus[1024];
    char sapStatus[1024];
    float previewComputeMs;
    float previewRenderMs;
    float previewTotalMs;
    int previewBatchCount;
    int previewInstanceCount;
    int previewBoxInstances;
    int previewSphereInstances;
    int previewCapsuleInstances;
    int previewCylinderInstances;
    int previewMeshAssetInstances;
    float predictionMs;
    float predictionMaxError;
    float predictionMaxAngularError;
    int predictionSamples;
    float velocityMs;
    float velocityMaxLinearError;
    float velocityMaxAngularError;
    int velocitySamples;
    float boundsMs;
    float boundsMaxError;
    int boundsSamples;
    float mortonMs;
    int mortonMismatches;
    int mortonSamples;
    float mortonSortMs;
    int mortonSortMismatches;
    int mortonSortCount;
    float pairMs;
    int pairCandidates;
    int pairSphereHits;
    int pairAllPairsSphereHits;
    int pairMissedSphereHits;
    int pairMismatches;
    float sapMs;
    int sapCandidates;
    int sapSphereHits;
    int sapAllPairsSphereHits;
    int sapMissedSphereHits;
    int sapMismatches;
    int sapBestAxis;
    int sapAxisCandidates[3];
    int sapAxisSphereHits[3];
    int sapAxisMissedSphereHits[3];
    WebGpuTimingStats predictionTiming;
    WebGpuTimingStats velocityTiming;
    WebGpuTimingStats boundsTiming;
    WebGpuTimingStats mortonTiming;
    WebGpuTimingStats mortonSortTiming;
    WebGpuTimingStats pairTiming;
    WebGpuTimingStats sapTiming;

    WebGpuContext();
    bool initializeDeviceOnly();
    bool initialize(SDL_Window *window);
    bool runOffscreenSmokeTest(int width, int height);
    bool runComputeSmokeTest();
    bool runBodyPredictionDiagnostic(const SimWorld &world, float dt, float gravity);
    bool runVelocityUpdateDiagnostic(const SimWorld &world, float dt);
    bool runBoundsDiagnostic(const SimWorld &world);
    bool runMortonDiagnostic(const SimWorld &world);
    bool runMortonSortDiagnostic(const SimWorld &world);
    bool runBroadphasePairDiagnostic(const SimWorld &world);
    bool runSweepAndPruneDiagnostic(const SimWorld &world);
    void resetDiagnosticTimingStats();
    bool clearSurface(int width, int height);
    bool clearWindowSurface(SDL_Window *window, int width, int height);
    bool renderTriangleSurface(int width, int height);
    bool renderInstancedPreviewSurface(const SimWorld &world, const WebGpuPreviewCamera &camera, int width, int height);
    bool renderInstancedPreviewSurface(const SimWorld &world, const WebGpuPreviewCamera &camera, const WebGpuRenderOptions &options, int width, int height);
    void shutdown();
    const char *statusText() const;
    const char *smokeStatusText() const;
    const char *computeStatusText() const;
    const char *presentStatusText() const;
    const char *predictionStatusText() const;
    const char *velocityStatusText() const;
    const char *boundsStatusText() const;
    const char *mortonStatusText() const;
    const char *mortonSortStatusText() const;
    const char *pairStatusText() const;
    const char *sapStatusText() const;
    float previewComputeMillis() const;
    float previewRenderMillis() const;
    float previewTotalMillis() const;
    int previewBatchCountValue() const;
    int previewInstanceCountValue() const;
    int previewBoxInstanceCount() const;
    int previewSphereInstanceCount() const;
    int previewCapsuleInstanceCount() const;
    int previewCylinderInstanceCount() const;
    int previewMeshAssetInstanceCount() const;
    float predictionMillis() const;
    float predictionMaxErrorValue() const;
    float predictionMaxAngularErrorValue() const;
    int predictionSampleCount() const;
    float velocityMillis() const;
    float velocityMaxLinearErrorValue() const;
    float velocityMaxAngularErrorValue() const;
    int velocitySampleCount() const;
    float boundsMillis() const;
    float boundsMaxErrorValue() const;
    int boundsSampleCount() const;
    float mortonMillis() const;
    int mortonMismatchCount() const;
    int mortonSampleCount() const;
    float mortonSortMillis() const;
    int mortonSortMismatchCount() const;
    int mortonSortItemCount() const;
    float pairMillis() const;
    int pairCandidateCount() const;
    int pairSphereHitCount() const;
    int pairAllPairsSphereHitCount() const;
    int pairMissedSphereHitCount() const;
    int pairMismatchCount() const;
    float sapMillis() const;
    int sapCandidateCount() const;
    int sapSphereHitCount() const;
    int sapAllPairsSphereHitCount() const;
    int sapMissedSphereHitCount() const;
    int sapMismatchCount() const;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    wgpu::Instance instance;
    wgpu::Surface surface;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::Buffer bodyInputBuffer;
    uint64_t bodyInputBufferBytes;
    wgpu::Buffer bodyInstanceBuffer;
    uint64_t bodyInstanceBufferBytes;
    wgpu::Buffer bodyPreviewCameraBuffer;
    wgpu::Texture bodyPreviewDepthTexture;
    wgpu::TextureView bodyPreviewDepthView;
    wgpu::Buffer predictionInputBuffer;
    uint64_t predictionInputBufferBytes;
    wgpu::Buffer predictionOutputBuffer;
    uint64_t predictionOutputBufferBytes;
    wgpu::Buffer predictionReadbackBuffer;
    uint64_t predictionReadbackBufferBytes;
    wgpu::Buffer predictionParamsBuffer;
    wgpu::BindGroupLayout predictionBindGroupLayout;
    wgpu::ComputePipeline predictionPipeline;
    wgpu::Buffer velocityInputBuffer;
    uint64_t velocityInputBufferBytes;
    wgpu::Buffer velocityOutputBuffer;
    uint64_t velocityOutputBufferBytes;
    wgpu::Buffer velocityReadbackBuffer;
    uint64_t velocityReadbackBufferBytes;
    wgpu::Buffer velocityParamsBuffer;
    wgpu::BindGroupLayout velocityBindGroupLayout;
    wgpu::ComputePipeline velocityPipeline;
    wgpu::Buffer boundsInputBuffer;
    uint64_t boundsInputBufferBytes;
    wgpu::Buffer boundsOutputBuffer;
    uint64_t boundsOutputBufferBytes;
    wgpu::Buffer boundsReadbackBuffer;
    uint64_t boundsReadbackBufferBytes;
    wgpu::Buffer boundsParamsBuffer;
    wgpu::BindGroupLayout boundsBindGroupLayout;
    wgpu::ComputePipeline boundsPipeline;
    wgpu::Buffer mortonInputBuffer;
    uint64_t mortonInputBufferBytes;
    wgpu::Buffer mortonOutputBuffer;
    uint64_t mortonOutputBufferBytes;
    wgpu::Buffer mortonReadbackBuffer;
    uint64_t mortonReadbackBufferBytes;
    wgpu::Buffer mortonParamsBuffer;
    wgpu::BindGroupLayout mortonBindGroupLayout;
    wgpu::ComputePipeline mortonPipeline;
    wgpu::Buffer mortonSortBuffer;
    uint64_t mortonSortBufferBytes;
    wgpu::Buffer mortonSortReadbackBuffer;
    uint64_t mortonSortReadbackBufferBytes;
    wgpu::Buffer mortonSortParamsBuffer;
    wgpu::BindGroupLayout mortonSortBindGroupLayout;
    wgpu::ComputePipeline mortonSortPipeline;
    wgpu::Buffer pairBodyBuffer;
    uint64_t pairBodyBufferBytes;
    wgpu::Buffer pairItemBuffer;
    uint64_t pairItemBufferBytes;
    wgpu::Buffer pairCountersBuffer;
    uint64_t pairCountersBufferBytes;
    wgpu::Buffer pairReadbackBuffer;
    uint64_t pairReadbackBufferBytes;
    wgpu::Buffer pairParamsBuffer;
    wgpu::BindGroupLayout pairBindGroupLayout;
    wgpu::ComputePipeline pairPipeline;
    wgpu::Buffer sapIntervalBuffer;
    uint64_t sapIntervalBufferBytes;
    wgpu::Buffer sapCountersBuffer;
    uint64_t sapCountersBufferBytes;
    wgpu::Buffer sapReadbackBuffer;
    uint64_t sapReadbackBufferBytes;
    wgpu::Buffer sapParamsBuffer;
    wgpu::BindGroupLayout sapSortBindGroupLayout;
    wgpu::ComputePipeline sapSortPipeline;
    wgpu::BindGroupLayout sapPairBindGroupLayout;
    wgpu::ComputePipeline sapPairPipeline;
    wgpu::BindGroupLayout bodyPreviewComputeBindGroupLayout;
    wgpu::ComputePipeline bodyPreviewComputePipeline;
    wgpu::BindGroupLayout bodyPreviewBindGroupLayout;
    wgpu::RenderPipeline bodyPreviewGridPipeline;
    wgpu::RenderPipeline bodyPreviewPipeline;
    wgpu::TextureFormat bodyPreviewFormat;
    bool bodyPreviewSurfaceConfigured;
    int bodyPreviewWidth;
    int bodyPreviewHeight;
#endif
};

struct WebGpuPhysicsBackend : PhysicsBackend
{
    const char *name() const override;
    void step(Solver &solver) override;
};

struct WebGpuRenderBackend : RenderBackend
{
    const char *name() const override;
    bool available() const;
};
