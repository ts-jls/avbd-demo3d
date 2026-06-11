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

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdarg.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <iomanip>
#include <sstream>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

#ifdef TARGET_OS_MAC
#include <OpenGL/GL.h>
#else
#include <GL/gl.h>
#endif

#include <SDL2/SDL.h>
#include <imgui.h>
#include <backends/imgui_impl_sdl2.h>
#include <backends/imgui_impl_opengl3.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "maths.h"
#include "solver.h"
#include "scenes.h"
#include "simulation_host.h"
#include "webgpu_backend.h"
#include "viewer_bridge.h"

#define WinWidth 1280
#define WinHeight 720

bool Running = 1;
bool FullScreen = 0;
SDL_Window *Window;
SDL_GLContext Context;
int WindowFlags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE;

SimulationHost simulationHost;
Solver *solver = &simulationHost.solver();
WebGpuContext webgpuContext;
WebGpuContext webgpuClearContext;
WebGpuContext webgpuMainViewContext;
ViewerBridge viewerBridge;
Joint *drag = 0;
int &currScene = simulationHost.sceneIndexRef();
float3 boxSize = {1, 1, 1};
float sphereRadius = 0.5f;
float capsuleRadius = 0.35f;
float capsuleHalfLength = 0.9f;
float cylinderRadius = 0.45f;
float cylinderHalfLength = 0.6f;
float boxVelocity = 10.0f;
float boxFriction = 0.5f;
float boxDensity = 1.0f;
bool &paused = simulationHost.pausedRef();
bool shootRequested = false;
Uint32 lastTapTicks = 0;
Uint32 ignoreEscapeUntilTicks = 0;
float2 lastTapPos = {0, 0};
float &lastPhysicsMs = simulationHost.lastPhysicsMsRef();
bool nativeRenderBodies = true;
bool nativeRenderProjectedShadows = true;
bool nativeRenderDebugForces = true;
float lastNativeRenderMs = 0.0f;
float lastNativeStaticBodiesMs = 0.0f;
float lastNativeProjectedShadowsMs = 0.0f;
float lastNativeDynamicBodiesMs = 0.0f;
float lastNativeDebugForcesMs = 0.0f;
float lastImGuiRenderMs = 0.0f;
float lastSwapPresentMs = 0.0f;
int lastNativeStaticBodiesDrawn = 0;
int lastNativeDynamicBodiesDrawn = 0;
int lastNativeDebugForcesDrawn = 0;
bool webgpuDiagnosticsEnabled = false;
bool webgpuBodyPredictionDiagnostic = false;
int webgpuBodyPredictionCadence = 30;
int webgpuBodyPredictionFrame = 0;
bool webgpuVelocityUpdateDiagnostic = false;
int webgpuVelocityUpdateCadence = 30;
int webgpuVelocityUpdateFrame = 0;
bool webgpuBoundsDiagnostic = false;
int webgpuBoundsCadence = 30;
int webgpuBoundsFrame = 0;
bool webgpuMortonDiagnostic = false;
int webgpuMortonCadence = 30;
int webgpuMortonFrame = 0;
bool webgpuMortonSortDiagnostic = false;
int webgpuMortonSortCadence = 120;
int webgpuMortonSortFrame = 0;
bool webgpuPairDiagnostic = false;
int webgpuPairCadence = 120;
int webgpuPairFrame = 0;
bool webgpuSapDiagnostic = false;
int webgpuSapCadence = 120;
int webgpuSapFrame = 0;

enum RenderBackendMode
{
    RENDER_BACKEND_OPENGL,
    RENDER_BACKEND_WEBGPU_INSTANCED_PREVIEW,
    RENDER_BACKEND_COUNT
};

enum MainViewMode
{
    MAIN_VIEW_OPENGL,
    MAIN_VIEW_WEBGPU_EXPERIMENTAL,
    MAIN_VIEW_COUNT
};

enum PhysicsBackendMode
{
    PHYSICS_BACKEND_CPU_REFERENCE,
    PHYSICS_BACKEND_WEBGPU_EXPERIMENTAL,
    PHYSICS_BACKEND_WEBGPU_EXPERIMENTAL_FAST,
    PHYSICS_BACKEND_WEBGPU_COUNTERLESS_FAST,
    PHYSICS_BACKEND_WEBGPU_DIRECT_FAST,
    PHYSICS_BACKEND_WEBGPU_RESIDENT_GROUND_FAST,
    PHYSICS_BACKEND_WEBGPU_CONTACT_DIRECT,
    PHYSICS_BACKEND_WEBGPU_CONTACT_RESIDENT,
    PHYSICS_BACKEND_WEBGPU_CONTACT_RESIDENT_ASYNC,
    PHYSICS_BACKEND_WEBGPU_JOINT_DIRECT,
    PHYSICS_BACKEND_COUNT
};

RenderBackendMode renderBackendMode = RENDER_BACKEND_OPENGL;
const char *renderBackendNames[RENDER_BACKEND_COUNT] = {
    "OpenGL Reference",
    "WebGPU Instanced Preview"};
MainViewMode mainViewMode = MAIN_VIEW_OPENGL;
const char *mainViewNames[MAIN_VIEW_COUNT] = {
    "OpenGL Main View",
    "WebGPU Main View Experimental"};
PhysicsBackendMode physicsBackendMode = PHYSICS_BACKEND_CPU_REFERENCE;
const char *physicsBackendNames[PHYSICS_BACKEND_COUNT] = {
    "CPU Reference",
    "WebGPU Physics Experimental",
    "WebGPU Physics Experimental Fast",
    "WebGPU Physics Experimental Counterless Fast",
    "WebGPU Physics Experimental Direct Fast",
    "WebGPU Physics Experimental Resident Ground Fast",
    "WebGPU Physics Experimental Contact Direct",
    "WebGPU Physics Experimental Contact Resident",
    "WebGPU Physics Experimental Contact Resident Async",
    "WebGPU Physics Experimental Joint Direct"};
SDL_Window *webgpuInstancedPreviewWindow = 0;
SDL_Window *webgpuMainViewWindow = 0;
WebGpuRenderOptions webgpuRenderOptions;

void setPhysicsBackendMode(PhysicsBackendMode mode)
{
    if ((mode == PHYSICS_BACKEND_WEBGPU_EXPERIMENTAL ||
         mode == PHYSICS_BACKEND_WEBGPU_EXPERIMENTAL_FAST ||
         mode == PHYSICS_BACKEND_WEBGPU_COUNTERLESS_FAST ||
         mode == PHYSICS_BACKEND_WEBGPU_DIRECT_FAST ||
         mode == PHYSICS_BACKEND_WEBGPU_RESIDENT_GROUND_FAST ||
         mode == PHYSICS_BACKEND_WEBGPU_CONTACT_DIRECT ||
         mode == PHYSICS_BACKEND_WEBGPU_CONTACT_RESIDENT ||
         mode == PHYSICS_BACKEND_WEBGPU_CONTACT_RESIDENT_ASYNC ||
         mode == PHYSICS_BACKEND_WEBGPU_JOINT_DIRECT) &&
        webgpuContext.deviceReady)
    {
        physicsBackendMode = mode;
        WebGpuPhysicsOptions options;
        if (mode == PHYSICS_BACKEND_WEBGPU_EXPERIMENTAL_FAST ||
            mode == PHYSICS_BACKEND_WEBGPU_COUNTERLESS_FAST ||
            mode == PHYSICS_BACKEND_WEBGPU_DIRECT_FAST ||
            mode == PHYSICS_BACKEND_WEBGPU_RESIDENT_GROUND_FAST)
        {
            options.applyContactPositions = true;
            options.applyGroundContacts = true;
            options.validateResidentContacts = false;
            options.useResidentGroundContacts = mode == PHYSICS_BACKEND_WEBGPU_RESIDENT_GROUND_FAST;
            options.useCpuFallbackPairs = mode == PHYSICS_BACKEND_WEBGPU_EXPERIMENTAL_FAST ||
                                          mode == PHYSICS_BACKEND_WEBGPU_COUNTERLESS_FAST ||
                                          mode == PHYSICS_BACKEND_WEBGPU_DIRECT_FAST;
            options.skipSapCounterReadback = mode == PHYSICS_BACKEND_WEBGPU_COUNTERLESS_FAST;
            options.useDirectSpherePositionSolve = mode == PHYSICS_BACKEND_WEBGPU_DIRECT_FAST;
        }
        if (mode == PHYSICS_BACKEND_WEBGPU_JOINT_DIRECT)
        {
            options.useJointProposals = true;
            options.directJointSolveOnly = true;
            options.jointProposalIterations = 2;
        }
        if (mode == PHYSICS_BACKEND_WEBGPU_CONTACT_DIRECT ||
            mode == PHYSICS_BACKEND_WEBGPU_CONTACT_RESIDENT ||
            mode == PHYSICS_BACKEND_WEBGPU_CONTACT_RESIDENT_ASYNC)
        {
            options.directContactSolveOnly = true;
            options.useResidentPrimitiveContacts = mode == PHYSICS_BACKEND_WEBGPU_CONTACT_RESIDENT ||
                                                   mode == PHYSICS_BACKEND_WEBGPU_CONTACT_RESIDENT_ASYNC;
            options.validateResidentContacts = mode == PHYSICS_BACKEND_WEBGPU_CONTACT_RESIDENT;
            if (mode == PHYSICS_BACKEND_WEBGPU_CONTACT_RESIDENT_ASYNC)
            {
                options.disableResidentContactReadbacks = true;
                options.useResidentCounterlessContacts = true;
                options.asyncFinalPositionReadback = true;
                options.residentContactIterations = 4;
                options.residentContactRelaxation = 0.02f;
            }
        }
        solver->physicsBackend.reset(new WebGpuPhysicsBackend(&webgpuContext, options));
        return;
    }

    physicsBackendMode = PHYSICS_BACKEND_CPU_REFERENCE;
    solver->physicsBackend = makeCpuReferencePhysicsBackend();
}

void startupLog(const char *format, ...)
{
    FILE *file = 0;
#ifdef _WIN32
    fopen_s(&file, "avbd_startup.log", "a");
#else
    file = fopen("avbd_startup.log", "a");
#endif
    if (!file)
        return;

    va_list args;
    va_start(args, format);
    vfprintf(file, format, args);
    va_end(args);
    fprintf(file, "\n");
    fclose(file);
}

std::string normalizeSceneName(const char *text)
{
    std::string out;
    if (!text)
        return out;
    for (const char *c = text; *c; ++c)
    {
        unsigned char ch = (unsigned char)*c;
        if (isalnum(ch))
            out.push_back((char)tolower(ch));
    }
    return out;
}

bool setStartupSceneByName(const char *name)
{
    std::string desired = normalizeSceneName(name);
    if (desired.empty())
        return false;
    for (int i = 0; i < sceneCount; ++i)
    {
        if (normalizeSceneName(sceneNames[i]) == desired)
        {
            currScene = i;
            return true;
        }
    }
    return false;
}

void applyStartupSceneOverride(int argc, char *argv[])
{
    const char *requestedScene = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--scene") == 0 && i + 1 < argc)
        {
            requestedScene = argv[i + 1];
            break;
        }
        const char prefix[] = "--scene=";
        if (strncmp(argv[i], prefix, sizeof(prefix) - 1) == 0)
        {
            requestedScene = argv[i] + sizeof(prefix) - 1;
            break;
        }
    }

#ifdef _WIN32
    if (!requestedScene)
    {
        static char envScene[256] = {};
        DWORD length = GetEnvironmentVariableA("AVBD_START_SCENE", envScene, (DWORD)sizeof(envScene));
        if (length > 0 && length < sizeof(envScene))
            requestedScene = envScene;
    }
#else
    if (!requestedScene)
        requestedScene = getenv("AVBD_START_SCENE");
#endif

    if (!requestedScene)
        return;

    if (setStartupSceneByName(requestedScene))
        startupLog("Startup: scene override '%s' -> %s", requestedScene, sceneNames[currScene]);
    else
        startupLog("Startup: unknown scene override '%s'", requestedScene);
}

#ifdef _WIN32
void formatWindowsError(DWORD error, char *buffer, size_t bufferSize)
{
    if (!buffer || bufferSize == 0)
        return;

    buffer[0] = 0;
    DWORD length = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                  0, error, 0, buffer, (DWORD)bufferSize, 0);
    if (length == 0)
    {
        snprintf(buffer, bufferSize, "FormatMessage failed");
        return;
    }

    while (length > 0 && (buffer[length - 1] == '\n' || buffer[length - 1] == '\r' || buffer[length - 1] == ' '))
    {
        buffer[length - 1] = 0;
        --length;
    }
}

int countRunningAvbdProcesses()
{
    int count = 0;
    DWORD processes[2048];
    DWORD bytesReturned = 0;
    if (!EnumProcesses(processes, sizeof(processes), &bytesReturned))
        return -1;

    DWORD processCount = bytesReturned / sizeof(DWORD);
    for (DWORD i = 0; i < processCount; ++i)
    {
        if (processes[i] == 0)
            continue;

        HANDLE process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processes[i]);
        if (!process)
            continue;

        char moduleName[MAX_PATH] = {};
        if (GetModuleBaseNameA(process, 0, moduleName, MAX_PATH) &&
            _stricmp(moduleName, "avbd_demo3d.exe") == 0)
        {
            ++count;
        }
        CloseHandle(process);
    }

    return count;
}

void logWindowCreationDiagnostics(const char *label, int attempt, DWORD lastError)
{
    char errorText[256];
    formatWindowsError(lastError, errorText, sizeof(errorText));

    char exePath[MAX_PATH] = {};
    GetModuleFileNameA(0, exePath, MAX_PATH);

    char cwd[MAX_PATH] = {};
    GetCurrentDirectoryA(MAX_PATH, cwd);

    HANDLE process = GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS memory = {};
    DWORD userHandles = GetGuiResources(process, GR_USEROBJECTS);
    DWORD gdiHandles = GetGuiResources(process, GR_GDIOBJECTS);
    int avbdProcesses = countRunningAvbdProcesses();

    startupLog("Startup diagnostics: %s attempt %d", label, attempt);
    startupLog("  Win32 last error: %lu (%s)", (unsigned long)lastError, errorText);
    startupLog("  USER handles: %lu", (unsigned long)userHandles);
    startupLog("  GDI handles: %lu", (unsigned long)gdiHandles);
    if (GetProcessMemoryInfo(process, &memory, sizeof(memory)))
    {
        startupLog("  Working set: %llu KB", (unsigned long long)(memory.WorkingSetSize / 1024));
        startupLog("  Pagefile usage: %llu KB", (unsigned long long)(memory.PagefileUsage / 1024));
    }
    else
    {
        startupLog("  Process memory: unavailable");
    }
    startupLog("  avbd_demo3d.exe processes: %d", avbdProcesses);
    startupLog("  Executable: %s", exePath);
    startupLog("  Working directory: %s", cwd);
}
#endif

void configureOpenGlAttributes()
{
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1); // Enable multisampling
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

#ifdef __EMSCRIPTEN__
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3); // OpenGL ES 3.0 (WebGL2)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0); // No forward-compatible flag
#endif
}

SDL_Window *createWindowWithRetry()
{
    const int maxAttempts = 3;
    for (int attempt = 1; attempt <= maxAttempts; ++attempt)
    {
        SDL_Window *window = SDL_CreateWindow("AVBD 3D", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                              WinWidth, WinHeight, WindowFlags);
        if (window)
        {
            if (attempt > 1)
                startupLog("Startup: window created on retry %d", attempt);
            return window;
        }

#ifdef _WIN32
        DWORD lastError = GetLastError();
#endif
        const char *sdlError = SDL_GetError();
        startupLog("Startup: SDL_CreateWindow attempt %d failed: %s", attempt, sdlError);
#ifdef _WIN32
        logWindowCreationDiagnostics("SDL_CreateWindow", attempt, lastError);
#endif
        if (attempt < maxAttempts)
            SDL_Delay(250);
    }

    return 0;
}

SDL_Window *createWindowWithVideoResetFallback()
{
    SDL_Window *window = createWindowWithRetry();
    if (window)
        return window;

    startupLog("Startup: resetting SDL video subsystem after window creation failures");
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
    SDL_Delay(750);

    if (SDL_InitSubSystem(SDL_INIT_VIDEO) < 0)
    {
        startupLog("Startup failed: SDL video reset: %s", SDL_GetError());
        return 0;
    }

    configureOpenGlAttributes();
    window = createWindowWithRetry();
    if (window)
        startupLog("Startup: window created after SDL video subsystem reset");

    return window;
}


const char *broadphaseNames[BROADPHASE_COUNT] = {
    "All Pairs",
    "Uniform Grid",
    "Sweep and Prune"};

const float spatialHashCellSizePresets[] = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
const char *spatialHashCellSizeNames[] = {"0.25", "0.5", "1.0", "2.0", "4.0"};
const int spatialHashCellSizeCount = sizeof(spatialHashCellSizePresets) / sizeof(spatialHashCellSizePresets[0]);

float camDistance = 50.0f;
float camAzimuth = rad(90.0f);
float camElevation = 0.35f;
float3 camTarget = {0, 0, 5.0f};
float3 camEye = {0, 0, 0};

const float kPi = 3.14159265358979323846f;
const float kFovY_deg = 45.0f;
const float kNear = 0.1f;
const float kFar = 2000.0f;

enum ShootShape
{
    SHOOT_SHAPE_BOX,
    SHOOT_SHAPE_SPHERE,
    SHOOT_SHAPE_CAPSULE,
    SHOOT_SHAPE_CYLINDER,
    SHOOT_SHAPE_COUNT
};

ShootShape shootShape = SHOOT_SHAPE_BOX;
const char *shootShapeNames[SHOOT_SHAPE_COUNT] = {
    "Box",
    "Sphere",
    "Capsule",
    "Cylinder"};

bool touchOnly = false;
std::map<SDL_FingerID, float2> activeFingers;
float2 prevGestureCenter;
bool hasPrevGestureCenter = false;
float dragRayDistance = 0.0f;
SDL_FingerID dragFingerId = 0;
bool touchHoldCandidate = false;
SDL_FingerID touchHoldFingerId = 0;
Uint32 touchHoldStartTicks = 0;
float2 touchHoldStartPos = {0, 0};

enum DragMode
{
    DRAG_MODE_NONE,
    DRAG_MODE_MOUSE,
    DRAG_MODE_TOUCH
};

DragMode dragMode = DRAG_MODE_NONE;

void makePlaneFromPointNormal(const float3 &p, const float3 &n, GLfloat plane[4]);
void makeShadowMatrix(GLfloat out[16], const GLfloat light[4], const GLfloat plane[4]);
void drawProjectedShadows();
void releaseDrag();

void closeWebGpuMainViewWindow()
{
    if (webgpuMainViewWindow)
    {
        SDL_DestroyWindow(webgpuMainViewWindow);
        webgpuMainViewWindow = 0;
        webgpuMainViewContext.shutdown();
        startupLog("Render: WebGPU main view window closed");
    }
    mainViewMode = MAIN_VIEW_OPENGL;
}

void openWebGpuMainViewWindow()
{
    if (!webgpuContext.deviceReady)
    {
        startupLog("Render: WebGPU main view skipped because device is not ready");
        mainViewMode = MAIN_VIEW_OPENGL;
        return;
    }

    if (webgpuMainViewWindow)
    {
        SDL_RaiseWindow(webgpuMainViewWindow);
        return;
    }

    webgpuMainViewWindow = SDL_CreateWindow("AVBD 3D - WebGPU Main View",
                                            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                            960, 540, SDL_WINDOW_RESIZABLE);
    if (!webgpuMainViewWindow)
    {
        startupLog("Render failed: WebGPU main view window: %s", SDL_GetError());
        mainViewMode = MAIN_VIEW_OPENGL;
        return;
    }

    if (!webgpuMainViewContext.initialize(webgpuMainViewWindow))
    {
        startupLog("Render failed: %s", webgpuMainViewContext.statusText());
        SDL_DestroyWindow(webgpuMainViewWindow);
        webgpuMainViewWindow = 0;
        mainViewMode = MAIN_VIEW_OPENGL;
        return;
    }

    startupLog("Render: WebGPU main view window opened");
}

void drawWebGpuMainViewWindow()
{
    if (mainViewMode != MAIN_VIEW_WEBGPU_EXPERIMENTAL)
        return;

    if (!webgpuMainViewWindow)
        openWebGpuMainViewWindow();
    if (!webgpuMainViewWindow)
        return;

    int w = 0;
    int h = 0;
    SDL_GetWindowSize(webgpuMainViewWindow, &w, &h);
    WebGpuPreviewCamera camera = {camEye, camTarget, float3{0, 0, 1}, kFovY_deg};
    if (!webgpuMainViewContext.renderInstancedPreviewSurface(solver->world, camera, webgpuRenderOptions, w, h))
        startupLog("Render: %s", webgpuMainViewContext.presentStatusText());
}

void closeWebGpuInstancedPreviewWindow()
{
    if (webgpuInstancedPreviewWindow)
    {
        SDL_DestroyWindow(webgpuInstancedPreviewWindow);
        webgpuInstancedPreviewWindow = 0;
        webgpuClearContext.shutdown();
        startupLog("Render: WebGPU instanced preview window closed");
    }
}

void openWebGpuInstancedPreviewWindow()
{
    if (!webgpuContext.deviceReady)
    {
        startupLog("Render: WebGPU instanced preview skipped because device is not ready");
        return;
    }

    if (webgpuInstancedPreviewWindow)
    {
        SDL_RaiseWindow(webgpuInstancedPreviewWindow);
        return;
    }

    webgpuInstancedPreviewWindow = SDL_CreateWindow("AVBD 3D - WebGPU Instanced Preview",
                                                    SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                                    640, 360, SDL_WINDOW_RESIZABLE);
    if (!webgpuInstancedPreviewWindow)
    {
        startupLog("Render failed: WebGPU instanced preview window: %s", SDL_GetError());
        return;
    }

    if (!webgpuClearContext.initialize(webgpuInstancedPreviewWindow))
    {
        startupLog("Render failed: %s", webgpuClearContext.statusText());
        SDL_DestroyWindow(webgpuInstancedPreviewWindow);
        webgpuInstancedPreviewWindow = 0;
        return;
    }

    startupLog("Render: WebGPU instanced preview window opened");
}

void drawWebGpuInstancedPreviewWindow()
{
    if (!webgpuInstancedPreviewWindow)
        return;

    int w = 0;
    int h = 0;
    SDL_GetWindowSize(webgpuInstancedPreviewWindow, &w, &h);
    WebGpuPreviewCamera camera = {camEye, camTarget, float3{0, 0, 1}, kFovY_deg};
    if (!webgpuClearContext.renderInstancedPreviewSurface(solver->world, camera, webgpuRenderOptions, w, h))
        startupLog("Render: %s", webgpuClearContext.presentStatusText());
}

float elapsedMs(Uint64 begin, Uint64 end)
{
    return (float)((double)(end - begin) * 1000.0 / (double)SDL_GetPerformanceFrequency());
}

void stepSolverTimed()
{
    simulationHost.stepFrame();
}

void broadcastViewerSnapshot()
{
    viewerBridge.broadcastSnapshot(simulationHost.world(), simulationHost.currentSceneName(), simulationHost.nextSnapshotFrame());
}

void processViewerCommands()
{
    SimulationCommand command;
    while (viewerBridge.pollCommand(command))
    {
        std::string normalized = SimulationHost::normalizeSceneName(command.command.c_str());
        if (normalized == "scene" || normalized == "loadscene" || normalized == "reset")
            releaseDrag();
        simulationHost.applyCommand(command);
    }
}

int currentSpatialHashCellSizePreset()
{
    for (int i = 0; i < spatialHashCellSizeCount; ++i)
    {
        if (solver->spatialHashCellSize == spatialHashCellSizePresets[i])
            return i;
    }
    return 3;
}

std::string performanceMetricsText()
{
    const SolverStats &s = solver->stats;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);

    out << "Scene: " << sceneNames[currScene] << "\n";
    out << "Physics backend: " << solver->physicsBackend->name() << "\n";
    out << "Physics backend mode: " << physicsBackendNames[(int)physicsBackendMode] << "\n";
    out << "Main view: " << mainViewNames[(int)mainViewMode] << "\n";
    out << "Render backend: " << renderBackendNames[(int)renderBackendMode] << "\n";
    out << "Viewer bridge: " << viewerBridge.statusText() << "\n";
    out << "Viewer clients: " << viewerBridge.clientCountValue() << "\n";
    out << "Viewer snapshot mode: " << (viewerBridge.statsSnapshot().binarySnapshotMode ? "Binary" : "JSON") << "\n";
    out << "WebGPU: " << webgpuContext.statusText() << "\n";
    out << "WebGPU smoke: " << webgpuContext.smokeStatusText() << "\n";
    out << "WebGPU compute: " << webgpuContext.computeStatusText() << "\n";
    out << "WebGPU runtime: " << webgpuContext.runtimeStatusText() << "\n";
    out << "WebGPU runtime total: " << webgpuContext.runtimeTotalMillis() << " ms\n";
    out << "WebGPU runtime prediction: " << webgpuContext.runtimePredictionMillis() << " ms\n";
    out << "WebGPU runtime velocity: " << webgpuContext.runtimeVelocityMillis() << " ms\n";
    out << "WebGPU runtime CPU fallback: " << webgpuContext.runtimeCpuFallbackMillis() << " ms\n";
    out << "WebGPU runtime max linear error: " << webgpuContext.runtimeMaxLinearErrorValue() << "\n";
    out << "WebGPU runtime max angular error: " << webgpuContext.runtimeMaxAngularErrorValue() << "\n";
    out << "WebGPU runtime frames: " << webgpuContext.runtimeFrameCount() << "\n";
    out << "WebGPU runtime fallbacks: " << webgpuContext.runtimeFallbackCount() << "\n";
    out << "WebGPU SAP emitted pairs: " << webgpuContext.sapSphereHitCount() << "\n";
    out << "WebGPU SAP counter readback bytes: " << webgpuContext.sapCounterReadbackByteCount() << "\n";
    out << "WebGPU SAP counter readback: " << webgpuContext.sapCounterReadbackMillis() << " ms\n";
    out << "WebGPU SAP pair readback bytes: " << webgpuContext.sapPairReadbackByteCount() << "\n";
    out << "WebGPU SAP pair readback: " << webgpuContext.sapPairReadbackMillis() << " ms\n";
    out << "WebGPU sphere contacts: " << webgpuContext.sphereContactCountValue() << "\n";
    out << "WebGPU external contacts: " << webgpuContext.sphereContactExternalContactCount() << "\n";
    out << "WebGPU external ground contacts: " << webgpuContext.sphereContactExternalGroundContactCount() << "\n";
    out << "WebGPU sphere contact readback bytes: " << webgpuContext.sphereContactReadbackByteCount() << "\n";
    out << "WebGPU sphere contact total: " << webgpuContext.sphereContactMillis() << " ms\n";
    out << "WebGPU sphere contact readback: " << webgpuContext.sphereContactReadbackMillis() << " ms\n";
    out << "WebGPU sphere contact body refs: " << webgpuContext.sphereContactBodyRefCount() << "\n";
    out << "WebGPU sphere contact active bodies: " << webgpuContext.sphereContactActiveBodyCount() << "\n";
    out << "WebGPU sphere contact max per body: " << webgpuContext.sphereContactMaxPerBodyCount() << "\n";
    out << "WebGPU sphere contact avg per active body: " << webgpuContext.sphereContactAvgPerActiveBodyValue() << "\n";
    out << "WebGPU sphere contact adjacency readback bytes: " << webgpuContext.sphereContactAdjacencyReadbackByteCount() << "\n";
    out << "WebGPU sphere contact adjacency buffer bytes: " << webgpuContext.sphereContactAdjacencyBufferByteCount() << "\n";
    out << "WebGPU sphere contact adjacency capacity: " << webgpuContext.sphereContactAdjacencyCapacityValue() << "\n";
    out << "WebGPU sphere contact adjacency written refs: " << webgpuContext.sphereContactAdjacencyWrittenRefCount() << "\n";
    out << "WebGPU sphere contact adjacency overflow refs: " << webgpuContext.sphereContactAdjacencyOverflowRefCount() << "\n";
    out << "WebGPU sphere contact adjacency: " << webgpuContext.sphereContactAdjacencyMillis() << " ms\n";
    out << "WebGPU sphere contact gather refs: " << webgpuContext.sphereContactGatherRefCount() << "\n";
    out << "WebGPU sphere contact gather active bodies: " << webgpuContext.sphereContactGatherActiveBodyCount() << "\n";
    out << "WebGPU sphere contact gather max per body: " << webgpuContext.sphereContactGatherMaxPerBodyCount() << "\n";
    out << "WebGPU sphere contact gather mismatches: " << webgpuContext.sphereContactGatherMismatchCount() << "\n";
    out << "WebGPU sphere contact gather readback bytes: " << webgpuContext.sphereContactGatherReadbackByteCount() << "\n";
    out << "WebGPU sphere contact gather normal checksum: " << webgpuContext.sphereContactGatherNormalChecksumValue() << "\n";
    out << "WebGPU sphere contact proposal active bodies: " << webgpuContext.sphereContactProposalActiveBodyCount() << "\n";
    out << "WebGPU sphere contact proposal max correction: " << webgpuContext.sphereContactProposalMaxCorrectionValue() << "\n";
    out << "WebGPU sphere contact proposal correction checksum: " << webgpuContext.sphereContactProposalCorrectionChecksumValue() << "\n";
    out << "WebGPU sphere contact gather: " << webgpuContext.sphereContactGatherMillis() << " ms\n";
    out << "WebGPU sphere contact proposal output active bodies: " << webgpuContext.sphereContactProposalOutputActiveBodyCount() << "\n";
    out << "WebGPU sphere contact proposal output readback bytes: " << webgpuContext.sphereContactProposalOutputReadbackByteCount() << "\n";
    out << "WebGPU sphere contact proposal output max delta: " << webgpuContext.sphereContactProposalOutputMaxDeltaValue() << "\n";
    out << "WebGPU sphere contact proposal output checksum: " << webgpuContext.sphereContactProposalOutputChecksumValue() << "\n";
    out << "WebGPU sphere contact proposal output: " << webgpuContext.sphereContactProposalOutputMillis() << " ms\n";
    out << "WebGPU sphere contact proposal residual readback bytes: " << webgpuContext.sphereContactProposalResidualReadbackByteCount() << "\n";
    out << "WebGPU sphere contact proposal residual before max: " << webgpuContext.sphereContactProposalResidualBeforeMaxValue() << "\n";
    out << "WebGPU sphere contact proposal residual after max: " << webgpuContext.sphereContactProposalResidualAfterMaxValue() << "\n";
    out << "WebGPU sphere contact proposal residual before checksum: " << webgpuContext.sphereContactProposalResidualBeforeChecksumValue() << "\n";
    out << "WebGPU sphere contact proposal residual after checksum: " << webgpuContext.sphereContactProposalResidualAfterChecksumValue() << "\n";
    out << "WebGPU sphere contact proposal residual: " << webgpuContext.sphereContactProposalResidualMillis() << " ms\n";
    out << "WebGPU sphere contact iterations: " << webgpuContext.sphereContactIterationCountValue() << "\n";
    out << "WebGPU sphere contact iteration residual after max: " << webgpuContext.sphereContactIterationResidualAfterMaxValue() << "\n";
    out << "WebGPU sphere contact iteration residual after checksum: " << webgpuContext.sphereContactIterationResidualAfterChecksumValue() << "\n";
    out << "WebGPU sphere contact iteration: " << webgpuContext.sphereContactIterationMillis() << " ms\n";
    out << "WebGPU sphere contact final position ready: " << webgpuContext.sphereContactFinalPositionReadyValue() << "\n";
    out << "WebGPU sphere contact final position bodies: " << webgpuContext.sphereContactFinalPositionBodyCountValue() << "\n";
    out << "WebGPU sphere contact final position bytes: " << webgpuContext.sphereContactFinalPositionByteCount() << "\n";
    out << "WebGPU sphere contact final position source: " << webgpuContext.sphereContactFinalPositionSourceText() << "\n";
    out << "WebGPU sphere contact applied position bodies: " << webgpuContext.sphereContactAppliedPositionBodyCount() << "\n";
    out << "WebGPU sphere contact applied position readback bytes: " << webgpuContext.sphereContactAppliedPositionReadbackByteCount() << "\n";
    out << "WebGPU sphere contact applied position max delta: " << webgpuContext.sphereContactAppliedPositionMaxDeltaValue() << "\n";
    out << "WebGPU sphere contact applied position checksum: " << webgpuContext.sphereContactAppliedPositionChecksumValue() << "\n";
    out << "WebGPU sphere contact applied position: " << webgpuContext.sphereContactAppliedPositionMillis() << " ms\n";
    out << "WebGPU sphere ground receivers: " << webgpuContext.sphereGroundReceiverCountValue() << "\n";
    out << "WebGPU ground dynamic bodies: " << webgpuContext.sphereGroundDynamicSphereCountValue() << "\n";
    out << "WebGPU sphere ground candidates: " << webgpuContext.sphereGroundCandidateCountValue() << "\n";
    out << "WebGPU direct sphere-cylinder bodies: " << webgpuContext.directSphereCylinderBodyCountValue() << "\n";
    out << "WebGPU direct sphere-cylinder candidates: " << webgpuContext.directSphereCylinderCandidateCountValue() << "\n";
    out << "WebGPU direct sphere-box bodies: " << webgpuContext.directSphereBoxBodyCountValue() << "\n";
    out << "WebGPU direct sphere-box candidates: " << webgpuContext.directSphereBoxCandidateCountValue() << "\n";
    out << "WebGPU direct box bodies: " << webgpuContext.directBoxBodyCountValue() << "\n";
    out << "WebGPU direct box-box candidates: " << webgpuContext.directBoxPairCandidateCountValue() << "\n";
    out << "WebGPU direct sphere contact applied bodies: " << webgpuContext.directSphereContactAppliedPositionBodyCount() << "\n";
    out << "WebGPU direct ground applied bodies: " << webgpuContext.directGroundAppliedPositionBodyCount() << "\n";
    out << "WebGPU sphere ground top: " << webgpuContext.sphereGroundTopValue() << "\n";
    out << "WebGPU sphere ground: " << webgpuContext.sphereGroundMillis() << " ms\n";
    out << "WebGPU diagnostics enabled: " << (webgpuDiagnosticsEnabled ? "On" : "Off") << "\n";
    out << "WebGPU ground grid: " << (webgpuRenderOptions.showGroundGrid ? "On" : "Off") << "\n";
    out << "WebGPU shape edges: " << (webgpuRenderOptions.showShapeEdges ? "On" : "Off") << "\n";
    out << "Native render bodies: " << (nativeRenderBodies ? "On" : "Off") << "\n";
    out << "Native render shadows: " << (nativeRenderProjectedShadows ? "On" : "Off") << "\n";
    out << "Native render debug forces: " << (nativeRenderDebugForces ? "On" : "Off") << "\n";
    bool anyWebGpuDiagnostic =
        webgpuBodyPredictionDiagnostic ||
        webgpuVelocityUpdateDiagnostic ||
        webgpuBoundsDiagnostic ||
        webgpuMortonDiagnostic ||
        webgpuMortonSortDiagnostic ||
        webgpuPairDiagnostic ||
        webgpuSapDiagnostic;
    if (webgpuDiagnosticsEnabled && !anyWebGpuDiagnostic)
        out << "WebGPU diagnostics selected: None\n";
    if (webgpuDiagnosticsEnabled && webgpuBodyPredictionDiagnostic)
    {
    out << "WebGPU body prediction diagnostic: " << (webgpuBodyPredictionDiagnostic ? "On" : "Off") << "\n";
    out << "WebGPU body prediction cadence: " << webgpuBodyPredictionCadence << "\n";
    out << "WebGPU body prediction: " << webgpuContext.predictionStatusText() << "\n";
    out << "WebGPU body prediction ms: " << webgpuContext.predictionMillis() << " ms\n";
    out << "WebGPU body prediction avg ms: " << webgpuContext.predictionTiming.avgMs << "\n";
    out << "WebGPU body prediction recent avg ms: " << webgpuContext.predictionTiming.recentAvgMs << "\n";
    out << "WebGPU body prediction min ms: " << webgpuContext.predictionTiming.minMs << "\n";
    out << "WebGPU body prediction max ms: " << webgpuContext.predictionTiming.maxMs << "\n";
    out << "WebGPU body prediction timing samples: " << webgpuContext.predictionTiming.samples << "\n";
    out << "WebGPU body prediction recent samples: " << webgpuContext.predictionTiming.recentSamples << "\n";
    out << "WebGPU body prediction max linear error: " << webgpuContext.predictionMaxErrorValue() << "\n";
    out << "WebGPU body prediction max angular error: " << webgpuContext.predictionMaxAngularErrorValue() << "\n";
    out << "WebGPU body prediction samples: " << webgpuContext.predictionSampleCount() << "\n";
    }
    if (webgpuDiagnosticsEnabled && webgpuVelocityUpdateDiagnostic)
    {
    out << "WebGPU velocity update diagnostic: " << (webgpuVelocityUpdateDiagnostic ? "On" : "Off") << "\n";
    out << "WebGPU velocity update cadence: " << webgpuVelocityUpdateCadence << "\n";
    out << "WebGPU velocity update: " << webgpuContext.velocityStatusText() << "\n";
    out << "WebGPU velocity update ms: " << webgpuContext.velocityMillis() << " ms\n";
    out << "WebGPU velocity update avg ms: " << webgpuContext.velocityTiming.avgMs << "\n";
    out << "WebGPU velocity update recent avg ms: " << webgpuContext.velocityTiming.recentAvgMs << "\n";
    out << "WebGPU velocity update min ms: " << webgpuContext.velocityTiming.minMs << "\n";
    out << "WebGPU velocity update max ms: " << webgpuContext.velocityTiming.maxMs << "\n";
    out << "WebGPU velocity update timing samples: " << webgpuContext.velocityTiming.samples << "\n";
    out << "WebGPU velocity update recent samples: " << webgpuContext.velocityTiming.recentSamples << "\n";
    out << "WebGPU velocity update max linear error: " << webgpuContext.velocityMaxLinearErrorValue() << "\n";
    out << "WebGPU velocity update max angular error: " << webgpuContext.velocityMaxAngularErrorValue() << "\n";
    out << "WebGPU velocity update samples: " << webgpuContext.velocitySampleCount() << "\n";
    }
    if (webgpuDiagnosticsEnabled && webgpuBoundsDiagnostic)
    {
    out << "WebGPU bounds diagnostic: " << (webgpuBoundsDiagnostic ? "On" : "Off") << "\n";
    out << "WebGPU bounds cadence: " << webgpuBoundsCadence << "\n";
    out << "WebGPU bounds: " << webgpuContext.boundsStatusText() << "\n";
    out << "WebGPU bounds ms: " << webgpuContext.boundsMillis() << " ms\n";
    out << "WebGPU bounds avg ms: " << webgpuContext.boundsTiming.avgMs << "\n";
    out << "WebGPU bounds recent avg ms: " << webgpuContext.boundsTiming.recentAvgMs << "\n";
    out << "WebGPU bounds min ms: " << webgpuContext.boundsTiming.minMs << "\n";
    out << "WebGPU bounds max ms: " << webgpuContext.boundsTiming.maxMs << "\n";
    out << "WebGPU bounds timing samples: " << webgpuContext.boundsTiming.samples << "\n";
    out << "WebGPU bounds recent samples: " << webgpuContext.boundsTiming.recentSamples << "\n";
    out << "WebGPU bounds max error: " << webgpuContext.boundsMaxErrorValue() << "\n";
    out << "WebGPU bounds samples: " << webgpuContext.boundsSampleCount() << "\n";
    }
    if (webgpuDiagnosticsEnabled && webgpuMortonDiagnostic)
    {
    out << "WebGPU morton diagnostic: " << (webgpuMortonDiagnostic ? "On" : "Off") << "\n";
    out << "WebGPU morton cadence: " << webgpuMortonCadence << "\n";
    out << "WebGPU morton: " << webgpuContext.mortonStatusText() << "\n";
    out << "WebGPU morton ms: " << webgpuContext.mortonMillis() << " ms\n";
    out << "WebGPU morton avg ms: " << webgpuContext.mortonTiming.avgMs << "\n";
    out << "WebGPU morton recent avg ms: " << webgpuContext.mortonTiming.recentAvgMs << "\n";
    out << "WebGPU morton min ms: " << webgpuContext.mortonTiming.minMs << "\n";
    out << "WebGPU morton max ms: " << webgpuContext.mortonTiming.maxMs << "\n";
    out << "WebGPU morton timing samples: " << webgpuContext.mortonTiming.samples << "\n";
    out << "WebGPU morton recent samples: " << webgpuContext.mortonTiming.recentSamples << "\n";
    out << "WebGPU morton mismatches: " << webgpuContext.mortonMismatchCount() << "\n";
    out << "WebGPU morton samples: " << webgpuContext.mortonSampleCount() << "\n";
    }
    if (webgpuDiagnosticsEnabled && webgpuMortonSortDiagnostic)
    {
    out << "WebGPU morton sort diagnostic: " << (webgpuMortonSortDiagnostic ? "On" : "Off") << "\n";
    out << "WebGPU morton sort cadence: " << webgpuMortonSortCadence << "\n";
    out << "WebGPU morton sort: " << webgpuContext.mortonSortStatusText() << "\n";
    out << "WebGPU morton sort ms: " << webgpuContext.mortonSortMillis() << " ms\n";
    out << "WebGPU morton sort avg ms: " << webgpuContext.mortonSortTiming.avgMs << "\n";
    out << "WebGPU morton sort recent avg ms: " << webgpuContext.mortonSortTiming.recentAvgMs << "\n";
    out << "WebGPU morton sort min ms: " << webgpuContext.mortonSortTiming.minMs << "\n";
    out << "WebGPU morton sort max ms: " << webgpuContext.mortonSortTiming.maxMs << "\n";
    out << "WebGPU morton sort timing samples: " << webgpuContext.mortonSortTiming.samples << "\n";
    out << "WebGPU morton sort recent samples: " << webgpuContext.mortonSortTiming.recentSamples << "\n";
    out << "WebGPU morton sort mismatches: " << webgpuContext.mortonSortMismatchCount() << "\n";
    out << "WebGPU morton sort items: " << webgpuContext.mortonSortItemCount() << "\n";
    }
    if (webgpuDiagnosticsEnabled && webgpuPairDiagnostic)
    {
    out << "WebGPU pair diagnostic: " << (webgpuPairDiagnostic ? "On" : "Off") << "\n";
    out << "WebGPU pair cadence: " << webgpuPairCadence << "\n";
    out << "WebGPU pair: " << webgpuContext.pairStatusText() << "\n";
    out << "WebGPU pair ms: " << webgpuContext.pairMillis() << " ms\n";
    out << "WebGPU pair avg ms: " << webgpuContext.pairTiming.avgMs << "\n";
    out << "WebGPU pair recent avg ms: " << webgpuContext.pairTiming.recentAvgMs << "\n";
    out << "WebGPU pair min ms: " << webgpuContext.pairTiming.minMs << "\n";
    out << "WebGPU pair max ms: " << webgpuContext.pairTiming.maxMs << "\n";
    out << "WebGPU pair timing samples: " << webgpuContext.pairTiming.samples << "\n";
    out << "WebGPU pair recent samples: " << webgpuContext.pairTiming.recentSamples << "\n";
    out << "WebGPU pair candidates: " << webgpuContext.pairCandidateCount() << "\n";
    out << "WebGPU pair sphere hits: " << webgpuContext.pairSphereHitCount() << "\n";
    out << "WebGPU pair all-pairs sphere hits: " << webgpuContext.pairAllPairsSphereHitCount() << "\n";
    out << "WebGPU pair missed sphere hits: " << webgpuContext.pairMissedSphereHitCount() << "\n";
    out << "WebGPU pair mismatches: " << webgpuContext.pairMismatchCount() << "\n";
    }
    if (webgpuDiagnosticsEnabled && webgpuSapDiagnostic)
    {
    out << "WebGPU SAP diagnostic: " << (webgpuSapDiagnostic ? "On" : "Off") << "\n";
    out << "WebGPU SAP cadence: " << webgpuSapCadence << "\n";
    out << "WebGPU SAP: " << webgpuContext.sapStatusText() << "\n";
    out << "WebGPU SAP ms: " << webgpuContext.sapMillis() << " ms\n";
    out << "WebGPU SAP avg ms: " << webgpuContext.sapTiming.avgMs << "\n";
    out << "WebGPU SAP recent avg ms: " << webgpuContext.sapTiming.recentAvgMs << "\n";
    out << "WebGPU SAP min ms: " << webgpuContext.sapTiming.minMs << "\n";
    out << "WebGPU SAP max ms: " << webgpuContext.sapTiming.maxMs << "\n";
    out << "WebGPU SAP timing samples: " << webgpuContext.sapTiming.samples << "\n";
    out << "WebGPU SAP recent samples: " << webgpuContext.sapTiming.recentSamples << "\n";
    out << "WebGPU SAP candidates: " << webgpuContext.sapCandidateCount() << "\n";
    out << "WebGPU SAP sphere hits: " << webgpuContext.sapSphereHitCount() << "\n";
    out << "WebGPU SAP all-pairs sphere hits: " << webgpuContext.sapAllPairsSphereHitCount() << "\n";
    out << "WebGPU SAP missed sphere hits: " << webgpuContext.sapMissedSphereHitCount() << "\n";
    out << "WebGPU SAP mismatches: " << webgpuContext.sapMismatchCount() << "\n";
    out << "WebGPU SAP best axis: " << (webgpuContext.sapBestAxis == 0 ? "X" : (webgpuContext.sapBestAxis == 1 ? "Y" : "Z")) << "\n";
    out << "WebGPU SAP X candidates: " << webgpuContext.sapAxisCandidates[0] << "\n";
    out << "WebGPU SAP X hits: " << webgpuContext.sapAxisSphereHits[0] << "\n";
    out << "WebGPU SAP X missed: " << webgpuContext.sapAxisMissedSphereHits[0] << "\n";
    out << "WebGPU SAP Y candidates: " << webgpuContext.sapAxisCandidates[1] << "\n";
    out << "WebGPU SAP Y hits: " << webgpuContext.sapAxisSphereHits[1] << "\n";
    out << "WebGPU SAP Y missed: " << webgpuContext.sapAxisMissedSphereHits[1] << "\n";
    out << "WebGPU SAP Z candidates: " << webgpuContext.sapAxisCandidates[2] << "\n";
    out << "WebGPU SAP Z hits: " << webgpuContext.sapAxisSphereHits[2] << "\n";
    out << "WebGPU SAP Z missed: " << webgpuContext.sapAxisMissedSphereHits[2] << "\n";
    }
    out << "WebGPU main view window: " << (webgpuMainViewWindow ? "Open" : "Closed") << "\n";
    out << "WebGPU main view device: " << webgpuMainViewContext.statusText() << "\n";
    out << "WebGPU main view present: " << webgpuMainViewContext.presentStatusText() << "\n";
    out << "WebGPU main view total: " << webgpuMainViewContext.previewTotalMillis() << " ms\n";
    out << "WebGPU main view compute submit: " << webgpuMainViewContext.previewComputeMillis() << " ms\n";
    out << "WebGPU main view render submit: " << webgpuMainViewContext.previewRenderMillis() << " ms\n";
    out << "WebGPU main view batches: " << webgpuMainViewContext.previewBatchCountValue() << "\n";
    out << "WebGPU main view instances: " << webgpuMainViewContext.previewInstanceCountValue() << "\n";
    out << "WebGPU main view boxes: " << webgpuMainViewContext.previewBoxInstanceCount() << "\n";
    out << "WebGPU main view spheres: " << webgpuMainViewContext.previewSphereInstanceCount() << "\n";
    out << "WebGPU main view capsules: " << webgpuMainViewContext.previewCapsuleInstanceCount() << "\n";
    out << "WebGPU main view cylinders: " << webgpuMainViewContext.previewCylinderInstanceCount() << "\n";
    out << "WebGPU main view mesh assets: " << webgpuMainViewContext.previewMeshAssetInstanceCount() << "\n";
    out << "WebGPU instanced preview window: " << (webgpuInstancedPreviewWindow ? "Open" : "Closed") << "\n";
    out << "WebGPU instanced preview device: " << webgpuClearContext.statusText() << "\n";
    out << "WebGPU instanced preview present: " << webgpuClearContext.presentStatusText() << "\n";
    out << "WebGPU instanced preview total: " << webgpuClearContext.previewTotalMillis() << " ms\n";
    out << "WebGPU instanced preview compute submit: " << webgpuClearContext.previewComputeMillis() << " ms\n";
    out << "WebGPU instanced preview render submit: " << webgpuClearContext.previewRenderMillis() << " ms\n";
    out << "WebGPU instanced preview batches: " << webgpuClearContext.previewBatchCountValue() << "\n";
    out << "WebGPU instanced preview instances: " << webgpuClearContext.previewInstanceCountValue() << "\n";
    out << "WebGPU instanced preview boxes: " << webgpuClearContext.previewBoxInstanceCount() << "\n";
    out << "WebGPU instanced preview spheres: " << webgpuClearContext.previewSphereInstanceCount() << "\n";
    out << "WebGPU instanced preview capsules: " << webgpuClearContext.previewCapsuleInstanceCount() << "\n";
    out << "WebGPU instanced preview cylinders: " << webgpuClearContext.previewCylinderInstanceCount() << "\n";
    out << "WebGPU instanced preview mesh assets: " << webgpuClearContext.previewMeshAssetInstanceCount() << "\n";
    out << "Broadphase: " << broadphaseNames[(int)solver->broadphaseMode] << "\n";
    out << "Selected cell size: " << solver->spatialHashCellSize << "\n";
    out << "Skip Ignore Solver Work: " << (solver->skipIgnoreCollisionSolverWork ? "On" : "Off") << "\n";
    out << "Deep Profiling: " << (solver->deepProfiling ? "On" : "Off") << "\n\n";

    out << "Broadphase\n";
    out << "Bodies: " << s.bodyCount << "\n";
    out << "Pair checks: " << s.pairChecks << "\n";
    out << "Sphere hits: " << s.sphereHits << "\n";
    out << "Created manifolds: " << s.manifoldsCreated << "\n";
    out << "Broadphase: " << s.broadphaseMs << " ms\n";
    out << "Hash build: " << s.spatialHashBuildMs << " ms\n";
    out << "Candidates: " << s.spatialHashCandidateMs << " ms\n";
    out << "Constrained checks: " << s.constrainedChecks << "\n";
    out << "Constrained hits: " << s.constrainedHits << "\n\n";

    out << "Solver\n";
    out << "Dense bodies: " << solver->world.bodies.size() << "\n";
    out << "Dense constraints: " << solver->world.constraints.size() << "\n";
    out << "Active bodies: " << s.activeBodyCount << "\n";
    out << "Forces: " << s.forceCount << "\n";
    out << "Joints: " << s.jointCount << "\n";
    out << "Springs: " << s.springCount << "\n";
    out << "Manifolds: " << s.manifoldCount << "\n";
    out << "Ignores: " << s.ignoreCollisionCount << "\n";
    out << "Primal visits: " << s.primalForceVisits << "\n";
    out << "Dual visits: " << s.dualForceVisits << "\n";
    out << "Primal ignore skipped: " << s.primalIgnoreCollisionSkipped << "\n";
    out << "Dual ignore skipped: " << s.dualIgnoreCollisionSkipped << "\n\n";

    if (solver->deepProfiling)
    {
        out << "Deep Broadphase\n";
        out << "Constraint: " << s.constrainedMs << " ms\n";
        out << "Manifold alloc: " << s.manifoldAllocMs << " ms\n";
        out << "Force scan visits: " << s.constrainedForceVisits << "\n\n";

        out << "Spatial Hash\n";
        out << "Cell size: " << s.spatialHashCellSize << "\n";
        out << "Cells: " << s.spatialHashOccupiedCells << "\n";
        out << "Cell insertions: " << s.spatialHashCellInsertions << "\n";
        out << "Max occupancy: " << s.spatialHashMaxCellOccupancy << "\n";
        out << "Avg occupancy: " << s.spatialHashAvgCellOccupancy << "\n";
        out << "Pair attempts: " << s.spatialHashPairAttempts << "\n";
        out << "Duplicate pairs: " << s.spatialHashDuplicatePairs << "\n";
        out << "Global bodies: " << s.spatialHashGlobalBodies << "\n";
        out << "Global attempts: " << s.spatialHashGlobalPairAttempts << "\n";
        out << "Dedup: " << s.spatialHashDedupMs << " ms\n\n";

        out << "Primal Detail\n";
        out << "Joint visits: " << s.primalJointVisits << "\n";
        out << "Joint: " << s.primalJointMs << " ms\n";
        out << "Spring visits: " << s.primalSpringVisits << "\n";
        out << "Spring: " << s.primalSpringMs << " ms\n";
        out << "Manifold visits: " << s.primalManifoldVisits << "\n";
        out << "Manifold: " << s.primalManifoldMs << " ms\n";
        out << "Ignore visits: " << s.primalIgnoreCollisionVisits << "\n";
        out << "Ignore: " << s.primalIgnoreCollisionMs << " ms\n";
        out << "Body solves: " << s.bodySolveCount << "\n";
        out << "Body solve: " << s.bodySolveMs << " ms\n";
        out << "Avg attached forces: " << s.avgAttachedForces << "\n";
        out << "Max attached forces: " << s.maxAttachedForces << "\n\n";

        out << "Dual Detail\n";
        out << "Joint visits: " << s.dualJointVisits << "\n";
        out << "Joint: " << s.dualJointMs << " ms\n";
        out << "Spring visits: " << s.dualSpringVisits << "\n";
        out << "Spring: " << s.dualSpringMs << " ms\n";
        out << "Manifold visits: " << s.dualManifoldVisits << "\n";
        out << "Manifold: " << s.dualManifoldMs << " ms\n";
        out << "Ignore visits: " << s.dualIgnoreCollisionVisits << "\n";
        out << "Ignore: " << s.dualIgnoreCollisionMs << " ms\n\n";
    }

    out << "Native Render\n";
    out << "Native render total: " << lastNativeRenderMs << " ms\n";
    out << "Static bodies: " << lastNativeStaticBodiesMs << " ms\n";
    out << "Static bodies drawn: " << lastNativeStaticBodiesDrawn << "\n";
    out << "Projected shadows: " << lastNativeProjectedShadowsMs << " ms\n";
    out << "Dynamic bodies: " << lastNativeDynamicBodiesMs << " ms\n";
    out << "Dynamic bodies drawn: " << lastNativeDynamicBodiesDrawn << "\n";
    out << "Debug forces: " << lastNativeDebugForcesMs << " ms\n";
    out << "Debug forces drawn: " << lastNativeDebugForcesDrawn << "\n";
    out << "ImGui render: " << lastImGuiRenderMs << " ms\n";
    out << "Swap/present: " << lastSwapPresentMs << " ms\n\n";

    out << "Timing\n";
    out << "Physics total: " << lastPhysicsMs << " ms\n";
    out << "Force init: " << s.forceInitMs << " ms\n";
    out << "Body init: " << s.bodyInitMs << " ms\n";
    out << "Primal solve: " << s.primalSolveMs << " ms\n";
    out << "Dual update: " << s.dualUpdateMs << " ms\n";
    out << "Velocity: " << s.velocityUpdateMs << " ms\n";
    out << std::setprecision(1);
    out << "FPS: " << ImGui::GetIO().Framerate << "\n";
    return out.str();
}

bool findShadowPlane(float3 &planePoint, float3 &planeNormal)
{
    const float3 up = {0, 0, 1};
    float bestScore = 0.0f;
    bool found = false;

    for (Rigid *body = solver->bodies; body != 0; body = body->next)
    {
        if (body->mass > 0.0f || body->shape.type != RIGID_SHAPE_BOX)
            continue;

        float3 half = body->size * 0.5f;
        float3 axes[3] = {
            rotate(body->positionAng, float3{1, 0, 0}),
            rotate(body->positionAng, float3{0, 1, 0}),
            rotate(body->positionAng, float3{0, 0, 1})};

        for (int axis = 0; axis < 3; ++axis)
        {
            int i1 = (axis + 1) % 3;
            int i2 = (axis + 2) % 3;
            float area = 4.0f * half[i1] * half[i2];
            if (area <= 0.0f)
                continue;

            for (int s = 0; s < 2; ++s)
            {
                float sign = s == 0 ? -1.0f : 1.0f;
                float3 n = axes[axis] * sign;
                float upness = dot(n, up);
                if (upness <= 0.15f)
                    continue;

                float score = area * upness;
                if (!found || score > bestScore)
                {
                    found = true;
                    bestScore = score;
                    planeNormal = n;
                    planePoint = body->positionLin + n * half[axis];
                }
            }
        }
    }

    return found;
}

float3 bodyVertexWorld(const Rigid *body, const GLfloat v[3])
{
    float3 local = {v[0] * body->size.x, v[1] * body->size.y, v[2] * body->size.z};
    return transform(body->positionLin, body->positionAng, local);
}

float3 sphereVertexWorld(const Rigid *body, float theta, float phi)
{
    float cp = cosf(phi);
    float3 local = {
        body->shape.radius * cp * cosf(theta),
        body->shape.radius * cp * sinf(theta),
        body->shape.radius * sinf(phi)};
    return transform(body->positionLin, body->positionAng, local);
}

float3 capsuleVertexWorld(const Rigid *body, float theta, float z, float radius)
{
    float3 local = {radius * cosf(theta), radius * sinf(theta), z};
    return transform(body->positionLin, body->positionAng, local);
}

float3 capsuleCapVertexWorld(const Rigid *body, float theta, float phi, float capSign)
{
    float cp = cosf(phi);
    float3 local = {
        body->shape.radius * cp * cosf(theta),
        body->shape.radius * cp * sinf(theta),
        capSign * body->shape.halfLength + body->shape.radius * sinf(phi)};
    return transform(body->positionLin, body->positionAng, local);
}

float3 cylinderVertexWorld(const Rigid *body, float theta, float z)
{
    float3 local = {
        body->shape.radius * cosf(theta),
        body->shape.radius * sinf(theta),
        z};
    return transform(body->positionLin, body->positionAng, local);
}

float3 applyProjectiveMatrix(const GLfloat m[16], const float3 &p)
{
    float x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
    float y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
    float z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
    float w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
    if (fabsf(w) > 1.0e-6f)
    {
        x /= w;
        y /= w;
        z /= w;
    }
    return {x, y, z};
}

void drawBoxSolid(const Rigid *body)
{
    static const GLfloat V[8][3] = {
        {-0.5f, -0.5f, -0.5f}, {+0.5f, -0.5f, -0.5f}, {+0.5f, +0.5f, -0.5f}, {-0.5f, +0.5f, -0.5f}, {-0.5f, -0.5f, +0.5f}, {+0.5f, -0.5f, +0.5f}, {+0.5f, +0.5f, +0.5f}, {-0.5f, +0.5f, +0.5f}};

    static const unsigned T[12][3] = {
        {0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6}, {1, 5, 6}, {1, 6, 2}, {4, 0, 3}, {4, 3, 7}, {3, 2, 6}, {3, 6, 7}, {4, 5, 1}, {4, 1, 0}};

    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 12; ++i)
    {
        float3 a = bodyVertexWorld(body, V[T[i][0]]);
        float3 b = bodyVertexWorld(body, V[T[i][1]]);
        float3 c = bodyVertexWorld(body, V[T[i][2]]);
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(b.x, b.y, b.z);
        glVertex3f(c.x, c.y, c.z);
    }
    glEnd();
}

void drawSphereSolid(const Rigid *body)
{
    const int slices = 20;
    const int stacks = 12;

    glBegin(GL_TRIANGLES);
    for (int stack = 0; stack < stacks; ++stack)
    {
        float phi0 = -0.5f * kPi + kPi * (float)stack / (float)stacks;
        float phi1 = -0.5f * kPi + kPi * (float)(stack + 1) / (float)stacks;

        for (int slice = 0; slice < slices; ++slice)
        {
            float theta0 = 2.0f * kPi * (float)slice / (float)slices;
            float theta1 = 2.0f * kPi * (float)(slice + 1) / (float)slices;

            float3 a = sphereVertexWorld(body, theta0, phi0);
            float3 b = sphereVertexWorld(body, theta1, phi0);
            float3 c = sphereVertexWorld(body, theta1, phi1);
            float3 d = sphereVertexWorld(body, theta0, phi1);

            glVertex3f(a.x, a.y, a.z);
            glVertex3f(b.x, b.y, b.z);
            glVertex3f(c.x, c.y, c.z);

            glVertex3f(a.x, a.y, a.z);
            glVertex3f(c.x, c.y, c.z);
            glVertex3f(d.x, d.y, d.z);
        }
    }
    glEnd();
}

void drawCapsuleSolid(const Rigid *body)
{
    const int slices = 20;
    const int stacks = 6;
    float h = body->shape.halfLength;
    float r = body->shape.radius;

    glBegin(GL_TRIANGLES);
    for (int slice = 0; slice < slices; ++slice)
    {
        float theta0 = 2.0f * kPi * (float)slice / (float)slices;
        float theta1 = 2.0f * kPi * (float)(slice + 1) / (float)slices;
        float3 a = capsuleVertexWorld(body, theta0, -h, r);
        float3 b = capsuleVertexWorld(body, theta1, -h, r);
        float3 c = capsuleVertexWorld(body, theta1, h, r);
        float3 d = capsuleVertexWorld(body, theta0, h, r);
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(b.x, b.y, b.z);
        glVertex3f(c.x, c.y, c.z);
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(c.x, c.y, c.z);
        glVertex3f(d.x, d.y, d.z);
    }

    for (int cap = 0; cap < 2; ++cap)
    {
        float sign = cap == 0 ? -1.0f : 1.0f;
        float phiStart = cap == 0 ? -0.5f * kPi : 0.0f;
        float phiEnd = cap == 0 ? 0.0f : 0.5f * kPi;
        for (int stack = 0; stack < stacks; ++stack)
        {
            float phi0 = phiStart + (phiEnd - phiStart) * (float)stack / (float)stacks;
            float phi1 = phiStart + (phiEnd - phiStart) * (float)(stack + 1) / (float)stacks;
            for (int slice = 0; slice < slices; ++slice)
            {
                float theta0 = 2.0f * kPi * (float)slice / (float)slices;
                float theta1 = 2.0f * kPi * (float)(slice + 1) / (float)slices;
                float3 a = capsuleCapVertexWorld(body, theta0, phi0, sign);
                float3 b = capsuleCapVertexWorld(body, theta1, phi0, sign);
                float3 c = capsuleCapVertexWorld(body, theta1, phi1, sign);
                float3 d = capsuleCapVertexWorld(body, theta0, phi1, sign);
                glVertex3f(a.x, a.y, a.z);
                glVertex3f(b.x, b.y, b.z);
                glVertex3f(c.x, c.y, c.z);
                glVertex3f(a.x, a.y, a.z);
                glVertex3f(c.x, c.y, c.z);
                glVertex3f(d.x, d.y, d.z);
            }
        }
    }
    glEnd();
}

void drawCylinderSolid(const Rigid *body)
{
    const int slices = 24;
    float h = body->shape.halfLength;

    glBegin(GL_TRIANGLES);
    for (int slice = 0; slice < slices; ++slice)
    {
        float theta0 = 2.0f * kPi * (float)slice / (float)slices;
        float theta1 = 2.0f * kPi * (float)(slice + 1) / (float)slices;
        float3 a = cylinderVertexWorld(body, theta0, -h);
        float3 b = cylinderVertexWorld(body, theta1, -h);
        float3 c = cylinderVertexWorld(body, theta1, h);
        float3 d = cylinderVertexWorld(body, theta0, h);
        float3 bottom = transform(body->positionLin, body->positionAng, {0.0f, 0.0f, -h});
        float3 top = transform(body->positionLin, body->positionAng, {0.0f, 0.0f, h});

        glVertex3f(a.x, a.y, a.z);
        glVertex3f(b.x, b.y, b.z);
        glVertex3f(c.x, c.y, c.z);
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(c.x, c.y, c.z);
        glVertex3f(d.x, d.y, d.z);

        glVertex3f(bottom.x, bottom.y, bottom.z);
        glVertex3f(b.x, b.y, b.z);
        glVertex3f(a.x, a.y, a.z);

        glVertex3f(top.x, top.y, top.z);
        glVertex3f(d.x, d.y, d.z);
        glVertex3f(c.x, c.y, c.z);
    }
    glEnd();
}

void drawBodySolid(const Rigid *body)
{
    if (body->shape.type == RIGID_SHAPE_SPHERE)
        drawSphereSolid(body);
    else if (body->shape.type == RIGID_SHAPE_CAPSULE)
        drawCapsuleSolid(body);
    else if (body->shape.type == RIGID_SHAPE_CYLINDER)
        drawCylinderSolid(body);
    else
        drawBoxSolid(body);
}

void drawBoxSolidProjected(const Rigid *body, const GLfloat shadowMat[16])
{
    static const GLfloat V[8][3] = {
        {-0.5f, -0.5f, -0.5f}, {+0.5f, -0.5f, -0.5f}, {+0.5f, +0.5f, -0.5f}, {-0.5f, +0.5f, -0.5f}, {-0.5f, -0.5f, +0.5f}, {+0.5f, -0.5f, +0.5f}, {+0.5f, +0.5f, +0.5f}, {-0.5f, +0.5f, +0.5f}};

    static const unsigned T[12][3] = {
        {0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6}, {1, 5, 6}, {1, 6, 2}, {4, 0, 3}, {4, 3, 7}, {3, 2, 6}, {3, 6, 7}, {4, 5, 1}, {4, 1, 0}};

    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 12; ++i)
    {
        float3 a = applyProjectiveMatrix(shadowMat, bodyVertexWorld(body, V[T[i][0]]));
        float3 b = applyProjectiveMatrix(shadowMat, bodyVertexWorld(body, V[T[i][1]]));
        float3 c = applyProjectiveMatrix(shadowMat, bodyVertexWorld(body, V[T[i][2]]));
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(b.x, b.y, b.z);
        glVertex3f(c.x, c.y, c.z);
    }
    glEnd();
}

void drawSphereSolidProjected(const Rigid *body, const GLfloat shadowMat[16])
{
    const int slices = 20;
    const int stacks = 12;

    glBegin(GL_TRIANGLES);
    for (int stack = 0; stack < stacks; ++stack)
    {
        float phi0 = -0.5f * kPi + kPi * (float)stack / (float)stacks;
        float phi1 = -0.5f * kPi + kPi * (float)(stack + 1) / (float)stacks;

        for (int slice = 0; slice < slices; ++slice)
        {
            float theta0 = 2.0f * kPi * (float)slice / (float)slices;
            float theta1 = 2.0f * kPi * (float)(slice + 1) / (float)slices;

            float3 a = applyProjectiveMatrix(shadowMat, sphereVertexWorld(body, theta0, phi0));
            float3 b = applyProjectiveMatrix(shadowMat, sphereVertexWorld(body, theta1, phi0));
            float3 c = applyProjectiveMatrix(shadowMat, sphereVertexWorld(body, theta1, phi1));
            float3 d = applyProjectiveMatrix(shadowMat, sphereVertexWorld(body, theta0, phi1));

            glVertex3f(a.x, a.y, a.z);
            glVertex3f(b.x, b.y, b.z);
            glVertex3f(c.x, c.y, c.z);

            glVertex3f(a.x, a.y, a.z);
            glVertex3f(c.x, c.y, c.z);
            glVertex3f(d.x, d.y, d.z);
        }
    }
    glEnd();
}

void drawCapsuleSolidProjected(const Rigid *body, const GLfloat shadowMat[16])
{
    const int slices = 20;
    float h = body->shape.halfLength;
    float r = body->shape.radius;

    glBegin(GL_TRIANGLES);
    for (int slice = 0; slice < slices; ++slice)
    {
        float theta0 = 2.0f * kPi * (float)slice / (float)slices;
        float theta1 = 2.0f * kPi * (float)(slice + 1) / (float)slices;
        float3 a = applyProjectiveMatrix(shadowMat, capsuleVertexWorld(body, theta0, -h, r));
        float3 b = applyProjectiveMatrix(shadowMat, capsuleVertexWorld(body, theta1, -h, r));
        float3 c = applyProjectiveMatrix(shadowMat, capsuleVertexWorld(body, theta1, h, r));
        float3 d = applyProjectiveMatrix(shadowMat, capsuleVertexWorld(body, theta0, h, r));
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(b.x, b.y, b.z);
        glVertex3f(c.x, c.y, c.z);
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(c.x, c.y, c.z);
        glVertex3f(d.x, d.y, d.z);
    }
    glEnd();
}

void drawCylinderSolidProjected(const Rigid *body, const GLfloat shadowMat[16])
{
    const int slices = 24;
    float h = body->shape.halfLength;

    glBegin(GL_TRIANGLES);
    for (int slice = 0; slice < slices; ++slice)
    {
        float theta0 = 2.0f * kPi * (float)slice / (float)slices;
        float theta1 = 2.0f * kPi * (float)(slice + 1) / (float)slices;
        float3 a = applyProjectiveMatrix(shadowMat, cylinderVertexWorld(body, theta0, -h));
        float3 b = applyProjectiveMatrix(shadowMat, cylinderVertexWorld(body, theta1, -h));
        float3 c = applyProjectiveMatrix(shadowMat, cylinderVertexWorld(body, theta1, h));
        float3 d = applyProjectiveMatrix(shadowMat, cylinderVertexWorld(body, theta0, h));
        float3 bottom = applyProjectiveMatrix(shadowMat, transform(body->positionLin, body->positionAng, {0.0f, 0.0f, -h}));
        float3 top = applyProjectiveMatrix(shadowMat, transform(body->positionLin, body->positionAng, {0.0f, 0.0f, h}));

        glVertex3f(a.x, a.y, a.z);
        glVertex3f(b.x, b.y, b.z);
        glVertex3f(c.x, c.y, c.z);
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(c.x, c.y, c.z);
        glVertex3f(d.x, d.y, d.z);

        glVertex3f(bottom.x, bottom.y, bottom.z);
        glVertex3f(b.x, b.y, b.z);
        glVertex3f(a.x, a.y, a.z);

        glVertex3f(top.x, top.y, top.z);
        glVertex3f(d.x, d.y, d.z);
        glVertex3f(c.x, c.y, c.z);
    }
    glEnd();
}

void drawBodySolidProjected(const Rigid *body, const GLfloat shadowMat[16])
{
    if (body->shape.type == RIGID_SHAPE_SPHERE)
        drawSphereSolidProjected(body, shadowMat);
    else if (body->shape.type == RIGID_SHAPE_CAPSULE)
        drawCapsuleSolidProjected(body, shadowMat);
    else if (body->shape.type == RIGID_SHAPE_CYLINDER)
        drawCylinderSolidProjected(body, shadowMat);
    else
        drawBoxSolidProjected(body, shadowMat);
}

void drawBody(const Rigid *body)
{
    static const GLfloat V[8][3] = {
        {-0.5f, -0.5f, -0.5f},
        {+0.5f, -0.5f, -0.5f},
        {+0.5f, +0.5f, -0.5f},
        {-0.5f, +0.5f, -0.5f},
        {-0.5f, -0.5f, +0.5f},
        {+0.5f, -0.5f, +0.5f},
        {+0.5f, +0.5f, +0.5f},
        {-0.5f, +0.5f, +0.5f}};

    static const unsigned T[12][3] = {
        {0, 1, 2}, {0, 2, 3}, // -Z
        {4, 6, 5},
        {4, 7, 6}, // +Z
        {1, 5, 6},
        {1, 6, 2}, // +X
        {4, 0, 3},
        {4, 3, 7}, // -X
        {3, 2, 6},
        {3, 6, 7}, // +Y
        {4, 5, 1},
        {4, 1, 0} // -Y
    };

    static const unsigned E[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

    glColor4f(0.80f, 0.84f, 0.90f, 1.0f);

    // Push filled faces slightly behind wireframe edges to avoid z-fighting.
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);

    if (body->shape.type == RIGID_SHAPE_SPHERE)
    {
        drawSphereSolid(body);
    }
    else if (body->shape.type == RIGID_SHAPE_CAPSULE)
    {
        drawCapsuleSolid(body);
    }
    else if (body->shape.type == RIGID_SHAPE_CYLINDER)
    {
        drawCylinderSolid(body);
    }
    else
    {
        glBegin(GL_TRIANGLES);
        for (int i = 0; i < 12; ++i)
        {
            float3 a = bodyVertexWorld(body, V[T[i][0]]);
            float3 b = bodyVertexWorld(body, V[T[i][1]]);
            float3 c = bodyVertexWorld(body, V[T[i][2]]);
            glVertex3f(a.x, a.y, a.z);
            glVertex3f(b.x, b.y, b.z);
            glVertex3f(c.x, c.y, c.z);
        }
        glEnd();
    }

    glDisable(GL_POLYGON_OFFSET_FILL);

    glColor4f(0.10f, 0.12f, 0.14f, 1.0f);
    if (body->shape.type == RIGID_SHAPE_SPHERE)
    {
        const int slices = 20;
        const int stacks = 12;
        glBegin(GL_LINES);
        for (int stack = 1; stack < stacks; ++stack)
        {
            float phi = -0.5f * kPi + kPi * (float)stack / (float)stacks;
            for (int slice = 0; slice < slices; ++slice)
            {
                float theta0 = 2.0f * kPi * (float)slice / (float)slices;
                float theta1 = 2.0f * kPi * (float)(slice + 1) / (float)slices;
                float3 a = sphereVertexWorld(body, theta0, phi);
                float3 b = sphereVertexWorld(body, theta1, phi);
                glVertex3f(a.x, a.y, a.z);
                glVertex3f(b.x, b.y, b.z);
            }
        }
        for (int slice = 0; slice < slices; ++slice)
        {
            float theta = 2.0f * kPi * (float)slice / (float)slices;
            for (int stack = 0; stack < stacks; ++stack)
            {
                float phi0 = -0.5f * kPi + kPi * (float)stack / (float)stacks;
                float phi1 = -0.5f * kPi + kPi * (float)(stack + 1) / (float)stacks;
                float3 a = sphereVertexWorld(body, theta, phi0);
                float3 b = sphereVertexWorld(body, theta, phi1);
                glVertex3f(a.x, a.y, a.z);
                glVertex3f(b.x, b.y, b.z);
            }
        }
        glEnd();
    }
    else if (body->shape.type == RIGID_SHAPE_CAPSULE)
    {
        const int slices = 20;
        const int capStacks = 6;
        float h = body->shape.halfLength;
        float r = body->shape.radius;
        glBegin(GL_LINES);
        for (int slice = 0; slice < slices; ++slice)
        {
            float theta0 = 2.0f * kPi * (float)slice / (float)slices;
            float theta1 = 2.0f * kPi * (float)(slice + 1) / (float)slices;
            float3 a = capsuleVertexWorld(body, theta0, -h, r);
            float3 b = capsuleVertexWorld(body, theta1, -h, r);
            float3 c = capsuleVertexWorld(body, theta0, h, r);
            float3 d = capsuleVertexWorld(body, theta1, h, r);
            glVertex3f(a.x, a.y, a.z);
            glVertex3f(b.x, b.y, b.z);
            glVertex3f(c.x, c.y, c.z);
            glVertex3f(d.x, d.y, d.z);
            glVertex3f(a.x, a.y, a.z);
            glVertex3f(c.x, c.y, c.z);
        }
        for (int cap = 0; cap < 2; ++cap)
        {
            float sign = cap == 0 ? -1.0f : 1.0f;
            float phiStart = cap == 0 ? -0.5f * kPi : 0.0f;
            float phiEnd = cap == 0 ? 0.0f : 0.5f * kPi;
            for (int stack = 1; stack < capStacks; ++stack)
            {
                float phi = phiStart + (phiEnd - phiStart) * (float)stack / (float)capStacks;
                for (int slice = 0; slice < slices; ++slice)
                {
                    float theta0 = 2.0f * kPi * (float)slice / (float)slices;
                    float theta1 = 2.0f * kPi * (float)(slice + 1) / (float)slices;
                    float3 a = capsuleCapVertexWorld(body, theta0, phi, sign);
                    float3 b = capsuleCapVertexWorld(body, theta1, phi, sign);
                    glVertex3f(a.x, a.y, a.z);
                    glVertex3f(b.x, b.y, b.z);
                }
            }
            for (int slice = 0; slice < slices; slice += 2)
            {
                float theta = 2.0f * kPi * (float)slice / (float)slices;
                for (int stack = 0; stack < capStacks; ++stack)
                {
                    float phi0 = phiStart + (phiEnd - phiStart) * (float)stack / (float)capStacks;
                    float phi1 = phiStart + (phiEnd - phiStart) * (float)(stack + 1) / (float)capStacks;
                    float3 a = capsuleCapVertexWorld(body, theta, phi0, sign);
                    float3 b = capsuleCapVertexWorld(body, theta, phi1, sign);
                    glVertex3f(a.x, a.y, a.z);
                    glVertex3f(b.x, b.y, b.z);
                }
            }
        }
        glEnd();
    }
    else if (body->shape.type == RIGID_SHAPE_CYLINDER)
    {
        const int slices = 24;
        float h = body->shape.halfLength;
        glBegin(GL_LINES);
        for (int slice = 0; slice < slices; ++slice)
        {
            float theta0 = 2.0f * kPi * (float)slice / (float)slices;
            float theta1 = 2.0f * kPi * (float)(slice + 1) / (float)slices;
            float3 a = cylinderVertexWorld(body, theta0, -h);
            float3 b = cylinderVertexWorld(body, theta1, -h);
            float3 c = cylinderVertexWorld(body, theta0, h);
            float3 d = cylinderVertexWorld(body, theta1, h);
            glVertex3f(a.x, a.y, a.z);
            glVertex3f(b.x, b.y, b.z);
            glVertex3f(c.x, c.y, c.z);
            glVertex3f(d.x, d.y, d.z);
            if (slice % 3 == 0)
            {
                glVertex3f(a.x, a.y, a.z);
                glVertex3f(c.x, c.y, c.z);
            }
        }
        glEnd();
    }
    else
    {
        glBegin(GL_LINES);
        for (int i = 0; i < 12; ++i)
        {
            float3 a = bodyVertexWorld(body, V[E[i][0]]);
            float3 b = bodyVertexWorld(body, V[E[i][1]]);
            glVertex3f(a.x, a.y, a.z);
            glVertex3f(b.x, b.y, b.z);
        }
        glEnd();
    }
}

void drawJoint(const Joint *joint)
{
    float3 v0 = joint->bodyA ? transform(joint->bodyA->positionLin, joint->bodyA->positionAng, joint->rA) : joint->rA;
    float3 v1 = transform(joint->bodyB->positionLin, joint->bodyB->positionAng, joint->rB);

    glColor3f(0.75f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(v0.x, v0.y, v0.z);
    glVertex3f(v1.x, v1.y, v1.z);
    glEnd();
}

void drawSpring(const Spring *spring)
{
    float3 v0 = transform(spring->bodyA->positionLin, spring->bodyA->positionAng, spring->rA);
    float3 v1 = transform(spring->bodyB->positionLin, spring->bodyB->positionAng, spring->rB);

    glColor3f(0.75f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(v0.x, v0.y, v0.z);
    glVertex3f(v1.x, v1.y, v1.z);
    glEnd();
}

void drawManifold(const Manifold *manifold)
{
    if (!SHOW_CONTACTS)
        return;

    glColor3f(0.75f, 0.0f, 0.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < manifold->numContacts; ++i)
    {
        float3 v0 = transform(manifold->bodyA->positionLin, manifold->bodyA->positionAng, manifold->contacts[i].rA);
        float3 v1 = transform(manifold->bodyB->positionLin, manifold->bodyB->positionAng, manifold->contacts[i].rB);
        glVertex3f(v0.x, v0.y, v0.z);
        glVertex3f(v1.x, v1.y, v1.z);
    }
    glEnd();
}

void drawSolver(const Solver *state)
{
    Uint64 totalBegin = SDL_GetPerformanceCounter();
    lastNativeStaticBodiesMs = 0.0f;
    lastNativeProjectedShadowsMs = 0.0f;
    lastNativeDynamicBodiesMs = 0.0f;
    lastNativeDebugForcesMs = 0.0f;
    lastNativeStaticBodiesDrawn = 0;
    lastNativeDynamicBodiesDrawn = 0;
    lastNativeDebugForcesDrawn = 0;

    if (nativeRenderBodies)
    {
        // Draw static receivers first so shadows depth-test against them.
        Uint64 begin = SDL_GetPerformanceCounter();
        for (const Rigid *body = state->bodies; body != 0; body = body->next)
        {
            if (body->mass <= 0.0f)
            {
                drawBody(body);
                ++lastNativeStaticBodiesDrawn;
            }
        }
        lastNativeStaticBodiesMs = elapsedMs(begin, SDL_GetPerformanceCounter());

        if (nativeRenderProjectedShadows)
        {
            begin = SDL_GetPerformanceCounter();
            drawProjectedShadows();
            lastNativeProjectedShadowsMs = elapsedMs(begin, SDL_GetPerformanceCounter());
        }

        // Draw dynamic bodies after shadows so they appear cleanly on top.
        begin = SDL_GetPerformanceCounter();
        for (const Rigid *body = state->bodies; body != 0; body = body->next)
        {
            if (body->mass > 0.0f)
            {
                drawBody(body);
                ++lastNativeDynamicBodiesDrawn;
            }
        }
        lastNativeDynamicBodiesMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    }

    if (nativeRenderDebugForces)
    {
        Uint64 begin = SDL_GetPerformanceCounter();
        for (const Force *force = state->forces; force != 0; force = force->next)
        {
            if (const Joint *joint = dynamic_cast<const Joint *>(force))
            {
                drawJoint(joint);
                ++lastNativeDebugForcesDrawn;
            }
            else if (const Spring *spring = dynamic_cast<const Spring *>(force))
            {
                drawSpring(spring);
                ++lastNativeDebugForcesDrawn;
            }
            else if (const Manifold *manifold = dynamic_cast<const Manifold *>(force))
            {
                drawManifold(manifold);
                ++lastNativeDebugForcesDrawn;
            }
        }
        lastNativeDebugForcesMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    }

    lastNativeRenderMs = elapsedMs(totalBegin, SDL_GetPerformanceCounter());
}

void drawProjectedShadows()
{
    GLint stencilBits = 0;
    glGetIntegerv(GL_STENCIL_BITS, &stencilBits);
    bool useStencil = stencilBits > 0;

    float3 planePoint;
    float3 planeNormal;
    if (!findShadowPlane(planePoint, planeNormal))
        return;

    GLfloat plane[4];
    makePlaneFromPointNormal(planePoint, planeNormal, plane);

    // Directional light (w=0). Normalized to keep stable projection scale.
    float3 l = normalize(float3{0.45f, 0.95f, 1.0f});
    GLfloat light[4] = {l.x, l.y, l.z, 0.0f};

    GLfloat shadowMat[16];
    makeShadowMatrix(shadowMat, light, plane);

    GLboolean lightingEnabled = glIsEnabled(GL_LIGHTING);
    GLboolean blendEnabled = glIsEnabled(GL_BLEND);
    GLboolean cullEnabled = glIsEnabled(GL_CULL_FACE);
    GLboolean polyOffsetEnabled = glIsEnabled(GL_POLYGON_OFFSET_FILL);
    GLboolean stencilEnabled = glIsEnabled(GL_STENCIL_TEST);
    GLboolean depthWriteEnabled = GL_TRUE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &depthWriteEnabled);
    GLboolean colorMask[4] = {GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE};
    glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);

    auto drawProjectedCasters = [&]()
    {
        for (Rigid *body = solver->bodies; body != 0; body = body->next)
        {
            if (body->mass <= 0.0f)
                continue;
            drawBodySolidProjected(body, shadowMat);
        }
    };

    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-1.0f, -1.0f);
    glDepthMask(GL_FALSE);

    glDisable(GL_BLEND);
    if (useStencil)
    {
        glEnable(GL_STENCIL_TEST);
        glClear(GL_STENCIL_BUFFER_BIT);

        // Pass 1: mark all shadowed pixels in stencil.
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glStencilMask(0xFF);
        glStencilFunc(GL_ALWAYS, 1, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        drawProjectedCasters();

        // Pass 2: draw shadow once per pixel (no overlap darkening).
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glColor3f(0.72f, 0.72f, 0.72f);
        glStencilMask(0xFF);
        glStencilFunc(GL_EQUAL, 1, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
        drawProjectedCasters();
    }
    else
    {
        // No stencil available: draw a flat shadow color directly.
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glColor3f(0.72f, 0.72f, 0.72f);
        drawProjectedCasters();
    }

    glDepthMask(depthWriteEnabled);
    glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
    if (useStencil)
        glStencilMask(0xFF);
    if (polyOffsetEnabled)
        glEnable(GL_POLYGON_OFFSET_FILL);
    else
        glDisable(GL_POLYGON_OFFSET_FILL);
    if (cullEnabled)
        glEnable(GL_CULL_FACE);
    else
        glDisable(GL_CULL_FACE);
    if (lightingEnabled)
        glEnable(GL_LIGHTING);
    else
        glDisable(GL_LIGHTING);
    if (blendEnabled)
        glEnable(GL_BLEND);
    else
        glDisable(GL_BLEND);
    if (stencilEnabled && useStencil)
        glEnable(GL_STENCIL_TEST);
    else
        glDisable(GL_STENCIL_TEST);
}

float3 orbitEye()
{
    float ce = cosf(camElevation), se = sinf(camElevation);
    float ca = cosf(camAzimuth), sa = sinf(camAzimuth);
    // Spherical with +Z up
    float3 off = {camDistance * ce * ca, camDistance * ce * sa, camDistance * se};
    return camTarget + off;
}

void shootBox()
{
    float3 forward = normalize(camTarget - camEye);
    float shapeRadius = 0.5f * length(boxSize);
    if (shootShape == SHOOT_SHAPE_SPHERE)
        shapeRadius = sphereRadius;
    else if (shootShape == SHOOT_SHAPE_CAPSULE)
        shapeRadius = capsuleRadius + capsuleHalfLength;
    else if (shootShape == SHOOT_SHAPE_CYLINDER)
        shapeRadius = sqrtf(cylinderRadius * cylinderRadius + cylinderHalfLength * cylinderHalfLength);
    float spawnOffset = 2.0f + shapeRadius;
    float3 spawnPos = camEye + forward * spawnOffset;
    float3 velocity = forward * boxVelocity;
    if (shootShape == SHOOT_SHAPE_SPHERE)
        Rigid::makeSphere(solver, sphereRadius, boxDensity, boxFriction, spawnPos, velocity);
    else if (shootShape == SHOOT_SHAPE_CAPSULE)
        Rigid::makeCapsule(solver, capsuleRadius, capsuleHalfLength, boxDensity, boxFriction, spawnPos, velocity);
    else if (shootShape == SHOOT_SHAPE_CYLINDER)
        Rigid::makeCylinder(solver, cylinderRadius, cylinderHalfLength, boxDensity, boxFriction, spawnPos, velocity);
    else
        new Rigid(solver, boxSize, boxDensity, boxFriction, spawnPos, velocity);
}

void releaseDrag()
{
    if (drag)
    {
        delete drag;
        drag = 0;
    }

    dragMode = DRAG_MODE_NONE;
    dragFingerId = 0;
}

bool screenToWorldRay(float2 screenPos, float3 &origin, float3 &dir)
{
    int w, h;
    SDL_GetWindowSize(Window, &w, &h);
    if (w <= 0 || h <= 0)
        return false;

    float aspect = (float)w / (float)h;
    float ndcX = screenPos.x / (float)w * 2.0f - 1.0f;
    float ndcY = 1.0f - screenPos.y / (float)h * 2.0f;

    float3 forward = normalize(camTarget - camEye);
    float3 upHint = {0, 0, 1};
    float3 right = cross(forward, upHint);
    if (lengthSq(right) < 1.0e-8f)
        right = cross(forward, float3{0, 1, 0});
    right = normalize(right);
    float3 up = cross(right, forward);

    float tanHalfFovY = tanf(0.5f * rad(kFovY_deg));
    float px = ndcX * aspect * tanHalfFovY;
    float py = ndcY * tanHalfFovY;

    origin = camEye;
    dir = normalize(forward + right * px + up * py);
    return true;
}

bool beginDragAtScreen(float2 screenPos, DragMode mode, SDL_FingerID fingerId = 0)
{
    float3 rayOrigin;
    float3 rayDir;
    if (!screenToWorldRay(screenPos, rayOrigin, rayDir))
        return false;

    float3 localHit;
    Rigid *body = solver->pick(rayOrigin, rayDir, localHit);
    if (!body)
        return false;

    float3 worldHit = transform(body->positionLin, body->positionAng, localHit);
    dragRayDistance = max(dot(worldHit - rayOrigin, rayDir), 0.1f);

    const float dragStiffness = 5000.0f;
    drag = new Joint(solver, 0, body, worldHit, localHit, dragStiffness, 0.0f);
    dragMode = mode;
    dragFingerId = fingerId;
    return true;
}

void updateDragAtScreen(float2 screenPos)
{
    if (!drag)
        return;

    float3 rayOrigin;
    float3 rayDir;
    if (!screenToWorldRay(screenPos, rayOrigin, rayDir))
        return;

    drag->rA = rayOrigin + rayDir * dragRayDistance;
}

void setPerspective(float fovY_deg, float aspect, float zNear, float zFar)
{
    float top = zNear * tanf(0.5f * rad(fovY_deg));
    float right = top * aspect;
    glFrustum(-right, right, -top, top, zNear, zFar); // fixed-function perspective
}

void setLookAt(const float3 &eye, const float3 &center, const float3 &upW)
{
    float3 f = normalize(center - eye);
    float3 s = normalize(cross(f, upW)); // right
    float3 u = cross(s, f);              // corrected up

    GLfloat m[16] = {
        s.x, u.x, -f.x, 0,
        s.y, u.y, -f.y, 0,
        s.z, u.z, -f.z, 0,
        0, 0, 0, 1};
    glMultMatrixf(m);
    glTranslatef(-eye.x, -eye.y, -eye.z);
}

void makePlaneFromPointNormal(const float3 &p, const float3 &n, GLfloat plane[4])
{
    // Plane in form Ax + By + Cz + D = 0
    float3 nn = normalize(n);
    plane[0] = nn.x;
    plane[1] = nn.y;
    plane[2] = nn.z;
    plane[3] = -(nn.x * p.x + nn.y * p.y + nn.z * p.z);
}

void makeShadowMatrix(GLfloat out[16], const GLfloat light[4], const GLfloat plane[4])
{
    // M = (plane dot light) * I - light * plane^T  (column-major for OpenGL)
    const float dot = plane[0] * light[0] + plane[1] * light[1] + plane[2] * light[2] + plane[3] * light[3];
#define M(r, c) out[(c) * 4 + (r)]
    M(0, 0) = dot - light[0] * plane[0];
    M(0, 1) = -light[0] * plane[1];
    M(0, 2) = -light[0] * plane[2];
    M(0, 3) = -light[0] * plane[3];
    M(1, 0) = -light[1] * plane[0];
    M(1, 1) = dot - light[1] * plane[1];
    M(1, 2) = -light[1] * plane[2];
    M(1, 3) = -light[1] * plane[3];
    M(2, 0) = -light[2] * plane[0];
    M(2, 1) = -light[2] * plane[1];
    M(2, 2) = dot - light[2] * plane[2];
    M(2, 3) = -light[2] * plane[3];
    M(3, 0) = -light[3] * plane[0];
    M(3, 1) = -light[3] * plane[1];
    M(3, 2) = -light[3] * plane[2];
    M(3, 3) = dot - light[3] * plane[3];
#undef M
}

void ui()
{
    // Draw the ImGui UI
    ImGui::Begin("Controls");
    if (touchOnly)
    {
        ImGui::Text("Orbit Cam: Two-Finger Drag");
        ImGui::Text("Zoom Cam: Pinch");
        ImGui::Text("Shoot Shape: Double Tap");
        ImGui::Text("Drag Shape: Tap and Hold");
    }
    else
    {
        ImGui::Text("Orbit Cam: Right Mouse (W/A/S/D)");
        ImGui::Text("Zoom Cam: Mouse Wheel (Q/E)");
        ImGui::Text("Shoot Shape: Middle Mouse (Space)");
        ImGui::Text("Drag Shape: Left Mouse");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    int scene = currScene;
    if (ImGui::BeginCombo("Scene", sceneNames[scene]))
    {
        for (int i = 0; i < sceneCount; i++)
        {
            bool selected = scene == i;
            if (ImGui::Selectable(sceneNames[i], selected) && i != currScene)
            {
                releaseDrag();
                simulationHost.loadScene(i);
            }
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    if (ImGui::Button(" Reset "))
    {
        releaseDrag();
        simulationHost.resetScene();
    }
    ImGui::SameLine();
    if (ImGui::Button("Default"))
        solver->defaultParams();

    ImGui::Checkbox("Pause", &paused);
    if (paused)
    {
        ImGui::SameLine();
        if (ImGui::Button("Step"))
            stepSolverTimed();
    }

    ImGui::Spacing();
    int shape = (int)shootShape;
    if (ImGui::BeginCombo("Shoot Shape", shootShapeNames[shape]))
    {
        for (int i = 0; i < SHOOT_SHAPE_COUNT; ++i)
        {
            bool selected = shape == i;
            if (ImGui::Selectable(shootShapeNames[i], selected))
                shootShape = (ShootShape)i;
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::SliderFloat("Shape Friction", &boxFriction, 0.0f, 2.0f);
    ImGui::SliderFloat3("Box Size", &boxSize.x, 0.1f, 5.0f);
    ImGui::SliderFloat("Sphere Radius", &sphereRadius, 0.1f, 5.0f);
    ImGui::SliderFloat("Capsule Radius", &capsuleRadius, 0.1f, 2.0f);
    ImGui::SliderFloat("Capsule Half Length", &capsuleHalfLength, 0.0f, 5.0f);
    ImGui::SliderFloat("Cylinder Radius", &cylinderRadius, 0.1f, 3.0f);
    ImGui::SliderFloat("Cylinder Half Length", &cylinderHalfLength, 0.05f, 5.0f);
    ImGui::SliderFloat("Shape Velocity", &boxVelocity, 0.0f, 20.0f);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SliderFloat("Gravity", &solver->gravity, -20.0f, 20.0f);
    ImGui::SliderFloat("Dt", &solver->dt, 0.001f, 0.1f);
    ImGui::SliderInt("Iterations", &solver->iterations, 1, 50);

    ImGui::End();
}

void performanceOverlay()
{
    int w, h;
    SDL_GetWindowSize(Window, &w, &h);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;

    ImGui::SetNextWindowPos(ImVec2((float)w - 10.0f, 10.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.78f);
    if (ImGui::Begin("Performance", 0, flags))
    {
        int broadphase = (int)solver->broadphaseMode;
        ImGui::SetNextItemWidth(170.0f);
        if (ImGui::BeginCombo("Broadphase", broadphaseNames[broadphase]))
        {
            for (int i = 0; i < BROADPHASE_COUNT; ++i)
            {
                bool selected = broadphase == i;
                if (ImGui::Selectable(broadphaseNames[i], selected))
                    solver->broadphaseMode = (BroadphaseMode)i;
                if (selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        int cellSizeIndex = currentSpatialHashCellSizePreset();
        ImGui::SetNextItemWidth(170.0f);
        if (ImGui::BeginCombo("Cell Size", spatialHashCellSizeNames[cellSizeIndex]))
        {
            for (int i = 0; i < spatialHashCellSizeCount; ++i)
            {
                bool selected = cellSizeIndex == i;
                if (ImGui::Selectable(spatialHashCellSizeNames[i], selected))
                    solver->spatialHashCellSize = spatialHashCellSizePresets[i];
                if (selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (ImGui::Button("Copy Metrics"))
        {
            std::string metrics = performanceMetricsText();
            SDL_SetClipboardText(metrics.c_str());
        }

        ImGui::Checkbox("Skip Ignore Solver Work", &solver->skipIgnoreCollisionSolverWork);
        ImGui::Checkbox("Deep Profiling", &solver->deepProfiling);

        ImGui::SeparatorText("Backend");
        int physicsBackendIndex = (int)physicsBackendMode;
        ImGui::SetNextItemWidth(220.0f);
        if (ImGui::BeginCombo("Physics Backend", physicsBackendNames[physicsBackendIndex]))
        {
            for (int i = 0; i < PHYSICS_BACKEND_COUNT; ++i)
            {
                bool selected = physicsBackendIndex == i;
                bool webgpuMode = i != PHYSICS_BACKEND_CPU_REFERENCE;
                if (webgpuMode && !webgpuContext.deviceReady)
                    ImGui::BeginDisabled();
                if (ImGui::Selectable(physicsBackendNames[i], selected))
                    setPhysicsBackendMode((PhysicsBackendMode)i);
                if (webgpuMode && !webgpuContext.deviceReady)
                    ImGui::EndDisabled();
                if (selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        int mainViewIndex = (int)mainViewMode;
        ImGui::SetNextItemWidth(220.0f);
        if (ImGui::BeginCombo("Main View", mainViewNames[mainViewIndex]))
        {
            for (int i = 0; i < MAIN_VIEW_COUNT; ++i)
            {
                bool selected = mainViewIndex == i;
                bool webgpuMode = i == MAIN_VIEW_WEBGPU_EXPERIMENTAL;
                if (webgpuMode && !webgpuContext.deviceReady)
                    ImGui::BeginDisabled();
                if (ImGui::Selectable(mainViewNames[i], selected))
                {
                    mainViewMode = (MainViewMode)i;
                    if (mainViewMode == MAIN_VIEW_OPENGL)
                        closeWebGpuMainViewWindow();
                    else
                        openWebGpuMainViewWindow();
                }
                if (webgpuMode && !webgpuContext.deviceReady)
                    ImGui::EndDisabled();
                if (selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        int renderBackendIndex = (int)renderBackendMode;
        ImGui::SetNextItemWidth(220.0f);
        if (ImGui::BeginCombo("Render Backend", renderBackendNames[renderBackendIndex]))
        {
            for (int i = 0; i < RENDER_BACKEND_COUNT; ++i)
            {
                bool selected = renderBackendIndex == i;
                if (ImGui::Selectable(renderBackendNames[i], selected))
                {
                    renderBackendMode = (RenderBackendMode)i;
                    if (renderBackendMode == RENDER_BACKEND_OPENGL)
                        closeWebGpuInstancedPreviewWindow();
                }
                if (selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        bool webgpuPreviewSelected = renderBackendMode == RENDER_BACKEND_WEBGPU_INSTANCED_PREVIEW;
        if (webgpuContext.deviceReady && webgpuPreviewSelected)
        {
            if (ImGui::Button(webgpuInstancedPreviewWindow ? "Focus Instanced Preview" : "Open Instanced Preview"))
                openWebGpuInstancedPreviewWindow();
            if (webgpuInstancedPreviewWindow)
            {
                ImGui::SameLine();
                if (ImGui::Button("Close Instanced Preview"))
                    closeWebGpuInstancedPreviewWindow();
            }
        }
        else
        {
            ImGui::BeginDisabled();
            ImGui::Button(webgpuContext.deviceReady ? "Open Instanced Preview" : "WebGPU Unavailable");
            ImGui::EndDisabled();
        }
        ImGui::Text("Physics: %s", solver->physicsBackend->name());
        ImGui::Text("Main View: %s", mainViewNames[(int)mainViewMode]);
        ImGui::Text("Render: %s", renderBackendNames[(int)renderBackendMode]);
        ImGui::Text("Viewer Bridge: %s", viewerBridge.statusText());
        ImGui::Text("Viewer Clients: %d", viewerBridge.clientCountValue());
        ImGui::Text("WebGPU Main View Window: %s", webgpuMainViewWindow ? "Open" : "Closed");
        ImGui::Text("Instanced Preview Window: %s", webgpuInstancedPreviewWindow ? "Open" : "Closed");
        ImGui::Text("WebGPU: %s", webgpuContext.statusText());
        ImGui::Text("WebGPU Smoke: %s", webgpuContext.smokeStatusText());
        ImGui::Text("WebGPU Compute: %s", webgpuContext.computeStatusText());
        ImGui::Text("WebGPU Runtime: %s", webgpuContext.runtimeStatusText());
        ImGui::Text("  Runtime %.2f ms, CPU fallback %.2f ms",
                    webgpuContext.runtimeTotalMillis(),
                    webgpuContext.runtimeCpuFallbackMillis());
        ImGui::Text("  Prediction %.2f ms, velocity %.2f ms",
                    webgpuContext.runtimePredictionMillis(),
                    webgpuContext.runtimeVelocityMillis());
        ImGui::Text("  Error lin %.6f ang %.6f frames %d fallbacks %d",
                    webgpuContext.runtimeMaxLinearErrorValue(),
                    webgpuContext.runtimeMaxAngularErrorValue(),
                    webgpuContext.runtimeFrameCount(),
                    webgpuContext.runtimeFallbackCount());
        ImGui::Text("  SAP emitted pairs: %d", webgpuContext.sapSphereHitCount());
        ImGui::Text("  SAP counter sync: %.2f ms, %.1f KB",
                    webgpuContext.sapCounterReadbackMillis(),
                    (double)webgpuContext.sapCounterReadbackByteCount() / 1024.0);
        ImGui::Text("  SAP pair readback: %.2f ms, %.1f KB",
                    webgpuContext.sapPairReadbackMillis(),
                    (double)webgpuContext.sapPairReadbackByteCount() / 1024.0);
        ImGui::Checkbox("WebGPU Ground Grid", &webgpuRenderOptions.showGroundGrid);
        ImGui::Checkbox("WebGPU Shape Edges", &webgpuRenderOptions.showShapeEdges);
        ImGui::SeparatorText("Native Render");
        ImGui::Checkbox("Native Bodies", &nativeRenderBodies);
        ImGui::Checkbox("Native Shadows", &nativeRenderProjectedShadows);
        ImGui::Checkbox("Native Debug Forces", &nativeRenderDebugForces);
        ImGui::Text("Native render total: %.2f ms", lastNativeRenderMs);
        ImGui::Text("  Static bodies: %.2f ms (%d)", lastNativeStaticBodiesMs, lastNativeStaticBodiesDrawn);
        ImGui::Text("  Shadows: %.2f ms", lastNativeProjectedShadowsMs);
        ImGui::Text("  Dynamic bodies: %.2f ms (%d)", lastNativeDynamicBodiesMs, lastNativeDynamicBodiesDrawn);
        ImGui::Text("  Debug forces: %.2f ms (%d)", lastNativeDebugForcesMs, lastNativeDebugForcesDrawn);
        ImGui::Text("  ImGui render: %.2f ms", lastImGuiRenderMs);
        ImGui::Text("  Swap/present: %.2f ms", lastSwapPresentMs);
        ImGui::Checkbox("GPU Diagnostics", &webgpuDiagnosticsEnabled);
        if (webgpuDiagnosticsEnabled)
        {
            if (ImGui::Button("Reset GPU Timing Stats"))
                webgpuContext.resetDiagnosticTimingStats();
            ImGui::Checkbox("Body Prediction Diagnostic", &webgpuBodyPredictionDiagnostic);
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderInt("Prediction Cadence", &webgpuBodyPredictionCadence, 1, 300);
            ImGui::Text("WebGPU Body Prediction: %s", webgpuContext.predictionStatusText());
            ImGui::Text("  Prediction: %.2f ms, lin %.6f, ang %.6f",
                        webgpuContext.predictionMillis(),
                        webgpuContext.predictionMaxErrorValue(),
                        webgpuContext.predictionMaxAngularErrorValue());
            ImGui::Text("  Recent %.2f avg %.2f min %.2f max %.2f n %d",
                        webgpuContext.predictionTiming.recentAvgMs,
                        webgpuContext.predictionTiming.avgMs,
                        webgpuContext.predictionTiming.minMs,
                        webgpuContext.predictionTiming.maxMs,
                        webgpuContext.predictionTiming.samples);
            ImGui::Checkbox("Velocity Update Diagnostic", &webgpuVelocityUpdateDiagnostic);
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderInt("Velocity Cadence", &webgpuVelocityUpdateCadence, 1, 300);
            ImGui::Text("WebGPU Velocity Update: %s", webgpuContext.velocityStatusText());
            ImGui::Text("  Velocity: %.2f ms, lin %.6f, ang %.6f",
                        webgpuContext.velocityMillis(),
                        webgpuContext.velocityMaxLinearErrorValue(),
                        webgpuContext.velocityMaxAngularErrorValue());
            ImGui::Text("  Recent %.2f avg %.2f min %.2f max %.2f n %d",
                        webgpuContext.velocityTiming.recentAvgMs,
                        webgpuContext.velocityTiming.avgMs,
                        webgpuContext.velocityTiming.minMs,
                        webgpuContext.velocityTiming.maxMs,
                        webgpuContext.velocityTiming.samples);
            ImGui::Checkbox("Bounds Diagnostic", &webgpuBoundsDiagnostic);
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderInt("Bounds Cadence", &webgpuBoundsCadence, 1, 300);
            ImGui::Text("WebGPU Bounds: %s", webgpuContext.boundsStatusText());
            ImGui::Text("  Bounds: %.2f ms, max %.6f",
                        webgpuContext.boundsMillis(),
                        webgpuContext.boundsMaxErrorValue());
            ImGui::Text("  Recent %.2f avg %.2f min %.2f max %.2f n %d",
                        webgpuContext.boundsTiming.recentAvgMs,
                        webgpuContext.boundsTiming.avgMs,
                        webgpuContext.boundsTiming.minMs,
                        webgpuContext.boundsTiming.maxMs,
                        webgpuContext.boundsTiming.samples);
            ImGui::Checkbox("Morton Diagnostic", &webgpuMortonDiagnostic);
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderInt("Morton Cadence", &webgpuMortonCadence, 1, 300);
            ImGui::Text("WebGPU Morton: %s", webgpuContext.mortonStatusText());
            ImGui::Text("  Morton: %.2f ms, mismatches %d",
                        webgpuContext.mortonMillis(),
                        webgpuContext.mortonMismatchCount());
            ImGui::Text("  Recent %.2f avg %.2f min %.2f max %.2f n %d",
                        webgpuContext.mortonTiming.recentAvgMs,
                        webgpuContext.mortonTiming.avgMs,
                        webgpuContext.mortonTiming.minMs,
                        webgpuContext.mortonTiming.maxMs,
                        webgpuContext.mortonTiming.samples);
            ImGui::Checkbox("Morton Sort Diagnostic", &webgpuMortonSortDiagnostic);
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderInt("Morton Sort Cadence", &webgpuMortonSortCadence, 1, 600);
            ImGui::Text("WebGPU Morton Sort: %s", webgpuContext.mortonSortStatusText());
            ImGui::Text("  Morton sort: %.2f ms, mismatches %d",
                        webgpuContext.mortonSortMillis(),
                        webgpuContext.mortonSortMismatchCount());
            ImGui::Text("  Recent %.2f avg %.2f min %.2f max %.2f n %d",
                        webgpuContext.mortonSortTiming.recentAvgMs,
                        webgpuContext.mortonSortTiming.avgMs,
                        webgpuContext.mortonSortTiming.minMs,
                        webgpuContext.mortonSortTiming.maxMs,
                        webgpuContext.mortonSortTiming.samples);
            ImGui::Checkbox("Pair Diagnostic", &webgpuPairDiagnostic);
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderInt("Pair Cadence", &webgpuPairCadence, 1, 600);
            ImGui::Text("WebGPU Pair: %s", webgpuContext.pairStatusText());
            ImGui::Text("  Pair: %.2f ms, candidates %d, hits %d, missed %d",
                        webgpuContext.pairMillis(),
                        webgpuContext.pairCandidateCount(),
                        webgpuContext.pairSphereHitCount(),
                        webgpuContext.pairMissedSphereHitCount());
            ImGui::Text("  Recent %.2f avg %.2f min %.2f max %.2f n %d",
                        webgpuContext.pairTiming.recentAvgMs,
                        webgpuContext.pairTiming.avgMs,
                        webgpuContext.pairTiming.minMs,
                        webgpuContext.pairTiming.maxMs,
                        webgpuContext.pairTiming.samples);
            ImGui::Checkbox("SAP Diagnostic", &webgpuSapDiagnostic);
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderInt("SAP Cadence", &webgpuSapCadence, 1, 600);
            ImGui::Text("WebGPU SAP: %s", webgpuContext.sapStatusText());
            ImGui::Text("  SAP: %.2f ms, candidates %d, hits %d, missed %d",
                        webgpuContext.sapMillis(),
                        webgpuContext.sapCandidateCount(),
                        webgpuContext.sapSphereHitCount(),
                        webgpuContext.sapMissedSphereHitCount());
            ImGui::Text("  Recent %.2f avg %.2f min %.2f max %.2f n %d",
                        webgpuContext.sapTiming.recentAvgMs,
                        webgpuContext.sapTiming.avgMs,
                        webgpuContext.sapTiming.minMs,
                        webgpuContext.sapTiming.maxMs,
                        webgpuContext.sapTiming.samples);
            ImGui::Text("  Best axis: %s", webgpuContext.sapBestAxis == 0 ? "X" : (webgpuContext.sapBestAxis == 1 ? "Y" : "Z"));
            ImGui::Text("  X %d cand %d hits %d missed",
                        webgpuContext.sapAxisCandidates[0],
                        webgpuContext.sapAxisSphereHits[0],
                        webgpuContext.sapAxisMissedSphereHits[0]);
            ImGui::Text("  Y %d cand %d hits %d missed",
                        webgpuContext.sapAxisCandidates[1],
                        webgpuContext.sapAxisSphereHits[1],
                        webgpuContext.sapAxisMissedSphereHits[1]);
            ImGui::Text("  Z %d cand %d hits %d missed",
                        webgpuContext.sapAxisCandidates[2],
                        webgpuContext.sapAxisSphereHits[2],
                        webgpuContext.sapAxisMissedSphereHits[2]);
        }
        ImGui::Text("WebGPU Main View Device: %s", webgpuMainViewContext.statusText());
        ImGui::Text("WebGPU Main View Present: %s", webgpuMainViewContext.presentStatusText());
        ImGui::Text("WebGPU Main View Total: %.2f ms", webgpuMainViewContext.previewTotalMillis());
        ImGui::Text("  Compute submit: %.2f ms", webgpuMainViewContext.previewComputeMillis());
        ImGui::Text("  Render submit: %.2f ms", webgpuMainViewContext.previewRenderMillis());
        ImGui::Text("  Batches: %d, instances: %d", webgpuMainViewContext.previewBatchCountValue(), webgpuMainViewContext.previewInstanceCountValue());
        ImGui::Text("  Box %d Sphere %d Capsule %d Cylinder %d Mesh %d",
                    webgpuMainViewContext.previewBoxInstanceCount(),
                    webgpuMainViewContext.previewSphereInstanceCount(),
                    webgpuMainViewContext.previewCapsuleInstanceCount(),
                    webgpuMainViewContext.previewCylinderInstanceCount(),
                    webgpuMainViewContext.previewMeshAssetInstanceCount());
        ImGui::Text("WebGPU Instanced Device: %s", webgpuClearContext.statusText());
        ImGui::Text("WebGPU Instanced Present: %s", webgpuClearContext.presentStatusText());
        ImGui::Text("WebGPU Instanced Total: %.2f ms", webgpuClearContext.previewTotalMillis());
        ImGui::Text("  Compute submit: %.2f ms", webgpuClearContext.previewComputeMillis());
        ImGui::Text("  Render submit: %.2f ms", webgpuClearContext.previewRenderMillis());
        ImGui::Text("  Batches: %d, instances: %d", webgpuClearContext.previewBatchCountValue(), webgpuClearContext.previewInstanceCountValue());
        ImGui::Text("  Box %d Sphere %d Capsule %d Cylinder %d Mesh %d",
                    webgpuClearContext.previewBoxInstanceCount(),
                    webgpuClearContext.previewSphereInstanceCount(),
                    webgpuClearContext.previewCapsuleInstanceCount(),
                    webgpuClearContext.previewCylinderInstanceCount(),
                    webgpuClearContext.previewMeshAssetInstanceCount());
        ImGui::Text("Dense bodies: %d", (int)solver->world.bodies.size());
        ImGui::Text("Dense constraints: %d", (int)solver->world.constraints.size());

        ImGui::SeparatorText("Broadphase");
        ImGui::Text("Bodies: %d", solver->stats.bodyCount);
        ImGui::Text("Pair checks: %d", solver->stats.pairChecks);
        ImGui::Text("Sphere hits: %d", solver->stats.sphereHits);
        ImGui::Text("Created manifolds: %d", solver->stats.manifoldsCreated);
        ImGui::Text("Broadphase: %.2f ms", solver->stats.broadphaseMs);
        ImGui::Text("  Hash build: %.2f ms", solver->stats.spatialHashBuildMs);
        ImGui::Text("  Candidates: %.2f ms", solver->stats.spatialHashCandidateMs);
        ImGui::Text("Constrained checks: %d", solver->stats.constrainedChecks);
        ImGui::Text("Constrained hits: %d", solver->stats.constrainedHits);

        if (solver->deepProfiling)
        {
            ImGui::Text("  Constraint: %.2f ms", solver->stats.constrainedMs);
            ImGui::Text("  Manifold alloc: %.2f ms", solver->stats.manifoldAllocMs);
            ImGui::Text("Force scan visits: %d", solver->stats.constrainedForceVisits);

            ImGui::SeparatorText("Spatial Hash");
            ImGui::Text("Cell size: %.2f", solver->spatialHashCellSize);
            ImGui::Text("Cells: %d", solver->stats.spatialHashOccupiedCells);
            ImGui::Text("Cell insertions: %d", solver->stats.spatialHashCellInsertions);
            ImGui::Text("Max occupancy: %d", solver->stats.spatialHashMaxCellOccupancy);
            ImGui::Text("Avg occupancy: %.2f", solver->stats.spatialHashAvgCellOccupancy);
            ImGui::Text("Pair attempts: %d", solver->stats.spatialHashPairAttempts);
            ImGui::Text("Duplicate pairs: %d", solver->stats.spatialHashDuplicatePairs);
            ImGui::Text("Global bodies: %d", solver->stats.spatialHashGlobalBodies);
            ImGui::Text("Global attempts: %d", solver->stats.spatialHashGlobalPairAttempts);
            ImGui::Text("Dedup: %.2f ms", solver->stats.spatialHashDedupMs);
        }

        ImGui::SeparatorText("Solver");
        ImGui::Text("Active bodies: %d", solver->stats.activeBodyCount);
        ImGui::Text("Forces: %d", solver->stats.forceCount);
        ImGui::Text("  Joints: %d", solver->stats.jointCount);
        ImGui::Text("  Springs: %d", solver->stats.springCount);
        ImGui::Text("  Manifolds: %d", solver->stats.manifoldCount);
        ImGui::Text("  Ignores: %d", solver->stats.ignoreCollisionCount);
        ImGui::Text("Primal visits: %d", solver->stats.primalForceVisits);
        ImGui::Text("Dual visits: %d", solver->stats.dualForceVisits);
        ImGui::Text("Primal ignore skipped: %d", solver->stats.primalIgnoreCollisionSkipped);
        ImGui::Text("Dual ignore skipped: %d", solver->stats.dualIgnoreCollisionSkipped);

        if (solver->deepProfiling)
        {
            ImGui::SeparatorText("Primal Detail");
            ImGui::Text("Joints: %d / %.2f ms", solver->stats.primalJointVisits, solver->stats.primalJointMs);
            ImGui::Text("Springs: %d / %.2f ms", solver->stats.primalSpringVisits, solver->stats.primalSpringMs);
            ImGui::Text("Manifolds: %d / %.2f ms", solver->stats.primalManifoldVisits, solver->stats.primalManifoldMs);
            ImGui::Text("Ignores: %d / %.2f ms", solver->stats.primalIgnoreCollisionVisits, solver->stats.primalIgnoreCollisionMs);
            ImGui::Text("Body solves: %d / %.2f ms", solver->stats.bodySolveCount, solver->stats.bodySolveMs);
            ImGui::Text("Attached avg/max: %.2f / %d", solver->stats.avgAttachedForces, solver->stats.maxAttachedForces);

            ImGui::SeparatorText("Dual Detail");
            ImGui::Text("Joints: %d / %.2f ms", solver->stats.dualJointVisits, solver->stats.dualJointMs);
            ImGui::Text("Springs: %d / %.2f ms", solver->stats.dualSpringVisits, solver->stats.dualSpringMs);
            ImGui::Text("Manifolds: %d / %.2f ms", solver->stats.dualManifoldVisits, solver->stats.dualManifoldMs);
            ImGui::Text("Ignores: %d / %.2f ms", solver->stats.dualIgnoreCollisionVisits, solver->stats.dualIgnoreCollisionMs);
        }

        ImGui::SeparatorText("Timing");
        ImGui::Text("Physics total: %.2f ms", lastPhysicsMs);
        ImGui::Text("Force init: %.2f ms", solver->stats.forceInitMs);
        ImGui::Text("Body init: %.2f ms", solver->stats.bodyInitMs);
        ImGui::Text("Primal solve: %.2f ms", solver->stats.primalSolveMs);
        ImGui::Text("Dual update: %.2f ms", solver->stats.dualUpdateMs);
        ImGui::Text("Velocity: %.2f ms", solver->stats.velocityUpdateMs);
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    }
    ImGui::End();
}

void input()
{
    auto &io = ImGui::GetIO();

    // --- Orbit controls ---
    const float orbitSpeedMouse = 0.005f;
    const float orbitSpeedKeys = rad(120.0f);
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right))
    {
        camAzimuth -= io.MouseDelta.x * orbitSpeedMouse;   // right drag -> yaw
        camElevation += io.MouseDelta.y * orbitSpeedMouse; // up drag   -> pitch
        camElevation = clamp(camElevation, rad(-89.0f), rad(89.0f));
    }

    if (!io.WantCaptureKeyboard)
    {
        float orbitDelta = orbitSpeedKeys * io.DeltaTime;
        if (ImGui::IsKeyDown(ImGuiKey_A))
            camAzimuth -= orbitDelta;
        if (ImGui::IsKeyDown(ImGuiKey_D))
            camAzimuth += orbitDelta;
        if (ImGui::IsKeyDown(ImGuiKey_W))
            camElevation += orbitDelta;
        if (ImGui::IsKeyDown(ImGuiKey_S))
            camElevation -= orbitDelta;
        camElevation = clamp(camElevation, rad(-89.0f), rad(89.0f));

        if (ImGui::IsKeyDown(ImGuiKey_E))
            camDistance *= 1.025f;
        if (ImGui::IsKeyDown(ImGuiKey_Q))
            camDistance /= 1.025f;
    }

    // Mouse wheel zoom (scroll up = closer)
    if (io.MouseWheel != 0.0f)
    {
        camDistance /= powf(1.1f, io.MouseWheel);
    }
    camDistance = clamp(camDistance, 0.2f, 1000.0f);

    bool shootFromMouse = ImGui::IsMouseClicked(ImGuiMouseButton_Middle) && !io.WantCaptureMouse;
    bool shootFromKeyboard = ImGui::IsKeyPressed(ImGuiKey_Space, false) && !io.WantCaptureKeyboard;
    bool shootFromTouch = touchOnly && shootRequested;
    if (shootFromMouse || shootFromKeyboard || shootFromTouch)
        shootBox();
    shootRequested = false;

    if (!touchOnly)
    {
        bool mouseDragDown = ImGui::IsMouseDown(ImGuiMouseButton_Left) && !io.WantCaptureMouse;
        float2 mousePos = {io.MousePos.x, io.MousePos.y};
        if (mouseDragDown)
        {
            if (dragMode == DRAG_MODE_MOUSE && drag)
                updateDragAtScreen(mousePos);
            else if (!drag)
                beginDragAtScreen(mousePos, DRAG_MODE_MOUSE);
        }
        else if (dragMode == DRAG_MODE_MOUSE)
        {
            releaseDrag();
        }
    }

    if (touchOnly)
    {
        const Uint32 kHoldMs = 180;
        const float kHoldMovePx = 20.0f;

        if (activeFingers.size() == 1 && !io.WantCaptureMouse)
        {
            auto it = activeFingers.begin();
            SDL_FingerID fingerId = it->first;
            float2 pos = it->second;

            if (dragMode == DRAG_MODE_TOUCH && drag && fingerId == dragFingerId)
            {
                updateDragAtScreen(pos);
            }
            else if (!drag && touchHoldCandidate && fingerId == touchHoldFingerId)
            {
                float2 move = pos - touchHoldStartPos;
                if (lengthSq(move) > kHoldMovePx * kHoldMovePx)
                {
                    touchHoldCandidate = false;
                }
                else if (SDL_GetTicks() - touchHoldStartTicks >= kHoldMs)
                {
                    if (beginDragAtScreen(pos, DRAG_MODE_TOUCH, fingerId))
                        touchHoldCandidate = false;
                }
            }
        }
        else
        {
            touchHoldCandidate = false;
            if (dragMode == DRAG_MODE_TOUCH)
                releaseDrag();
        }
    }
}

void mainLoop()
{
    // Event loop
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        ImGui_ImplSDL2_ProcessEvent(&event);

        if (event.type == SDL_KEYDOWN)
        {
            if (event.key.keysym.sym == SDLK_RETURN && (event.key.keysym.mod & KMOD_ALT))
            {
                FullScreen = !FullScreen;
                Uint32 fullscreenFlag = FullScreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0;
                SDL_SetWindowFullscreen(Window, fullscreenFlag);
            }

            if (event.key.keysym.sym == SDLK_ESCAPE)
            {
                if (SDL_GetTicks() < ignoreEscapeUntilTicks)
                    continue;

                if (webgpuMainViewWindow)
                {
                    closeWebGpuMainViewWindow();
                    ignoreEscapeUntilTicks = SDL_GetTicks() + 500;
                    continue;
                }

                if (webgpuInstancedPreviewWindow)
                {
                    closeWebGpuInstancedPreviewWindow();
                    ignoreEscapeUntilTicks = SDL_GetTicks() + 500;
                    continue;
                }
#ifndef __EMSCRIPTEN__
                Running = 0;
#endif
            }
        }
        else if (event.type == SDL_QUIT)
        {
#ifndef __EMSCRIPTEN__
            Running = 0;
#endif
        }
        else if (event.type == SDL_WINDOWEVENT)
        {
            if (webgpuMainViewWindow &&
                event.window.windowID == SDL_GetWindowID(webgpuMainViewWindow) &&
                event.window.event == SDL_WINDOWEVENT_CLOSE)
            {
                closeWebGpuMainViewWindow();
                continue;
            }

            if (webgpuInstancedPreviewWindow &&
                event.window.windowID == SDL_GetWindowID(webgpuInstancedPreviewWindow) &&
                event.window.event == SDL_WINDOWEVENT_CLOSE)
            {
                closeWebGpuInstancedPreviewWindow();
                continue;
            }
        }
        else if (event.type == SDL_FINGERDOWN)
        {
            int w, h;
            SDL_GetWindowSize(Window, &w, &h);
            SDL_FingerID id = event.tfinger.fingerId;
            float2 pos = {event.tfinger.x * w, event.tfinger.y * h};

            if (touchOnly && activeFingers.empty())
            {
                const Uint32 kDoubleTapMs = 300;
                const float kDoubleTapDistance = 40.0f;
                Uint32 now = SDL_GetTicks();
                float2 deltaTap = pos - lastTapPos;
                bool closeInTime = lastTapTicks > 0 && (now - lastTapTicks) <= kDoubleTapMs;
                bool closeInSpace = lengthSq(deltaTap) <= kDoubleTapDistance * kDoubleTapDistance;
                if (closeInTime && closeInSpace)
                {
                    shootRequested = true;
                    lastTapTicks = 0;
                }
                else
                {
                    lastTapTicks = now;
                    lastTapPos = pos;
                }
            }

            activeFingers[id] = pos;
            if (touchOnly)
            {
                if (activeFingers.size() == 1)
                {
                    touchHoldCandidate = true;
                    touchHoldFingerId = id;
                    touchHoldStartTicks = SDL_GetTicks();
                    touchHoldStartPos = pos;
                }
                else
                {
                    touchHoldCandidate = false;
                    if (dragMode == DRAG_MODE_TOUCH)
                        releaseDrag();
                }
            }
            if (activeFingers.size() != 2)
            {
                hasPrevGestureCenter = false;
            }
        }
        else if (event.type == SDL_FINGERMOTION)
        {
            int w, h;
            SDL_GetWindowSize(Window, &w, &h);
            SDL_FingerID id = event.tfinger.fingerId;
            auto it = activeFingers.find(id);
            if (it != activeFingers.end())
                it->second = {event.tfinger.x * w, event.tfinger.y * h};
        }
        else if (event.type == SDL_FINGERUP)
        {
            if (touchOnly)
            {
                if (touchHoldCandidate && event.tfinger.fingerId == touchHoldFingerId)
                    touchHoldCandidate = false;

                if (dragMode == DRAG_MODE_TOUCH && event.tfinger.fingerId == dragFingerId)
                    releaseDrag();
            }

            activeFingers.erase(event.tfinger.fingerId);
            hasPrevGestureCenter = false;
        }
        else if (event.type == SDL_MULTIGESTURE)
        {
            if (event.mgesture.numFingers == 2)
            {
                int w, h;
                SDL_GetWindowSize(Window, &w, &h);
                float2 center = {event.mgesture.x * w, event.mgesture.y * h};

                // Two-finger drag -> orbit
                if (hasPrevGestureCenter)
                {
                    float2 d = center - prevGestureCenter;
                    camAzimuth -= d.x * 0.005f;
                    camElevation += d.y * 0.005f;
                    camElevation = clamp(camElevation, rad(-89.0f), rad(89.0f));
                }
                prevGestureCenter = center;
                hasPrevGestureCenter = true;

                // Pinch -> zoom
                float dDist = event.mgesture.dDist; // positive = fingers move apart
                if (dDist != 0.0f)
                {
                    float zoomFactor = 1.0f + dDist * 2.0f; // gentle
                    if (zoomFactor > 0.01f)
                        camDistance = clamp(camDistance / zoomFactor, 0.2f, 1000.0f);
                }
            }
        }
    }

    // Setup GL
    int w, h;
    SDL_GetWindowSize(Window, &w, &h);

    glEnable(GL_LINE_SMOOTH);
    glLineWidth(2.0f);
    glPointSize(3.0f);
    glViewport(0, 0, w, h);
    glClearColor(1, 1, 1, 1);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glShadeModel(GL_FLAT);
    glEnable(GL_NORMALIZE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Camera matrices
    camEye = orbitEye();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    setPerspective(kFovY_deg, (float)w / (float)h, kNear, kFar);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    setLookAt(camEye, camTarget, float3{0, 0, 1});

    // ImGUI setup
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    input();
    ui();
    processViewerCommands();

    // Step solver and draw it
    if (!paused)
        stepSolverTimed();
    broadcastViewerSnapshot();
    static int viewerStatusFrame = 0;
    if ((viewerStatusFrame++ % 30) == 0)
        viewerBridge.broadcastStatus(simulationHost.metricsText().c_str());
    if (webgpuDiagnosticsEnabled && webgpuBodyPredictionDiagnostic)
    {
        int cadence = max(webgpuBodyPredictionCadence, 1);
        if ((webgpuBodyPredictionFrame++ % cadence) == 0)
            webgpuContext.runBodyPredictionDiagnostic(solver->world, solver->dt, solver->gravity);
    }
    if (webgpuDiagnosticsEnabled && webgpuVelocityUpdateDiagnostic)
    {
        int cadence = max(webgpuVelocityUpdateCadence, 1);
        if ((webgpuVelocityUpdateFrame++ % cadence) == 0)
            webgpuContext.runVelocityUpdateDiagnostic(solver->world, solver->dt);
    }
    if (webgpuDiagnosticsEnabled && webgpuBoundsDiagnostic)
    {
        int cadence = max(webgpuBoundsCadence, 1);
        if ((webgpuBoundsFrame++ % cadence) == 0)
            webgpuContext.runBoundsDiagnostic(solver->world);
    }
    if (webgpuDiagnosticsEnabled && webgpuMortonDiagnostic)
    {
        int cadence = max(webgpuMortonCadence, 1);
        if ((webgpuMortonFrame++ % cadence) == 0)
            webgpuContext.runMortonDiagnostic(solver->world);
    }
    if (webgpuDiagnosticsEnabled && webgpuMortonSortDiagnostic)
    {
        int cadence = max(webgpuMortonSortCadence, 1);
        if ((webgpuMortonSortFrame++ % cadence) == 0)
            webgpuContext.runMortonSortDiagnostic(solver->world);
    }
    if (webgpuDiagnosticsEnabled && webgpuPairDiagnostic)
    {
        int cadence = max(webgpuPairCadence, 1);
        if ((webgpuPairFrame++ % cadence) == 0)
            webgpuContext.runBroadphasePairDiagnostic(solver->world);
    }
    if (webgpuDiagnosticsEnabled && webgpuSapDiagnostic)
    {
        int cadence = max(webgpuSapCadence, 1);
        if ((webgpuSapFrame++ % cadence) == 0)
            webgpuContext.runSweepAndPruneDiagnostic(solver->world);
    }
    drawSolver(solver);
    performanceOverlay();

    // ImGUI rendering
    Uint64 imguiBegin = SDL_GetPerformanceCounter();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    lastImGuiRenderMs = elapsedMs(imguiBegin, SDL_GetPerformanceCounter());

    Uint64 presentBegin = SDL_GetPerformanceCounter();
    SDL_GL_SwapWindow(Window);
    lastSwapPresentMs = elapsedMs(presentBegin, SDL_GetPerformanceCounter());
    drawWebGpuMainViewWindow();
    drawWebGpuInstancedPreviewWindow();
}

int main(int argc, char *argv[])
{
    startupLog("Startup: begin");
    applyStartupSceneOverride(argc, argv);

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        startupLog("Startup failed: SDL_Init: %s", SDL_GetError());
        printf("Failed to initialize SDL: %s\n", SDL_GetError());
        return -1;
    }
    startupLog("Startup: SDL initialized");

#ifdef __EMSCRIPTEN__
    touchOnly = (bool)emscripten_run_script_int("window.matchMedia('(pointer:coarse)').matches ? 1 : 0");
#endif

    configureOpenGlAttributes();

    // Create the SDL window
    Window = createWindowWithVideoResetFallback();
    if (!Window)
    {
        startupLog("Startup failed: SDL_CreateWindow: %s", SDL_GetError());
        printf("Failed to create window: %s\n", SDL_GetError());
        return -1;
    }
    startupLog("Startup: window created");

    // Create the OpenGL context
    Context = SDL_GL_CreateContext(Window);
    if (!Context)
    {
        startupLog("Startup failed: SDL_GL_CreateContext: %s", SDL_GetError());
        printf("Failed to create OpenGL context: %s\n", SDL_GetError());
        SDL_DestroyWindow(Window);
        SDL_Quit();
        return -1;
    }
    startupLog("Startup: OpenGL context created");

    SDL_GL_MakeCurrent(Window, Context);
    SDL_GL_SetSwapInterval(1); // Enable vsync
    webgpuContext.initializeDeviceOnly();
    int webgpuSmokeWidth = 0;
    int webgpuSmokeHeight = 0;
    SDL_GetWindowSize(Window, &webgpuSmokeWidth, &webgpuSmokeHeight);
    webgpuContext.runOffscreenSmokeTest(webgpuSmokeWidth, webgpuSmokeHeight);
    webgpuContext.runComputeSmokeTest();
    startupLog("Startup: %s", webgpuContext.statusText());
    startupLog("Startup: %s", webgpuContext.smokeStatusText());
    startupLog("Startup: %s", webgpuContext.computeStatusText());

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;

// Scale UI higher for mobile devices
#ifdef __EMSCRIPTEN__
    const float uiScale = touchOnly ? 2.0f : 1.0f;
#else
    const float uiScale = 1.0f;
#endif

    ImFontConfig font_config;
    font_config.SizePixels = 13.0f * uiScale;
    io.Fonts->AddFontDefault(&font_config);

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Scale all style elements
    ImGuiStyle &style = ImGui::GetStyle();
    style.ScaleAllSizes(uiScale);

    // Setup Platform/Renderer bindings
    ImGui_ImplSDL2_InitForOpenGL(Window, Context);
#ifdef __EMSCRIPTEN__
    ImGui_ImplOpenGL3_Init("#version 300 es"); // WebAssembly
#else
    ImGui_ImplOpenGL3_Init("#version 150"); // Desktop OpenGL
#endif
    startupLog("Startup: ImGui initialized");

    // Load scene
    simulationHost.loadScene(currScene);
    viewerBridge.start(8765);
    startupLog("Startup: %s", viewerBridge.statusText());
    startupLog("Startup: initial scene loaded");

#ifdef __EMSCRIPTEN__
    // Use Emscripten's main loop for the web
    emscripten_set_main_loop(mainLoop, 0, 1);
#else
    // For native builds, use a while loop
    while (Running)
    {
        mainLoop();
    }
#endif

    // Cleanup
    closeWebGpuMainViewWindow();
    closeWebGpuInstancedPreviewWindow();
    viewerBridge.stop();
    webgpuContext.shutdown();
    SDL_GL_DeleteContext(Context);
    SDL_DestroyWindow(Window);
    SDL_Quit();
    startupLog("Shutdown: complete");

    return 0;
}
