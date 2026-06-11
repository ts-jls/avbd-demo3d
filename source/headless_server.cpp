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

#include "avbd_gpu_solver.h"
#include "simulation_host.h"
#include "viewer_bridge.h"
#include "webgpu_device.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace
{
std::atomic<bool> running(true);

#ifdef _WIN32
BOOL WINAPI consoleHandler(DWORD event)
{
    if (event == CTRL_C_EVENT || event == CTRL_CLOSE_EVENT || event == CTRL_BREAK_EVENT)
    {
        running = false;
        return TRUE;
    }
    return FALSE;
}
#endif

const char *argValue(int argc, char **argv, const char *name)
{
    size_t nameLength = strlen(name);
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], name) == 0 && i + 1 < argc)
            return argv[i + 1];
        if (strncmp(argv[i], name, nameLength) == 0 && argv[i][nameLength] == '=')
            return argv[i] + nameLength + 1;
    }
    return 0;
}

bool hasFlag(int argc, char **argv, const char *name)
{
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], name) == 0)
            return true;
    }
    return false;
}

std::string jsonString(const char *value)
{
    std::string out;
    out.push_back('"');
    for (const char *p = value ? value : ""; *p; ++p)
    {
        switch (*p)
        {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out.push_back(*p); break;
        }
    }
    out.push_back('"');
    return out;
}

struct RunningStat
{
    double total = 0.0;
    double min = 0.0;
    double max = 0.0;
    int count = 0;

    void add(double value)
    {
        if (count == 0)
        {
            min = value;
            max = value;
        }
        else
        {
            if (value < min)
                min = value;
            if (value > max)
                max = value;
        }
        total += value;
        count++;
    }

    double avg() const { return count > 0 ? total / (double)count : 0.0; }
};

struct SphereGroundClearance
{
    bool hasGround = false;
    float groundTop = 0.0f;
    int dynamicGroundBodies = 0;
    int belowGroundBodies = 0;
    int dynamicSpheres = 0;
    int belowGroundSpheres = 0;
    float minGroundBodyClearance = 0.0f;
    float minClearance = 0.0f;
    float maxGroundBodyPenetration = 0.0f;
    float maxPenetration = 0.0f;
};

struct FinalStateSignature
{
    int activeBodies = 0;
    int dynamicBodies = 0;
    int staticBodies = 0;
    int boxes = 0;
    int spheres = 0;
    int capsules = 0;
    int cylinders = 0;
    double positionChecksum = 0.0;
    double positionAbsChecksum = 0.0;
    double rotationChecksum = 0.0;
    double linearVelocityChecksum = 0.0;
    double angularVelocityChecksum = 0.0;
    double maxSpeed = 0.0;
    double maxAngularSpeed = 0.0;
    double aabbMinX = 0.0;
    double aabbMinY = 0.0;
    double aabbMinZ = 0.0;
    double aabbMaxX = 0.0;
    double aabbMaxY = 0.0;
    double aabbMaxZ = 0.0;
    double dynamicAabbMinX = 0.0;
    double dynamicAabbMinY = 0.0;
    double dynamicAabbMinZ = 0.0;
    double dynamicAabbMaxX = 0.0;
    double dynamicAabbMaxY = 0.0;
    double dynamicAabbMaxZ = 0.0;
};

struct WorldComparison
{
    int referenceBodies = 0;
    int candidateBodies = 0;
    int comparedBodies = 0;
    int bodyCountMismatch = 0;
    double maxPositionError = 0.0;
    double rmsPositionError = 0.0;
    double maxRotationError = 0.0;
    double rmsRotationError = 0.0;
    double maxLinearVelocityError = 0.0;
    double rmsLinearVelocityError = 0.0;
    double maxAngularVelocityError = 0.0;
    double rmsAngularVelocityError = 0.0;
};

double finiteOrZero(float value)
{
    return std::isfinite(value) ? (double)value : 0.0;
}

double square(double value)
{
    return value * value;
}

double lengthOf(float3 value)
{
    double x = finiteOrZero(value.x);
    double y = finiteOrZero(value.y);
    double z = finiteOrZero(value.z);
    return std::sqrt(x * x + y * y + z * z);
}

double distanceOf(float3 a, float3 b)
{
    double dx = finiteOrZero(a.x) - finiteOrZero(b.x);
    double dy = finiteOrZero(a.y) - finiteOrZero(b.y);
    double dz = finiteOrZero(a.z) - finiteOrZero(b.z);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double distanceOf(quat a, quat b)
{
    double dx = finiteOrZero(a.x) - finiteOrZero(b.x);
    double dy = finiteOrZero(a.y) - finiteOrZero(b.y);
    double dz = finiteOrZero(a.z) - finiteOrZero(b.z);
    double dw = finiteOrZero(a.w) - finiteOrZero(b.w);
    double direct = std::sqrt(dx * dx + dy * dy + dz * dz + dw * dw);

    double sx = finiteOrZero(a.x) + finiteOrZero(b.x);
    double sy = finiteOrZero(a.y) + finiteOrZero(b.y);
    double sz = finiteOrZero(a.z) + finiteOrZero(b.z);
    double sw = finiteOrZero(a.w) + finiteOrZero(b.w);
    double flipped = std::sqrt(sx * sx + sy * sy + sz * sz + sw * sw);
    return std::min(direct, flipped);
}

FinalStateSignature measureFinalStateSignature(const SimWorld &world)
{
    FinalStateSignature result;
    bool hasBounds = false;
    bool hasDynamicBounds = false;
    for (size_t i = 0; i < world.bodies.size(); ++i)
    {
        const SimBodyData &body = world.bodies[i];
        if (!body.active)
            continue;

        result.activeBodies++;
        bool dynamicBody = body.mass > 0.0f;
        if (dynamicBody)
            result.dynamicBodies++;
        else
            result.staticBodies++;

        switch (body.shape.type)
        {
        case RIGID_SHAPE_BOX:
            result.boxes++;
            break;
        case RIGID_SHAPE_SPHERE:
            result.spheres++;
            break;
        case RIGID_SHAPE_CAPSULE:
            result.capsules++;
            break;
        case RIGID_SHAPE_CYLINDER:
            result.cylinders++;
            break;
        default:
            break;
        }

        double indexWeight = (double)(i + 1);
        double shapeWeight = 1.0 + (double)body.shape.type * 0.125;
        double x = finiteOrZero(body.positionLin.x);
        double y = finiteOrZero(body.positionLin.y);
        double z = finiteOrZero(body.positionLin.z);
        double qx = finiteOrZero(body.positionAng.x);
        double qy = finiteOrZero(body.positionAng.y);
        double qz = finiteOrZero(body.positionAng.z);
        double qw = finiteOrZero(body.positionAng.w);
        double vx = finiteOrZero(body.velocityLin.x);
        double vy = finiteOrZero(body.velocityLin.y);
        double vz = finiteOrZero(body.velocityLin.z);
        double wx = finiteOrZero(body.velocityAng.x);
        double wy = finiteOrZero(body.velocityAng.y);
        double wz = finiteOrZero(body.velocityAng.z);

        result.positionChecksum += indexWeight * shapeWeight * (x * 0.73 + y * 1.31 + z * 1.91);
        result.positionAbsChecksum += indexWeight * shapeWeight * (std::fabs(x) + std::fabs(y) + std::fabs(z));
        result.rotationChecksum += indexWeight * (qx * 0.59 + qy * 0.83 + qz * 1.17 + qw * 1.43);
        result.linearVelocityChecksum += indexWeight * (vx * 0.67 + vy * 1.07 + vz * 1.61);
        result.angularVelocityChecksum += indexWeight * (wx * 0.71 + wy * 1.19 + wz * 1.73);

        double speed = lengthOf(body.velocityLin);
        double angularSpeed = lengthOf(body.velocityAng);
        if (speed > result.maxSpeed)
            result.maxSpeed = speed;
        if (angularSpeed > result.maxAngularSpeed)
            result.maxAngularSpeed = angularSpeed;

        double radius = finiteOrZero(body.radius);
        if (!hasBounds)
        {
            result.aabbMinX = x - radius;
            result.aabbMinY = y - radius;
            result.aabbMinZ = z - radius;
            result.aabbMaxX = x + radius;
            result.aabbMaxY = y + radius;
            result.aabbMaxZ = z + radius;
            hasBounds = true;
        }
        else
        {
            result.aabbMinX = std::min(result.aabbMinX, x - radius);
            result.aabbMinY = std::min(result.aabbMinY, y - radius);
            result.aabbMinZ = std::min(result.aabbMinZ, z - radius);
            result.aabbMaxX = std::max(result.aabbMaxX, x + radius);
            result.aabbMaxY = std::max(result.aabbMaxY, y + radius);
            result.aabbMaxZ = std::max(result.aabbMaxZ, z + radius);
        }

        if (dynamicBody)
        {
            if (!hasDynamicBounds)
            {
                result.dynamicAabbMinX = x - radius;
                result.dynamicAabbMinY = y - radius;
                result.dynamicAabbMinZ = z - radius;
                result.dynamicAabbMaxX = x + radius;
                result.dynamicAabbMaxY = y + radius;
                result.dynamicAabbMaxZ = z + radius;
                hasDynamicBounds = true;
            }
            else
            {
                result.dynamicAabbMinX = std::min(result.dynamicAabbMinX, x - radius);
                result.dynamicAabbMinY = std::min(result.dynamicAabbMinY, y - radius);
                result.dynamicAabbMinZ = std::min(result.dynamicAabbMinZ, z - radius);
                result.dynamicAabbMaxX = std::max(result.dynamicAabbMaxX, x + radius);
                result.dynamicAabbMaxY = std::max(result.dynamicAabbMaxY, y + radius);
                result.dynamicAabbMaxZ = std::max(result.dynamicAabbMaxZ, z + radius);
            }
        }
    }
    return result;
}

WorldComparison compareWorldStates(const SimWorld &reference, const SimWorld &candidate)
{
    WorldComparison result;
    result.referenceBodies = (int)reference.bodies.size();
    result.candidateBodies = (int)candidate.bodies.size();
    result.bodyCountMismatch = result.referenceBodies == result.candidateBodies ? 0 : 1;

    size_t count = std::min(reference.bodies.size(), candidate.bodies.size());
    double positionSumSq = 0.0;
    double rotationSumSq = 0.0;
    double linearVelocitySumSq = 0.0;
    double angularVelocitySumSq = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        const SimBodyData &a = reference.bodies[i];
        const SimBodyData &b = candidate.bodies[i];
        if (!a.active && !b.active)
            continue;

        result.comparedBodies++;
        double positionError = distanceOf(a.positionLin, b.positionLin);
        double rotationError = distanceOf(a.positionAng, b.positionAng);
        double linearVelocityError = distanceOf(a.velocityLin, b.velocityLin);
        double angularVelocityError = distanceOf(a.velocityAng, b.velocityAng);

        result.maxPositionError = std::max(result.maxPositionError, positionError);
        result.maxRotationError = std::max(result.maxRotationError, rotationError);
        result.maxLinearVelocityError = std::max(result.maxLinearVelocityError, linearVelocityError);
        result.maxAngularVelocityError = std::max(result.maxAngularVelocityError, angularVelocityError);

        positionSumSq += square(positionError);
        rotationSumSq += square(rotationError);
        linearVelocitySumSq += square(linearVelocityError);
        angularVelocitySumSq += square(angularVelocityError);
    }

    if (result.comparedBodies > 0)
    {
        double invCount = 1.0 / (double)result.comparedBodies;
        result.rmsPositionError = std::sqrt(positionSumSq * invCount);
        result.rmsRotationError = std::sqrt(rotationSumSq * invCount);
        result.rmsLinearVelocityError = std::sqrt(linearVelocitySumSq * invCount);
        result.rmsAngularVelocityError = std::sqrt(angularVelocitySumSq * invCount);
    }
    return result;
}

bool isBenchmarkGroundReceiver(const SimBodyData &body)
{
    if (!body.active || body.mass > 0.0f || body.shape.type != RIGID_SHAPE_BOX)
        return false;
    if (body.shape.size.x < 8.0f || body.shape.size.y < 8.0f || body.shape.size.z > 2.0f)
        return false;

    float3 normal = rotate(body.positionAng, float3{0.0f, 0.0f, 1.0f});
    return normal.z > 0.98f;
}

bool isBenchmarkGroundClearanceBody(const SimBodyData &body)
{
    if (!body.active || body.mass <= 0.0f)
        return false;
    return body.shape.type == RIGID_SHAPE_BOX ||
           body.shape.type == RIGID_SHAPE_SPHERE ||
           body.shape.type == RIGID_SHAPE_CAPSULE ||
           body.shape.type == RIGID_SHAPE_CYLINDER;
}

float benchmarkGroundContactExtentZ(const SimBodyData &body)
{
    if (body.shape.type == RIGID_SHAPE_SPHERE)
        return body.shape.radius > 0.0f ? body.shape.radius : body.radius;
    if (body.shape.type == RIGID_SHAPE_BOX)
    {
        float3 axisX = rotate(body.positionAng, float3{1.0f, 0.0f, 0.0f});
        float3 axisY = rotate(body.positionAng, float3{0.0f, 1.0f, 0.0f});
        float3 axisZ = rotate(body.positionAng, float3{0.0f, 0.0f, 1.0f});
        return fabsf(axisX.z) * std::max(0.0f, body.shape.size.x * 0.5f) +
               fabsf(axisY.z) * std::max(0.0f, body.shape.size.y * 0.5f) +
               fabsf(axisZ.z) * std::max(0.0f, body.shape.size.z * 0.5f);
    }
    if (body.shape.type == RIGID_SHAPE_CAPSULE)
    {
        float3 axis = rotate(body.positionAng, float3{0.0f, 0.0f, 1.0f});
        return fabsf(axis.z) * std::max(0.0f, body.shape.halfLength) +
               std::max(0.0f, body.shape.radius);
    }
    if (body.shape.type == RIGID_SHAPE_CYLINDER)
    {
        float3 axis = rotate(body.positionAng, float3{0.0f, 0.0f, 1.0f});
        float radialZ = sqrtf(std::max(0.0f, 1.0f - axis.z * axis.z));
        return fabsf(axis.z) * std::max(0.0f, body.shape.halfLength) +
               radialZ * std::max(0.0f, body.shape.radius);
    }
    return body.radius;
}

SphereGroundClearance measureSphereGroundClearance(const SimWorld &world)
{
    SphereGroundClearance result;
    for (const SimBodyData &body : world.bodies)
    {
        if (!isBenchmarkGroundReceiver(body))
            continue;
        float top = body.positionLin.z + body.shape.size.z * 0.5f;
        if (!result.hasGround || top > result.groundTop)
            result.groundTop = top;
        result.hasGround = true;
    }

    bool foundSphere = false;
    bool foundGroundBody = false;
    const float groundPenetrationEpsilon = 1.0e-5f;
    for (const SimBodyData &body : world.bodies)
    {
        if (!isBenchmarkGroundClearanceBody(body))
            continue;

        result.dynamicGroundBodies++;
        if (!result.hasGround)
            continue;

        float extentZ = benchmarkGroundContactExtentZ(body);
        float groundBodyClearance = body.positionLin.z - extentZ - result.groundTop;
        if (fabsf(groundBodyClearance) < groundPenetrationEpsilon)
            groundBodyClearance = 0.0f;
        if (!foundGroundBody || groundBodyClearance < result.minGroundBodyClearance)
            result.minGroundBodyClearance = groundBodyClearance;
        foundGroundBody = true;
        if (groundBodyClearance < -groundPenetrationEpsilon)
        {
            result.belowGroundBodies++;
            float penetration = -groundBodyClearance;
            if (penetration > result.maxGroundBodyPenetration)
                result.maxGroundBodyPenetration = penetration;
        }

        if (body.shape.type == RIGID_SHAPE_SPHERE)
        {
            result.dynamicSpheres++;
            if (!foundSphere || groundBodyClearance < result.minClearance)
                result.minClearance = groundBodyClearance;
            foundSphere = true;
            if (groundBodyClearance < -groundPenetrationEpsilon)
            {
                result.belowGroundSpheres++;
                float penetration = -groundBodyClearance;
                if (penetration > result.maxPenetration)
                    result.maxPenetration = penetration;
            }
        }
    }
    return result;
}

std::string bridgeMetricsText(const ViewerBridgeStats &stats)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "Viewer clients: " << stats.clients << "\n";
    out << "Command queue: " << stats.queuedCommands << "\n";
    out << "Snapshots sent: " << stats.snapshotCount << "\n";
    out << "Snapshot mode: " << (stats.binarySnapshotMode ? "Binary" : "JSON") << "\n";
    out << "Snapshot bytes: " << stats.lastSnapshotBytes << "\n";
    out << "Snapshot KB: " << ((double)stats.lastSnapshotBytes / 1024.0) << "\n";
    out << "Snapshot MB total: " << ((double)stats.totalSnapshotBytes / (1024.0 * 1024.0)) << "\n";
    out << "Snapshot serialize: " << stats.lastSerializeMs << " ms\n";
    out << "Snapshot serialize avg: " << stats.avgSerializeMs << " ms\n";
    out << "WebSocket send: " << stats.lastSendMs << " ms\n";
    out << "WebSocket send avg: " << stats.avgSendMs << " ms\n";
    out << "WebSocket sent clients: " << stats.lastSentClients << "\n";
    out << "WebSocket send failures: " << stats.sendFailures << "\n";
    return out.str();
}

std::string avbdGpuMetricsText()
{
    const AvbdGpuSolverStats &gpuStats = avbdGpuSolverStats();
    if (!gpuStats.active)
        return std::string();
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "AVBD GPU iterate: " << gpuStats.gpuIterateMs << " ms\n";
    out << "AVBD GPU bodies: " << gpuStats.bodies << "\n";
    out << "AVBD GPU contacts: " << gpuStats.contacts << "\n";
    out << "AVBD GPU joints: " << gpuStats.joints << "\n";
    out << "AVBD GPU springs: " << gpuStats.springs << "\n";
    out << "AVBD GPU colors: " << gpuStats.colors << "\n";
    out << "AVBD GPU CPU-iterate fallbacks: " << gpuStats.cpuIterateFallbacks << "\n";
    return out.str();
}

std::string combinedMetricsText(const SimulationHost &host, const ViewerBridge &bridge, int tickOverruns)
{
    std::ostringstream out;
    out << host.metricsText();
    out << avbdGpuMetricsText();
    out << bridgeMetricsText(bridge.statsSnapshot());
    out << "Tick overruns: " << tickOverruns << "\n";
    return out.str();
}

bool configurePhysicsBackend(SimulationHost &host, const char *physicsBackend, WebGpuDevice &webgpuDevice)
{
    if (!physicsBackend || strcmp(physicsBackend, "cpu") == 0 || strcmp(physicsBackend, "CPU") == 0)
    {
        host.solver().physicsBackend = makeCpuReferencePhysicsBackend();
        return true;
    }

    // Two lanes: the CPU reference solver and the paper-faithful GPU AVBD
    // solver. The historical experimental lanes (fast/counterless/resident/
    // joint-*/contact-direct and the "auto" router) were removed after
    // webgpu-avbd superseded them.
    bool avbdWebGpuBackend = strcmp(physicsBackend, "webgpu-avbd") == 0 ||
                             strcmp(physicsBackend, "WebGPU AVBD") == 0;
    if (!avbdWebGpuBackend)
    {
        std::fprintf(stderr, "Unknown physics backend '%s' (expected cpu or webgpu-avbd)\n", physicsBackend);
        return false;
    }

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    webgpuDevice.initialize();
    if (webgpuDevice.deviceReady)
        webgpuDevice.runComputeSmokeTest();
    if (!webgpuDevice.deviceReady)
    {
        std::fprintf(stderr, "WebGPU physics unavailable: %s\n", webgpuDevice.statusText());
        host.solver().physicsBackend = makeCpuReferencePhysicsBackend();
        return false;
    }
    host.solver().physicsBackend = makeWebGpuAvbdPhysicsBackend(&webgpuDevice);
    return true;
#else
    (void)webgpuDevice;
    std::fprintf(stderr, "WebGPU physics unavailable: build does not include Dawn/WebGPU\n");
    host.solver().physicsBackend = makeCpuReferencePhysicsBackend();
    return false;
#endif
}

void applyBroadphaseMode(Solver &solver, const char *mode)
{
    if (!mode)
        return;
    if (strcmp(mode, "allpairs") == 0)
        solver.broadphaseMode = BROADPHASE_ALL_PAIRS;
    else if (strcmp(mode, "grid") == 0 || strcmp(mode, "uniformgrid") == 0 || strcmp(mode, "spatialhash") == 0)
        solver.broadphaseMode = BROADPHASE_UNIFORM_GRID;
    else if (strcmp(mode, "sap") == 0 || strcmp(mode, "sweepandprune") == 0)
        solver.broadphaseMode = BROADPHASE_SWEEP_AND_PRUNE;
    else
        std::fprintf(stderr, "Unknown broadphase '%s' (expected allpairs, grid, or sap)\n", mode);
}

bool isPhysicsBackendCommand(const SimulationCommand &command)
{
    std::string name;
    for (char c : command.command)
    {
        unsigned char ch = (unsigned char)c;
        if (std::isalnum(ch))
            name.push_back((char)std::tolower(ch));
    }
    return name == "physicsbackend" || name == "setphysicsbackend";
}

int runBenchmark(const char *scene, int frames, int warmupFrames, bool resetAfterWarmup, bool noStream, uint16_t port, const char *physicsBackend, bool collisionOnly, int iterations, bool hasGravityOverride, float gravityOverride, const char *broadphase)
{
    if (frames <= 0)
        frames = 600;
    if (warmupFrames < 0)
        warmupFrames = 0;

    SimulationHost host;
    WebGpuDevice webgpuDevice;
    if (scene && !host.loadSceneByName(scene))
    {
        std::fprintf(stderr, "Unknown benchmark scene '%s'\n", scene);
        return 2;
    }
    else if (!scene)
    {
        host.loadScene(host.sceneIndex());
    }
    configurePhysicsBackend(host, physicsBackend, webgpuDevice);
    applyBroadphaseMode(host.solver(), broadphase);
    if (iterations > 0)
        host.solver().iterations = iterations;
    if (hasGravityOverride)
        host.solver().gravity = gravityOverride;
    host.solver().world.syncFromLegacy(host.solver());
    ViewerBridge bridge;
    if (!noStream)
        bridge.start(port);

    RunningStat frameMs;
    RunningStat physicsMs;
    RunningStat simWorldSyncMs;
    RunningStat broadphaseMs;
    RunningStat forceInitMs;
    RunningStat bodyInitMs;
    RunningStat primalMs;
    RunningStat dualMs;
    RunningStat velocityUpdateMs;
    RunningStat forceInitGatherMs;
    RunningStat forceInitParallelMs;
    RunningStat forceInitCleanupMs;
    RunningStat manifoldAllocMsStat;
    RunningStat avbdIterateMs;
    RunningStat avbdBuildMs;
    RunningStat avbdSubmitMs;
    RunningStat avbdWaitMs;
    RunningStat avbdApplyMs;
    RunningStat serializeMs;
    RunningStat sendMs;
    uint64_t lastBytes = 0;

    using Clock = std::chrono::high_resolution_clock;
    for (int frame = 0; frame < warmupFrames; ++frame)
    {
        if (collisionOnly)
        {
            host.solver().benchmarkBroadphaseOnly();
        }
        else
        {
            host.stepFrame();
        }
    }
    if (resetAfterWarmup && warmupFrames > 0)
    {
        host.resetScene();
        if (iterations > 0)
            host.solver().iterations = iterations;
        if (hasGravityOverride)
            host.solver().gravity = gravityOverride;
        host.solver().world.syncFromLegacy(host.solver());
    }

    auto totalBegin = Clock::now();
    for (int frame = 0; frame < frames; ++frame)
    {
        auto frameBegin = Clock::now();
        if (collisionOnly)
        {
            host.solver().benchmarkBroadphaseOnly();
        }
        else
        {
            host.stepFrame();
        }
        if (!noStream && !collisionOnly)
            bridge.broadcastSnapshot(host.world(), host.currentSceneName(), host.nextSnapshotFrame());
        auto frameEnd = Clock::now();

        const SolverStats &stats = host.solver().stats;
        ViewerBridgeStats bridgeStats = bridge.statsSnapshot();
        double frameElapsed = std::chrono::duration<double, std::milli>(frameEnd - frameBegin).count();
        frameMs.add(frameElapsed);
        physicsMs.add(collisionOnly ? frameElapsed : host.lastPhysicsMs());
        simWorldSyncMs.add(stats.simWorldSyncMs);
        broadphaseMs.add(stats.broadphaseMs);
        forceInitMs.add(stats.forceInitMs);
        forceInitGatherMs.add(stats.forceInitGatherMs);
        forceInitParallelMs.add(stats.forceInitParallelMs);
        forceInitCleanupMs.add(stats.forceInitCleanupMs);
        manifoldAllocMsStat.add(stats.manifoldAllocMs);
        bodyInitMs.add(stats.bodyInitMs);
        primalMs.add(stats.primalSolveMs);
        dualMs.add(stats.dualUpdateMs);
        velocityUpdateMs.add(stats.velocityUpdateMs);
        const AvbdGpuSolverStats &frameAvbdStats = avbdGpuSolverStats();
        if (frameAvbdStats.active)
        {
            avbdIterateMs.add(frameAvbdStats.gpuIterateMs);
            avbdBuildMs.add(frameAvbdStats.buildFrameMs);
            avbdSubmitMs.add(frameAvbdStats.submitMs);
            avbdWaitMs.add(frameAvbdStats.waitMs);
            avbdApplyMs.add(frameAvbdStats.applyMs);
        }
        serializeMs.add(bridgeStats.lastSerializeMs);
        sendMs.add(bridgeStats.lastSendMs);
        lastBytes = bridgeStats.lastSnapshotBytes;
    }
    double totalMs = std::chrono::duration<double, std::milli>(Clock::now() - totalBegin).count();
    ViewerBridgeStats bridgeStats = bridge.statsSnapshot();
    if (!noStream)
        bridge.stop();
    SphereGroundClearance sphereGroundClearance = measureSphereGroundClearance(host.world());
    FinalStateSignature finalState = measureFinalStateSignature(host.world());

    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{";
    out << "\"type\":\"headlessBenchmark\",";
    out << "\"scene\":\"" << host.currentSceneName() << "\",";
    out << "\"physicsBackend\":\"" << host.solver().physicsBackend->name() << "\",";
    out << "\"collisionOnly\":" << (collisionOnly ? "true" : "false") << ",";
    out << "\"frames\":" << frames << ",";
    out << "\"warmupFrames\":" << warmupFrames << ",";
    out << "\"resetAfterWarmup\":" << (resetAfterWarmup ? "true" : "false") << ",";
    out << "\"iterations\":" << host.solver().iterations << ",";
    out << "\"gravity\":" << host.solver().gravity << ",";
    out << "\"bodies\":" << host.world().bodies.size() << ",";
    out << "\"constraints\":" << host.world().constraints.size() << ",";
    out << "\"finalActiveBodies\":" << finalState.activeBodies << ",";
    out << "\"finalDynamicBodies\":" << finalState.dynamicBodies << ",";
    out << "\"finalStaticBodies\":" << finalState.staticBodies << ",";
    out << "\"finalBoxes\":" << finalState.boxes << ",";
    out << "\"finalSpheres\":" << finalState.spheres << ",";
    out << "\"finalCapsules\":" << finalState.capsules << ",";
    out << "\"finalCylinders\":" << finalState.cylinders << ",";
    out << "\"finalPositionChecksum\":" << finalState.positionChecksum << ",";
    out << "\"finalPositionAbsChecksum\":" << finalState.positionAbsChecksum << ",";
    out << "\"finalRotationChecksum\":" << finalState.rotationChecksum << ",";
    out << "\"finalLinearVelocityChecksum\":" << finalState.linearVelocityChecksum << ",";
    out << "\"finalAngularVelocityChecksum\":" << finalState.angularVelocityChecksum << ",";
    out << "\"finalMaxSpeed\":" << finalState.maxSpeed << ",";
    out << "\"finalMaxAngularSpeed\":" << finalState.maxAngularSpeed << ",";
    out << "\"finalAabbMin\":[" << finalState.aabbMinX << "," << finalState.aabbMinY << "," << finalState.aabbMinZ << "],";
    out << "\"finalAabbMax\":[" << finalState.aabbMaxX << "," << finalState.aabbMaxY << "," << finalState.aabbMaxZ << "],";
    out << "\"finalDynamicAabbMin\":[" << finalState.dynamicAabbMinX << "," << finalState.dynamicAabbMinY << "," << finalState.dynamicAabbMinZ << "],";
    out << "\"finalDynamicAabbMax\":[" << finalState.dynamicAabbMaxX << "," << finalState.dynamicAabbMaxY << "," << finalState.dynamicAabbMaxZ << "],";
    out << "\"finalDynamicMaxHeight\":" << finalState.dynamicAabbMaxZ << ",";
    out << "\"forceCount\":" << host.solver().stats.forceCount << ",";
    out << "\"manifoldCount\":" << host.solver().stats.manifoldCount << ",";
    out << "\"manifoldsCreated\":" << host.solver().stats.manifoldsCreated << ",";
    out << "\"jointInitializationSkipped\":" << host.solver().stats.jointInitializationSkipped << ",";
    out << "\"ignoreInitializationSkipped\":" << host.solver().stats.ignoreCollisionInitializationSkipped << ",";
    out << "\"pairChecks\":" << host.solver().stats.pairChecks << ",";
    out << "\"sphereHits\":" << host.solver().stats.sphereHits << ",";
    out << "\"constrainedHits\":" << host.solver().stats.constrainedHits << ",";
    out << "\"primalJointVisits\":" << host.solver().stats.primalJointVisits << ",";
    out << "\"dualJointVisits\":" << host.solver().stats.dualJointVisits << ",";
    out << "\"primalJointSkipped\":" << host.solver().stats.primalJointSkipped << ",";
    out << "\"dualJointSkipped\":" << host.solver().stats.dualJointSkipped << ",";
    out << "\"primalIgnoreVisits\":" << host.solver().stats.primalIgnoreCollisionVisits << ",";
    out << "\"dualIgnoreVisits\":" << host.solver().stats.dualIgnoreCollisionVisits << ",";
    out << "\"primalIgnoreSkipped\":" << host.solver().stats.primalIgnoreCollisionSkipped << ",";
    out << "\"dualIgnoreSkipped\":" << host.solver().stats.dualIgnoreCollisionSkipped << ",";
    out << "\"finalGroundFound\":" << (sphereGroundClearance.hasGround ? "true" : "false") << ",";
    out << "\"finalGroundTop\":" << sphereGroundClearance.groundTop << ",";
    out << "\"finalDynamicGroundBodies\":" << sphereGroundClearance.dynamicGroundBodies << ",";
    out << "\"finalBelowGroundBodies\":" << sphereGroundClearance.belowGroundBodies << ",";
    out << "\"finalMinGroundBodyClearance\":" << sphereGroundClearance.minGroundBodyClearance << ",";
    out << "\"finalMaxGroundBodyPenetration\":" << sphereGroundClearance.maxGroundBodyPenetration << ",";
    out << "\"finalDynamicSpheres\":" << sphereGroundClearance.dynamicSpheres << ",";
    out << "\"finalBelowGroundSpheres\":" << sphereGroundClearance.belowGroundSpheres << ",";
    out << "\"finalMinSphereGroundClearance\":" << sphereGroundClearance.minClearance << ",";
    out << "\"finalMaxSphereGroundPenetration\":" << sphereGroundClearance.maxPenetration << ",";
    out << "\"streaming\":" << (noStream ? "false" : "true") << ",";
    out << "\"totalMs\":" << totalMs << ",";
    out << "\"frameAvgMs\":" << frameMs.avg() << ",";
    out << "\"frameMinMs\":" << frameMs.min << ",";
    out << "\"frameMaxMs\":" << frameMs.max << ",";
    out << "\"physicsAvgMs\":" << physicsMs.avg() << ",";
    out << "\"physicsMinMs\":" << physicsMs.min << ",";
    out << "\"physicsMaxMs\":" << physicsMs.max << ",";
    out << "\"simWorldSyncAvgMs\":" << simWorldSyncMs.avg() << ",";
    out << "\"broadphaseAvgMs\":" << broadphaseMs.avg() << ",";
    out << "\"forceInitAvgMs\":" << forceInitMs.avg() << ",";
    out << "\"forceInitGatherAvgMs\":" << forceInitGatherMs.avg() << ",";
    out << "\"forceInitParallelAvgMs\":" << forceInitParallelMs.avg() << ",";
    out << "\"forceInitCleanupAvgMs\":" << forceInitCleanupMs.avg() << ",";
    out << "\"manifoldAllocAvgMs\":" << manifoldAllocMsStat.avg() << ",";
    out << "\"bodyInitAvgMs\":" << bodyInitMs.avg() << ",";
    out << "\"primalAvgMs\":" << primalMs.avg() << ",";
    out << "\"dualAvgMs\":" << dualMs.avg() << ",";
    out << "\"velocityUpdateAvgMs\":" << velocityUpdateMs.avg() << ",";
    const AvbdGpuSolverStats &avbdStats = avbdGpuSolverStats();
    out << "\"webgpuStatus\":" << jsonString(webgpuDevice.statusText()) << ",";
    out << "\"webgpuRuntimeFallbacks\":" << avbdStats.cpuIterateFallbacks << ",";
    out << "\"avbdGpuActive\":" << (avbdStats.active ? "true" : "false") << ",";
    out << "\"avbdGpuIterateMs\":" << avbdStats.gpuIterateMs << ",";
    out << "\"avbdGpuIterateAvgMs\":" << avbdIterateMs.avg() << ",";
    out << "\"avbdGpuBuildAvgMs\":" << avbdBuildMs.avg() << ",";
    out << "\"avbdGpuSubmitAvgMs\":" << avbdSubmitMs.avg() << ",";
    out << "\"avbdGpuWaitAvgMs\":" << avbdWaitMs.avg() << ",";
    out << "\"avbdGpuApplyAvgMs\":" << avbdApplyMs.avg() << ",";
    out << "\"avbdGpuBodies\":" << avbdStats.bodies << ",";
    out << "\"avbdGpuContacts\":" << avbdStats.contacts << ",";
    out << "\"avbdGpuSpherePairs\":" << avbdStats.spherePairs << ",";
    out << "\"avbdGpuJoints\":" << avbdStats.joints << ",";
    out << "\"avbdGpuSprings\":" << avbdStats.springs << ",";
    out << "\"avbdGpuColors\":" << avbdStats.colors << ",";
    out << "\"snapshotSerializeAvgMs\":" << serializeMs.avg() << ",";
    out << "\"webSocketSendAvgMs\":" << sendMs.avg() << ",";
    out << "\"snapshotMode\":\"" << (bridgeStats.binarySnapshotMode ? "binary" : "json") << "\",";
    out << "\"snapshotBytes\":" << lastBytes << ",";
    out << "\"snapshotsSent\":" << bridgeStats.snapshotCount;
    out << "}\n";
    std::printf("%s", out.str().c_str());
    return 0;
}

int runBackendComparison(const char *scene, int frames, const char *candidateBackend, int iterations)
{
    if (frames <= 0)
        frames = 120;
    if (!candidateBackend)
        candidateBackend = "webgpu";

    SimulationHost referenceHost;
    SimulationHost candidateHost;
    WebGpuDevice candidateWebGpuDevice;

    if (scene)
    {
        if (!referenceHost.loadSceneByName(scene) || !candidateHost.loadSceneByName(scene))
        {
            std::fprintf(stderr, "Unknown compare scene '%s'\n", scene);
            return 2;
        }
    }
    else
    {
        referenceHost.loadScene(referenceHost.sceneIndex());
        candidateHost.loadScene(candidateHost.sceneIndex());
    }

    referenceHost.solver().physicsBackend = makeCpuReferencePhysicsBackend();
    if (!configurePhysicsBackend(candidateHost, candidateBackend, candidateWebGpuDevice))
        return 2;
    if (iterations > 0)
    {
        referenceHost.solver().iterations = iterations;
        candidateHost.solver().iterations = iterations;
    }
    referenceHost.solver().world.syncFromLegacy(referenceHost.solver());
    candidateHost.solver().world.syncFromLegacy(candidateHost.solver());

    RunningStat referencePhysicsMs;
    RunningStat candidatePhysicsMs;
    for (int frame = 0; frame < frames; ++frame)
    {
        referenceHost.stepFrame();
        candidateHost.stepFrame();
        referencePhysicsMs.add(referenceHost.lastPhysicsMs());
        candidatePhysicsMs.add(candidateHost.lastPhysicsMs());
    }

    WorldComparison comparison = compareWorldStates(referenceHost.world(), candidateHost.world());
    FinalStateSignature referenceSignature = measureFinalStateSignature(referenceHost.world());
    FinalStateSignature candidateSignature = measureFinalStateSignature(candidateHost.world());
    SphereGroundClearance referenceGround = measureSphereGroundClearance(referenceHost.world());
    SphereGroundClearance candidateGround = measureSphereGroundClearance(candidateHost.world());

    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{";
    out << "\"type\":\"backendComparison\",";
    out << "\"scene\":\"" << referenceHost.currentSceneName() << "\",";
    out << "\"frames\":" << frames << ",";
    out << "\"iterations\":" << referenceHost.solver().iterations << ",";
    out << "\"referenceBackend\":\"" << referenceHost.solver().physicsBackend->name() << "\",";
    out << "\"candidateBackend\":\"" << candidateHost.solver().physicsBackend->name() << "\",";
    out << "\"referencePhysicsAvgMs\":" << referencePhysicsMs.avg() << ",";
    out << "\"candidatePhysicsAvgMs\":" << candidatePhysicsMs.avg() << ",";
    out << "\"candidateWebgpuRuntimeFallbacks\":" << avbdGpuSolverStats().cpuIterateFallbacks << ",";
    out << "\"candidateWebgpuStatus\":" << jsonString(candidateWebGpuDevice.statusText()) << ",";
    out << "\"referenceBodies\":" << comparison.referenceBodies << ",";
    out << "\"candidateBodies\":" << comparison.candidateBodies << ",";
    out << "\"comparedBodies\":" << comparison.comparedBodies << ",";
    out << "\"bodyCountMismatch\":" << comparison.bodyCountMismatch << ",";
    out << "\"maxPositionError\":" << comparison.maxPositionError << ",";
    out << "\"rmsPositionError\":" << comparison.rmsPositionError << ",";
    out << "\"maxRotationError\":" << comparison.maxRotationError << ",";
    out << "\"rmsRotationError\":" << comparison.rmsRotationError << ",";
    out << "\"maxLinearVelocityError\":" << comparison.maxLinearVelocityError << ",";
    out << "\"rmsLinearVelocityError\":" << comparison.rmsLinearVelocityError << ",";
    out << "\"maxAngularVelocityError\":" << comparison.maxAngularVelocityError << ",";
    out << "\"rmsAngularVelocityError\":" << comparison.rmsAngularVelocityError << ",";
    out << "\"referencePositionChecksum\":" << referenceSignature.positionChecksum << ",";
    out << "\"candidatePositionChecksum\":" << candidateSignature.positionChecksum << ",";
    out << "\"referenceRotationChecksum\":" << referenceSignature.rotationChecksum << ",";
    out << "\"candidateRotationChecksum\":" << candidateSignature.rotationChecksum << ",";
    out << "\"referenceLinearVelocityChecksum\":" << referenceSignature.linearVelocityChecksum << ",";
    out << "\"candidateLinearVelocityChecksum\":" << candidateSignature.linearVelocityChecksum << ",";
    out << "\"referenceAngularVelocityChecksum\":" << referenceSignature.angularVelocityChecksum << ",";
    out << "\"candidateAngularVelocityChecksum\":" << candidateSignature.angularVelocityChecksum << ",";
    out << "\"referenceBelowGroundSpheres\":" << referenceGround.belowGroundSpheres << ",";
    out << "\"candidateBelowGroundSpheres\":" << candidateGround.belowGroundSpheres << ",";
    out << "\"referenceMaxGroundPenetration\":" << referenceGround.maxPenetration << ",";
    out << "\"candidateMaxGroundPenetration\":" << candidateGround.maxPenetration << ",";
    out << "\"referenceBelowGroundBodies\":" << referenceGround.belowGroundBodies << ",";
    out << "\"candidateBelowGroundBodies\":" << candidateGround.belowGroundBodies << ",";
    out << "\"referenceMaxGroundBodyPenetration\":" << referenceGround.maxGroundBodyPenetration << ",";
    out << "\"candidateMaxGroundBodyPenetration\":" << candidateGround.maxGroundBodyPenetration;
    out << "}\n";
    std::printf("%s", out.str().c_str());
    return 0;
}
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleCtrlHandler(consoleHandler, TRUE);
#endif

    const char *scene = argValue(argc, argv, "--scene");
    const char *benchmarkScene = argValue(argc, argv, "--benchmark-scene");
    const char *compareScene = argValue(argc, argv, "--compare-scene");
    const char *physicsBackend = argValue(argc, argv, "--physics-backend");
    const char *compareBackend = argValue(argc, argv, "--compare-backend");
    int benchmarkFrames = atoi(argValue(argc, argv, "--benchmark-frames") ? argValue(argc, argv, "--benchmark-frames") : "0");
    int benchmarkWarmupFrames = atoi(argValue(argc, argv, "--warmup-frames") ? argValue(argc, argv, "--warmup-frames") : "0");
    bool resetAfterWarmup = hasFlag(argc, argv, "--reset-after-warmup");
    int compareFrames = atoi(argValue(argc, argv, "--compare-frames") ? argValue(argc, argv, "--compare-frames") : "0");
    int iterations = atoi(argValue(argc, argv, "--iterations") ? argValue(argc, argv, "--iterations") : "0");
    const char *gravityValue = argValue(argc, argv, "--gravity");
    bool hasGravityOverride = gravityValue != 0;
    float gravityOverride = (float)atof(gravityValue ? gravityValue : "0");
    uint16_t port = (uint16_t)atoi(argValue(argc, argv, "--port") ? argValue(argc, argv, "--port") : "8765");
    double metricsInterval = atof(argValue(argc, argv, "--metrics-interval") ? argValue(argc, argv, "--metrics-interval") : "0");
    bool noStream = hasFlag(argc, argv, "--no-stream");
    bool collisionOnly = hasFlag(argc, argv, "--collision-only");
    const char *broadphase = argValue(argc, argv, "--broadphase");
        double tickRate = atof(argValue(argc, argv, "--tick-rate") ? argValue(argc, argv, "--tick-rate") : "60");
    if (tickRate <= 0.0)
        tickRate = 60.0;

    if (benchmarkScene || benchmarkFrames > 0)
        return runBenchmark(benchmarkScene ? benchmarkScene : scene, benchmarkFrames, benchmarkWarmupFrames, resetAfterWarmup, noStream, port, physicsBackend, collisionOnly, iterations, hasGravityOverride, gravityOverride, broadphase);
    if (compareScene || compareFrames > 0)
        return runBackendComparison(compareScene ? compareScene : scene, compareFrames, compareBackend ? compareBackend : physicsBackend, iterations);

    SimulationHost host;
    WebGpuDevice webgpuDevice;
    if (scene && !host.loadSceneByName(scene))
    {
        std::printf("Unknown scene '%s', loading default scene '%s'\n", scene, host.currentSceneName());
        host.loadScene(host.sceneIndex());
    }
    else
    {
        host.loadScene(host.sceneIndex());
    }
    host.setPaused(hasFlag(argc, argv, "--paused"));
    configurePhysicsBackend(host, physicsBackend, webgpuDevice);
    applyBroadphaseMode(host.solver(), broadphase);
    if (iterations > 0)
        host.solver().iterations = iterations;
    if (hasGravityOverride)
        host.solver().gravity = gravityOverride;

    ViewerBridge bridge;
    if (!noStream)
        bridge.start(port);

    std::printf("AVBD headless server\n");
    std::printf("Scene: %s\n", host.currentSceneName());
    std::printf("Physics backend: %s\n", host.solver().physicsBackend->name());
    std::printf("Gravity: %.3f\n", host.solver().gravity);
    if (physicsBackend && strstr(host.solver().physicsBackend->name(), "WebGPU"))
        std::printf("WebGPU runtime: %s\n", webgpuDevice.statusText());
    std::printf("Bridge: %s\n", noStream ? "disabled by --no-stream" : bridge.statusText());
    std::printf("Press Ctrl+C to stop.\n");

    using Clock = std::chrono::high_resolution_clock;
    const auto tick = std::chrono::duration<double>(1.0 / tickRate);
    auto nextTick = Clock::now();
    auto lastMetricsPrint = Clock::now();
    int statusFrame = 0;
    int tickOverruns = 0;

    while (running)
    {
        SimulationCommand command;
        while (bridge.pollCommand(command))
        {
            if (isPhysicsBackendCommand(command))
            {
                if (!command.stringValue.empty())
                    configurePhysicsBackend(host, command.stringValue.c_str(), webgpuDevice);
                continue;
            }
            host.applyCommand(command);
        }

        if (!host.isPaused())
            host.stepFrame();

        if (!noStream)
        {
            bridge.broadcastSnapshot(host.world(), host.currentSceneName(), host.nextSnapshotFrame());
            if ((statusFrame++ % 30) == 0)
            {
                std::string metrics = combinedMetricsText(host, bridge, tickOverruns);
                bridge.broadcastStatus(metrics.c_str());
            }
        }

        if (metricsInterval > 0.0)
        {
            auto now = Clock::now();
            double elapsed = std::chrono::duration<double>(now - lastMetricsPrint).count();
            if (elapsed >= metricsInterval)
            {
                ViewerBridgeStats bridgeStats = bridge.statsSnapshot();
                std::printf("Metrics: scene=\"%s\" physics=%.2fms simWorld=%.2fms serialize=%.2fms send=%.2fms snapshot=%.1fKB clients=%d overruns=%d\n",
                            host.currentSceneName(),
                            host.lastPhysicsMs(),
                            host.solver().stats.simWorldSyncMs,
                            bridgeStats.lastSerializeMs,
                            bridgeStats.lastSendMs,
                            (double)bridgeStats.lastSnapshotBytes / 1024.0,
                            bridgeStats.clients,
                            tickOverruns);
                lastMetricsPrint = now;
            }
        }

        nextTick += std::chrono::duration_cast<Clock::duration>(tick);
        std::this_thread::sleep_until(nextTick);
        if (Clock::now() > nextTick + std::chrono::milliseconds(250))
        {
            tickOverruns++;
            nextTick = Clock::now();
        }
    }

    bridge.stop();
    std::printf("Shutdown complete.\n");
    return 0;
}
