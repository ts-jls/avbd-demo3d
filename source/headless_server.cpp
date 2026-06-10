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
#include "webgpu_backend.h"

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

void setEnvironmentFlag(const char *name, bool enabled)
{
#ifdef _WIN32
    _putenv_s(name, enabled ? "1" : "0");
#else
    setenv(name, enabled ? "1" : "0", 1);
#endif
}

void setEnvironmentValue(const char *name, const char *value)
{
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    setenv(name, value ? value : "", 1);
#endif
}

bool environmentFlagEnabled(const char *name)
{
#ifdef _WIN32
    char *value = 0;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || !value)
        return false;
    bool enabled = strcmp(value, "0") != 0 &&
                   strcmp(value, "false") != 0 &&
                   strcmp(value, "False") != 0;
    free(value);
    return enabled;
#else
    const char *value = getenv(name);
    return value &&
           strcmp(value, "0") != 0 &&
           strcmp(value, "false") != 0 &&
           strcmp(value, "False") != 0;
#endif
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

std::string webGpuRuntimeMetricsText(const WebGpuContext &webgpuContext)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "WebGPU runtime: " << webgpuContext.runtimeStatusText() << "\n";
    out << "WebGPU runtime total: " << webgpuContext.runtimeTotalMillis() << " ms\n";
    out << "WebGPU runtime sync: " << webgpuContext.runtimeSyncMillis() << " ms\n";
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
    out << "WebGPU joint topology: " << webgpuContext.jointTopologyStatusText() << "\n";
    out << "WebGPU joint topology joints: " << webgpuContext.jointTopologyJointCount() << "\n";
    out << "WebGPU joint topology refs: " << webgpuContext.jointTopologyBodyRefCount() << "\n";
    out << "WebGPU joint topology active bodies: " << webgpuContext.jointTopologyActiveBodyCount() << "\n";
    out << "WebGPU joint topology max per body: " << webgpuContext.jointTopologyMaxPerBodyCount() << "\n";
    out << "WebGPU joint topology mismatches: " << webgpuContext.jointTopologyMismatchCount() << "\n";
    out << "WebGPU joint topology readback bytes: " << webgpuContext.jointTopologyReadbackByteCount() << "\n";
    out << "WebGPU joint topology time: " << webgpuContext.jointTopologyMillis() << " ms\n";
    out << "WebGPU joint colors: " << webgpuContext.jointColorCountValue() << "\n";
    out << "WebGPU joint color conflicts: " << webgpuContext.jointColorConflictCount() << "\n";
    out << "WebGPU joint color min bucket: " << webgpuContext.jointColorMinBucketCount() << "\n";
    out << "WebGPU joint color max bucket: " << webgpuContext.jointColorMaxBucketCount() << "\n";
    out << "WebGPU joint color readback bytes: " << webgpuContext.jointColorReadbackByteCount() << "\n";
    out << "WebGPU joint residual max: " << webgpuContext.jointResidualMaxValue() << "\n";
    out << "WebGPU joint residual RMS: " << webgpuContext.jointResidualRmsValue() << "\n";
    out << "WebGPU joint residual readback bytes: " << webgpuContext.jointResidualReadbackByteCount() << "\n";
    out << "WebGPU joint proposal max correction: " << webgpuContext.jointProposalMaxCorrectionValue() << "\n";
    out << "WebGPU joint proposal RMS correction: " << webgpuContext.jointProposalRmsCorrectionValue() << "\n";
    out << "WebGPU joint proposal active bodies: " << webgpuContext.jointProposalActiveBodyCount() << "\n";
    out << "WebGPU joint proposal max per body: " << webgpuContext.jointProposalMaxPerBodyCount() << "\n";
    out << "WebGPU joint proposal readback bytes: " << webgpuContext.jointProposalReadbackByteCount() << "\n";
    out << "WebGPU joint proposal iterations: " << webgpuContext.jointProposalIterationCount() << "\n";
    out << "WebGPU joint proposal residual after max: " << webgpuContext.jointProposalResidualAfterMaxValue() << "\n";
    out << "WebGPU joint proposal residual after RMS: " << webgpuContext.jointProposalResidualAfterRmsValue() << "\n";
    out << "WebGPU joint proposal residual readback bytes: " << webgpuContext.jointProposalResidualReadbackByteCount() << "\n";
    out << "WebGPU joint proposal final position ready: " << webgpuContext.jointProposalFinalPositionReadyValue() << "\n";
    out << "WebGPU joint proposal final position bodies: " << webgpuContext.jointProposalFinalPositionBodyCountValue() << "\n";
    out << "WebGPU joint proposal final position bytes: " << webgpuContext.jointProposalFinalPositionByteCount() << "\n";
    out << "WebGPU joint proposal final position absolute: " << webgpuContext.jointProposalFinalPositionAbsoluteValue() << "\n";
    out << "WebGPU joint proposal seeded from contact: " << webgpuContext.jointProposalSeededFromContactValue() << "\n";
    out << "WebGPU joint proposal applied position bodies: " << webgpuContext.jointProposalAppliedPositionBodyCount() << "\n";
    out << "WebGPU joint proposal applied position readback bytes: " << webgpuContext.jointProposalAppliedPositionReadbackByteCount() << "\n";
    out << "WebGPU joint proposal applied position max delta: " << webgpuContext.jointProposalAppliedPositionMaxDeltaValue() << "\n";
    out << "WebGPU joint proposal applied position checksum: " << webgpuContext.jointProposalAppliedPositionChecksumValue() << "\n";
    out << "WebGPU joint proposal applied position: " << webgpuContext.jointProposalAppliedPositionMillis() << " ms\n";
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
    out << "WebGPU sphere contact relaxation: " << webgpuContext.sphereContactIterationRelaxationValue() << "\n";
    out << "WebGPU sphere contact iteration residual after max: " << webgpuContext.sphereContactIterationResidualAfterMaxValue() << "\n";
    out << "WebGPU sphere contact iteration residual after checksum: " << webgpuContext.sphereContactIterationResidualAfterChecksumValue() << "\n";
    out << "WebGPU sphere contact iteration: " << webgpuContext.sphereContactIterationMillis() << " ms\n";
    out << "WebGPU sphere contact final position ready: " << webgpuContext.sphereContactFinalPositionReadyValue() << "\n";
    out << "WebGPU sphere contact final position bodies: " << webgpuContext.sphereContactFinalPositionBodyCountValue() << "\n";
    out << "WebGPU sphere contact final position bytes: " << webgpuContext.sphereContactFinalPositionByteCount() << "\n";
    out << "WebGPU sphere contact final position source: " << webgpuContext.sphereContactFinalPositionSourceText() << "\n";
    out << "WebGPU sphere contact final position readback deferred: " << webgpuContext.sphereContactFinalPositionReadbackDeferredValue() << "\n";
    out << "WebGPU sphere contact async readback pending: " << webgpuContext.sphereContactFinalPositionAsyncReadbackPendingValue() << "\n";
    out << "WebGPU sphere contact async readback scheduled: " << webgpuContext.sphereContactFinalPositionAsyncReadbackScheduledValue() << "\n";
    out << "WebGPU sphere contact async readback consumed: " << webgpuContext.sphereContactFinalPositionAsyncReadbackConsumedValue() << "\n";
    out << "WebGPU sphere contact async readback dropped: " << webgpuContext.sphereContactFinalPositionAsyncReadbackDroppedValue() << "\n";
    out << "WebGPU sphere contact async readback wait: " << webgpuContext.sphereContactFinalPositionAsyncReadbackWaitMillis() << " ms\n";
    out << "WebGPU sphere contact async readback bodies: " << webgpuContext.sphereContactFinalPositionAsyncReadbackBodyCountValue() << "\n";
    out << "WebGPU sphere contact async readback bytes: " << webgpuContext.sphereContactFinalPositionAsyncReadbackByteCount() << "\n";
    out << "WebGPU sphere contact async readback source: " << webgpuContext.sphereContactFinalPositionAsyncReadbackSourceText() << "\n";
    out << "WebGPU sphere contact applied position bodies: " << webgpuContext.sphereContactAppliedPositionBodyCount() << "\n";
    out << "WebGPU sphere contact applied position readback bytes: " << webgpuContext.sphereContactAppliedPositionReadbackByteCount() << "\n";
    out << "WebGPU sphere contact applied position max delta: " << webgpuContext.sphereContactAppliedPositionMaxDeltaValue() << "\n";
    out << "WebGPU sphere contact applied position checksum: " << webgpuContext.sphereContactAppliedPositionChecksumValue() << "\n";
    out << "WebGPU sphere contact applied position: " << webgpuContext.sphereContactAppliedPositionMillis() << " ms\n";
    out << "WebGPU sphere contact applied position wait: " << webgpuContext.sphereContactAppliedPositionWaitMillis() << " ms\n";
    out << "WebGPU sphere contact applied position CPU: " << webgpuContext.sphereContactAppliedPositionCpuMillis() << " ms\n";
    out << "WebGPU sphere ground receivers: " << webgpuContext.sphereGroundReceiverCountValue() << "\n";
    out << "WebGPU ground dynamic bodies: " << webgpuContext.sphereGroundDynamicSphereCountValue() << "\n";
    out << "WebGPU sphere ground candidates: " << webgpuContext.sphereGroundCandidateCountValue() << "\n";
    out << "WebGPU direct sphere-cylinder bodies: " << webgpuContext.directSphereCylinderBodyCountValue() << "\n";
    out << "WebGPU direct sphere-cylinder candidates: " << webgpuContext.directSphereCylinderCandidateCountValue() << "\n";
    out << "WebGPU direct sphere-capsule bodies: " << webgpuContext.directSphereCapsuleBodyCountValue() << "\n";
    out << "WebGPU direct sphere-capsule candidates: " << webgpuContext.directSphereCapsuleCandidateCountValue() << "\n";
    out << "WebGPU direct sphere-box bodies: " << webgpuContext.directSphereBoxBodyCountValue() << "\n";
    out << "WebGPU direct sphere-box candidates: " << webgpuContext.directSphereBoxCandidateCountValue() << "\n";
    out << "WebGPU direct box bodies: " << webgpuContext.directBoxBodyCountValue() << "\n";
    out << "WebGPU direct box-box candidates: " << webgpuContext.directBoxPairCandidateCountValue() << "\n";
    out << "WebGPU direct sphere contact applied bodies: " << webgpuContext.directSphereContactAppliedPositionBodyCount() << "\n";
    out << "WebGPU direct ground applied bodies: " << webgpuContext.directGroundAppliedPositionBodyCount() << "\n";
    out << "WebGPU sphere ground top: " << webgpuContext.sphereGroundTopValue() << "\n";
    out << "WebGPU sphere ground: " << webgpuContext.sphereGroundMillis() << " ms\n";
    return out.str();
}

std::string combinedMetricsText(const SimulationHost &host, const ViewerBridge &bridge, int tickOverruns, const WebGpuContext &webgpuContext)
{
    std::ostringstream out;
    out << host.metricsText();
    out << webGpuRuntimeMetricsText(webgpuContext);
    out << bridgeMetricsText(bridge.statsSnapshot());
    out << "Tick overruns: " << tickOverruns << "\n";
    return out.str();
}

bool configurePhysicsBackend(SimulationHost &host, const char *physicsBackend, WebGpuContext &webgpuContext)
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
    webgpuContext.initializeDeviceOnly();
    if (webgpuContext.deviceReady)
        webgpuContext.runComputeSmokeTest();
    if (!webgpuContext.deviceReady)
    {
        std::fprintf(stderr, "WebGPU physics unavailable: %s\n", webgpuContext.statusText());
        host.solver().physicsBackend = makeCpuReferencePhysicsBackend();
        return false;
    }
    host.solver().physicsBackend = makeWebGpuAvbdPhysicsBackend(&webgpuContext);
    return true;
#else
    (void)webgpuContext;
    std::fprintf(stderr, "WebGPU physics unavailable: build does not include Dawn/WebGPU\n");
    host.solver().physicsBackend = makeCpuReferencePhysicsBackend();
    return false;
#endif
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

bool perturbFirstJointBody(SimulationHost &host, float amount)
{
    if (amount == 0.0f)
        return false;

    SimWorld &world = host.solver().world;
    for (ForceId forceId : world.jointIds)
    {
        if (forceId == INVALID_FORCE_ID || forceId >= world.constraints.size())
            continue;
        const SimConstraintData &constraint = world.constraints[forceId];
        if (!constraint.active || constraint.type != SIM_CONSTRAINT_JOINT)
            continue;

        BodyId candidates[2] = {constraint.bodyA, constraint.bodyB};
        for (BodyId bodyId : candidates)
        {
            if (bodyId == INVALID_BODY_ID || bodyId >= world.bodies.size())
                continue;
            SimBodyData &bodyData = world.bodies[bodyId];
            if (!bodyData.active || bodyData.mass <= 0.0f || !bodyData.source)
                continue;

            bodyData.source->positionLin.x += amount;
            bodyData.source->initialLin = bodyData.source->positionLin;
            bodyData.source->inertialLin = bodyData.source->positionLin;
            world.syncFromLegacy(host.solver());
            return true;
        }
    }
    return false;
}

int runBenchmark(const char *scene, int frames, int warmupFrames, bool resetAfterWarmup, bool noStream, uint16_t port, const char *physicsBackend, bool collisionOnly, int iterations, bool hasGravityOverride, float gravityOverride, bool gpuJointTopologyDiagnostic, int gpuJointTopologyRepeats, int gpuJointProposalIterations, bool gpuApplyJointProposals, float gpuJointPerturb)
{
    if (frames <= 0)
        frames = 600;
    if (warmupFrames < 0)
        warmupFrames = 0;

    SimulationHost host;
    WebGpuContext webgpuContext;
    if (scene && !host.loadSceneByName(scene))
    {
        std::fprintf(stderr, "Unknown benchmark scene '%s'\n", scene);
        return 2;
    }
    else if (!scene)
    {
        host.loadScene(host.sceneIndex());
    }
    configurePhysicsBackend(host, physicsBackend, webgpuContext);
    if (iterations > 0)
        host.solver().iterations = iterations;
    if (hasGravityOverride)
        host.solver().gravity = gravityOverride;
    host.solver().world.syncFromLegacy(host.solver());
    bool gpuJointPerturbApplied = perturbFirstJointBody(host, gpuJointPerturb);
    bool jointTopologyDiagnosticPassed = false;
    RunningStat jointTopologyRepeatMs;
    if (gpuJointTopologyDiagnostic)
    {
        if (gpuJointTopologyRepeats <= 0)
            gpuJointTopologyRepeats = 1;
        for (int repeat = 0; repeat < gpuJointTopologyRepeats; ++repeat)
        {
            bool repeatPassed = webgpuContext.runJointTopologyDiagnostic(host.solver().world, gpuJointProposalIterations);
            jointTopologyRepeatMs.add(webgpuContext.jointTopologyMillis());
            jointTopologyDiagnosticPassed = repeat == 0 ? repeatPassed : (jointTopologyDiagnosticPassed && repeatPassed);
        }
    }
    else
    {
        gpuJointTopologyRepeats = 0;
    }
    bool gpuApplyJointProposalsOk = false;
    if (gpuApplyJointProposals)
        gpuApplyJointProposalsOk = webgpuContext.applyJointProposalFinalPositions(host.solver());

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
    RunningStat webgpuRuntimeMs;
    RunningStat webgpuRuntimeSyncMs;
    RunningStat webgpuSapMs;
    RunningStat webgpuSapCounterReadbackMs;
    RunningStat webgpuSapReadbackMs;
    RunningStat webgpuSphereContactMs;
    RunningStat webgpuSphereContactReadbackMs;
    RunningStat webgpuSphereContactAdjacencyMs;
    RunningStat webgpuSphereContactGatherMs;
    RunningStat webgpuSphereContactProposalOutputMs;
    RunningStat webgpuSphereContactProposalResidualMs;
    RunningStat webgpuSphereContactIterationMs;
    RunningStat webgpuDirectSphereCylinderCandidates;
    RunningStat webgpuDirectSphereCapsuleCandidates;
    RunningStat webgpuDirectSphereBoxCandidates;
    RunningStat webgpuDirectBoxPairCandidates;
    RunningStat webgpuDirectRoundPairCandidates;
    RunningStat webgpuDirectGpuContactRecords;
    RunningStat webgpuDirectGpuRoundPairCandidates;
    RunningStat webgpuDirectGpuBoxPairCandidates;
    RunningStat webgpuDirectSphereAppliedBodies;
    RunningStat webgpuDirectGroundAppliedBodies;
    RunningStat serializeMs;
    RunningStat sendMs;
    uint64_t lastBytes = 0;

    using Clock = std::chrono::high_resolution_clock;
    for (int frame = 0; frame < warmupFrames; ++frame)
    {
        if (collisionOnly)
        {
            const char *backendName = host.solver().physicsBackend ? host.solver().physicsBackend->name() : "";
            if (std::strstr(backendName, "WebGPU") && webgpuContext.deviceReady)
            {
                std::vector<BroadphasePair> ignoredPairs;
                webgpuContext.runSweepAndPrunePairs(host.solver().world, ignoredPairs, 0, true,
                                                    !environmentFlagEnabled("AVBD_GPU_CONTACT_SOLVE_NO_READBACK"),
                                                    false, false, false, false);
            }
            else
            {
                host.solver().benchmarkBroadphaseOnly();
            }
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
            const char *backendName = host.solver().physicsBackend ? host.solver().physicsBackend->name() : "";
            if (std::strstr(backendName, "WebGPU") && webgpuContext.deviceReady)
            {
                std::vector<BroadphasePair> ignoredPairs;
                webgpuContext.runSweepAndPrunePairs(host.solver().world, ignoredPairs, 0, true,
                                                    !environmentFlagEnabled("AVBD_GPU_CONTACT_SOLVE_NO_READBACK"),
                                                    false, false, false, false);
            }
            else
            {
                host.solver().benchmarkBroadphaseOnly();
            }
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
        bodyInitMs.add(stats.bodyInitMs);
        primalMs.add(stats.primalSolveMs);
        dualMs.add(stats.dualUpdateMs);
        velocityUpdateMs.add(stats.velocityUpdateMs);
        webgpuRuntimeMs.add(webgpuContext.runtimeTotalMillis());
        webgpuRuntimeSyncMs.add(webgpuContext.runtimeSyncMillis());
        webgpuSapMs.add(webgpuContext.sapMillis());
        webgpuSapCounterReadbackMs.add(webgpuContext.sapCounterReadbackMillis());
        webgpuSapReadbackMs.add(webgpuContext.sapPairReadbackMillis());
        webgpuSphereContactMs.add(webgpuContext.sphereContactMillis());
        webgpuSphereContactReadbackMs.add(webgpuContext.sphereContactReadbackMillis());
        webgpuSphereContactAdjacencyMs.add(webgpuContext.sphereContactAdjacencyMillis());
        webgpuSphereContactGatherMs.add(webgpuContext.sphereContactGatherMillis());
        webgpuSphereContactProposalOutputMs.add(webgpuContext.sphereContactProposalOutputMillis());
        webgpuSphereContactProposalResidualMs.add(webgpuContext.sphereContactProposalResidualMillis());
        webgpuSphereContactIterationMs.add(webgpuContext.sphereContactIterationMillis());
        webgpuDirectSphereCylinderCandidates.add((double)webgpuContext.directSphereCylinderCandidateCountValue());
        webgpuDirectSphereCapsuleCandidates.add((double)webgpuContext.directSphereCapsuleCandidateCountValue());
        webgpuDirectSphereBoxCandidates.add((double)webgpuContext.directSphereBoxCandidateCountValue());
        webgpuDirectBoxPairCandidates.add((double)webgpuContext.directBoxPairCandidateCountValue());
        webgpuDirectRoundPairCandidates.add((double)webgpuContext.directRoundPairCandidateCountValue());
        webgpuDirectGpuContactRecords.add((double)webgpuContext.directGpuContactRecordCountValue());
        webgpuDirectGpuRoundPairCandidates.add((double)webgpuContext.directGpuRoundPairCandidateCountValue());
        webgpuDirectGpuBoxPairCandidates.add((double)webgpuContext.directGpuBoxPairCandidateCountValue());
        webgpuDirectSphereAppliedBodies.add((double)webgpuContext.directSphereContactAppliedPositionBodyCount());
        webgpuDirectGroundAppliedBodies.add((double)webgpuContext.directGroundAppliedPositionBodyCount());
        serializeMs.add(bridgeStats.lastSerializeMs);
        sendMs.add(bridgeStats.lastSendMs);
        lastBytes = bridgeStats.lastSnapshotBytes;
    }
    double totalMs = std::chrono::duration<double, std::milli>(Clock::now() - totalBegin).count();
    ViewerBridgeStats bridgeStats = bridge.statsSnapshot();
    if (!noStream)
        bridge.stop();
    if (webgpuContext.sphereContactFinalPositionAsyncReadbackPendingValue())
        webgpuContext.consumePendingSphereContactFinalPositions(host.solver());
    if (webgpuContext.runSphereGroundCorrection(host.world(), false) &&
        webgpuContext.sphereGroundCandidateCountValue() > 0)
        webgpuContext.applySphereContactFinalPositions(host.solver());
    SphereGroundClearance sphereGroundClearance = measureSphereGroundClearance(host.world());
    FinalStateSignature finalState = measureFinalStateSignature(host.world());

    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{";
    out << "\"type\":\"headlessBenchmark\",";
    out << "\"scene\":\"" << host.currentSceneName() << "\",";
    out << "\"physicsBackend\":\"" << host.solver().physicsBackend->name() << "\",";
    out << "\"webgpuSphereContactsEnabled\":" << (environmentFlagEnabled("AVBD_GPU_SPHERE_CONTACTS") ? "true" : "false") << ",";
    out << "\"webgpuGroundContactFeed\":" << (environmentFlagEnabled("AVBD_GPU_GROUND_CONTACT_FEED") ? "true" : "false") << ",";
    out << "\"webgpuContactSolveDiagnostic\":" << (environmentFlagEnabled("AVBD_GPU_CONTACT_SOLVE_DIAGNOSTIC") ? "true" : "false") << ",";
    out << "\"webgpuJointTopologyDiagnostic\":" << (gpuJointTopologyDiagnostic ? "true" : "false") << ",";
    out << "\"webgpuJointTopologyDiagnosticPassed\":" << (jointTopologyDiagnosticPassed ? "true" : "false") << ",";
    out << "\"webgpuJointTopologyRepeats\":" << gpuJointTopologyRepeats << ",";
    out << "\"webgpuJointPerturb\":" << gpuJointPerturb << ",";
    out << "\"webgpuJointPerturbApplied\":" << (gpuJointPerturbApplied ? "true" : "false") << ",";
    out << "\"webgpuApplyJointProposals\":" << (gpuApplyJointProposals ? "true" : "false") << ",";
    out << "\"webgpuApplyJointProposalsOk\":" << (gpuApplyJointProposalsOk ? "true" : "false") << ",";
    const char *backendName = host.solver().physicsBackend->name();
    bool backendFastMode = std::strstr(backendName, "Fast") != 0;
    bool backendContactResidentAsyncMode = std::strstr(backendName, "Contact Resident Async") != 0;
    bool backendJointProposalMode = std::strstr(backendName, "Joint Proposal") != 0 ||
                                    std::strstr(backendName, "Joint Replace") != 0 ||
                                    std::strstr(backendName, "Joint Direct") != 0 ||
                                    std::strstr(backendName, "Joint Contact Direct") != 0;
    bool backendDirectJointMode = std::strstr(backendName, "Joint Direct") != 0 ||
                                  std::strstr(backendName, "Joint Contact Direct") != 0;
    out << "\"webgpuContactSolveNoReadback\":" << ((environmentFlagEnabled("AVBD_GPU_CONTACT_SOLVE_NO_READBACK") || backendFastMode || backendContactResidentAsyncMode) ? "true" : "false") << ",";
    out << "\"webgpuDirectCounterReadback\":" << (environmentFlagEnabled("AVBD_GPU_DIRECT_COUNTER_READBACK") ? "true" : "false") << ",";
    out << "\"webgpuRuntimeValidate\":" << (environmentFlagEnabled("AVBD_GPU_RUNTIME_VALIDATE") ? "true" : "false") << ",";
    out << "\"webgpuApplyPrediction\":" << (environmentFlagEnabled("AVBD_GPU_APPLY_PREDICTION") ? "true" : "false") << ",";
    out << "\"webgpuApplyVelocity\":" << (environmentFlagEnabled("AVBD_GPU_APPLY_VELOCITY") ? "true" : "false") << ",";
    out << "\"webgpuResidentCounterlessContacts\":" << ((environmentFlagEnabled("AVBD_GPU_RESIDENT_COUNTERLESS_CONTACTS") || backendContactResidentAsyncMode) ? "true" : "false") << ",";
    out << "\"webgpuAsyncFinalPositionReadback\":" << ((environmentFlagEnabled("AVBD_GPU_ASYNC_FINAL_POSITION_READBACK") || backendContactResidentAsyncMode) ? "true" : "false") << ",";
    out << "\"webgpuApplyContactPositions\":" << ((environmentFlagEnabled("AVBD_GPU_APPLY_CONTACT_POSITIONS") || backendFastMode) ? "true" : "false") << ",";
    out << "\"webgpuGroundContacts\":" << ((environmentFlagEnabled("AVBD_GPU_GROUND_CONTACTS") || backendFastMode) ? "true" : "false") << ",";
    out << "\"webgpuRuntimeJointProposals\":" << ((environmentFlagEnabled("AVBD_GPU_JOINT_PROPOSALS") || backendJointProposalMode) ? "true" : "false") << ",";
    out << "\"webgpuReplaceCpuJoints\":" << ((environmentFlagEnabled("AVBD_GPU_REPLACE_CPU_JOINTS") || std::strstr(backendName, "Joint Replace") != 0) ? "true" : "false") << ",";
    out << "\"webgpuDirectJointSolve\":" << ((environmentFlagEnabled("AVBD_GPU_JOINT_DIRECT") || backendDirectJointMode) ? "true" : "false") << ",";
    out << "\"webgpuDirectContactSolve\":" << ((environmentFlagEnabled("AVBD_GPU_CONTACT_DIRECT") ||
                                                std::strstr(backendName, "Contact Direct") != 0 ||
                                                std::strstr(backendName, "Contact Resident") != 0)
                                                   ? "true"
                                                   : "false")
        << ",";
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
    out << "\"bodyInitAvgMs\":" << bodyInitMs.avg() << ",";
    out << "\"primalAvgMs\":" << primalMs.avg() << ",";
    out << "\"dualAvgMs\":" << dualMs.avg() << ",";
    out << "\"velocityUpdateAvgMs\":" << velocityUpdateMs.avg() << ",";
    out << "\"webgpuRuntimeAvgMs\":" << webgpuRuntimeMs.avg() << ",";
    out << "\"webgpuRuntimeSyncAvgMs\":" << webgpuRuntimeSyncMs.avg() << ",";
    out << "\"webgpuStatus\":" << jsonString(webgpuContext.statusText()) << ",";
    out << "\"webgpuRuntimeStatus\":" << jsonString(webgpuContext.runtimeStatusText()) << ",";
    out << "\"webgpuSapStatus\":" << jsonString(webgpuContext.sapStatusText()) << ",";
    out << "\"webgpuRuntimeFrames\":" << webgpuContext.runtimeFrameCount() << ",";
    out << "\"webgpuRuntimeFallbacks\":" << webgpuContext.runtimeFallbackCount() << ",";
    out << "\"webgpuPredictionAppliedBodies\":" << webgpuContext.predictionAppliedBodyCount() << ",";
    out << "\"webgpuPredictionAppliedReadbackBytes\":" << webgpuContext.predictionAppliedReadbackByteCount() << ",";
    out << "\"webgpuPredictionAppliedMs\":" << webgpuContext.predictionAppliedMillis() << ",";
    out << "\"webgpuVelocityAppliedBodies\":" << webgpuContext.velocityAppliedBodyCount() << ",";
    out << "\"webgpuVelocityAppliedReadbackBytes\":" << webgpuContext.velocityAppliedReadbackByteCount() << ",";
    out << "\"webgpuVelocityAppliedMs\":" << webgpuContext.velocityAppliedMillis() << ",";
    out << "\"webgpuRuntimeMaxLinearError\":" << webgpuContext.runtimeMaxLinearErrorValue() << ",";
    out << "\"webgpuRuntimeMaxAngularError\":" << webgpuContext.runtimeMaxAngularErrorValue() << ",";
    out << "\"webgpuSapCandidates\":" << webgpuContext.sapCandidateCount() << ",";
    out << "\"webgpuSapSphereHits\":" << webgpuContext.sapSphereHitCount() << ",";
    out << "\"webgpuSapAvgMs\":" << webgpuSapMs.avg() << ",";
    out << "\"webgpuSapCounterReadbackAvgMs\":" << webgpuSapCounterReadbackMs.avg() << ",";
    out << "\"webgpuSapCounterReadbackBytes\":" << webgpuContext.sapCounterReadbackByteCount() << ",";
    out << "\"webgpuSapPairReadbackAvgMs\":" << webgpuSapReadbackMs.avg() << ",";
    out << "\"webgpuSapPairReadbackBytes\":" << webgpuContext.sapPairReadbackByteCount() << ",";
    out << "\"webgpuJointTopologyStatus\":\"" << webgpuContext.jointTopologyStatusText() << "\",";
    out << "\"webgpuJointTopologyMs\":" << webgpuContext.jointTopologyMillis() << ",";
    out << "\"webgpuJointTopologyAvgMs\":" << jointTopologyRepeatMs.avg() << ",";
    out << "\"webgpuJointTopologyMinMs\":" << jointTopologyRepeatMs.min << ",";
    out << "\"webgpuJointTopologyMaxMs\":" << jointTopologyRepeatMs.max << ",";
    out << "\"webgpuJointTopologyJoints\":" << webgpuContext.jointTopologyJointCount() << ",";
    out << "\"webgpuJointTopologyBodyRefs\":" << webgpuContext.jointTopologyBodyRefCount() << ",";
    out << "\"webgpuJointTopologyActiveBodies\":" << webgpuContext.jointTopologyActiveBodyCount() << ",";
    out << "\"webgpuJointTopologyMaxPerBody\":" << webgpuContext.jointTopologyMaxPerBodyCount() << ",";
    out << "\"webgpuJointTopologyMismatches\":" << webgpuContext.jointTopologyMismatchCount() << ",";
    out << "\"webgpuJointTopologyReadbackBytes\":" << webgpuContext.jointTopologyReadbackByteCount() << ",";
    out << "\"webgpuJointColorCount\":" << webgpuContext.jointColorCountValue() << ",";
    out << "\"webgpuJointColorConflicts\":" << webgpuContext.jointColorConflictCount() << ",";
    out << "\"webgpuJointColorMinBucket\":" << webgpuContext.jointColorMinBucketCount() << ",";
    out << "\"webgpuJointColorMaxBucket\":" << webgpuContext.jointColorMaxBucketCount() << ",";
    out << "\"webgpuJointColorReadbackBytes\":" << webgpuContext.jointColorReadbackByteCount() << ",";
    out << "\"webgpuJointResidualMax\":" << webgpuContext.jointResidualMaxValue() << ",";
    out << "\"webgpuJointResidualRms\":" << webgpuContext.jointResidualRmsValue() << ",";
    out << "\"webgpuJointResidualReadbackBytes\":" << webgpuContext.jointResidualReadbackByteCount() << ",";
    out << "\"webgpuJointProposalMaxCorrection\":" << webgpuContext.jointProposalMaxCorrectionValue() << ",";
    out << "\"webgpuJointProposalRmsCorrection\":" << webgpuContext.jointProposalRmsCorrectionValue() << ",";
    out << "\"webgpuJointProposalActiveBodies\":" << webgpuContext.jointProposalActiveBodyCount() << ",";
    out << "\"webgpuJointProposalMaxPerBody\":" << webgpuContext.jointProposalMaxPerBodyCount() << ",";
    out << "\"webgpuJointProposalReadbackBytes\":" << webgpuContext.jointProposalReadbackByteCount() << ",";
    out << "\"webgpuJointProposalIterations\":" << webgpuContext.jointProposalIterationCount() << ",";
    out << "\"webgpuJointProposalResidualAfterMax\":" << webgpuContext.jointProposalResidualAfterMaxValue() << ",";
    out << "\"webgpuJointProposalResidualAfterRms\":" << webgpuContext.jointProposalResidualAfterRmsValue() << ",";
    out << "\"webgpuJointProposalResidualReadbackBytes\":" << webgpuContext.jointProposalResidualReadbackByteCount() << ",";
    out << "\"webgpuJointProposalFinalPositionReady\":" << webgpuContext.jointProposalFinalPositionReadyValue() << ",";
    out << "\"webgpuJointProposalFinalPositionBodies\":" << webgpuContext.jointProposalFinalPositionBodyCountValue() << ",";
    out << "\"webgpuJointProposalFinalPositionBytes\":" << webgpuContext.jointProposalFinalPositionByteCount() << ",";
    out << "\"webgpuJointProposalFinalPositionAbsolute\":" << webgpuContext.jointProposalFinalPositionAbsoluteValue() << ",";
    out << "\"webgpuJointProposalSeededFromContact\":" << webgpuContext.jointProposalSeededFromContactValue() << ",";
    out << "\"webgpuJointProposalAppliedPositionBodies\":" << webgpuContext.jointProposalAppliedPositionBodyCount() << ",";
    out << "\"webgpuJointProposalAppliedPositionReadbackBytes\":" << webgpuContext.jointProposalAppliedPositionReadbackByteCount() << ",";
    out << "\"webgpuJointProposalAppliedPositionMaxDelta\":" << webgpuContext.jointProposalAppliedPositionMaxDeltaValue() << ",";
    out << "\"webgpuJointProposalAppliedPositionChecksum\":" << webgpuContext.jointProposalAppliedPositionChecksumValue() << ",";
    out << "\"webgpuJointProposalAppliedPositionMs\":" << webgpuContext.jointProposalAppliedPositionMillis() << ",";
    out << "\"webgpuJointProposalFinalPositionAsyncReadbackPending\":" << webgpuContext.jointProposalFinalPositionAsyncReadbackPendingValue() << ",";
    out << "\"webgpuJointProposalFinalPositionAsyncReadbackScheduled\":" << webgpuContext.jointProposalFinalPositionAsyncReadbackScheduledValue() << ",";
    out << "\"webgpuJointProposalFinalPositionAsyncReadbackConsumed\":" << webgpuContext.jointProposalFinalPositionAsyncReadbackConsumedValue() << ",";
    out << "\"webgpuJointProposalFinalPositionAsyncReadbackDropped\":" << webgpuContext.jointProposalFinalPositionAsyncReadbackDroppedValue() << ",";
    out << "\"webgpuJointProposalFinalPositionAsyncReadbackWaitMs\":" << webgpuContext.jointProposalFinalPositionAsyncReadbackWaitMillis() << ",";
    out << "\"webgpuJointProposalFinalPositionAsyncReadbackBodies\":" << webgpuContext.jointProposalFinalPositionAsyncReadbackBodyCountValue() << ",";
    out << "\"webgpuJointProposalFinalPositionAsyncReadbackBytes\":" << webgpuContext.jointProposalFinalPositionAsyncReadbackByteCount() << ",";
    out << "\"webgpuJointProposalFinalPositionAsyncReadbackAbsolute\":" << webgpuContext.jointProposalFinalPositionAsyncReadbackAbsoluteValue() << ",";
    out << "\"webgpuSphereContacts\":" << webgpuContext.sphereContactCountValue() << ",";
    out << "\"webgpuExternalContacts\":" << webgpuContext.sphereContactExternalContactCount() << ",";
    out << "\"webgpuExternalGroundContacts\":" << webgpuContext.sphereContactExternalGroundContactCount() << ",";
    out << "\"webgpuSphereContactAvgMs\":" << webgpuSphereContactMs.avg() << ",";
    out << "\"webgpuSphereContactReadbackAvgMs\":" << webgpuSphereContactReadbackMs.avg() << ",";
    out << "\"webgpuSphereContactReadbackBytes\":" << webgpuContext.sphereContactReadbackByteCount() << ",";
    out << "\"webgpuSphereContactBodyRefs\":" << webgpuContext.sphereContactBodyRefCount() << ",";
    out << "\"webgpuSphereContactActiveBodies\":" << webgpuContext.sphereContactActiveBodyCount() << ",";
    out << "\"webgpuSphereContactMaxPerBody\":" << webgpuContext.sphereContactMaxPerBodyCount() << ",";
    out << "\"webgpuSphereContactAvgPerActiveBody\":" << webgpuContext.sphereContactAvgPerActiveBodyValue() << ",";
    out << "\"webgpuSphereContactAdjacencyAvgMs\":" << webgpuSphereContactAdjacencyMs.avg() << ",";
    out << "\"webgpuSphereContactAdjacencyReadbackBytes\":" << webgpuContext.sphereContactAdjacencyReadbackByteCount() << ",";
    out << "\"webgpuSphereContactAdjacencyBufferBytes\":" << webgpuContext.sphereContactAdjacencyBufferByteCount() << ",";
    out << "\"webgpuSphereContactAdjacencyCapacity\":" << webgpuContext.sphereContactAdjacencyCapacityValue() << ",";
    out << "\"webgpuSphereContactAdjacencyWrittenRefs\":" << webgpuContext.sphereContactAdjacencyWrittenRefCount() << ",";
    out << "\"webgpuSphereContactAdjacencyOverflowRefs\":" << webgpuContext.sphereContactAdjacencyOverflowRefCount() << ",";
    out << "\"webgpuSphereContactGatherAvgMs\":" << webgpuSphereContactGatherMs.avg() << ",";
    out << "\"webgpuSphereContactGatherRefs\":" << webgpuContext.sphereContactGatherRefCount() << ",";
    out << "\"webgpuSphereContactGatherActiveBodies\":" << webgpuContext.sphereContactGatherActiveBodyCount() << ",";
    out << "\"webgpuSphereContactGatherMaxPerBody\":" << webgpuContext.sphereContactGatherMaxPerBodyCount() << ",";
    out << "\"webgpuSphereContactGatherMismatches\":" << webgpuContext.sphereContactGatherMismatchCount() << ",";
    out << "\"webgpuSphereContactGatherReadbackBytes\":" << webgpuContext.sphereContactGatherReadbackByteCount() << ",";
    out << "\"webgpuSphereContactGatherNormalChecksum\":" << webgpuContext.sphereContactGatherNormalChecksumValue() << ",";
    out << "\"webgpuSphereContactProposalActiveBodies\":" << webgpuContext.sphereContactProposalActiveBodyCount() << ",";
    out << "\"webgpuSphereContactProposalMaxCorrection\":" << webgpuContext.sphereContactProposalMaxCorrectionValue() << ",";
    out << "\"webgpuSphereContactProposalCorrectionChecksum\":" << webgpuContext.sphereContactProposalCorrectionChecksumValue() << ",";
    out << "\"webgpuSphereContactProposalOutputAvgMs\":" << webgpuSphereContactProposalOutputMs.avg() << ",";
    out << "\"webgpuSphereContactProposalOutputActiveBodies\":" << webgpuContext.sphereContactProposalOutputActiveBodyCount() << ",";
    out << "\"webgpuSphereContactProposalOutputReadbackBytes\":" << webgpuContext.sphereContactProposalOutputReadbackByteCount() << ",";
    out << "\"webgpuSphereContactProposalOutputMaxDelta\":" << webgpuContext.sphereContactProposalOutputMaxDeltaValue() << ",";
    out << "\"webgpuSphereContactProposalOutputChecksum\":" << webgpuContext.sphereContactProposalOutputChecksumValue() << ",";
    out << "\"webgpuSphereContactProposalResidualAvgMs\":" << webgpuSphereContactProposalResidualMs.avg() << ",";
    out << "\"webgpuSphereContactProposalResidualReadbackBytes\":" << webgpuContext.sphereContactProposalResidualReadbackByteCount() << ",";
    out << "\"webgpuSphereContactProposalResidualBeforeMax\":" << webgpuContext.sphereContactProposalResidualBeforeMaxValue() << ",";
    out << "\"webgpuSphereContactProposalResidualAfterMax\":" << webgpuContext.sphereContactProposalResidualAfterMaxValue() << ",";
    out << "\"webgpuSphereContactProposalResidualBeforeChecksum\":" << webgpuContext.sphereContactProposalResidualBeforeChecksumValue() << ",";
    out << "\"webgpuSphereContactProposalResidualAfterChecksum\":" << webgpuContext.sphereContactProposalResidualAfterChecksumValue() << ",";
    out << "\"webgpuSphereContactIterations\":" << webgpuContext.sphereContactIterationCountValue() << ",";
    out << "\"webgpuSphereContactRelaxation\":" << webgpuContext.sphereContactIterationRelaxationValue() << ",";
    out << "\"webgpuSphereContactIterationAvgMs\":" << webgpuSphereContactIterationMs.avg() << ",";
    out << "\"webgpuSphereContactIterationResidualAfterMax\":" << webgpuContext.sphereContactIterationResidualAfterMaxValue() << ",";
    out << "\"webgpuSphereContactIterationResidualAfterChecksum\":" << webgpuContext.sphereContactIterationResidualAfterChecksumValue() << ",";
    out << "\"webgpuSphereContactFinalPositionReady\":" << webgpuContext.sphereContactFinalPositionReadyValue() << ",";
    out << "\"webgpuSphereContactFinalPositionBodies\":" << webgpuContext.sphereContactFinalPositionBodyCountValue() << ",";
    out << "\"webgpuSphereContactFinalPositionBytes\":" << webgpuContext.sphereContactFinalPositionByteCount() << ",";
    out << "\"webgpuSphereContactFinalPositionSource\":\"" << webgpuContext.sphereContactFinalPositionSourceText() << "\",";
    out << "\"webgpuSphereContactFinalPositionReadbackDeferred\":" << webgpuContext.sphereContactFinalPositionReadbackDeferredValue() << ",";
    out << "\"webgpuSphereContactFinalPositionAsyncReadbackPending\":" << webgpuContext.sphereContactFinalPositionAsyncReadbackPendingValue() << ",";
    out << "\"webgpuSphereContactFinalPositionAsyncReadbackScheduled\":" << webgpuContext.sphereContactFinalPositionAsyncReadbackScheduledValue() << ",";
    out << "\"webgpuSphereContactFinalPositionAsyncReadbackConsumed\":" << webgpuContext.sphereContactFinalPositionAsyncReadbackConsumedValue() << ",";
    out << "\"webgpuSphereContactFinalPositionAsyncReadbackDropped\":" << webgpuContext.sphereContactFinalPositionAsyncReadbackDroppedValue() << ",";
    out << "\"webgpuSphereContactFinalPositionAsyncReadbackWaitMs\":" << webgpuContext.sphereContactFinalPositionAsyncReadbackWaitMillis() << ",";
    out << "\"webgpuSphereContactFinalPositionAsyncReadbackBodies\":" << webgpuContext.sphereContactFinalPositionAsyncReadbackBodyCountValue() << ",";
    out << "\"webgpuSphereContactFinalPositionAsyncReadbackBytes\":" << webgpuContext.sphereContactFinalPositionAsyncReadbackByteCount() << ",";
    out << "\"webgpuSphereContactFinalPositionAsyncReadbackSource\":\"" << webgpuContext.sphereContactFinalPositionAsyncReadbackSourceText() << "\",";
    out << "\"webgpuSphereContactAppliedPositionBodies\":" << webgpuContext.sphereContactAppliedPositionBodyCount() << ",";
    out << "\"webgpuSphereContactAppliedPositionReadbackBytes\":" << webgpuContext.sphereContactAppliedPositionReadbackByteCount() << ",";
    out << "\"webgpuSphereContactAppliedPositionMaxDelta\":" << webgpuContext.sphereContactAppliedPositionMaxDeltaValue() << ",";
    out << "\"webgpuSphereContactAppliedPositionChecksum\":" << webgpuContext.sphereContactAppliedPositionChecksumValue() << ",";
    out << "\"webgpuSphereContactAppliedPositionMs\":" << webgpuContext.sphereContactAppliedPositionMillis() << ",";
    out << "\"webgpuSphereContactAppliedPositionWaitMs\":" << webgpuContext.sphereContactAppliedPositionWaitMillis() << ",";
    out << "\"webgpuSphereContactAppliedPositionCpuMs\":" << webgpuContext.sphereContactAppliedPositionCpuMillis() << ",";
    out << "\"webgpuSphereGroundReceivers\":" << webgpuContext.sphereGroundReceiverCountValue() << ",";
    out << "\"webgpuSphereGroundDynamicSpheres\":" << webgpuContext.sphereGroundDynamicSphereCountValue() << ",";
    out << "\"webgpuGroundDynamicBodies\":" << webgpuContext.sphereGroundDynamicSphereCountValue() << ",";
    out << "\"webgpuSphereGroundCandidates\":" << webgpuContext.sphereGroundCandidateCountValue() << ",";
    out << "\"webgpuGroundCandidates\":" << webgpuContext.sphereGroundCandidateCountValue() << ",";
    out << "\"webgpuDirectSphereCylinderBodies\":" << webgpuContext.directSphereCylinderBodyCountValue() << ",";
    out << "\"webgpuDirectSphereCylinderCandidates\":" << webgpuContext.directSphereCylinderCandidateCountValue() << ",";
    out << "\"webgpuDirectSphereCylinderCandidatesMax\":" << webgpuDirectSphereCylinderCandidates.max << ",";
    out << "\"webgpuDirectSphereCylinderCandidatesAvg\":" << webgpuDirectSphereCylinderCandidates.avg() << ",";
    out << "\"webgpuDirectSphereCapsuleBodies\":" << webgpuContext.directSphereCapsuleBodyCountValue() << ",";
    out << "\"webgpuDirectSphereCapsuleCandidates\":" << webgpuContext.directSphereCapsuleCandidateCountValue() << ",";
    out << "\"webgpuDirectSphereCapsuleCandidatesMax\":" << webgpuDirectSphereCapsuleCandidates.max << ",";
    out << "\"webgpuDirectSphereCapsuleCandidatesAvg\":" << webgpuDirectSphereCapsuleCandidates.avg() << ",";
    out << "\"webgpuDirectSphereBoxBodies\":" << webgpuContext.directSphereBoxBodyCountValue() << ",";
    out << "\"webgpuDirectSphereBoxCandidates\":" << webgpuContext.directSphereBoxCandidateCountValue() << ",";
    out << "\"webgpuDirectSphereBoxCandidatesMax\":" << webgpuDirectSphereBoxCandidates.max << ",";
    out << "\"webgpuDirectSphereBoxCandidatesAvg\":" << webgpuDirectSphereBoxCandidates.avg() << ",";
    out << "\"webgpuDirectBoxBodies\":" << webgpuContext.directBoxBodyCountValue() << ",";
    out << "\"webgpuDirectBoxPairCandidates\":" << webgpuContext.directBoxPairCandidateCountValue() << ",";
    out << "\"webgpuDirectBoxPairCandidatesMax\":" << webgpuDirectBoxPairCandidates.max << ",";
    out << "\"webgpuDirectBoxPairCandidatesAvg\":" << webgpuDirectBoxPairCandidates.avg() << ",";
    out << "\"webgpuDirectRoundBodies\":" << webgpuContext.directRoundBodyCountValue() << ",";
    out << "\"webgpuDirectRoundPairCandidates\":" << webgpuContext.directRoundPairCandidateCountValue() << ",";
    out << "\"webgpuDirectRoundPairCandidatesMax\":" << webgpuDirectRoundPairCandidates.max << ",";
    out << "\"webgpuDirectRoundPairCandidatesAvg\":" << webgpuDirectRoundPairCandidates.avg() << ",";
    out << "\"webgpuDirectGpuContactRecords\":" << webgpuContext.directGpuContactRecordCountValue() << ",";
    out << "\"webgpuDirectGpuContactRecordsMax\":" << webgpuDirectGpuContactRecords.max << ",";
    out << "\"webgpuDirectGpuContactRecordsAvg\":" << webgpuDirectGpuContactRecords.avg() << ",";
    out << "\"webgpuDirectGpuRoundPairCandidates\":" << webgpuContext.directGpuRoundPairCandidateCountValue() << ",";
    out << "\"webgpuDirectGpuRoundPairCandidatesMax\":" << webgpuDirectGpuRoundPairCandidates.max << ",";
    out << "\"webgpuDirectGpuRoundPairCandidatesAvg\":" << webgpuDirectGpuRoundPairCandidates.avg() << ",";
    out << "\"webgpuDirectGpuBoxPairCandidates\":" << webgpuContext.directGpuBoxPairCandidateCountValue() << ",";
    out << "\"webgpuDirectGpuBoxPairCandidatesMax\":" << webgpuDirectGpuBoxPairCandidates.max << ",";
    out << "\"webgpuDirectGpuBoxPairCandidatesAvg\":" << webgpuDirectGpuBoxPairCandidates.avg() << ",";
    out << "\"webgpuDirectGpuCounterReadbackBytes\":" << webgpuContext.directGpuCounterReadbackByteCount() << ",";
    out << "\"webgpuDirectGpuCounterReadbackMs\":" << webgpuContext.directGpuCounterReadbackMillis() << ",";
    out << "\"webgpuDirectSphereContactAppliedPositionBodies\":" << webgpuContext.directSphereContactAppliedPositionBodyCount() << ",";
    out << "\"webgpuDirectSphereContactAppliedPositionBodiesMax\":" << webgpuDirectSphereAppliedBodies.max << ",";
    out << "\"webgpuDirectSphereContactAppliedPositionBodiesAvg\":" << webgpuDirectSphereAppliedBodies.avg() << ",";
    out << "\"webgpuDirectRoundContactAppliedPositionBodies\":" << webgpuContext.directSphereContactAppliedPositionBodyCount() << ",";
    out << "\"webgpuDirectRoundContactAppliedPositionBodiesMax\":" << webgpuDirectSphereAppliedBodies.max << ",";
    out << "\"webgpuDirectRoundContactAppliedPositionBodiesAvg\":" << webgpuDirectSphereAppliedBodies.avg() << ",";
    out << "\"webgpuDirectGroundAppliedPositionBodies\":" << webgpuContext.directGroundAppliedPositionBodyCount() << ",";
    out << "\"webgpuDirectGroundAppliedPositionBodiesMax\":" << webgpuDirectGroundAppliedBodies.max << ",";
    out << "\"webgpuDirectGroundAppliedPositionBodiesAvg\":" << webgpuDirectGroundAppliedBodies.avg() << ",";
    out << "\"webgpuSphereGroundTop\":" << webgpuContext.sphereGroundTopValue() << ",";
    out << "\"webgpuSphereGroundMs\":" << webgpuContext.sphereGroundMillis() << ",";
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
    WebGpuContext candidateWebGpuContext;

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
    if (!configurePhysicsBackend(candidateHost, candidateBackend, candidateWebGpuContext))
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
    out << "\"candidateWebgpuRuntimeAvgMs\":" << candidateWebGpuContext.runtimeTotalMillis() << ",";
    out << "\"candidateWebgpuRuntimeFrames\":" << candidateWebGpuContext.runtimeFrameCount() << ",";
    out << "\"candidateWebgpuRuntimeFallbacks\":" << candidateWebGpuContext.runtimeFallbackCount() << ",";
    out << "\"candidateWebgpuStatus\":" << jsonString(candidateWebGpuContext.statusText()) << ",";
    out << "\"candidateWebgpuRuntimeStatus\":" << jsonString(candidateWebGpuContext.runtimeStatusText()) << ",";
    out << "\"candidateWebgpuSapStatus\":" << jsonString(candidateWebGpuContext.sapStatusText()) << ",";
    out << "\"candidateWebgpuJointTopologyStatus\":" << jsonString(candidateWebGpuContext.jointTopologyStatusText()) << ",";
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
    bool gpuContactSolveDiagnostic = hasFlag(argc, argv, "--gpu-contact-solve-diagnostic");
    bool gpuContactSolveNoReadback = hasFlag(argc, argv, "--gpu-contact-solve-no-readback");
    bool gpuDirectCounterReadback = hasFlag(argc, argv, "--gpu-direct-counter-readback");
    bool gpuRuntimeValidate = hasFlag(argc, argv, "--gpu-runtime-validate");
    bool gpuApplyPrediction = hasFlag(argc, argv, "--gpu-apply-prediction");
    bool gpuApplyVelocity = hasFlag(argc, argv, "--gpu-apply-velocity");
    bool gpuResidentCounterlessContacts = hasFlag(argc, argv, "--gpu-resident-counterless-contacts");
    bool gpuDeferFinalPositionReadback = hasFlag(argc, argv, "--gpu-defer-final-position-readback");
    bool gpuAsyncFinalPositionReadback = hasFlag(argc, argv, "--gpu-async-final-position-readback");
    bool gpuApplyContactPositions = hasFlag(argc, argv, "--gpu-apply-contact-positions");
    bool gpuGroundContacts = hasFlag(argc, argv, "--gpu-ground-contacts");
    bool gpuSphereContacts = hasFlag(argc, argv, "--gpu-sphere-contacts");
    bool gpuGroundContactFeed = hasFlag(argc, argv, "--gpu-ground-contact-feed");
    bool gpuJointTopologyDiagnostic = hasFlag(argc, argv, "--gpu-joint-topology-diagnostic");
    bool gpuApplyJointProposals = hasFlag(argc, argv, "--gpu-apply-joint-proposals");
    bool gpuRuntimeJointProposals = hasFlag(argc, argv, "--gpu-runtime-joint-proposals");
    bool gpuReplaceCpuJoints = hasFlag(argc, argv, "--gpu-replace-cpu-joints");
    int gpuJointTopologyRepeats = atoi(argValue(argc, argv, "--gpu-joint-topology-repeats") ? argValue(argc, argv, "--gpu-joint-topology-repeats") : "1");
    int gpuJointProposalIterations = atoi(argValue(argc, argv, "--gpu-joint-proposal-iterations") ? argValue(argc, argv, "--gpu-joint-proposal-iterations") : "1");
    float gpuJointPerturb = (float)atof(argValue(argc, argv, "--gpu-joint-perturb") ? argValue(argc, argv, "--gpu-joint-perturb") : "0");
    double tickRate = atof(argValue(argc, argv, "--tick-rate") ? argValue(argc, argv, "--tick-rate") : "60");
    if (tickRate <= 0.0)
        tickRate = 60.0;

    if (gpuSphereContacts)
        setEnvironmentFlag("AVBD_GPU_SPHERE_CONTACTS", true);
    if (gpuGroundContactFeed)
        setEnvironmentFlag("AVBD_GPU_GROUND_CONTACT_FEED", true);
    if (gpuContactSolveDiagnostic)
        setEnvironmentFlag("AVBD_GPU_CONTACT_SOLVE_DIAGNOSTIC", true);
    if (gpuContactSolveNoReadback)
        setEnvironmentFlag("AVBD_GPU_CONTACT_SOLVE_NO_READBACK", true);
    if (gpuDirectCounterReadback)
        setEnvironmentFlag("AVBD_GPU_DIRECT_COUNTER_READBACK", true);
    if (gpuRuntimeValidate)
        setEnvironmentFlag("AVBD_GPU_RUNTIME_VALIDATE", true);
    if (gpuApplyPrediction)
        setEnvironmentFlag("AVBD_GPU_APPLY_PREDICTION", true);
    if (gpuApplyVelocity)
        setEnvironmentFlag("AVBD_GPU_APPLY_VELOCITY", true);
    if (gpuResidentCounterlessContacts)
        setEnvironmentFlag("AVBD_GPU_RESIDENT_COUNTERLESS_CONTACTS", true);
    if (gpuDeferFinalPositionReadback)
        setEnvironmentFlag("AVBD_GPU_DEFER_FINAL_POSITION_READBACK", true);
    if (gpuAsyncFinalPositionReadback)
        setEnvironmentFlag("AVBD_GPU_ASYNC_FINAL_POSITION_READBACK", true);
    if (gpuApplyContactPositions)
        setEnvironmentFlag("AVBD_GPU_APPLY_CONTACT_POSITIONS", true);
    if (gpuGroundContacts)
        setEnvironmentFlag("AVBD_GPU_GROUND_CONTACTS", true);
    if (gpuRuntimeJointProposals)
        setEnvironmentFlag("AVBD_GPU_JOINT_PROPOSALS", true);
    if (gpuReplaceCpuJoints)
        setEnvironmentFlag("AVBD_GPU_REPLACE_CPU_JOINTS", true);
    if (argValue(argc, argv, "--gpu-joint-proposal-iterations"))
        setEnvironmentValue("AVBD_GPU_JOINT_PROPOSAL_ITERATIONS", argValue(argc, argv, "--gpu-joint-proposal-iterations"));
    if (argValue(argc, argv, "--gpu-contact-iterations"))
        setEnvironmentValue("AVBD_GPU_CONTACT_ITERATIONS", argValue(argc, argv, "--gpu-contact-iterations"));
    if (argValue(argc, argv, "--gpu-contact-relaxation"))
        setEnvironmentValue("AVBD_GPU_CONTACT_RELAXATION", argValue(argc, argv, "--gpu-contact-relaxation"));

    if (benchmarkScene || benchmarkFrames > 0)
        return runBenchmark(benchmarkScene ? benchmarkScene : scene, benchmarkFrames, benchmarkWarmupFrames, resetAfterWarmup, noStream, port, physicsBackend, collisionOnly, iterations, hasGravityOverride, gravityOverride, gpuJointTopologyDiagnostic, gpuJointTopologyRepeats, gpuJointProposalIterations, gpuApplyJointProposals, gpuJointPerturb);
    if (compareScene || compareFrames > 0)
        return runBackendComparison(compareScene ? compareScene : scene, compareFrames, compareBackend ? compareBackend : physicsBackend, iterations);

    SimulationHost host;
    WebGpuContext webgpuContext;
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
    configurePhysicsBackend(host, physicsBackend, webgpuContext);
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
        std::printf("WebGPU runtime: %s\n", webgpuContext.statusText());
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
                    configurePhysicsBackend(host, command.stringValue.c_str(), webgpuContext);
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
                std::string metrics = combinedMetricsText(host, bridge, tickOverruns, webgpuContext);
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
