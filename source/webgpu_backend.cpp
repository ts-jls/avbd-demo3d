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
#include <stdlib.h>
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

int environmentIntValue(const char *name, int fallback)
{
#ifdef _WIN32
    char *value = 0;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || !value)
        return fallback;
    int parsed = atoi(value);
    free(value);
    return parsed;
#else
    const char *value = getenv(name);
    return value ? atoi(value) : fallback;
#endif
}

float environmentFloatValue(const char *name, float fallback)
{
#ifdef _WIN32
    char *value = 0;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || !value)
        return fallback;
    float parsed = (float)atof(value);
    free(value);
    return parsed;
#else
    const char *value = getenv(name);
    return value ? (float)atof(value) : fallback;
#endif
}

uint32_t floatToBits(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool isGpuStaticGroundReceiver(const SimBodyData &body)
{
    if (!body.active || body.mass > 0.0f || body.shape.type != RIGID_SHAPE_BOX)
        return false;
    if (body.shape.size.x < 8.0f || body.shape.size.y < 8.0f || body.shape.size.z > 2.0f)
        return false;

    float3 normal = rotate(body.positionAng, float3{0.0f, 0.0f, 1.0f});
    return normal.z > 0.98f;
}

bool findGpuStaticGroundTop(const SimWorld &world, float &groundTop, int *receiverCount = 0)
{
    bool found = false;
    groundTop = 0.0f;
    int count = 0;
    for (const SimBodyData &body : world.bodies)
    {
        if (!isGpuStaticGroundReceiver(body))
            continue;

        count++;
        float top = body.positionLin.z + body.shape.size.z * 0.5f;
        if (!found || top > groundTop)
            groundTop = top;
        found = true;
    }
    if (receiverCount)
        *receiverCount = count;
    return found;
}

bool isGpuDynamicSphereVsStaticGroundPair(const SimWorld &world, const BroadphasePair &pair)
{
    if (pair.bodyA >= world.bodies.size() || pair.bodyB >= world.bodies.size())
        return false;

    const SimBodyData &bodyA = world.bodies[pair.bodyA];
    const SimBodyData &bodyB = world.bodies[pair.bodyB];
    return (bodyA.mass > 0.0f && bodyA.shape.type == RIGID_SHAPE_SPHERE && isGpuStaticGroundReceiver(bodyB)) ||
           (bodyB.mass > 0.0f && bodyB.shape.type == RIGID_SHAPE_SPHERE && isGpuStaticGroundReceiver(bodyA));
}

bool isGpuDynamicSphere(const SimBodyData &body)
{
    return body.active && body.source && body.mass > 0.0f && body.shape.type == RIGID_SHAPE_SPHERE;
}

bool isGpuDynamicGroundCorrectionBody(const SimBodyData &body)
{
    return body.active &&
           body.source &&
           body.mass > 0.0f &&
           (body.shape.type == RIGID_SHAPE_SPHERE ||
            body.shape.type == RIGID_SHAPE_BOX ||
            body.shape.type == RIGID_SHAPE_CAPSULE ||
            body.shape.type == RIGID_SHAPE_CYLINDER);
}

bool isGpuDynamicGroundCorrectionBody(const SimBodyData &body, bool includeSpheres)
{
    if (!isGpuDynamicGroundCorrectionBody(body))
        return false;
    return includeSpheres || body.shape.type != RIGID_SHAPE_SPHERE;
}

float gpuGroundContactExtentZ(const SimBodyData &body)
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

bool shouldUseCpuFallbackPairsForGpuContacts(const SimWorld &world)
{
    int activeBodies = 0;
    int dynamicSpheres = 0;
    int nonSpheres = 0;
    for (const SimBodyData &body : world.bodies)
    {
        if (!body.active || !body.source)
            continue;
        ++activeBodies;
        if (isGpuDynamicSphere(body))
            ++dynamicSpheres;
        else
            ++nonSpheres;
    }

    return activeBodies >= 512 &&
           dynamicSpheres >= activeBodies * 3 / 4 &&
           nonSpheres <= 128;
}

void appendCpuFallbackPairsForGpuAppliedContacts(const SimWorld &world, std::vector<BroadphasePair> &pairs, bool removeGroundPairs)
{
    std::vector<BodyId> dynamicSpheres;
    std::vector<BodyId> otherBodies;
    dynamicSpheres.reserve(world.bodies.size());
    otherBodies.reserve(world.bodies.size());

    for (BodyId bodyId = 0; bodyId < world.bodies.size(); ++bodyId)
    {
        const SimBodyData &body = world.bodies[bodyId];
        if (!body.active || !body.source)
            continue;
        if (isGpuDynamicSphere(body))
            dynamicSpheres.push_back(bodyId);
        else
            otherBodies.push_back(bodyId);
    }

    auto appendIfOverlapping = [&](BodyId bodyA, BodyId bodyB)
    {
        if (bodyA == bodyB)
            return;
        if (bodyB < bodyA)
            std::swap(bodyA, bodyB);

        const SimBodyData &a = world.bodies[bodyA];
        const SimBodyData &b = world.bodies[bodyB];
        if (a.mass <= 0.0f && b.mass <= 0.0f)
            return;

        BroadphasePair pair{bodyA, bodyB};
        if (removeGroundPairs && isGpuDynamicSphereVsStaticGroundPair(world, pair))
            return;

        float3 dp = a.positionLin - b.positionLin;
        float radius = a.radius + b.radius;
        if (dot(dp, dp) <= radius * radius)
            pairs.push_back(pair);
    };

    for (BodyId sphereBody : dynamicSpheres)
    {
        for (BodyId otherBody : otherBodies)
            appendIfOverlapping(sphereBody, otherBody);
    }

    for (size_t i = 0; i < otherBodies.size(); ++i)
    {
        for (size_t j = i + 1; j < otherBodies.size(); ++j)
        {
            appendIfOverlapping(otherBodies[i], otherBodies[j]);
        }
    }
}

void removeSphereSpherePairs(const SimWorld &world, std::vector<BroadphasePair> &pairs)
{
    pairs.erase(std::remove_if(pairs.begin(), pairs.end(),
                               [&](const BroadphasePair &pair)
                               {
                                   if (pair.bodyA >= world.bodies.size() || pair.bodyB >= world.bodies.size())
                                       return false;
                                   const SimBodyData &bodyA = world.bodies[pair.bodyA];
                                   const SimBodyData &bodyB = world.bodies[pair.bodyB];
                                   return bodyA.shape.type == RIGID_SHAPE_SPHERE &&
                                          bodyB.shape.type == RIGID_SHAPE_SPHERE;
                               }),
                pairs.end());
}

void removeGpuAppliedContactPairs(const SimWorld &world, std::vector<BroadphasePair> &pairs, bool removeGroundPairs)
{
    pairs.erase(std::remove_if(pairs.begin(), pairs.end(),
                               [&](const BroadphasePair &pair)
                               {
                                   if (pair.bodyA >= world.bodies.size() || pair.bodyB >= world.bodies.size())
                                       return false;
                                   const SimBodyData &bodyA = world.bodies[pair.bodyA];
                                   const SimBodyData &bodyB = world.bodies[pair.bodyB];
                                   if (bodyA.shape.type == RIGID_SHAPE_SPHERE &&
                                       bodyB.shape.type == RIGID_SHAPE_SPHERE)
                                       return true;
                                   return removeGroundPairs && isGpuDynamicSphereVsStaticGroundPair(world, pair);
                               }),
                pairs.end());
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
    uint32_t shapeType;
    uint32_t pad1;
    uint32_t groundTopBits;
    uint32_t pad2;
    uint32_t shapeSizeXBits;
    uint32_t shapeSizeYBits;
    uint32_t shapeSizeZBits;
    float orientationX;
    float orientationY;
    float orientationZ;
    float orientationW;
};

struct GpuSapCounters
{
    uint32_t candidates;
    uint32_t sphereHits;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    uint32_t pad3;
    uint32_t pad4;
    uint32_t pad5;
};

struct GpuSapParams
{
    uint32_t count;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    uint32_t contactCapacity;
    uint32_t pad3;
    uint32_t pad4;
    uint32_t pad5;
};

struct GpuDirectSphereParams
{
    uint32_t intervalCount;
    uint32_t bodyCount;
    uint32_t axis;
    uint32_t pad0;
};

struct GpuDirectContactCounters
{
    uint32_t contactRecords;
    uint32_t roundPairCandidates;
    uint32_t sphereBoxCandidates;
    uint32_t sphereCylinderCandidates;
    uint32_t sphereCapsuleCandidates;
    uint32_t boxPairCandidates;
    uint32_t scannedPairs;
    uint32_t activeIntervals;
};

struct GpuJointTopologyInput
{
    uint32_t bodyA;
    uint32_t bodyB;
    uint32_t flags;
    uint32_t pad0;
};

struct GpuJointTopologyParams
{
    uint32_t jointCount;
    uint32_t bodyCount;
    uint32_t pad0;
    uint32_t pad1;
};

struct GpuJointResidualBody
{
    float position[4];
    float rotation[4];
};

struct GpuJointResidualInput
{
    uint32_t bodyA;
    uint32_t bodyB;
    uint32_t flags;
    uint32_t pad0;
    float rA[4];
    float rB[4];
};

struct GpuJointProposalOutput
{
    float correction[4];
    uint32_t jointCount;
    uint32_t color;
    uint32_t pad0;
    uint32_t pad1;
};

struct GpuSapPair
{
    uint32_t bodyA;
    uint32_t bodyB;
};

struct GpuContactBody
{
    float position[4];
    float orientation[4];
    float halfSizeRadius[4];
    float extra[4];
    uint32_t shapeType;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct GpuSphereContact
{
    uint32_t bodyA;
    uint32_t bodyB;
    uint32_t pad0;
    uint32_t pad1;
    float normal[4];
};

struct GpuContactAdjacencyParams
{
    uint32_t contactCount;
    uint32_t bodyCount;
    uint32_t capacityPerBody;
    float contactRelaxation;
};

struct GpuContactGatherOutput
{
    uint32_t count;
    uint32_t overflow;
    uint32_t pad0;
    uint32_t pad1;
    float normalSum[4];
    float correction[4];
};

struct GpuContactProposalOutput
{
    float position[4];
    float correction[4];
};

struct GpuContactProposalResidual
{
    float beforePenetration;
    float afterPenetration;
    float pad0;
    float pad1;
};

struct GpuSphereGroundParams
{
    uint32_t bodyCount;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    float ground[4];
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
    : initialized(false), deviceReady(false), smokeTestPassed(false), status{}, smokeStatus{}, computeStatus{}, presentStatus{}, predictionStatus{}, velocityStatus{}, boundsStatus{}, mortonStatus{}, mortonSortStatus{}, pairStatus{}, sapStatus{}, jointTopologyStatus{}, runtimeStatus{},
      previewComputeMs(0.0f), previewRenderMs(0.0f), previewTotalMs(0.0f),
      previewBatchCount(0), previewInstanceCount(0), previewBoxInstances(0), previewSphereInstances(0), previewCapsuleInstances(0), previewCylinderInstances(0), previewMeshAssetInstances(0),
      predictionMs(0.0f), predictionMaxError(0.0f), predictionMaxAngularError(0.0f), predictionSamples(0),
      velocityMs(0.0f), velocityMaxLinearError(0.0f), velocityMaxAngularError(0.0f), velocitySamples(0),
      boundsMs(0.0f), boundsMaxError(0.0f), boundsSamples(0),
      mortonMs(0.0f), mortonMismatches(0), mortonSamples(0),
      mortonSortMs(0.0f), mortonSortMismatches(0), mortonSortCount(0),
      pairMs(0.0f), pairCandidates(0), pairSphereHits(0), pairAllPairsSphereHits(0), pairMissedSphereHits(0), pairMismatches(0),
      sapMs(0.0f), sapCandidates(0), sapSphereHits(0), sapAllPairsSphereHits(0), sapMissedSphereHits(0), sapMismatches(0),
      sapBestAxis(0), sapAxisCandidates{}, sapAxisSphereHits{}, sapAxisMissedSphereHits{},
      sapCounterReadbackBytes(0), sapCounterReadbackMs(0.0f),
      sapPairReadbackBytes(0), sapPairReadbackMs(0.0f),
      jointTopologyMs(0.0f), jointTopologyJoints(0), jointTopologyBodyRefs(0), jointTopologyActiveBodies(0),
      jointTopologyMaxPerBody(0), jointTopologyMismatches(0), jointTopologyReadbackBytes(0),
      jointColorCount(0), jointColorConflicts(0), jointColorMinBucket(0), jointColorMaxBucket(0), jointColorReadbackBytes(0),
      jointResidualMax(0.0f), jointResidualRms(0.0f), jointResidualReadbackBytes(0),
      jointProposalMaxCorrection(0.0f), jointProposalRmsCorrection(0.0f), jointProposalActiveBodies(0),
      jointProposalMaxPerBody(0), jointProposalReadbackBytes(0), jointProposalIterations(0),
      jointProposalResidualAfterMax(0.0f), jointProposalResidualAfterRms(0.0f), jointProposalResidualReadbackBytes(0),
      jointProposalFinalPositionReady(0), jointProposalFinalPositionBodyCount(0), jointProposalFinalPositionBytes(0),
      jointProposalFinalPositionAbsolute(0), jointProposalSeededFromContact(0),
      jointProposalAppliedPositionBodies(0), jointProposalAppliedPositionReadbackBytes(0),
      jointProposalAppliedPositionMaxDelta(0.0f), jointProposalAppliedPositionChecksum(0.0f),
      jointProposalAppliedPositionMs(0.0f),
      jointProposalFinalPositionAsyncReadbackPending(0), jointProposalFinalPositionAsyncReadbackScheduled(0),
      jointProposalFinalPositionAsyncReadbackConsumed(0), jointProposalFinalPositionAsyncReadbackDropped(0),
      jointProposalFinalPositionAsyncReadbackWaitMs(0.0f), jointProposalFinalPositionAsyncReadbackBodyCount(0),
      jointProposalFinalPositionAsyncReadbackBytes(0), jointProposalFinalPositionAsyncReadbackAbsolute(0),
      sphereContactCount(0), sphereContactExternalContacts(0), sphereContactExternalGroundContacts(0),
      sphereContactReadbackBytes(0), sphereContactMs(0.0f), sphereContactReadbackMs(0.0f),
      sphereContactBodyRefs(0), sphereContactActiveBodies(0), sphereContactMaxPerBody(0), sphereContactAvgPerActiveBody(0.0f),
      sphereContactAdjacencyReadbackBytes(0), sphereContactAdjacencyListBytes(0), sphereContactAdjacencyCapacity(0),
      sphereContactAdjacencyWrittenRefs(0), sphereContactAdjacencyOverflowRefs(0), sphereContactAdjacencyMs(0.0f),
      sphereContactGatherRefs(0), sphereContactGatherActiveBodies(0), sphereContactGatherMaxPerBody(0),
      sphereContactGatherMismatches(0), sphereContactGatherReadbackBytes(0), sphereContactGatherNormalChecksum(0.0f),
      sphereContactProposalActiveBodies(0), sphereContactProposalMaxCorrection(0.0f), sphereContactProposalCorrectionChecksum(0.0f),
      sphereContactGatherMs(0.0f),
      sphereContactProposalOutputActiveBodies(0), sphereContactProposalOutputReadbackBytes(0),
      sphereContactProposalOutputMaxDelta(0.0f), sphereContactProposalOutputChecksum(0.0f),
      sphereContactProposalOutputMs(0.0f),
      sphereContactProposalResidualReadbackBytes(0), sphereContactProposalResidualBeforeMax(0.0f),
      sphereContactProposalResidualAfterMax(0.0f), sphereContactProposalResidualBeforeChecksum(0.0f),
      sphereContactProposalResidualAfterChecksum(0.0f), sphereContactProposalResidualMs(0.0f),
      sphereContactIterationCount(0), sphereContactIterationMs(0.0f), sphereContactIterationRelaxation(0.10f),
      sphereContactIterationResidualAfterMax(0.0f), sphereContactIterationResidualAfterChecksum(0.0f),
      sphereContactFinalPositionReady(0), sphereContactFinalPositionBodyCount(0),
      sphereContactFinalPositionBytes(0), sphereContactFinalPositionSource(0),
      sphereContactAppliedPositionBodies(0), sphereContactAppliedPositionReadbackBytes(0),
      sphereContactAppliedPositionMaxDelta(0.0f), sphereContactAppliedPositionChecksum(0.0f),
      sphereContactAppliedPositionMs(0.0f), sphereContactAppliedPositionWaitMs(0.0f),
      sphereContactAppliedPositionCpuMs(0.0f), sphereContactFinalPositionReadbackDeferred(0),
      sphereContactFinalPositionAsyncReadbackPending(0), sphereContactFinalPositionAsyncReadbackScheduled(0),
      sphereContactFinalPositionAsyncReadbackConsumed(0), sphereContactFinalPositionAsyncReadbackDropped(0),
      sphereContactFinalPositionAsyncReadbackWaitMs(0.0f), sphereContactFinalPositionAsyncReadbackBodyCount(0),
      sphereContactFinalPositionAsyncReadbackBytes(0), sphereContactFinalPositionAsyncReadbackSource(0),
      sphereGroundReceiverCount(0), sphereGroundDynamicSphereCount(0), sphereGroundCandidateCount(0),
      directSphereCylinderBodyCount(0), directSphereCylinderCandidateCount(0),
      directSphereCapsuleBodyCount(0), directSphereCapsuleCandidateCount(0),
      directSphereBoxBodyCount(0), directSphereBoxCandidateCount(0),
      directBoxBodyCount(0), directBoxPairCandidateCount(0),
      directRoundBodyCount(0), directRoundPairCandidateCount(0),
      directGpuContactRecordCount(0), directGpuRoundPairCandidateCount(0), directGpuBoxPairCandidateCount(0),
      directGpuCounterReadbackBytes(0), directGpuCounterReadbackMs(0.0f),
      directSphereContactAppliedPositionBodies(0), directGroundAppliedPositionBodies(0),
      predictionAppliedBodies(0), predictionAppliedReadbackBytes(0), predictionAppliedMs(0.0f),
      velocityAppliedBodies(0), velocityAppliedReadbackBytes(0), velocityAppliedMs(0.0f),
      sphereGroundTop(0.0f), sphereGroundMs(0.0f),
      runtimeTotalMs(0.0f), runtimeSyncMs(0.0f), runtimePredictionMs(0.0f), runtimeVelocityMs(0.0f), runtimeCpuFallbackMs(0.0f),
      runtimeMaxLinearError(0.0f), runtimeMaxAngularError(0.0f), runtimeFrames(0), runtimeFallbacks(0)
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
    sapSortPassParamsBufferBytes = 0;
    sapPairOutputBufferBytes = 0;
    sapPairReadbackBufferBytes = 0;
    sapDirectSphereParamsBuffer = nullptr;
    jointTopologyInputBufferBytes = 0;
    jointTopologyOutputBufferBytes = 0;
    jointTopologyReadbackBufferBytes = 0;
    jointColorBufferBytes = 0;
    jointColorCounterBufferBytes = 0;
    jointResidualInputBufferBytes = 0;
    jointResidualBodyBufferBytes = 0;
    jointResidualOutputBufferBytes = 0;
    jointProposalOutputBufferBytes = 0;
    jointProposalResidualBufferBytes = 0;
    jointProposalFinalReadbackBufferBytes = 0;
    jointProposalFinalAsyncReadbackBufferBytes = 0;
    sphereContactBodyBufferBytes = 0;
    sphereContactPairBufferBytes = 0;
    sphereContactCountersBufferBytes = 0;
    sphereContactCounterReadbackBufferBytes = 0;
    sphereContactOutputBufferBytes = 0;
    sphereContactReadbackBufferBytes = 0;
    sphereContactBodyCountBufferBytes = 0;
    sphereContactBodyCountReadbackBufferBytes = 0;
    sphereContactAdjacencyBufferBytes = 0;
    sphereContactGatherOutputBufferBytes = 0;
    sphereContactGatherReadbackBufferBytes = 0;
    sphereContactProposalOutputBufferBytes = 0;
    sphereContactProposalOutputReadbackBufferBytes = 0;
    sphereContactIterationOutputBufferBytes = 0;
    sphereContactIterationScratchBufferBytes = 0;
    sphereContactFinalPositionBufferBytes = 0;
    sphereContactFinalPositionAsyncReadbackBufferBytes = 0;
    sphereContactProposalResidualBufferBytes = 0;
    sphereContactProposalResidualReadbackBufferBytes = 0;
    sapDirectContactCountersBufferBytes = 0;
    sapDirectContactCountersReadbackBufferBytes = 0;
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
    setStatus(jointTopologyStatus, "Joint topology not run");
    setStatus(runtimeStatus, "WebGPU runtime not run");
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

bool WebGpuContext::runBodyPredictionDiagnostic(const SimWorld &world, float dt, float gravity, bool validateReadback)
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
    if (validateReadback && (predictionReadbackBuffer == nullptr || predictionReadbackBufferBytes < readbackBytes))
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
        (validateReadback && predictionReadbackBuffer == nullptr) || predictionParamsBuffer == nullptr)
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
    if (validateReadback)
        encoder.CopyBufferToBuffer(predictionOutputBuffer, 0, predictionReadbackBuffer, 0, readbackBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    if (!validateReadback)
    {
        predictionSamples = 0;
        predictionMaxError = 0.0f;
        predictionMaxAngularError = 0.0f;
        predictionMs = elapsedMs(begin, SDL_GetPerformanceCounter());
        recordTimingSample(predictionTiming, predictionMs);
        snprintf(predictionStatus, sizeof(predictionStatus), "Body prediction submitted: %zu bodies, validation skipped",
                 inputs.size());
        return true;
    }

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

bool WebGpuContext::applyBodyPredictionOutputs(Solver &solver)
{
    predictionAppliedBodies = 0;
    predictionAppliedReadbackBytes = 0;
    predictionAppliedMs = 0.0f;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(predictionStatus, "Body prediction apply skipped: WebGPU device not ready");
        return false;
    }
    if (predictionOutputBuffer == nullptr)
    {
        setStatus(predictionStatus, "Body prediction apply failed: no prediction output");
        return false;
    }

    int activeCount = 0;
    for (size_t i = 0; i < solver.world.bodies.size(); ++i)
    {
        if (solver.world.bodies[i].active)
            activeCount++;
    }
    if (activeCount <= 0)
    {
        setStatus(predictionStatus, "Body prediction apply skipped: no active bodies");
        return false;
    }

    const uint64_t readbackBytes = (uint64_t)(activeCount * (int)sizeof(GpuPredictionOutput));
    if (predictionReadbackBuffer == nullptr || predictionReadbackBufferBytes < readbackBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(readbackBytes, 4096);
        predictionReadbackBuffer = device.CreateBuffer(&desc);
        predictionReadbackBufferBytes = predictionReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (predictionReadbackBuffer == nullptr)
    {
        setStatus(predictionStatus, "Body prediction apply failed: readback allocation");
        return false;
    }

    Uint64 begin = SDL_GetPerformanceCounter();
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
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
        setStatus(predictionStatus, "Body prediction apply failed: readback map");
        return false;
    }

    const GpuPredictionOutput *outputs =
        (const GpuPredictionOutput *)predictionReadbackBuffer.GetConstMappedRange(0, readbackBytes);
    if (!outputs)
    {
        predictionReadbackBuffer.Unmap();
        setStatus(predictionStatus, "Body prediction apply failed: readback range");
        return false;
    }

    int outputIndex = 0;
    for (size_t i = 0; i < solver.world.bodies.size(); ++i)
    {
        const SimBodyData &bodyData = solver.world.bodies[i];
        if (!bodyData.active)
            continue;
        if (outputIndex >= activeCount)
            break;

        Rigid *body = bodyData.source;
        const GpuPredictionOutput &output = outputs[outputIndex++];
        if (!body)
            continue;

        body->inertialLin = float3{output.inertial[0], output.inertial[1], output.inertial[2]};
        body->inertialAng = quat{output.inertialAng[0], output.inertialAng[1], output.inertialAng[2], output.inertialAng[3]};
        body->initialLin = body->positionLin;
        body->initialAng = body->positionAng;
        if (body->mass > 0.0f)
        {
            body->positionLin = body->inertialLin;
            body->positionAng = body->inertialAng;
        }
        predictionAppliedBodies++;
    }

    predictionReadbackBuffer.Unmap();
    predictionAppliedReadbackBytes = (int)readbackBytes;
    predictionAppliedMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    snprintf(predictionStatus, sizeof(predictionStatus), "Body prediction applied: %d bodies, readback %.1f KB",
             predictionAppliedBodies, (double)predictionAppliedReadbackBytes / 1024.0);
    return predictionAppliedBodies > 0;
#elif AVBD_ENABLE_WEBGPU
    setStatus(predictionStatus, "Body prediction apply skipped: Dawn not linked");
    return false;
#else
    setStatus(predictionStatus, "Body prediction apply skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::runVelocityUpdateDiagnostic(const SimWorld &world, float dt, bool validateReadback)
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
    if (validateReadback && (velocityReadbackBuffer == nullptr || velocityReadbackBufferBytes < readbackBytes))
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
        (validateReadback && velocityReadbackBuffer == nullptr) || velocityParamsBuffer == nullptr)
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
    if (validateReadback)
        encoder.CopyBufferToBuffer(velocityOutputBuffer, 0, velocityReadbackBuffer, 0, readbackBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    if (!validateReadback)
    {
        velocitySamples = 0;
        velocityMaxLinearError = 0.0f;
        velocityMaxAngularError = 0.0f;
        velocityMs = elapsedMs(begin, SDL_GetPerformanceCounter());
        recordTimingSample(velocityTiming, velocityMs);
        snprintf(velocityStatus, sizeof(velocityStatus), "Velocity update submitted: %zu bodies, validation skipped",
                 inputs.size());
        return true;
    }

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

bool WebGpuContext::applyVelocityOutputs(Solver &solver)
{
    velocityAppliedBodies = 0;
    velocityAppliedReadbackBytes = 0;
    velocityAppliedMs = 0.0f;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(velocityStatus, "Velocity apply skipped: WebGPU device not ready");
        return false;
    }
    if (velocityOutputBuffer == nullptr || solver.dt == 0.0f)
    {
        setStatus(velocityStatus, "Velocity apply failed: no velocity output");
        return false;
    }

    int activeCount = 0;
    for (size_t i = 0; i < solver.world.activeBodyIds.size(); ++i)
    {
        BodyId bodyId = solver.world.activeBodyIds[i];
        if (bodyId == INVALID_BODY_ID || bodyId >= solver.world.bodies.size())
            continue;
        const SimBodyData &bodyData = solver.world.bodies[bodyId];
        if (bodyData.active && bodyData.source && bodyData.mass > 0.0f)
            activeCount++;
    }
    if (activeCount <= 0)
    {
        setStatus(velocityStatus, "Velocity apply skipped: no dynamic active bodies");
        return false;
    }

    const uint64_t readbackBytes = (uint64_t)activeCount * sizeof(GpuVelocityOutput);
    if (velocityReadbackBuffer == nullptr || velocityReadbackBufferBytes < readbackBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(readbackBytes, 4096);
        velocityReadbackBuffer = device.CreateBuffer(&desc);
        velocityReadbackBufferBytes = velocityReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (velocityReadbackBuffer == nullptr)
    {
        setStatus(velocityStatus, "Velocity apply failed: readback allocation");
        return false;
    }

    Uint64 begin = SDL_GetPerformanceCounter();
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
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
        setStatus(velocityStatus, "Velocity apply failed: readback map");
        return false;
    }

    const GpuVelocityOutput *outputs =
        (const GpuVelocityOutput *)velocityReadbackBuffer.GetConstMappedRange(0, readbackBytes);
    if (!outputs)
    {
        velocityReadbackBuffer.Unmap();
        setStatus(velocityStatus, "Velocity apply failed: readback range");
        return false;
    }

    int outputIndex = 0;
    for (size_t i = 0; i < solver.world.activeBodyIds.size(); ++i)
    {
        BodyId bodyId = solver.world.activeBodyIds[i];
        if (bodyId == INVALID_BODY_ID || bodyId >= solver.world.bodies.size())
            continue;
        const SimBodyData &bodyData = solver.world.bodies[bodyId];
        if (!bodyData.active || !bodyData.source || bodyData.mass <= 0.0f)
            continue;
        if (outputIndex >= activeCount)
            break;

        const GpuVelocityOutput &output = outputs[outputIndex++];
        Rigid *body = bodyData.source;
        body->velocityLin = float3{output.velocity[0], output.velocity[1], output.velocity[2]};
        body->velocityAng = float3{output.angularVelocity[0], output.angularVelocity[1], output.angularVelocity[2]};
        velocityAppliedBodies++;
    }

    velocityReadbackBuffer.Unmap();
    velocityAppliedReadbackBytes = (int)readbackBytes;
    velocityAppliedMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    snprintf(velocityStatus, sizeof(velocityStatus), "Velocity applied: %d bodies, readback %.1f KB",
             velocityAppliedBodies, (double)velocityAppliedReadbackBytes / 1024.0);
    return velocityAppliedBodies > 0;
#elif AVBD_ENABLE_WEBGPU
    setStatus(velocityStatus, "Velocity apply skipped: Dawn not linked");
    return false;
#else
    setStatus(velocityStatus, "Velocity apply skipped: WebGPU disabled");
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
    groundTopBits: u32,
    pad2: u32,
    shapeSizeXBits: u32,
    shapeSizeYBits: u32,
    shapeSizeZBits: u32,
    orientationX: f32,
    orientationY: f32,
    orientationZ: f32,
    orientationW: f32,
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
    groundTopBits: u32,
    pad2: u32,
    shapeSizeXBits: u32,
    shapeSizeYBits: u32,
    shapeSizeZBits: u32,
    orientationX: f32,
    orientationY: f32,
    orientationZ: f32,
    orientationW: f32,
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

bool WebGpuContext::runJointTopologyDiagnostic(const SimWorld &world, int proposalIterationsRequested, bool diagnosticReadback, bool seedFromContactFinalPositions)
{
    jointTopologyMs = 0.0f;
    jointTopologyJoints = 0;
    jointTopologyBodyRefs = 0;
    jointTopologyActiveBodies = 0;
    jointTopologyMaxPerBody = 0;
    jointTopologyMismatches = 0;
    jointTopologyReadbackBytes = 0;
    jointColorCount = 0;
    jointColorConflicts = 0;
    jointColorMinBucket = 0;
    jointColorMaxBucket = 0;
    jointColorReadbackBytes = 0;
    jointResidualMax = 0.0f;
    jointResidualRms = 0.0f;
    jointResidualReadbackBytes = 0;
    jointProposalMaxCorrection = 0.0f;
    jointProposalRmsCorrection = 0.0f;
    jointProposalActiveBodies = 0;
    jointProposalMaxPerBody = 0;
    jointProposalReadbackBytes = 0;
    jointProposalIterations = 0;
    jointProposalResidualAfterMax = 0.0f;
    jointProposalResidualAfterRms = 0.0f;
    jointProposalResidualReadbackBytes = 0;
    jointProposalFinalPositionReady = 0;
    jointProposalFinalPositionBodyCount = 0;
    jointProposalFinalPositionBytes = 0;
    jointProposalFinalPositionAbsolute = 0;
    jointProposalSeededFromContact = 0;
    jointProposalAppliedPositionBodies = 0;
    jointProposalAppliedPositionReadbackBytes = 0;
    jointProposalAppliedPositionMaxDelta = 0.0f;
    jointProposalAppliedPositionChecksum = 0.0f;
    jointProposalAppliedPositionMs = 0.0f;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    int proposalIterations = std::max(1, proposalIterationsRequested);
    jointProposalIterations = proposalIterations;
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(jointTopologyStatus, "Joint topology skipped: WebGPU device not ready");
        return false;
    }

    std::vector<GpuJointTopologyInput> joints;
    joints.reserve(world.jointIds.size());
    std::vector<GpuJointResidualInput> residualJoints;
    residualJoints.reserve(world.jointIds.size());
    std::vector<uint32_t> expectedBodyCounts(world.bodies.size(), 0);
    std::vector<std::vector<uint32_t>> jointAdjacency(world.bodies.size());
    for (ForceId forceId : world.jointIds)
    {
        if (forceId == INVALID_FORCE_ID || forceId >= world.constraints.size())
            continue;
        const SimConstraintData &constraint = world.constraints[forceId];
        if (!constraint.active || constraint.type != SIM_CONSTRAINT_JOINT)
            continue;

        GpuJointTopologyInput input = {};
        input.bodyA = constraint.bodyA;
        input.bodyB = constraint.bodyB;
        if (constraint.bodyA != INVALID_BODY_ID && constraint.bodyA < world.bodies.size())
        {
            input.flags |= 1u;
            expectedBodyCounts[constraint.bodyA]++;
        }
        if (constraint.bodyB != INVALID_BODY_ID && constraint.bodyB < world.bodies.size())
        {
            input.flags |= 2u;
            expectedBodyCounts[constraint.bodyB]++;
        }
        if ((input.flags & 3u) == 3u)
        {
            jointAdjacency[constraint.bodyA].push_back((uint32_t)constraint.bodyB);
            jointAdjacency[constraint.bodyB].push_back((uint32_t)constraint.bodyA);
        }
        joints.push_back(input);

        GpuJointResidualInput residualInput = {};
        residualInput.bodyA = input.bodyA;
        residualInput.bodyB = input.bodyB;
        residualInput.flags = input.flags;
        const Joint *joint = static_cast<const Joint *>(constraint.source);
        residualInput.rA[0] = joint->rA.x;
        residualInput.rA[1] = joint->rA.y;
        residualInput.rA[2] = joint->rA.z;
        residualInput.rA[3] = 0.0f;
        residualInput.rB[0] = joint->rB.x;
        residualInput.rB[1] = joint->rB.y;
        residualInput.rB[2] = joint->rB.z;
        residualInput.rB[3] = 0.0f;
        residualJoints.push_back(residualInput);
    }

    jointTopologyJoints = (int)joints.size();
    if (world.bodies.empty())
    {
        setStatus(jointTopologyStatus, "Joint topology skipped: no bodies");
        return false;
    }
    if (joints.empty())
    {
        setStatus(jointTopologyStatus, "Joint topology skipped: no joints");
        return false;
    }

    uint32_t expectedRefs = 0;
    uint32_t expectedActiveBodies = 0;
    uint32_t expectedMaxPerBody = 0;
    for (uint32_t count : expectedBodyCounts)
    {
        expectedRefs += count;
        if (count > 0)
            expectedActiveBodies++;
        expectedMaxPerBody = std::max(expectedMaxPerBody, count);
    }

    const uint32_t uncolored = 0xffffffffu;
    std::vector<uint32_t> bodyColors(world.bodies.size(), uncolored);
    std::vector<uint32_t> colorBuckets;
    std::vector<uint8_t> usedColors;
    for (size_t bodyIndex = 0; bodyIndex < jointAdjacency.size(); ++bodyIndex)
    {
        if (expectedBodyCounts[bodyIndex] == 0)
            continue;

        usedColors.assign(colorBuckets.size() + jointAdjacency[bodyIndex].size() + 1, 0);
        for (uint32_t neighbor : jointAdjacency[bodyIndex])
        {
            uint32_t color = bodyColors[neighbor];
            if (color != uncolored && color < usedColors.size())
                usedColors[color] = 1;
        }

        uint32_t chosenColor = 0;
        while (chosenColor < usedColors.size() && usedColors[chosenColor] != 0)
            chosenColor++;
        if (chosenColor >= colorBuckets.size())
            colorBuckets.resize((size_t)chosenColor + 1, 0);
        bodyColors[bodyIndex] = chosenColor;
        colorBuckets[chosenColor]++;
    }

    jointColorCount = (int)colorBuckets.size();
    jointColorMinBucket = 0;
    jointColorMaxBucket = 0;
    for (uint32_t bucket : colorBuckets)
    {
        if (bucket == 0)
            continue;
        if (jointColorMinBucket == 0 || bucket < (uint32_t)jointColorMinBucket)
            jointColorMinBucket = (int)bucket;
        jointColorMaxBucket = std::max(jointColorMaxBucket, (int)bucket);
    }

    std::vector<GpuJointResidualBody> residualBodies(world.bodies.size());
    for (size_t i = 0; i < world.bodies.size(); ++i)
    {
        const SimBodyData &body = world.bodies[i];
        residualBodies[i].position[0] = body.positionLin.x;
        residualBodies[i].position[1] = body.positionLin.y;
        residualBodies[i].position[2] = body.positionLin.z;
        residualBodies[i].position[3] = 0.0f;
        residualBodies[i].rotation[0] = body.positionAng.x;
        residualBodies[i].rotation[1] = body.positionAng.y;
        residualBodies[i].rotation[2] = body.positionAng.z;
        residualBodies[i].rotation[3] = body.positionAng.w;
    }

    auto begin = SDL_GetPerformanceCounter();
    uint64_t inputBytes = sizeof(GpuJointTopologyInput) * joints.size();
    uint64_t outputBytes = sizeof(uint32_t) * world.bodies.size();
    uint64_t topologyReadbackBytes = outputBytes;
    uint64_t colorReadbackBytes = sizeof(uint32_t);
    uint64_t colorBytes = sizeof(uint32_t) * bodyColors.size();
    uint64_t colorCounterBytes = sizeof(uint32_t);
    uint64_t residualBodyBytes = sizeof(GpuJointResidualBody) * residualBodies.size();
    uint64_t residualInputBytes = sizeof(GpuJointResidualInput) * residualJoints.size();
    uint64_t residualOutputBytes = sizeof(float) * residualJoints.size();
    uint64_t proposalOutputBytes = sizeof(GpuJointProposalOutput) * world.bodies.size();
    uint64_t proposalResidualBytes = sizeof(float) * residualJoints.size();
    uint64_t readbackBytes = topologyReadbackBytes + colorReadbackBytes + residualOutputBytes + proposalOutputBytes + proposalResidualBytes;
    GpuJointTopologyParams params = {};
    params.jointCount = (uint32_t)joints.size();
    params.bodyCount = (uint32_t)world.bodies.size();

    if (jointTopologyInputBuffer == nullptr || jointTopologyInputBufferBytes < inputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = inputBytes;
        desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        jointTopologyInputBuffer = device.CreateBuffer(&desc);
        jointTopologyInputBufferBytes = jointTopologyInputBuffer == nullptr ? 0 : desc.size;
    }
    if (jointTopologyOutputBuffer == nullptr || jointTopologyOutputBufferBytes < outputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = outputBytes;
        desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
        jointTopologyOutputBuffer = device.CreateBuffer(&desc);
        jointTopologyOutputBufferBytes = jointTopologyOutputBuffer == nullptr ? 0 : desc.size;
    }
    if (diagnosticReadback && (jointTopologyReadbackBuffer == nullptr || jointTopologyReadbackBufferBytes < readbackBytes))
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = readbackBytes;
        desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        jointTopologyReadbackBuffer = device.CreateBuffer(&desc);
        jointTopologyReadbackBufferBytes = jointTopologyReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (jointTopologyParamsBuffer == nullptr)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = sizeof(GpuJointTopologyParams);
        desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        jointTopologyParamsBuffer = device.CreateBuffer(&desc);
    }
    if (jointColorBuffer == nullptr || jointColorBufferBytes < colorBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = colorBytes;
        desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        jointColorBuffer = device.CreateBuffer(&desc);
        jointColorBufferBytes = jointColorBuffer == nullptr ? 0 : desc.size;
    }
    if (jointColorCounterBuffer == nullptr || jointColorCounterBufferBytes < colorCounterBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = colorCounterBytes;
        desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
        jointColorCounterBuffer = device.CreateBuffer(&desc);
        jointColorCounterBufferBytes = jointColorCounterBuffer == nullptr ? 0 : desc.size;
    }
    if (jointResidualInputBuffer == nullptr || jointResidualInputBufferBytes < residualInputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = residualInputBytes;
        desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        jointResidualInputBuffer = device.CreateBuffer(&desc);
        jointResidualInputBufferBytes = jointResidualInputBuffer == nullptr ? 0 : desc.size;
    }
    if (jointResidualBodyBuffer == nullptr || jointResidualBodyBufferBytes < residualBodyBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = residualBodyBytes;
        desc.usage = wgpu::BufferUsage::Storage |
                     wgpu::BufferUsage::CopyDst |
                     wgpu::BufferUsage::CopySrc;
        jointResidualBodyBuffer = device.CreateBuffer(&desc);
        jointResidualBodyBufferBytes = jointResidualBodyBuffer == nullptr ? 0 : desc.size;
    }
    if (jointResidualOutputBuffer == nullptr || jointResidualOutputBufferBytes < residualOutputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = residualOutputBytes;
        desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
        jointResidualOutputBuffer = device.CreateBuffer(&desc);
        jointResidualOutputBufferBytes = jointResidualOutputBuffer == nullptr ? 0 : desc.size;
    }
    if (jointProposalOutputBuffer == nullptr || jointProposalOutputBufferBytes < proposalOutputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = proposalOutputBytes;
        desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
        jointProposalOutputBuffer = device.CreateBuffer(&desc);
        jointProposalOutputBufferBytes = jointProposalOutputBuffer == nullptr ? 0 : desc.size;
    }
    if (jointProposalResidualBuffer == nullptr || jointProposalResidualBufferBytes < proposalResidualBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = proposalResidualBytes;
        desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
        jointProposalResidualBuffer = device.CreateBuffer(&desc);
        jointProposalResidualBufferBytes = jointProposalResidualBuffer == nullptr ? 0 : desc.size;
    }
    if (jointTopologyInputBuffer == nullptr || jointTopologyOutputBuffer == nullptr ||
        (diagnosticReadback && jointTopologyReadbackBuffer == nullptr) || jointTopologyParamsBuffer == nullptr ||
        jointColorBuffer == nullptr || jointColorCounterBuffer == nullptr ||
        jointResidualInputBuffer == nullptr || jointResidualBodyBuffer == nullptr ||
        jointResidualOutputBuffer == nullptr || jointProposalOutputBuffer == nullptr ||
        jointProposalResidualBuffer == nullptr)
    {
        setStatus(jointTopologyStatus, "Joint topology failed: buffer allocation");
        return false;
    }

    std::vector<uint32_t> zeros(world.bodies.size(), 0);
    queue.WriteBuffer(jointTopologyInputBuffer, 0, joints.data(), inputBytes);
    queue.WriteBuffer(jointTopologyOutputBuffer, 0, zeros.data(), outputBytes);
    queue.WriteBuffer(jointTopologyParamsBuffer, 0, &params, sizeof(params));
    queue.WriteBuffer(jointColorBuffer, 0, bodyColors.data(), colorBytes);
    queue.WriteBuffer(jointResidualInputBuffer, 0, residualJoints.data(), residualInputBytes);
    queue.WriteBuffer(jointResidualBodyBuffer, 0, residualBodies.data(), residualBodyBytes);
    uint32_t zeroCounter = 0;
    queue.WriteBuffer(jointColorCounterBuffer, 0, &zeroCounter, sizeof(zeroCounter));
    const bool useContactSeed = seedFromContactFinalPositions &&
                                sphereContactFinalPositionReady != 0 &&
                                sphereContactFinalPositionBuffer != nullptr &&
                                sphereContactFinalPositionBodyCount >= (int)world.bodies.size() &&
                                sphereContactFinalPositionBytes >= (int)((uint64_t)world.bodies.size() * sizeof(GpuContactProposalOutput));

    const char *shaderSource = R"(
struct JointInput {
    bodyA: u32,
    bodyB: u32,
    flags: u32,
    pad0: u32,
};

struct Params {
    jointCount: u32,
    bodyCount: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> joints: array<JointInput>;
@group(0) @binding(1) var<storage, read_write> bodyCounts: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.jointCount) {
        return;
    }
    let joint = joints[index];
    if ((joint.flags & 1u) != 0u && joint.bodyA < params.bodyCount) {
        atomicAdd(&bodyCounts[joint.bodyA], 1u);
    }
    if ((joint.flags & 2u) != 0u && joint.bodyB < params.bodyCount) {
        atomicAdd(&bodyCounts[joint.bodyB], 1u);
    }
}
)";

    if (jointTopologyPipeline == nullptr || jointTopologyBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = shaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint topology failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry layoutEntries[3] = {};
        layoutEntries[0].binding = 0;
        layoutEntries[0].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[0].buffer.minBindingSize = sizeof(GpuJointTopologyInput);
        layoutEntries[1].binding = 1;
        layoutEntries[1].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
        layoutEntries[1].buffer.minBindingSize = sizeof(uint32_t);
        layoutEntries[2].binding = 2;
        layoutEntries[2].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
        layoutEntries[2].buffer.minBindingSize = sizeof(GpuJointTopologyParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 3;
        layoutDesc.entries = layoutEntries;
        jointTopologyBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (jointTopologyBindGroupLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint topology failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &jointTopologyBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint topology failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        jointTopologyPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (jointTopologyPipeline == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint topology failed: compute pipeline");
            return false;
        }
    }

    const char *colorShaderSource = R"(
struct JointInput {
    bodyA: u32,
    bodyB: u32,
    flags: u32,
    pad0: u32,
};

struct ResidualJointInput {
    bodyA: u32,
    bodyB: u32,
    flags: u32,
    pad0: u32,
    rA: vec4f,
    rB: vec4f,
};

struct ResidualBody {
    position: vec4f,
    rotation: vec4f,
};

struct Params {
    jointCount: u32,
    bodyCount: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> joints: array<JointInput>;
@group(0) @binding(1) var<storage, read> bodyColors: array<u32>;
@group(0) @binding(2) var<storage, read_write> conflicts: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> residualJoints: array<ResidualJointInput>;
@group(0) @binding(5) var<storage, read> bodies: array<ResidualBody>;
@group(0) @binding(6) var<storage, read_write> residualSq: array<f32>;

fn rotateByQuat(q: vec4f, v: vec3f) -> vec3f {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.jointCount) {
        return;
    }
    let joint = joints[index];
    if ((joint.flags & 3u) == 3u &&
        joint.bodyA < params.bodyCount &&
        joint.bodyB < params.bodyCount &&
        bodyColors[joint.bodyA] == bodyColors[joint.bodyB]) {
        atomicAdd(&conflicts[0], 1u);
    }

    let residualJoint = residualJoints[index];
    if ((residualJoint.flags & 3u) == 3u &&
        residualJoint.bodyA < params.bodyCount &&
        residualJoint.bodyB < params.bodyCount) {
        let bodyA = bodies[residualJoint.bodyA];
        let bodyB = bodies[residualJoint.bodyB];
        let anchorA = bodyA.position.xyz + rotateByQuat(bodyA.rotation, residualJoint.rA.xyz);
        let anchorB = bodyB.position.xyz + rotateByQuat(bodyB.rotation, residualJoint.rB.xyz);
        let delta = anchorA - anchorB;
        residualSq[index] = dot(delta, delta);
    } else {
        residualSq[index] = 0.0;
    }
}
)";

    if (jointColorPipeline == nullptr || jointColorBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = colorShaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint coloring failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry layoutEntries[7] = {};
        layoutEntries[0].binding = 0;
        layoutEntries[0].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[0].buffer.minBindingSize = sizeof(GpuJointTopologyInput);
        layoutEntries[1].binding = 1;
        layoutEntries[1].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[1].buffer.minBindingSize = sizeof(uint32_t);
        layoutEntries[2].binding = 2;
        layoutEntries[2].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[2].buffer.type = wgpu::BufferBindingType::Storage;
        layoutEntries[2].buffer.minBindingSize = sizeof(uint32_t);
        layoutEntries[3].binding = 3;
        layoutEntries[3].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[3].buffer.type = wgpu::BufferBindingType::Uniform;
        layoutEntries[3].buffer.minBindingSize = sizeof(GpuJointTopologyParams);
        layoutEntries[4].binding = 4;
        layoutEntries[4].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[4].buffer.minBindingSize = sizeof(GpuJointResidualInput);
        layoutEntries[5].binding = 5;
        layoutEntries[5].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[5].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[5].buffer.minBindingSize = sizeof(GpuJointResidualBody);
        layoutEntries[6].binding = 6;
        layoutEntries[6].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[6].buffer.type = wgpu::BufferBindingType::Storage;
        layoutEntries[6].buffer.minBindingSize = sizeof(float);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 7;
        layoutDesc.entries = layoutEntries;
        jointColorBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (jointColorBindGroupLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint coloring failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &jointColorBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint coloring failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        jointColorPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (jointColorPipeline == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint coloring failed: compute pipeline");
            return false;
        }
    }

    const char *proposalShaderSource = R"(
struct ResidualJointInput {
    bodyA: u32,
    bodyB: u32,
    flags: u32,
    pad0: u32,
    rA: vec4f,
    rB: vec4f,
};

struct ResidualBody {
    position: vec4f,
    rotation: vec4f,
};

struct ProposalOutput {
    correction: vec4f,
    jointCount: u32,
    color: u32,
    pad0: u32,
    pad1: u32,
};

struct Params {
    jointCount: u32,
    bodyCount: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> residualJoints: array<ResidualJointInput>;
@group(0) @binding(1) var<storage, read> bodies: array<ResidualBody>;
@group(0) @binding(2) var<storage, read> bodyColors: array<u32>;
@group(0) @binding(3) var<storage, read_write> proposals: array<ProposalOutput>;
@group(0) @binding(4) var<uniform> params: Params;

fn rotateByQuat(q: vec4f, v: vec3f) -> vec3f {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let bodyIndex = gid.x;
    if (bodyIndex >= params.bodyCount) {
        return;
    }

    let bodyColor = bodyColors[bodyIndex];
    var correction = vec3f(0.0, 0.0, 0.0);
    var jointCount = 0u;
    if (bodyColor != 0xffffffffu) {
        for (var jointIndex = 0u; jointIndex < params.jointCount; jointIndex = jointIndex + 1u) {
            let joint = residualJoints[jointIndex];
            if ((joint.flags & 3u) != 3u ||
                joint.bodyA >= params.bodyCount ||
                joint.bodyB >= params.bodyCount) {
                continue;
            }

            if (joint.bodyA != bodyIndex && joint.bodyB != bodyIndex) {
                continue;
            }

            let bodyA = bodies[joint.bodyA];
            let bodyB = bodies[joint.bodyB];
            let anchorA = bodyA.position.xyz + rotateByQuat(bodyA.rotation, joint.rA.xyz);
            let anchorB = bodyB.position.xyz + rotateByQuat(bodyB.rotation, joint.rB.xyz);
            let delta = anchorA - anchorB;
            if (joint.bodyA == bodyIndex) {
                correction = correction - 0.5 * delta;
                jointCount = jointCount + 1u;
            }
            if (joint.bodyB == bodyIndex) {
                correction = correction + 0.5 * delta;
                jointCount = jointCount + 1u;
            }
        }
    }

    if (jointCount > 0u) {
        correction = correction / f32(jointCount);
    }
    proposals[bodyIndex].correction = vec4f(correction, 0.0);
    proposals[bodyIndex].jointCount = jointCount;
    proposals[bodyIndex].color = bodyColor;
    proposals[bodyIndex].pad0 = 0u;
    proposals[bodyIndex].pad1 = 0u;
}
)";

    if (jointProposalPipeline == nullptr || jointProposalBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = proposalShaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry layoutEntries[5] = {};
        layoutEntries[0].binding = 0;
        layoutEntries[0].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[0].buffer.minBindingSize = sizeof(GpuJointResidualInput);
        layoutEntries[1].binding = 1;
        layoutEntries[1].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[1].buffer.minBindingSize = sizeof(GpuJointResidualBody);
        layoutEntries[2].binding = 2;
        layoutEntries[2].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[2].buffer.minBindingSize = sizeof(uint32_t);
        layoutEntries[3].binding = 3;
        layoutEntries[3].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[3].buffer.type = wgpu::BufferBindingType::Storage;
        layoutEntries[3].buffer.minBindingSize = sizeof(GpuJointProposalOutput);
        layoutEntries[4].binding = 4;
        layoutEntries[4].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[4].buffer.type = wgpu::BufferBindingType::Uniform;
        layoutEntries[4].buffer.minBindingSize = sizeof(GpuJointTopologyParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 5;
        layoutDesc.entries = layoutEntries;
        jointProposalBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (jointProposalBindGroupLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &jointProposalBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        jointProposalPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (jointProposalPipeline == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal failed: compute pipeline");
            return false;
        }
    }

    const char *proposalApplyShaderSource = R"(
struct ResidualBody {
    position: vec4f,
    rotation: vec4f,
};

struct ProposalOutput {
    correction: vec4f,
    jointCount: u32,
    color: u32,
    pad0: u32,
    pad1: u32,
};

struct Params {
    jointCount: u32,
    bodyCount: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> proposals: array<ProposalOutput>;
@group(0) @binding(1) var<storage, read_write> bodies: array<ResidualBody>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let bodyIndex = gid.x;
    if (bodyIndex >= params.bodyCount) {
        return;
    }
    let proposal = proposals[bodyIndex];
    if (proposal.jointCount > 0u) {
        bodies[bodyIndex].position = vec4f(bodies[bodyIndex].position.xyz + proposal.correction.xyz, bodies[bodyIndex].position.w);
    }
}
)";

    if (jointProposalApplyPipeline == nullptr || jointProposalApplyBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = proposalApplyShaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal apply failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry layoutEntries[3] = {};
        layoutEntries[0].binding = 0;
        layoutEntries[0].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[0].buffer.minBindingSize = sizeof(GpuJointProposalOutput);
        layoutEntries[1].binding = 1;
        layoutEntries[1].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
        layoutEntries[1].buffer.minBindingSize = sizeof(GpuJointResidualBody);
        layoutEntries[2].binding = 2;
        layoutEntries[2].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
        layoutEntries[2].buffer.minBindingSize = sizeof(GpuJointTopologyParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 3;
        layoutDesc.entries = layoutEntries;
        jointProposalApplyBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (jointProposalApplyBindGroupLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal apply failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &jointProposalApplyBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal apply failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        jointProposalApplyPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (jointProposalApplyPipeline == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal apply failed: compute pipeline");
            return false;
        }
    }

    const char *proposalResidualShaderSource = R"(
struct ResidualJointInput {
    bodyA: u32,
    bodyB: u32,
    flags: u32,
    pad0: u32,
    rA: vec4f,
    rB: vec4f,
};

struct ResidualBody {
    position: vec4f,
    rotation: vec4f,
};

struct ProposalOutput {
    correction: vec4f,
    jointCount: u32,
    color: u32,
    pad0: u32,
    pad1: u32,
};

struct Params {
    jointCount: u32,
    bodyCount: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> residualJoints: array<ResidualJointInput>;
@group(0) @binding(1) var<storage, read> bodies: array<ResidualBody>;
@group(0) @binding(2) var<storage, read> proposals: array<ProposalOutput>;
@group(0) @binding(3) var<storage, read_write> residualAfterSq: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn rotateByQuat(q: vec4f, v: vec3f) -> vec3f {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let jointIndex = gid.x;
    if (jointIndex >= params.jointCount) {
        return;
    }
    let joint = residualJoints[jointIndex];
    if ((joint.flags & 3u) == 3u &&
        joint.bodyA < params.bodyCount &&
        joint.bodyB < params.bodyCount) {
        let bodyA = bodies[joint.bodyA];
        let bodyB = bodies[joint.bodyB];
        let anchorA = bodyA.position.xyz + proposals[joint.bodyA].correction.xyz + rotateByQuat(bodyA.rotation, joint.rA.xyz);
        let anchorB = bodyB.position.xyz + proposals[joint.bodyB].correction.xyz + rotateByQuat(bodyB.rotation, joint.rB.xyz);
        let delta = anchorA - anchorB;
        residualAfterSq[jointIndex] = dot(delta, delta);
    } else {
        residualAfterSq[jointIndex] = 0.0;
    }
}
)";

    if (jointProposalResidualPipeline == nullptr || jointProposalResidualBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = proposalResidualShaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal residual failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry layoutEntries[5] = {};
        layoutEntries[0].binding = 0;
        layoutEntries[0].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[0].buffer.minBindingSize = sizeof(GpuJointResidualInput);
        layoutEntries[1].binding = 1;
        layoutEntries[1].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[1].buffer.minBindingSize = sizeof(GpuJointResidualBody);
        layoutEntries[2].binding = 2;
        layoutEntries[2].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[2].buffer.minBindingSize = sizeof(GpuJointProposalOutput);
        layoutEntries[3].binding = 3;
        layoutEntries[3].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[3].buffer.type = wgpu::BufferBindingType::Storage;
        layoutEntries[3].buffer.minBindingSize = sizeof(float);
        layoutEntries[4].binding = 4;
        layoutEntries[4].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[4].buffer.type = wgpu::BufferBindingType::Uniform;
        layoutEntries[4].buffer.minBindingSize = sizeof(GpuJointTopologyParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 5;
        layoutDesc.entries = layoutEntries;
        jointProposalResidualBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (jointProposalResidualBindGroupLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal residual failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &jointProposalResidualBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal residual failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        jointProposalResidualPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (jointProposalResidualPipeline == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint proposal residual failed: compute pipeline");
            return false;
        }
    }

    const char *contactSeedShaderSource = R"(
struct ContactPosition {
    position: vec4f,
    correction: vec4f,
};

struct ResidualBody {
    position: vec4f,
    rotation: vec4f,
};

struct Params {
    jointCount: u32,
    bodyCount: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> contactPositions: array<ContactPosition>;
@group(0) @binding(1) var<storage, read_write> bodies: array<ResidualBody>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let bodyIndex = gid.x;
    if (bodyIndex >= params.bodyCount) {
        return;
    }
    let seeded = contactPositions[bodyIndex].position;
    bodies[bodyIndex].position = vec4f(seeded.xyz, bodies[bodyIndex].position.w);
}
)";

    if (useContactSeed && (jointContactSeedPipeline == nullptr || jointContactSeedBindGroupLayout == nullptr))
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = contactSeedShaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint contact seed failed: shader module");
            return false;
        }

        wgpu::BindGroupLayoutEntry layoutEntries[3] = {};
        layoutEntries[0].binding = 0;
        layoutEntries[0].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        layoutEntries[0].buffer.minBindingSize = sizeof(GpuContactProposalOutput);
        layoutEntries[1].binding = 1;
        layoutEntries[1].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
        layoutEntries[1].buffer.minBindingSize = sizeof(GpuJointResidualBody);
        layoutEntries[2].binding = 2;
        layoutEntries[2].visibility = wgpu::ShaderStage::Compute;
        layoutEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
        layoutEntries[2].buffer.minBindingSize = sizeof(GpuJointTopologyParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 3;
        layoutDesc.entries = layoutEntries;
        jointContactSeedBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        if (jointContactSeedBindGroupLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint contact seed failed: bind layout");
            return false;
        }

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &jointContactSeedBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        if (pipelineLayout == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint contact seed failed: pipeline layout");
            return false;
        }

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        jointContactSeedPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (jointContactSeedPipeline == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint contact seed failed: compute pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry bindEntries[3] = {};
    bindEntries[0].binding = 0;
    bindEntries[0].buffer = jointTopologyInputBuffer;
    bindEntries[0].offset = 0;
    bindEntries[0].size = inputBytes;
    bindEntries[1].binding = 1;
    bindEntries[1].buffer = jointTopologyOutputBuffer;
    bindEntries[1].offset = 0;
    bindEntries[1].size = outputBytes;
    bindEntries[2].binding = 2;
    bindEntries[2].buffer = jointTopologyParamsBuffer;
    bindEntries[2].offset = 0;
    bindEntries[2].size = sizeof(params);

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = jointTopologyBindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = bindEntries;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
    if (bindGroup == nullptr)
    {
        setStatus(jointTopologyStatus, "Joint topology failed: bind group");
        return false;
    }

    wgpu::BindGroupEntry colorBindEntries[7] = {};
    colorBindEntries[0].binding = 0;
    colorBindEntries[0].buffer = jointTopologyInputBuffer;
    colorBindEntries[0].offset = 0;
    colorBindEntries[0].size = inputBytes;
    colorBindEntries[1].binding = 1;
    colorBindEntries[1].buffer = jointColorBuffer;
    colorBindEntries[1].offset = 0;
    colorBindEntries[1].size = colorBytes;
    colorBindEntries[2].binding = 2;
    colorBindEntries[2].buffer = jointColorCounterBuffer;
    colorBindEntries[2].offset = 0;
    colorBindEntries[2].size = colorCounterBytes;
    colorBindEntries[3].binding = 3;
    colorBindEntries[3].buffer = jointTopologyParamsBuffer;
    colorBindEntries[3].offset = 0;
    colorBindEntries[3].size = sizeof(params);
    colorBindEntries[4].binding = 4;
    colorBindEntries[4].buffer = jointResidualInputBuffer;
    colorBindEntries[4].offset = 0;
    colorBindEntries[4].size = residualInputBytes;
    colorBindEntries[5].binding = 5;
    colorBindEntries[5].buffer = jointResidualBodyBuffer;
    colorBindEntries[5].offset = 0;
    colorBindEntries[5].size = residualBodyBytes;
    colorBindEntries[6].binding = 6;
    colorBindEntries[6].buffer = jointResidualOutputBuffer;
    colorBindEntries[6].offset = 0;
    colorBindEntries[6].size = residualOutputBytes;

    wgpu::BindGroupDescriptor colorBindGroupDesc = {};
    colorBindGroupDesc.layout = jointColorBindGroupLayout;
    colorBindGroupDesc.entryCount = 7;
    colorBindGroupDesc.entries = colorBindEntries;
    wgpu::BindGroup colorBindGroup = device.CreateBindGroup(&colorBindGroupDesc);
    if (colorBindGroup == nullptr)
    {
        setStatus(jointTopologyStatus, "Joint coloring failed: bind group");
        return false;
    }

    wgpu::BindGroupEntry proposalBindEntries[5] = {};
    proposalBindEntries[0].binding = 0;
    proposalBindEntries[0].buffer = jointResidualInputBuffer;
    proposalBindEntries[0].offset = 0;
    proposalBindEntries[0].size = residualInputBytes;
    proposalBindEntries[1].binding = 1;
    proposalBindEntries[1].buffer = jointResidualBodyBuffer;
    proposalBindEntries[1].offset = 0;
    proposalBindEntries[1].size = residualBodyBytes;
    proposalBindEntries[2].binding = 2;
    proposalBindEntries[2].buffer = jointColorBuffer;
    proposalBindEntries[2].offset = 0;
    proposalBindEntries[2].size = colorBytes;
    proposalBindEntries[3].binding = 3;
    proposalBindEntries[3].buffer = jointProposalOutputBuffer;
    proposalBindEntries[3].offset = 0;
    proposalBindEntries[3].size = proposalOutputBytes;
    proposalBindEntries[4].binding = 4;
    proposalBindEntries[4].buffer = jointTopologyParamsBuffer;
    proposalBindEntries[4].offset = 0;
    proposalBindEntries[4].size = sizeof(params);

    wgpu::BindGroupDescriptor proposalBindGroupDesc = {};
    proposalBindGroupDesc.layout = jointProposalBindGroupLayout;
    proposalBindGroupDesc.entryCount = 5;
    proposalBindGroupDesc.entries = proposalBindEntries;
    wgpu::BindGroup proposalBindGroup = device.CreateBindGroup(&proposalBindGroupDesc);
    if (proposalBindGroup == nullptr)
    {
        setStatus(jointTopologyStatus, "Joint proposal failed: bind group");
        return false;
    }

    wgpu::BindGroupEntry proposalApplyBindEntries[3] = {};
    proposalApplyBindEntries[0].binding = 0;
    proposalApplyBindEntries[0].buffer = jointProposalOutputBuffer;
    proposalApplyBindEntries[0].offset = 0;
    proposalApplyBindEntries[0].size = proposalOutputBytes;
    proposalApplyBindEntries[1].binding = 1;
    proposalApplyBindEntries[1].buffer = jointResidualBodyBuffer;
    proposalApplyBindEntries[1].offset = 0;
    proposalApplyBindEntries[1].size = residualBodyBytes;
    proposalApplyBindEntries[2].binding = 2;
    proposalApplyBindEntries[2].buffer = jointTopologyParamsBuffer;
    proposalApplyBindEntries[2].offset = 0;
    proposalApplyBindEntries[2].size = sizeof(params);

    wgpu::BindGroupDescriptor proposalApplyBindGroupDesc = {};
    proposalApplyBindGroupDesc.layout = jointProposalApplyBindGroupLayout;
    proposalApplyBindGroupDesc.entryCount = 3;
    proposalApplyBindGroupDesc.entries = proposalApplyBindEntries;
    wgpu::BindGroup proposalApplyBindGroup = device.CreateBindGroup(&proposalApplyBindGroupDesc);
    if (proposalApplyBindGroup == nullptr)
    {
        setStatus(jointTopologyStatus, "Joint proposal apply failed: bind group");
        return false;
    }

    wgpu::BindGroupEntry proposalResidualBindEntries[5] = {};
    proposalResidualBindEntries[0].binding = 0;
    proposalResidualBindEntries[0].buffer = jointResidualInputBuffer;
    proposalResidualBindEntries[0].offset = 0;
    proposalResidualBindEntries[0].size = residualInputBytes;
    proposalResidualBindEntries[1].binding = 1;
    proposalResidualBindEntries[1].buffer = jointResidualBodyBuffer;
    proposalResidualBindEntries[1].offset = 0;
    proposalResidualBindEntries[1].size = residualBodyBytes;
    proposalResidualBindEntries[2].binding = 2;
    proposalResidualBindEntries[2].buffer = jointProposalOutputBuffer;
    proposalResidualBindEntries[2].offset = 0;
    proposalResidualBindEntries[2].size = proposalOutputBytes;
    proposalResidualBindEntries[3].binding = 3;
    proposalResidualBindEntries[3].buffer = jointProposalResidualBuffer;
    proposalResidualBindEntries[3].offset = 0;
    proposalResidualBindEntries[3].size = proposalResidualBytes;
    proposalResidualBindEntries[4].binding = 4;
    proposalResidualBindEntries[4].buffer = jointTopologyParamsBuffer;
    proposalResidualBindEntries[4].offset = 0;
    proposalResidualBindEntries[4].size = sizeof(params);

    wgpu::BindGroupDescriptor proposalResidualBindGroupDesc = {};
    proposalResidualBindGroupDesc.layout = jointProposalResidualBindGroupLayout;
    proposalResidualBindGroupDesc.entryCount = 5;
    proposalResidualBindGroupDesc.entries = proposalResidualBindEntries;
    wgpu::BindGroup proposalResidualBindGroup = device.CreateBindGroup(&proposalResidualBindGroupDesc);
    if (proposalResidualBindGroup == nullptr)
    {
        setStatus(jointTopologyStatus, "Joint proposal residual failed: bind group");
        return false;
    }

    wgpu::BindGroup contactSeedBindGroup = nullptr;
    if (useContactSeed)
    {
        wgpu::BindGroupEntry contactSeedEntries[3] = {};
        contactSeedEntries[0].binding = 0;
        contactSeedEntries[0].buffer = sphereContactFinalPositionBuffer;
        contactSeedEntries[0].offset = 0;
        contactSeedEntries[0].size = (uint64_t)world.bodies.size() * sizeof(GpuContactProposalOutput);
        contactSeedEntries[1].binding = 1;
        contactSeedEntries[1].buffer = jointResidualBodyBuffer;
        contactSeedEntries[1].offset = 0;
        contactSeedEntries[1].size = residualBodyBytes;
        contactSeedEntries[2].binding = 2;
        contactSeedEntries[2].buffer = jointTopologyParamsBuffer;
        contactSeedEntries[2].offset = 0;
        contactSeedEntries[2].size = sizeof(params);

        wgpu::BindGroupDescriptor contactSeedBindGroupDesc = {};
        contactSeedBindGroupDesc.layout = jointContactSeedBindGroupLayout;
        contactSeedBindGroupDesc.entryCount = 3;
        contactSeedBindGroupDesc.entries = contactSeedEntries;
        contactSeedBindGroup = device.CreateBindGroup(&contactSeedBindGroupDesc);
        if (contactSeedBindGroup == nullptr)
        {
            setStatus(jointTopologyStatus, "Joint contact seed failed: bind group");
            return false;
        }

        device.PushErrorScope(wgpu::ErrorFilter::Validation);
        wgpu::CommandEncoder seedEncoder = device.CreateCommandEncoder();
        wgpu::ComputePassEncoder seedPass = seedEncoder.BeginComputePass();
        seedPass.SetPipeline(jointContactSeedPipeline);
        seedPass.SetBindGroup(0, contactSeedBindGroup);
        seedPass.DispatchWorkgroups((uint32_t)((world.bodies.size() + 63) / 64));
        seedPass.End();
        wgpu::CommandBuffer seedCommands = seedEncoder.Finish();
        queue.Submit(1, &seedCommands);
        wgpu::PopErrorScopeStatus seedScopeStatus = wgpu::PopErrorScopeStatus::Error;
        wgpu::ErrorType seedErrorType = wgpu::ErrorType::Unknown;
        std::string seedErrorMessage;
        wgpu::Future seedScopeFuture = device.PopErrorScope(
            wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message)
            {
                seedScopeStatus = status;
                seedErrorType = type;
                seedErrorMessage = toString(message);
            });
        if (instance.WaitAny(seedScopeFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            seedScopeStatus != wgpu::PopErrorScopeStatus::Success)
        {
            setStatus(jointTopologyStatus, "Joint contact seed failed: validation scope");
            return false;
        }
        if (seedErrorType != wgpu::ErrorType::NoError)
        {
            snprintf(status, sizeof(status), "WebGPU validation error: %s", seedErrorMessage.c_str());
            snprintf(jointTopologyStatus, sizeof(jointTopologyStatus),
                     "Joint contact seed validation failed: %s", seedErrorMessage.c_str());
            return false;
        }
    }

    device.PushErrorScope(wgpu::ErrorFilter::Validation);
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(jointTopologyPipeline);
        pass.SetBindGroup(0, bindGroup);
        pass.DispatchWorkgroups((uint32_t)((joints.size() + 63) / 64));
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(jointColorPipeline);
        pass.SetBindGroup(0, colorBindGroup);
        pass.DispatchWorkgroups((uint32_t)((joints.size() + 63) / 64));
        pass.End();
    }
    for (int proposalIteration = 0; proposalIteration < proposalIterations; ++proposalIteration)
    {
        {
            wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
            pass.SetPipeline(jointProposalPipeline);
            pass.SetBindGroup(0, proposalBindGroup);
            pass.DispatchWorkgroups((uint32_t)((world.bodies.size() + 63) / 64));
            pass.End();
        }
        if (proposalIteration + 1 < proposalIterations)
        {
            wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
            pass.SetPipeline(jointProposalApplyPipeline);
            pass.SetBindGroup(0, proposalApplyBindGroup);
            pass.DispatchWorkgroups((uint32_t)((world.bodies.size() + 63) / 64));
            pass.End();
        }
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(jointProposalResidualPipeline);
        pass.SetBindGroup(0, proposalResidualBindGroup);
        pass.DispatchWorkgroups((uint32_t)((joints.size() + 63) / 64));
        pass.End();
    }
    if (useContactSeed)
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(jointProposalApplyPipeline);
        pass.SetBindGroup(0, proposalApplyBindGroup);
        pass.DispatchWorkgroups((uint32_t)((world.bodies.size() + 63) / 64));
        pass.End();
    }
    if (diagnosticReadback)
    {
        encoder.CopyBufferToBuffer(jointTopologyOutputBuffer, 0, jointTopologyReadbackBuffer, 0, topologyReadbackBytes);
        encoder.CopyBufferToBuffer(jointColorCounterBuffer, 0, jointTopologyReadbackBuffer, topologyReadbackBytes, colorReadbackBytes);
        encoder.CopyBufferToBuffer(jointResidualOutputBuffer, 0, jointTopologyReadbackBuffer, topologyReadbackBytes + colorReadbackBytes, residualOutputBytes);
        encoder.CopyBufferToBuffer(jointProposalOutputBuffer, 0, jointTopologyReadbackBuffer, topologyReadbackBytes + colorReadbackBytes + residualOutputBytes, proposalOutputBytes);
        encoder.CopyBufferToBuffer(jointProposalResidualBuffer, 0, jointTopologyReadbackBuffer, topologyReadbackBytes + colorReadbackBytes + residualOutputBytes + proposalOutputBytes, proposalResidualBytes);
    }
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
    wgpu::PopErrorScopeStatus jointScopeStatus = wgpu::PopErrorScopeStatus::Error;
    wgpu::ErrorType jointErrorType = wgpu::ErrorType::Unknown;
    std::string jointErrorMessage;
    wgpu::Future jointScopeFuture = device.PopErrorScope(
        wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message)
        {
            jointScopeStatus = status;
            jointErrorType = type;
            jointErrorMessage = toString(message);
        });
    if (instance.WaitAny(jointScopeFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        jointScopeStatus != wgpu::PopErrorScopeStatus::Success)
    {
        setStatus(jointTopologyStatus, "Joint proposal runtime failed: validation scope");
        return false;
    }
    if (jointErrorType != wgpu::ErrorType::NoError)
    {
        snprintf(status, sizeof(status), "WebGPU validation error: %s", jointErrorMessage.c_str());
        snprintf(jointTopologyStatus, sizeof(jointTopologyStatus),
                 "Joint proposal runtime validation failed: %s", jointErrorMessage.c_str());
        return false;
    }

    if (!diagnosticReadback)
    {
        jointTopologyBodyRefs = (int)expectedRefs;
        jointTopologyActiveBodies = (int)expectedActiveBodies;
        jointTopologyMaxPerBody = (int)expectedMaxPerBody;
        jointTopologyMismatches = 0;
        jointTopologyReadbackBytes = 0;
        jointColorReadbackBytes = 0;
        jointResidualReadbackBytes = 0;
        jointResidualMax = 0.0f;
        jointResidualRms = 0.0f;
        jointProposalReadbackBytes = 0;
        jointProposalActiveBodies = (int)expectedActiveBodies;
        jointProposalMaxPerBody = (int)expectedMaxPerBody;
        jointProposalMaxCorrection = 0.0f;
        jointProposalRmsCorrection = 0.0f;
        jointProposalResidualReadbackBytes = 0;
        jointProposalResidualAfterMax = 0.0f;
        jointProposalResidualAfterRms = 0.0f;
        jointProposalFinalPositionReady = 1;
        jointProposalFinalPositionBodyCount = (int)world.bodies.size();
        jointProposalFinalPositionBytes = (int)(useContactSeed ? residualBodyBytes : proposalOutputBytes);
        jointProposalFinalPositionAbsolute = useContactSeed ? 1 : 0;
        jointProposalSeededFromContact = useContactSeed ? 1 : 0;
        jointTopologyMs = elapsedMs(begin, SDL_GetPerformanceCounter());
        snprintf(jointTopologyStatus, sizeof(jointTopologyStatus),
                 "Joint proposal runtime passed: %d joints, %d refs, %d active bodies, colors %d, %d proposal iterations%s",
                 jointTopologyJoints, jointTopologyBodyRefs, jointTopologyActiveBodies, jointColorCount, jointProposalIterations,
                 useContactSeed ? ", contact seeded" : "");
        return true;
    }

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = jointTopologyReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, readbackBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
    {
        setStatus(jointTopologyStatus, "Joint topology failed: readback map");
        return false;
    }

    const uint32_t *gpuCounts = (const uint32_t *)jointTopologyReadbackBuffer.GetConstMappedRange(0, readbackBytes);
    if (!gpuCounts)
    {
        jointTopologyReadbackBuffer.Unmap();
        setStatus(jointTopologyStatus, "Joint topology failed: readback range");
        return false;
    }

    uint32_t gpuRefs = 0;
    uint32_t gpuActiveBodies = 0;
    uint32_t gpuMaxPerBody = 0;
    int mismatches = 0;
    for (size_t i = 0; i < world.bodies.size(); ++i)
    {
        uint32_t count = gpuCounts[i];
        gpuRefs += count;
        if (count > 0)
            gpuActiveBodies++;
        gpuMaxPerBody = std::max(gpuMaxPerBody, count);
        if (count != expectedBodyCounts[i])
            mismatches++;
    }
    const uint32_t *gpuColorConflictCount = (const uint32_t *)((const uint8_t *)gpuCounts + topologyReadbackBytes);
    jointColorConflicts = gpuColorConflictCount ? (int)(*gpuColorConflictCount) : 1;
    const float *gpuResidualSq = (const float *)((const uint8_t *)gpuCounts + topologyReadbackBytes + colorReadbackBytes);
    double residualSumSq = 0.0;
    float residualMaxSq = 0.0f;
    if (gpuResidualSq)
    {
        for (size_t i = 0; i < residualJoints.size(); ++i)
        {
            float value = gpuResidualSq[i];
            if (!std::isfinite(value) || value < 0.0f)
                value = 0.0f;
            residualSumSq += (double)value;
            if (value > residualMaxSq)
                residualMaxSq = value;
        }
    }
    const GpuJointProposalOutput *gpuProposals =
        (const GpuJointProposalOutput *)((const uint8_t *)gpuCounts + topologyReadbackBytes + colorReadbackBytes + residualOutputBytes);
    double proposalSumSq = 0.0;
    float proposalMaxSq = 0.0f;
    int proposalActiveBodies = 0;
    int proposalMaxPerBody = 0;
    if (gpuProposals)
    {
        for (size_t i = 0; i < world.bodies.size(); ++i)
        {
            const GpuJointProposalOutput &proposal = gpuProposals[i];
            if (proposal.jointCount == 0)
                continue;
            proposalActiveBodies++;
            proposalMaxPerBody = std::max(proposalMaxPerBody, (int)proposal.jointCount);
            float x = proposal.correction[0];
            float y = proposal.correction[1];
            float z = proposal.correction[2];
            if (!std::isfinite(x))
                x = 0.0f;
            if (!std::isfinite(y))
                y = 0.0f;
            if (!std::isfinite(z))
                z = 0.0f;
            float correctionSq = x * x + y * y + z * z;
            proposalSumSq += (double)correctionSq;
            if (correctionSq > proposalMaxSq)
                proposalMaxSq = correctionSq;
        }
    }
    const float *gpuProposalResidualSq =
        (const float *)((const uint8_t *)gpuCounts + topologyReadbackBytes + colorReadbackBytes + residualOutputBytes + proposalOutputBytes);
    double proposalResidualSumSq = 0.0;
    float proposalResidualMaxSq = 0.0f;
    if (gpuProposalResidualSq)
    {
        for (size_t i = 0; i < residualJoints.size(); ++i)
        {
            float value = gpuProposalResidualSq[i];
            if (!std::isfinite(value) || value < 0.0f)
                value = 0.0f;
            proposalResidualSumSq += (double)value;
            if (value > proposalResidualMaxSq)
                proposalResidualMaxSq = value;
        }
    }
    jointTopologyReadbackBuffer.Unmap();

    jointTopologyBodyRefs = (int)gpuRefs;
    jointTopologyActiveBodies = (int)gpuActiveBodies;
    jointTopologyMaxPerBody = (int)gpuMaxPerBody;
    jointTopologyMismatches = mismatches +
                              (gpuRefs == expectedRefs ? 0 : 1) +
                              (gpuActiveBodies == expectedActiveBodies ? 0 : 1) +
                              (gpuMaxPerBody == expectedMaxPerBody ? 0 : 1);
    jointTopologyReadbackBytes = (int)topologyReadbackBytes;
    jointColorReadbackBytes = (int)colorReadbackBytes;
    jointResidualReadbackBytes = (int)residualOutputBytes;
    jointResidualMax = sqrtf(residualMaxSq);
    jointResidualRms = residualJoints.empty() ? 0.0f : (float)std::sqrt(residualSumSq / (double)residualJoints.size());
    jointProposalReadbackBytes = (int)proposalOutputBytes;
    jointProposalActiveBodies = proposalActiveBodies;
    jointProposalMaxPerBody = proposalMaxPerBody;
    jointProposalMaxCorrection = sqrtf(proposalMaxSq);
    jointProposalRmsCorrection = proposalActiveBodies == 0 ? 0.0f : (float)std::sqrt(proposalSumSq / (double)proposalActiveBodies);
    jointProposalResidualReadbackBytes = (int)proposalResidualBytes;
    jointProposalResidualAfterMax = sqrtf(proposalResidualMaxSq);
    jointProposalResidualAfterRms = residualJoints.empty() ? 0.0f : (float)std::sqrt(proposalResidualSumSq / (double)residualJoints.size());
    jointProposalFinalPositionReady = 1;
    jointProposalFinalPositionBodyCount = (int)world.bodies.size();
    jointProposalFinalPositionBytes = (int)(useContactSeed ? residualBodyBytes : proposalOutputBytes);
    jointProposalFinalPositionAbsolute = useContactSeed ? 1 : 0;
    jointProposalSeededFromContact = useContactSeed ? 1 : 0;
    jointTopologyMs = elapsedMs(begin, SDL_GetPerformanceCounter());

    snprintf(jointTopologyStatus, sizeof(jointTopologyStatus),
             "Joint topology passed: %d joints, %d refs, %d active bodies, max %d, mismatches %d, colors %d, conflicts %d, residual %.6f -> %.6f, proposal max %.6f%s",
             jointTopologyJoints, jointTopologyBodyRefs, jointTopologyActiveBodies,
             jointTopologyMaxPerBody, jointTopologyMismatches, jointColorCount, jointColorConflicts,
             jointResidualMax, jointProposalResidualAfterMax, jointProposalMaxCorrection,
             useContactSeed ? ", contact seeded" : "");
    return jointTopologyMismatches == 0 && jointColorConflicts == 0;
#elif AVBD_ENABLE_WEBGPU
    setStatus(jointTopologyStatus, "Joint topology skipped: Dawn not linked");
    return false;
#else
    setStatus(jointTopologyStatus, "Joint topology skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::runSweepAndPrunePairs(const SimWorld &world, std::vector<BroadphasePair> &pairs, std::vector<ExternalManifoldContact> *sphereContacts, bool counterOnly, bool validateResidentContacts, bool suppressSphereSpherePairs, bool suppressSphereGroundPairs, bool emitSphereGroundContacts, bool runResidentContactSolve, bool suppressNonSpherePairOutput, bool directSpherePositionSolve, int residentContactIterations, float residentContactRelaxation, bool forceResidentCounterlessContacts)
{
    pairs.clear();
    if (sphereContacts)
        sphereContacts->clear();
    sapMs = 0.0f;
    sapCandidates = 0;
    sapSphereHits = 0;
    sapAllPairsSphereHits = 0;
    sapMissedSphereHits = 0;
    sapMismatches = 0;
    sapCounterReadbackBytes = 0;
    sapCounterReadbackMs = 0.0f;
    sapPairReadbackBytes = 0;
    sapPairReadbackMs = 0.0f;
    sphereContactCount = 0;
    sphereContactExternalContacts = 0;
    sphereContactExternalGroundContacts = 0;
    sphereContactReadbackBytes = 0;
    sphereContactMs = 0.0f;
    sphereContactReadbackMs = 0.0f;
    sphereContactBodyRefs = 0;
    sphereContactActiveBodies = 0;
    sphereContactMaxPerBody = 0;
    sphereContactAvgPerActiveBody = 0.0f;
    sphereContactAdjacencyReadbackBytes = 0;
    sphereContactAdjacencyListBytes = 0;
    sphereContactAdjacencyCapacity = 0;
    sphereContactAdjacencyWrittenRefs = 0;
    sphereContactAdjacencyOverflowRefs = 0;
    sphereContactAdjacencyMs = 0.0f;
    sphereContactGatherRefs = 0;
    sphereContactGatherActiveBodies = 0;
    sphereContactGatherMaxPerBody = 0;
    sphereContactGatherMismatches = 0;
    sphereContactGatherReadbackBytes = 0;
    sphereContactGatherNormalChecksum = 0.0f;
    sphereContactProposalActiveBodies = 0;
    sphereContactProposalMaxCorrection = 0.0f;
    sphereContactProposalCorrectionChecksum = 0.0f;
    sphereContactGatherMs = 0.0f;
    sphereContactProposalOutputActiveBodies = 0;
    sphereContactProposalOutputReadbackBytes = 0;
    sphereContactProposalOutputMaxDelta = 0.0f;
    sphereContactProposalOutputChecksum = 0.0f;
    sphereContactProposalOutputMs = 0.0f;
    sphereContactProposalResidualReadbackBytes = 0;
    sphereContactProposalResidualBeforeMax = 0.0f;
    sphereContactProposalResidualAfterMax = 0.0f;
    sphereContactProposalResidualBeforeChecksum = 0.0f;
    sphereContactProposalResidualAfterChecksum = 0.0f;
    sphereContactProposalResidualMs = 0.0f;
    sphereContactIterationCount = 0;
    sphereContactIterationMs = 0.0f;
    sphereContactIterationRelaxation = 0.10f;
    sphereContactIterationResidualAfterMax = 0.0f;
    sphereContactIterationResidualAfterChecksum = 0.0f;
    sphereContactFinalPositionReady = 0;
    sphereContactFinalPositionBodyCount = 0;
    sphereContactFinalPositionBytes = 0;
    sphereContactFinalPositionSource = 0;
    sphereContactFinalReferencePositions.clear();
    sphereContactAppliedPositionBodies = 0;
    sphereContactAppliedPositionReadbackBytes = 0;
    sphereContactAppliedPositionMaxDelta = 0.0f;
    sphereContactAppliedPositionChecksum = 0.0f;
    sphereContactAppliedPositionMs = 0.0f;
    sphereContactAppliedPositionWaitMs = 0.0f;
    sphereContactAppliedPositionCpuMs = 0.0f;
    sphereContactFinalPositionReadbackDeferred = 0;
    sphereContactFinalPositionAsyncReadbackScheduled = 0;
    sphereContactFinalPositionAsyncReadbackConsumed = 0;
    sphereContactFinalPositionAsyncReadbackDropped = 0;
    sphereContactFinalPositionAsyncReadbackWaitMs = 0.0f;
    sphereGroundReceiverCount = 0;
    sphereGroundDynamicSphereCount = 0;
    sphereGroundCandidateCount = 0;
    directSphereCylinderBodyCount = 0;
    directSphereCylinderCandidateCount = 0;
    directSphereCapsuleBodyCount = 0;
    directSphereCapsuleCandidateCount = 0;
    directSphereBoxBodyCount = 0;
    directSphereBoxCandidateCount = 0;
    directBoxBodyCount = 0;
    directBoxPairCandidateCount = 0;
    directRoundBodyCount = 0;
    directRoundPairCandidateCount = 0;
    directGpuContactRecordCount = 0;
    directGpuRoundPairCandidateCount = 0;
    directGpuBoxPairCandidateCount = 0;
    directGpuCounterReadbackBytes = 0;
    directGpuCounterReadbackMs = 0.0f;
    directSphereContactAppliedPositionBodies = 0;
    directGroundAppliedPositionBodies = 0;
    sphereGroundTop = 0.0f;
    sphereGroundMs = 0.0f;
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    sphereContactFinalPositionBuffer = nullptr;
    sphereContactFinalPositionBufferBytes = 0;
#endif
    for (int axis = 0; axis < 3; ++axis)
    {
        sapAxisCandidates[axis] = 0;
        sapAxisSphereHits[axis] = 0;
        sapAxisMissedSphereHits[axis] = 0;
    }

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
    {
        setStatus(sapStatus, "SAP pairs skipped: WebGPU device not ready");
        return false;
    }

    std::vector<GpuPairBody> bodies;
    std::vector<BodyId> bodyIds;
    bodies.reserve(world.bodies.size());
    bodyIds.reserve(world.bodies.size());
    float minCenter[3] = {INFINITY, INFINITY, INFINITY};
    float maxCenter[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (BodyId bodyId = 0; bodyId < world.bodies.size(); ++bodyId)
    {
        const SimBodyData &body = world.bodies[bodyId];
        if (!body.active || !body.source)
            continue;

        GpuPairBody gpuBody = {};
        gpuBody.position[0] = body.positionLin.x;
        gpuBody.position[1] = body.positionLin.y;
        gpuBody.position[2] = body.positionLin.z;
        gpuBody.position[3] = body.radius;
        bodies.push_back(gpuBody);
        bodyIds.push_back(bodyId);
        for (int axis = 0; axis < 3; ++axis)
        {
            minCenter[axis] = std::min(minCenter[axis], gpuBody.position[axis]);
            maxCenter[axis] = std::max(maxCenter[axis], gpuBody.position[axis]);
        }
    }

    if (bodies.size() < 2)
    {
        setStatus(sapStatus, "SAP pairs skipped: fewer than two bodies");
        return false;
    }

    sapBestAxis = 0;
    float bestSpan = maxCenter[0] - minCenter[0];
    for (int axis = 1; axis < 3; ++axis)
    {
        float span = maxCenter[axis] - minCenter[axis];
        if (span > bestSpan)
        {
            bestSpan = span;
            sapBestAxis = axis;
        }
    }

    int otherA = (sapBestAxis + 1) % 3;
    int otherB = (sapBestAxis + 2) % 3;
    std::vector<GpuSapInterval> intervals;
    intervals.reserve(bodies.size());
    int intervalGroundReceivers = 0;
    int intervalDynamicSpheres = 0;
    for (uint32_t i = 0; i < bodies.size(); ++i)
    {
        const GpuPairBody &body = bodies[i];
        float center = body.position[sapBestAxis];
        float radius = body.position[3];
        GpuSapInterval interval = {};
        interval.minX = center - radius;
        interval.maxX = center + radius;
        interval.y = body.position[otherA];
        interval.z = body.position[otherB];
        interval.radius = radius;
        interval.index = (uint32_t)bodyIds[i];
        interval.shapeType = (uint32_t)world.bodies[bodyIds[i]].shape.type;
        const SimBodyData &simBody = world.bodies[bodyIds[i]];
        interval.pad2 = floatToBits(simBody.shape.radius > 0.0f ? simBody.shape.radius : simBody.radius);
        interval.shapeSizeXBits = floatToBits(std::max(0.0f, simBody.shape.size.x * 0.5f));
        interval.shapeSizeYBits = floatToBits(std::max(0.0f, simBody.shape.size.y * 0.5f));
        interval.shapeSizeZBits = floatToBits(std::max(0.0f, simBody.shape.size.z * 0.5f));
        interval.orientationX = simBody.positionAng.x;
        interval.orientationY = simBody.positionAng.y;
        interval.orientationZ = simBody.positionAng.z;
        interval.orientationW = simBody.positionAng.w;
        if (simBody.shape.type == RIGID_SHAPE_CYLINDER ||
            simBody.shape.type == RIGID_SHAPE_CAPSULE)
            interval.groundTopBits = floatToBits(std::max(0.0f, simBody.shape.halfLength));
        if (simBody.mass > 0.0f && simBody.shape.type == RIGID_SHAPE_SPHERE)
            ++intervalDynamicSpheres;
        if (isGpuStaticGroundReceiver(simBody))
        {
            interval.pad1 = 1u;
            interval.groundTopBits = floatToBits(simBody.positionLin.z + simBody.shape.size.z * 0.5f);
            ++intervalGroundReceivers;
        }
        intervals.push_back(interval);
    }
    if (emitSphereGroundContacts)
    {
        sphereGroundReceiverCount = intervalGroundReceivers;
        sphereGroundDynamicSphereCount = intervalDynamicSpheres;
    }

    std::vector<GpuSapPair> directBoxPairs;
    if (directSpherePositionSolve)
    {
        std::vector<BodyId> dynamicSphereIds;
        std::vector<BodyId> dynamicRoundIds;
        std::vector<BodyId> cylinderIds;
        std::vector<BodyId> capsuleIds;
        std::vector<BodyId> boxIds;
        dynamicSphereIds.reserve(intervalDynamicSpheres);
        dynamicRoundIds.reserve(world.bodies.size());
        for (BodyId bodyId = 0; bodyId < world.bodies.size(); ++bodyId)
        {
            const SimBodyData &simBody = world.bodies[bodyId];
            if (!simBody.active || !simBody.source)
                continue;
            if (simBody.mass > 0.0f &&
                (simBody.shape.type == RIGID_SHAPE_SPHERE ||
                 simBody.shape.type == RIGID_SHAPE_CAPSULE ||
                 simBody.shape.type == RIGID_SHAPE_CYLINDER))
                dynamicRoundIds.push_back(bodyId);
            if (simBody.mass > 0.0f && simBody.shape.type == RIGID_SHAPE_SPHERE)
                dynamicSphereIds.push_back(bodyId);
            else if (simBody.shape.type == RIGID_SHAPE_CYLINDER)
                cylinderIds.push_back(bodyId);
            else if (simBody.shape.type == RIGID_SHAPE_CAPSULE)
                capsuleIds.push_back(bodyId);
            else if (simBody.shape.type == RIGID_SHAPE_BOX && !isGpuStaticGroundReceiver(simBody))
                boxIds.push_back(bodyId);
        }

        directSphereCylinderBodyCount = (int)cylinderIds.size();
        directSphereCapsuleBodyCount = (int)capsuleIds.size();
        directSphereBoxBodyCount = (int)boxIds.size();
        directBoxBodyCount = (int)boxIds.size();
        directRoundBodyCount = (int)dynamicRoundIds.size();
        int cylinderCandidates = 0;
        int capsuleCandidates = 0;
        int boxCandidates = 0;
        int boxPairCandidates = 0;
        int roundPairCandidates = 0;
        auto roundRadius = [&](const SimBodyData &body) -> float {
            if (body.shape.type == RIGID_SHAPE_SPHERE)
                return body.shape.radius > 0.0f ? body.shape.radius : body.radius;
            if (body.shape.type == RIGID_SHAPE_CAPSULE || body.shape.type == RIGID_SHAPE_CYLINDER)
                return body.shape.radius > 0.0f ? body.shape.radius : body.radius;
            return 0.0f;
        };
        auto roundEndpoint = [&](const SimBodyData &body, float signValue) -> float3 {
            float halfLength = body.shape.type == RIGID_SHAPE_CAPSULE || body.shape.type == RIGID_SHAPE_CYLINDER
                ? std::max(0.0f, body.shape.halfLength)
                : 0.0f;
            if (halfLength <= 0.0f)
                return body.positionLin;
            return body.positionLin + rotate(body.positionAng, float3{0.0f, 0.0f, signValue * halfLength});
        };
        auto segmentDistanceSquared = [](float3 p1, float3 q1, float3 p2, float3 q2) -> float {
            float3 d1 = q1 - p1;
            float3 d2 = q2 - p2;
            float3 r = p1 - p2;
            float a = dot(d1, d1);
            float e = dot(d2, d2);
            float f = dot(d2, r);
            float s = 0.0f;
            float t = 0.0f;
            if (a <= 0.000001f && e <= 0.000001f)
            {
                s = 0.0f;
                t = 0.0f;
            }
            else if (a <= 0.000001f)
            {
                s = 0.0f;
                t = clamp(f / std::max(e, 0.000001f), 0.0f, 1.0f);
            }
            else
            {
                float c = dot(d1, r);
                if (e <= 0.000001f)
                {
                    t = 0.0f;
                    s = clamp(-c / std::max(a, 0.000001f), 0.0f, 1.0f);
                }
                else
                {
                    float b = dot(d1, d2);
                    float denom = a * e - b * b;
                    if (fabsf(denom) > 0.000001f)
                        s = clamp((b * f - c * e) / denom, 0.0f, 1.0f);
                    else
                        s = 0.0f;
                    t = (b * s + f) / std::max(e, 0.000001f);
                    if (t < 0.0f)
                    {
                        t = 0.0f;
                        s = clamp(-c / std::max(a, 0.000001f), 0.0f, 1.0f);
                    }
                    else if (t > 1.0f)
                    {
                        t = 1.0f;
                        s = clamp((b - c) / std::max(a, 0.000001f), 0.0f, 1.0f);
                    }
                }
            }
            float3 closestA = p1 + d1 * s;
            float3 closestB = p2 + d2 * t;
            float3 delta = closestA - closestB;
            return dot(delta, delta);
        };
        auto sphereBoxOverlaps = [](float3 spherePos, float sphereRadius, const SimBodyData &boxBody) -> bool {
            float hx = std::max(0.0f, boxBody.shape.size.x * 0.5f);
            float hy = std::max(0.0f, boxBody.shape.size.y * 0.5f);
            float hz = std::max(0.0f, boxBody.shape.size.z * 0.5f);
            if (sphereRadius <= 0.0f || hx <= 0.0f || hy <= 0.0f || hz <= 0.0f)
                return false;

            float3 local = rotate(conjugate(boxBody.positionAng), spherePos - boxBody.positionLin);
            float gapX = std::max(std::fabs(local.x) - hx, 0.0f);
            float gapY = std::max(std::fabs(local.y) - hy, 0.0f);
            float gapZ = std::max(std::fabs(local.z) - hz, 0.0f);
            return gapX * gapX + gapY * gapY + gapZ * gapZ < sphereRadius * sphereRadius;
        };
        auto boxAxis = [](const SimBodyData &boxBody, int axis) -> float3 {
            if (axis == 0)
                return rotate(boxBody.positionAng, float3{1.0f, 0.0f, 0.0f});
            if (axis == 1)
                return rotate(boxBody.positionAng, float3{0.0f, 1.0f, 0.0f});
            return rotate(boxBody.positionAng, float3{0.0f, 0.0f, 1.0f});
        };
        auto boxHalfSize = [](const SimBodyData &boxBody, int axis) -> float {
            if (axis == 0)
                return std::max(0.0f, boxBody.shape.size.x * 0.5f);
            if (axis == 1)
                return std::max(0.0f, boxBody.shape.size.y * 0.5f);
            return std::max(0.0f, boxBody.shape.size.z * 0.5f);
        };
        auto boxProjectionRadius = [&](const SimBodyData &boxBody, float3 axis) -> float {
            return boxHalfSize(boxBody, 0) * std::fabs(dot(boxAxis(boxBody, 0), axis)) +
                   boxHalfSize(boxBody, 1) * std::fabs(dot(boxAxis(boxBody, 1), axis)) +
                   boxHalfSize(boxBody, 2) * std::fabs(dot(boxAxis(boxBody, 2), axis));
        };
        auto boxBoxOverlaps = [&](const SimBodyData &a, const SimBodyData &b) -> bool {
            if (boxHalfSize(a, 0) <= 0.0f || boxHalfSize(a, 1) <= 0.0f || boxHalfSize(a, 2) <= 0.0f ||
                boxHalfSize(b, 0) <= 0.0f || boxHalfSize(b, 1) <= 0.0f || boxHalfSize(b, 2) <= 0.0f)
                return false;
            float3 delta = b.positionLin - a.positionLin;
            auto separatedOnAxis = [&](float3 axis) -> bool {
                float axisLength = length(axis);
                if (axisLength <= 0.000001f)
                    return false;
                axis = axis / axisLength;
                float separation = std::fabs(dot(delta, axis));
                return separation >= boxProjectionRadius(a, axis) + boxProjectionRadius(b, axis);
            };
            for (int axisIndex = 0; axisIndex < 3; ++axisIndex)
            {
                if (separatedOnAxis(boxAxis(a, axisIndex)))
                    return false;
            }
            for (int axisIndex = 0; axisIndex < 3; ++axisIndex)
            {
                if (separatedOnAxis(boxAxis(b, axisIndex)))
                    return false;
            }
            for (int axisA = 0; axisA < 3; ++axisA)
                for (int axisB = 0; axisB < 3; ++axisB)
                    if (separatedOnAxis(cross(boxAxis(a, axisA), boxAxis(b, axisB))))
                        return false;
            return true;
        };
        // This CPU SAT prefilter is only a small-scene bridge until the GPU
        // broadphase emits box pairs directly.  Dense soft bodies can have
        // thousands of boxes, which turns this into an accidental O(n^2) CPU
        // broadphase inside the experimental GPU path.
        static const size_t maxCpuDirectBoxPairPrefilterBodies = 512;
        if (boxIds.size() <= maxCpuDirectBoxPairPrefilterBodies)
        {
            directBoxPairs.reserve(boxIds.size());
            for (size_t i = 0; i < boxIds.size(); ++i)
            {
                const SimBodyData &a = world.bodies[boxIds[i]];
                for (size_t j = i + 1; j < boxIds.size(); ++j)
                {
                    const SimBodyData &b = world.bodies[boxIds[j]];
                    if (boxBoxOverlaps(a, b))
                    {
                        ++boxPairCandidates;
                        directBoxPairs.push_back(GpuSapPair{boxIds[i], boxIds[j]});
                    }
                }
            }
        }
        if (dynamicRoundIds.size() <= 2048)
        {
            for (size_t i = 0; i < dynamicRoundIds.size(); ++i)
            {
                const SimBodyData &a = world.bodies[dynamicRoundIds[i]];
                float radiusA = roundRadius(a);
                if (radiusA <= 0.0f)
                    continue;
                for (size_t j = i + 1; j < dynamicRoundIds.size(); ++j)
                {
                    const SimBodyData &b = world.bodies[dynamicRoundIds[j]];
                    float radiusB = roundRadius(b);
                    if (radiusB <= 0.0f)
                        continue;
                    float radius = radiusA + radiusB;
                    if (segmentDistanceSquared(roundEndpoint(a, -1.0f), roundEndpoint(a, 1.0f),
                                               roundEndpoint(b, -1.0f), roundEndpoint(b, 1.0f)) < radius * radius)
                        ++roundPairCandidates;
                }
                if (a.shape.type == RIGID_SHAPE_CAPSULE || a.shape.type == RIGID_SHAPE_CYLINDER)
                {
                    for (BodyId boxId : boxIds)
                    {
                        const SimBodyData &boxBody = world.bodies[boxId];
                        if (sphereBoxOverlaps(a.positionLin, radiusA, boxBody) ||
                            sphereBoxOverlaps(roundEndpoint(a, -1.0f), radiusA, boxBody) ||
                            sphereBoxOverlaps(roundEndpoint(a, 1.0f), radiusA, boxBody))
                            ++roundPairCandidates;
                    }
                }
            }
        }
        for (BodyId sphereId : dynamicSphereIds)
        {
            const SimBodyData &sphereBody = world.bodies[sphereId];
            float sphereRadius = sphereBody.shape.radius > 0.0f ? sphereBody.shape.radius : sphereBody.radius;
            for (BodyId cylinderId : cylinderIds)
            {
                const SimBodyData &cylinderBody = world.bodies[cylinderId];
                float cylinderRadius = cylinderBody.shape.radius > 0.0f ? cylinderBody.shape.radius : cylinderBody.radius;
                float cylinderHalfLength = std::max(0.0f, cylinderBody.shape.halfLength);
                if (sphereRadius <= 0.0f || cylinderRadius <= 0.0f || cylinderHalfLength <= 0.0f)
                    continue;

                float3 local = rotate(conjugate(cylinderBody.positionAng), sphereBody.positionLin - cylinderBody.positionLin);
                float radialDistance = std::sqrt(local.x * local.x + local.y * local.y);
                float radialGap = std::max(radialDistance - cylinderRadius, 0.0f);
                float capGap = std::max(std::fabs(local.z) - cylinderHalfLength, 0.0f);
                if (radialGap * radialGap + capGap * capGap < sphereRadius * sphereRadius)
                    ++cylinderCandidates;
            }
            for (BodyId capsuleId : capsuleIds)
            {
                const SimBodyData &capsuleBody = world.bodies[capsuleId];
                float capsuleRadius = capsuleBody.shape.radius > 0.0f ? capsuleBody.shape.radius : capsuleBody.radius;
                float capsuleHalfLength = std::max(0.0f, capsuleBody.shape.halfLength);
                if (sphereRadius <= 0.0f || capsuleRadius <= 0.0f || capsuleHalfLength <= 0.0f)
                    continue;

                float3 local = rotate(conjugate(capsuleBody.positionAng), sphereBody.positionLin - capsuleBody.positionLin);
                float closestZ = clamp(local.z, -capsuleHalfLength, capsuleHalfLength);
                float3 delta = local - float3{0.0f, 0.0f, closestZ};
                float radius = sphereRadius + capsuleRadius;
                if (dot(delta, delta) < radius * radius)
                    ++capsuleCandidates;
            }
            for (BodyId boxId : boxIds)
            {
                const SimBodyData &boxBody = world.bodies[boxId];
                float hx = std::max(0.0f, boxBody.shape.size.x * 0.5f);
                float hy = std::max(0.0f, boxBody.shape.size.y * 0.5f);
                float hz = std::max(0.0f, boxBody.shape.size.z * 0.5f);
                if (sphereRadius <= 0.0f || hx <= 0.0f || hy <= 0.0f || hz <= 0.0f)
                    continue;

                float3 local = rotate(conjugate(boxBody.positionAng), sphereBody.positionLin - boxBody.positionLin);
                float gapX = std::max(std::fabs(local.x) - hx, 0.0f);
                float gapY = std::max(std::fabs(local.y) - hy, 0.0f);
                float gapZ = std::max(std::fabs(local.z) - hz, 0.0f);
                if (gapX * gapX + gapY * gapY + gapZ * gapZ < sphereRadius * sphereRadius)
                    ++boxCandidates;
            }
        }
        directSphereCylinderCandidateCount = cylinderCandidates;
        directSphereCapsuleCandidateCount = capsuleCandidates;
        directSphereBoxCandidateCount = boxCandidates;
        directBoxPairCandidateCount = boxPairCandidates;
        directRoundPairCandidateCount = roundPairCandidates;
    }

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
        setStatus(sapStatus, "SAP pairs failed: buffer allocation");
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
    groundTopBits: u32,
    pad2: u32,
    shapeSizeXBits: u32,
    shapeSizeYBits: u32,
    shapeSizeZBits: u32,
    orientationX: f32,
    orientationY: f32,
    orientationZ: f32,
    orientationW: f32,
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
            setStatus(sapStatus, "SAP pairs failed: sort shader module");
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
        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &sapSortBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        sapSortPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (sapSortPipeline == nullptr)
        {
            setStatus(sapStatus, "SAP pairs failed: sort pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry sortEntries[2] = {};
    sortEntries[0].binding = 0;
    sortEntries[0].buffer = sapIntervalBuffer;
    sortEntries[0].size = intervalBytes;
    sortEntries[1].binding = 1;
    sortEntries[1].buffer = sapParamsBuffer;
    sortEntries[1].size = sizeof(GpuSortParams);
    wgpu::BindGroupDescriptor sortBindGroupDesc = {};
    sortBindGroupDesc.layout = sapSortBindGroupLayout;
    sortBindGroupDesc.entryCount = 2;
    sortBindGroupDesc.entries = sortEntries;
    wgpu::BindGroup sortBindGroup = device.CreateBindGroup(&sortBindGroupDesc);
    if (sortBindGroup == nullptr)
    {
        setStatus(sapStatus, "SAP pairs failed: sort bind group");
        return false;
    }

    const bool pureCounterOnly = counterOnly && !runResidentContactSolve;
    const bool skipPairOutput = pureCounterOnly || suppressNonSpherePairOutput;
    const bool emitExternalSphereContacts = sphereContacts != nullptr && suppressSphereSpherePairs;
    const bool residentCounterlessContacts =
        runResidentContactSolve &&
        !validateResidentContacts &&
        !counterOnly &&
        sphereContacts == nullptr &&
        (forceResidentCounterlessContacts || environmentFlagEnabled("AVBD_GPU_RESIDENT_COUNTERLESS_CONTACTS"));
    GpuSapParams pairParams = {};
    pairParams.count = itemCount;
    pairParams.pad0 = (emitExternalSphereContacts || (runResidentContactSolve && (counterOnly || suppressSphereSpherePairs))) ? 1u : 0u;
    pairParams.pad1 = emitSphereGroundContacts ? 1u : 0u;
    pairParams.pad2 = (suppressSphereGroundPairs ? 1u : 0u) |
                      (skipPairOutput ? 2u : 0u) |
                      ((uint32_t)sapBestAxis << 8u);
    if (sapPairParamsBuffer == nullptr)
    {
        wgpu::BufferDescriptor pairParamsDesc = {};
        pairParamsDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                           (uint64_t)wgpu::BufferUsage::CopyDst);
        pairParamsDesc.size = sizeof(GpuSapParams);
        sapPairParamsBuffer = device.CreateBuffer(&pairParamsDesc);
    }
    if (sapPairParamsBuffer == nullptr)
    {
        setStatus(sapStatus, "SAP pairs failed: pair params buffer");
        return false;
    }

    const uint64_t maxPairBytes = 256ull * 1024ull * 1024ull;
    uint64_t maxPairCount = (uint64_t)itemCount * (uint64_t)(itemCount - 1u) / 2ull;
    const uint64_t fullPairCapacityBytes = maxPairCount * sizeof(GpuSapPair);
    const uint64_t compactContactCapacitySlots = std::max<uint64_t>(1ull, fullPairCapacityBytes / sizeof(GpuSphereContact));
    const uint64_t compactContactCapacityBytes = compactContactCapacitySlots * sizeof(GpuSphereContact);
    uint64_t pairCapacityBytes = skipPairOutput ? sizeof(GpuSapPair) : fullPairCapacityBytes;
    if (fullPairCapacityBytes > maxPairBytes)
    {
        snprintf(sapStatus, sizeof(sapStatus), "SAP pairs fallback: %llu possible pairs exceed output cap",
                 (unsigned long long)maxPairCount);
        return false;
    }
    if (pairCapacityBytes == 0)
        pairCapacityBytes = sizeof(GpuSapPair);

    if (sapPairOutputBuffer == nullptr || sapPairOutputBufferBytes < pairCapacityBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopySrc);
        desc.size = alignUp(pairCapacityBytes, 4096);
        sapPairOutputBuffer = device.CreateBuffer(&desc);
        sapPairOutputBufferBytes = sapPairOutputBuffer == nullptr ? 0 : desc.size;
    }
    if (sapPairOutputBuffer == nullptr)
    {
        setStatus(sapStatus, "SAP pairs failed: pair output buffer allocation");
        return false;
    }
    const bool contactOutputEnabled = pairParams.pad0 != 0u || pairParams.pad1 != 0u;
    const uint64_t contactCapacityBytes = contactOutputEnabled
        ? compactContactCapacityBytes
        : (uint64_t)sizeof(GpuSphereContact);
    const uint64_t contactCapacitySlots = std::max<uint64_t>(1ull, contactCapacityBytes / sizeof(GpuSphereContact));
    pairParams.contactCapacity = contactCapacitySlots > UINT32_MAX ? UINT32_MAX : (uint32_t)contactCapacitySlots;
    queue.WriteBuffer(sapPairParamsBuffer, 0, &pairParams, sizeof(pairParams));
    if (sphereContactOutputBuffer == nullptr || sphereContactOutputBufferBytes < contactCapacityBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopyDst |
                                 (uint64_t)wgpu::BufferUsage::CopySrc);
        desc.size = alignUp(contactCapacityBytes, 4096);
        sphereContactOutputBuffer = device.CreateBuffer(&desc);
        sphereContactOutputBufferBytes = sphereContactOutputBuffer == nullptr ? 0 : desc.size;
    }
    if (sphereContactOutputBuffer == nullptr)
    {
        setStatus(sapStatus, "SAP pairs failed: contact output buffer allocation");
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

    const uint64_t passParamBytes = alignUp((uint64_t)passParamData.size(), 256);
    if (sapSortPassParamsBuffer == nullptr || sapSortPassParamsBufferBytes < passParamBytes)
    {
        wgpu::BufferDescriptor passParamsDesc = {};
        passParamsDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::CopySrc |
                                           (uint64_t)wgpu::BufferUsage::CopyDst);
        passParamsDesc.size = passParamBytes;
        sapSortPassParamsBuffer = device.CreateBuffer(&passParamsDesc);
        sapSortPassParamsBufferBytes = sapSortPassParamsBuffer == nullptr ? 0 : passParamsDesc.size;
    }
    if (sapSortPassParamsBuffer == nullptr)
    {
        setStatus(sapStatus, "SAP pairs failed: pass params buffer");
        return false;
    }
    queue.WriteBuffer(sapSortPassParamsBuffer, 0, passParamData.data(), passParamData.size());

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex)
    {
        encoder.CopyBufferToBuffer(sapSortPassParamsBuffer, (uint64_t)passIndex * passParamStride,
                                   sapParamsBuffer, 0, sizeof(GpuSortParams));
        wgpu::ComputePassEncoder sortPass = encoder.BeginComputePass();
        sortPass.SetPipeline(sapSortPipeline);
        sortPass.SetBindGroup(0, sortBindGroup);
        sortPass.DispatchWorkgroups((paddedCount + 63u) / 64u);
        sortPass.End();
    }

    if (directSpherePositionSolve)
    {
        device.PushErrorScope(wgpu::ErrorFilter::Validation);
        const uint32_t bodyCount = (uint32_t)world.bodies.size();
        const uint64_t outputBytes = (uint64_t)bodyCount * sizeof(GpuContactProposalOutput);
        if (sphereContactProposalOutputBuffer == nullptr || sphereContactProposalOutputBufferBytes < outputBytes)
        {
            wgpu::BufferDescriptor desc = {};
            desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                     (uint64_t)wgpu::BufferUsage::CopyDst |
                                     (uint64_t)wgpu::BufferUsage::CopySrc);
            desc.size = alignUp(outputBytes, 4096);
            sphereContactProposalOutputBuffer = device.CreateBuffer(&desc);
            sphereContactProposalOutputBufferBytes = sphereContactProposalOutputBuffer == nullptr ? 0 : desc.size;
        }
        if (sphereContactProposalOutputReadbackBuffer == nullptr ||
            sphereContactProposalOutputReadbackBufferBytes < outputBytes)
        {
            wgpu::BufferDescriptor desc = {};
            desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                     (uint64_t)wgpu::BufferUsage::CopyDst);
            desc.size = alignUp(outputBytes, 4096);
            sphereContactProposalOutputReadbackBuffer = device.CreateBuffer(&desc);
            sphereContactProposalOutputReadbackBufferBytes = sphereContactProposalOutputReadbackBuffer == nullptr ? 0 : desc.size;
        }
        if (sapDirectSphereParamsBuffer == nullptr)
        {
            wgpu::BufferDescriptor desc = {};
            desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                     (uint64_t)wgpu::BufferUsage::CopyDst);
            desc.size = sizeof(GpuDirectSphereParams);
            sapDirectSphereParamsBuffer = device.CreateBuffer(&desc);
        }
        const uint64_t directCounterBytes = sizeof(GpuDirectContactCounters);
        if (sapDirectContactCountersBuffer == nullptr || sapDirectContactCountersBufferBytes < directCounterBytes)
        {
            wgpu::BufferDescriptor desc = {};
            desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                     (uint64_t)wgpu::BufferUsage::CopyDst |
                                     (uint64_t)wgpu::BufferUsage::CopySrc);
            desc.size = alignUp(directCounterBytes, 256);
            sapDirectContactCountersBuffer = device.CreateBuffer(&desc);
            sapDirectContactCountersBufferBytes = sapDirectContactCountersBuffer == nullptr ? 0 : desc.size;
        }
        const bool directCounterReadbackEnabled =
            environmentFlagEnabled("AVBD_GPU_DIRECT_COUNTER_READBACK");
        if (directCounterReadbackEnabled &&
            (sapDirectContactCountersReadbackBuffer == nullptr ||
             sapDirectContactCountersReadbackBufferBytes < directCounterBytes))
        {
            wgpu::BufferDescriptor desc = {};
            desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                     (uint64_t)wgpu::BufferUsage::CopyDst);
            desc.size = alignUp(directCounterBytes, 256);
            sapDirectContactCountersReadbackBuffer = device.CreateBuffer(&desc);
            sapDirectContactCountersReadbackBufferBytes = sapDirectContactCountersReadbackBuffer == nullptr ? 0 : desc.size;
        }
        if (sphereContactProposalOutputBuffer == nullptr ||
            sphereContactProposalOutputReadbackBuffer == nullptr ||
            sapDirectSphereParamsBuffer == nullptr ||
            sapDirectContactCountersBuffer == nullptr ||
            (directCounterReadbackEnabled && sapDirectContactCountersReadbackBuffer == nullptr))
        {
            setStatus(sapStatus, "SAP direct sphere solve failed: buffer allocation");
            return false;
        }

        std::vector<GpuContactProposalOutput> directOutputs(bodyCount);
        sphereContactFinalReferencePositions.resize(world.bodies.size());
        for (BodyId bodyId = 0; bodyId < world.bodies.size(); ++bodyId)
        {
            const SimBodyData &body = world.bodies[bodyId];
            directOutputs[bodyId].position[0] = body.positionLin.x;
            directOutputs[bodyId].position[1] = body.positionLin.y;
            directOutputs[bodyId].position[2] = body.positionLin.z;
            directOutputs[bodyId].position[3] = body.radius;
            directOutputs[bodyId].correction[0] = 0.0f;
            directOutputs[bodyId].correction[1] = 0.0f;
            directOutputs[bodyId].correction[2] = 0.0f;
            directOutputs[bodyId].correction[3] = 0.0f;
            sphereContactFinalReferencePositions[bodyId] = body.positionLin;
        }
        if (!directOutputs.empty())
            queue.WriteBuffer(sphereContactProposalOutputBuffer, 0, directOutputs.data(), outputBytes);

        GpuDirectSphereParams directParams = {};
        directParams.intervalCount = itemCount;
        directParams.bodyCount = bodyCount;
        directParams.axis = (uint32_t)sapBestAxis;
        queue.WriteBuffer(sapDirectSphereParamsBuffer, 0, &directParams, sizeof(directParams));
        GpuDirectContactCounters zeroDirectCounters = {};
        queue.WriteBuffer(sapDirectContactCountersBuffer, 0, &zeroDirectCounters, sizeof(zeroDirectCounters));

        static const char *directSphereShaderSource = R"(
struct Interval {
    minX: f32,
    maxX: f32,
    y: f32,
    z: f32,
    radius: f32,
    index: u32,
    shapeType: u32,
    flags: u32,
    groundTopBits: u32,
    pad2: u32,
    shapeSizeXBits: u32,
    shapeSizeYBits: u32,
    shapeSizeZBits: u32,
    orientationX: f32,
    orientationY: f32,
    orientationZ: f32,
    orientationW: f32,
};

struct Params {
    intervalCount: u32,
    bodyCount: u32,
    axis: u32,
    pad0: u32,
};

struct ProposalOutput {
    position: vec4f,
    correction: vec4f,
};

struct DirectCounters {
    contactRecords: atomic<u32>,
    roundPairCandidates: atomic<u32>,
    sphereBoxCandidates: atomic<u32>,
    sphereCylinderCandidates: atomic<u32>,
    sphereCapsuleCandidates: atomic<u32>,
    boxPairCandidates: atomic<u32>,
    scannedPairs: atomic<u32>,
    activeIntervals: atomic<u32>,
};

@group(0) @binding(0) var<storage, read> intervals: array<Interval>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputs: array<ProposalOutput>;
@group(0) @binding(3) var<storage, read_write> directCounters: DirectCounters;

fn intervalCenter(value: Interval) -> f32 {
    return (value.minX + value.maxX) * 0.5;
}


fn intervalPosition(value: Interval) -> vec3f {
    let center = intervalCenter(value);
    if (params.axis == 0u) {
        return vec3f(center, value.y, value.z);
    }
    if (params.axis == 1u) {
        return vec3f(value.z, center, value.y);
    }
    return vec3f(value.y, value.z, center);
}

fn intervalOrientation(value: Interval) -> vec4f {
    return vec4f(value.orientationX, value.orientationY, value.orientationZ, value.orientationW);
}

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn quatInverseRotate(q: vec4f, v: vec3f) -> vec3f {
    return quatRotate(vec4f(-q.x, -q.y, -q.z, q.w), v);
}

fn sphereBoxCorrection(selfPos: vec3f, selfRadius: f32, other: Interval) -> vec3f {
    let boxPos = intervalPosition(other);
    let boxOrientation = intervalOrientation(other);
    let halfSize = vec3f(
        bitcast<f32>(other.shapeSizeXBits),
        bitcast<f32>(other.shapeSizeYBits),
        bitcast<f32>(other.shapeSizeZBits));
    if (halfSize.x <= 0.0 || halfSize.y <= 0.0 || halfSize.z <= 0.0) {
        return vec3f(0.0, 0.0, 0.0);
    }

    let local = quatInverseRotate(boxOrientation, selfPos - boxPos);
    let closestLocal = clamp(local, -halfSize, halfSize);
    let closest = boxPos + quatRotate(boxOrientation, closestLocal);
    let delta = selfPos - closest;
    let distSq = dot(delta, delta);
    if (distSq >= selfRadius * selfRadius) {
        return vec3f(0.0, 0.0, 0.0);
    }

    var normal = vec3f(1.0, 0.0, 0.0);
    var penetration = selfRadius;
    if (distSq > 0.000001) {
        let dist = sqrt(distSq);
        normal = delta / dist;
        penetration = selfRadius - dist;
    } else {
        let clearance = halfSize - abs(local);
        if (clearance.x <= clearance.y && clearance.x <= clearance.z) {
            normal = vec3f(select(-1.0, 1.0, local.x >= 0.0), 0.0, 0.0);
            penetration = selfRadius + max(clearance.x, 0.0);
        } else if (clearance.y <= clearance.z) {
            normal = vec3f(0.0, select(-1.0, 1.0, local.y >= 0.0), 0.0);
            penetration = selfRadius + max(clearance.y, 0.0);
        } else {
            normal = vec3f(0.0, 0.0, select(-1.0, 1.0, local.z >= 0.0));
            penetration = selfRadius + max(clearance.z, 0.0);
        }
    }
    return quatRotate(boxOrientation, normal) * min(penetration * 0.02, 0.0025);
}

fn sphereCylinderCorrection(selfPos: vec3f, selfRadius: f32, other: Interval) -> vec3f {
    let cylPos = intervalPosition(other);
    let cylOrientation = intervalOrientation(other);
    let cylRadius = bitcast<f32>(other.pad2);
    let cylHalfLength = bitcast<f32>(other.groundTopBits);
    if (cylRadius <= 0.0 || cylHalfLength <= 0.0) {
        return vec3f(0.0, 0.0, 0.0);
    }

    let local = quatInverseRotate(cylOrientation, selfPos - cylPos);
    let radial = vec2f(local.x, local.y);
    let radialLen = length(radial);
    let clampedZ = clamp(local.z, -cylHalfLength, cylHalfLength);
    var closestRadial = radial;
    if (radialLen > cylRadius) {
        closestRadial = radial * (cylRadius / max(radialLen, 0.000001));
    }
    let closestLocal = vec3f(closestRadial.x, closestRadial.y, clampedZ);
    var delta = local - closestLocal;
    var distSq = dot(delta, delta);
    if (distSq >= selfRadius * selfRadius) {
        return vec3f(0.0, 0.0, 0.0);
    }

    var normal = vec3f(0.0, 0.0, 1.0);
    var penetration = selfRadius;
    if (distSq > 0.000001) {
        let dist = sqrt(distSq);
        normal = delta / dist;
        penetration = selfRadius - dist;
    } else {
        let sideClearance = cylRadius - radialLen;
        let capClearance = cylHalfLength - abs(local.z);
        if (sideClearance < capClearance && radialLen > 0.000001) {
            normal = vec3f(radial.x / radialLen, radial.y / radialLen, 0.0);
            penetration = selfRadius + max(sideClearance, 0.0);
        } else {
            normal = vec3f(0.0, 0.0, select(-1.0, 1.0, local.z >= 0.0));
            penetration = selfRadius + max(capClearance, 0.0);
        }
    }
    return quatRotate(cylOrientation, normal) * min(penetration * 0.02, 0.0025);
}

fn sphereCapsuleCorrection(selfPos: vec3f, selfRadius: f32, other: Interval) -> vec3f {
    let capsulePos = intervalPosition(other);
    let capsuleOrientation = intervalOrientation(other);
    let capsuleRadius = bitcast<f32>(other.pad2);
    let capsuleHalfLength = bitcast<f32>(other.groundTopBits);
    if (capsuleRadius <= 0.0 || capsuleHalfLength <= 0.0) {
        return vec3f(0.0, 0.0, 0.0);
    }

    let local = quatInverseRotate(capsuleOrientation, selfPos - capsulePos);
    let closestLocal = vec3f(0.0, 0.0, clamp(local.z, -capsuleHalfLength, capsuleHalfLength));
    var delta = local - closestLocal;
    let radius = selfRadius + capsuleRadius;
    var distSq = dot(delta, delta);
    if (distSq >= radius * radius) {
        return vec3f(0.0, 0.0, 0.0);
    }

    var normal = vec3f(1.0, 0.0, 0.0);
    var penetration = radius;
    if (distSq > 0.000001) {
        let dist = sqrt(distSq);
        normal = delta / dist;
        penetration = radius - dist;
    } else {
        normal = vec3f(1.0, 0.0, 0.0);
    }
    return quatRotate(capsuleOrientation, normal) * min(penetration * 0.02, 0.0025);
}

fn roundRadius(value: Interval) -> f32 {
    if (value.shapeType == 1u) {
        return value.radius;
    }
    if (value.shapeType == 2u || value.shapeType == 3u) {
        return bitcast<f32>(value.pad2);
    }
    return 0.0;
}

fn roundHalfLength(value: Interval) -> f32 {
    if (value.shapeType == 2u || value.shapeType == 3u) {
        return bitcast<f32>(value.groundTopBits);
    }
    return 0.0;
}

fn roundEndpoint(value: Interval, signValue: f32) -> vec3f {
    let halfLength = roundHalfLength(value);
    if (halfLength <= 0.0) {
        return intervalPosition(value);
    }
    return intervalPosition(value) + quatRotate(intervalOrientation(value), vec3f(0.0, 0.0, signValue * halfLength));
}

fn roundBoxCorrection(roundValue: Interval, boxValue: Interval) -> vec3f {
    let radius = roundRadius(roundValue);
    if (radius <= 0.0 || boxValue.shapeType != 0u) {
        return vec3f(0.0, 0.0, 0.0);
    }

    let centerCorrection = sphereBoxCorrection(intervalPosition(roundValue), radius, boxValue);
    let halfLength = roundHalfLength(roundValue);
    if (halfLength <= 0.0) {
        return centerCorrection;
    }

    let aCorrection = sphereBoxCorrection(roundEndpoint(roundValue, -1.0), radius, boxValue);
    let bCorrection = sphereBoxCorrection(roundEndpoint(roundValue, 1.0), radius, boxValue);
    let correction = centerCorrection + aCorrection + bCorrection;
    let len = length(correction);
    if (len <= 0.000001) {
        return vec3f(0.0, 0.0, 0.0);
    }
    return correction / max(1.0, select(1.0, 3.0, length(centerCorrection) > 0.000001 && length(aCorrection) > 0.000001 && length(bCorrection) > 0.000001));
}

fn roundPairCorrection(selfValue: Interval, other: Interval) -> vec3f {
    let selfRadius = roundRadius(selfValue);
    let otherRadius = roundRadius(other);
    if (selfRadius <= 0.0 || otherRadius <= 0.0) {
        return vec3f(0.0, 0.0, 0.0);
    }

    let p1 = roundEndpoint(selfValue, -1.0);
    let q1 = roundEndpoint(selfValue, 1.0);
    let p2 = roundEndpoint(other, -1.0);
    let q2 = roundEndpoint(other, 1.0);
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    var s = 0.0;
    var t = 0.0;
    if (a <= 0.000001 && e <= 0.000001) {
        s = 0.0;
        t = 0.0;
    } else if (a <= 0.000001) {
        s = 0.0;
        t = clamp(f / max(e, 0.000001), 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (e <= 0.000001) {
            t = 0.0;
            s = clamp(-c / max(a, 0.000001), 0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            if (abs(denom) > 0.000001) {
                s = clamp((b * f - c * e) / denom, 0.0, 1.0);
            } else {
                s = 0.0;
            }
            t = (b * s + f) / max(e, 0.000001);
            if (t < 0.0) {
                t = 0.0;
                s = clamp(-c / max(a, 0.000001), 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((b - c) / max(a, 0.000001), 0.0, 1.0);
            }
        }
    }

    let closestSelf = p1 + d1 * s;
    let closestOther = p2 + d2 * t;
    let delta = closestSelf - closestOther;
    let distSq = dot(delta, delta);
    let radius = selfRadius + otherRadius;
    if (distSq >= radius * radius) {
        return vec3f(0.0, 0.0, 0.0);
    }

    var normal = vec3f(1.0, 0.0, 0.0);
    var dist = 0.0;
    if (distSq > 0.000001) {
        dist = sqrt(distSq);
        normal = delta / dist;
    }
    let penetration = radius - dist;
    return normal * min(penetration * 0.02, 0.0025);
}

)" R"(
fn boxHalfSize(value: Interval) -> vec3f {
    return vec3f(
        bitcast<f32>(value.shapeSizeXBits),
        bitcast<f32>(value.shapeSizeYBits),
        bitcast<f32>(value.shapeSizeZBits));
}

fn boxAxis(value: Interval, axisIndex: u32) -> vec3f {
    let q = intervalOrientation(value);
    if (axisIndex == 0u) {
        return quatRotate(q, vec3f(1.0, 0.0, 0.0));
    }
    if (axisIndex == 1u) {
        return quatRotate(q, vec3f(0.0, 1.0, 0.0));
    }
    return quatRotate(q, vec3f(0.0, 0.0, 1.0));
}

fn boxProjectionRadius(value: Interval, axis: vec3f) -> f32 {
    let h = boxHalfSize(value);
    return h.x * abs(dot(boxAxis(value, 0u), axis)) +
           h.y * abs(dot(boxAxis(value, 1u), axis)) +
           h.z * abs(dot(boxAxis(value, 2u), axis));
}

fn boxBoxAxisCorrection(selfValue: Interval, other: Interval, rawAxis: vec3f, bestOverlap: ptr<function, f32>, bestAxis: ptr<function, vec3f>) -> bool {
    let axisLength = length(rawAxis);
    if (axisLength <= 0.000001) {
        return true;
    }
    let axis = rawAxis / axisLength;
    let selfPos = intervalPosition(selfValue);
    let otherPos = intervalPosition(other);
    let delta = selfPos - otherPos;
    let overlap = boxProjectionRadius(selfValue, axis) + boxProjectionRadius(other, axis) - abs(dot(delta, axis));
    if (overlap <= 0.0) {
        return false;
    }
    if (overlap < *bestOverlap) {
        *bestOverlap = overlap;
        var directedAxis = axis;
        if (dot(delta, axis) < 0.0) {
            directedAxis = -axis;
        }
        *bestAxis = directedAxis;
    }
    return true;
}

fn boxBoxCorrection(selfValue: Interval, other: Interval) -> vec3f {
    if (selfValue.shapeType != 0u || other.shapeType != 0u ||
        selfValue.radius <= 0.0 || other.radius <= 0.0) {
        return vec3f(0.0, 0.0, 0.0);
    }

    var bestOverlap = 1000000.0;
    var bestAxis = vec3f(1.0, 0.0, 0.0);
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(selfValue, 0u), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(selfValue, 1u), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(selfValue, 2u), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(other, 0u), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(other, 1u), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(other, 2u), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 0u), boxAxis(other, 0u)), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 0u), boxAxis(other, 1u)), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 0u), boxAxis(other, 2u)), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 1u), boxAxis(other, 0u)), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 1u), boxAxis(other, 1u)), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 1u), boxAxis(other, 2u)), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 2u), boxAxis(other, 0u)), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 2u), boxAxis(other, 1u)), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 2u), boxAxis(other, 2u)), &bestOverlap, &bestAxis)) {
        return vec3f(0.0, 0.0, 0.0);
    }

    return bestAxis * min(bestOverlap * 0.02, 0.0025);
}

fn pairCorrection(selfValue: Interval, other: Interval) -> vec3f {
    if (other.index == 0xffffffffu) {
        return vec3f(0.0, 0.0, 0.0);
    }
    let selfPos = intervalPosition(selfValue);
    let selfRadius = roundRadius(selfValue);
    if (selfValue.shapeType == 0u && other.shapeType == 0u) {
        return boxBoxCorrection(selfValue, other);
    }
    if (selfValue.shapeType == 1u && other.shapeType == 0u && other.flags != 1u) {
        return sphereBoxCorrection(selfPos, selfRadius, other);
    }
    if (selfValue.shapeType == 0u && selfValue.flags == 0u && other.shapeType == 1u) {
        return -sphereBoxCorrection(intervalPosition(other), other.radius, selfValue);
    }
    if ((selfValue.shapeType == 2u || selfValue.shapeType == 3u) && other.shapeType == 0u && other.flags != 1u) {
        return roundBoxCorrection(selfValue, other);
    }
    if (selfValue.shapeType == 0u && selfValue.flags == 0u && (other.shapeType == 2u || other.shapeType == 3u)) {
        return -roundBoxCorrection(other, selfValue);
    }
    if (other.shapeType != 1u && other.shapeType != 2u && other.shapeType != 3u) {
        return vec3f(0.0, 0.0, 0.0);
    }
    return roundPairCorrection(selfValue, other);
}

fn radiusPairCorrection(selfValue: Interval, other: Interval) -> vec3f {
    if (selfValue.radius <= 0.0 || other.radius <= 0.0 || selfValue.radius >= 8.0 || other.radius >= 8.0) {
        return vec3f(0.0, 0.0, 0.0);
    }
    let delta = intervalPosition(selfValue) - intervalPosition(other);
    let distSq = dot(delta, delta);
    let radius = selfValue.radius + other.radius;
    if (distSq >= radius * radius) {
        return vec3f(0.0, 0.0, 0.0);
    }
    var normal = vec3f(1.0, 0.0, 0.0);
    var dist = 0.0;
    if (distSq > 0.000001) {
        dist = sqrt(distSq);
        normal = delta / dist;
    }
    return normal * min((radius - dist) * 0.02, 0.0025);
}

fn recordDirectContact(selfValue: Interval, other: Interval, correction: vec3f) {
    if (length(correction) <= 0.000001) {
        return;
    }
    atomicAdd(&directCounters.contactRecords, 1u);
    if (selfValue.shapeType == 0u && other.shapeType == 0u) {
        atomicAdd(&directCounters.boxPairCandidates, 1u);
        return;
    }
    if ((selfValue.shapeType == 1u && other.shapeType == 0u) ||
        (selfValue.shapeType == 0u && other.shapeType == 1u)) {
        atomicAdd(&directCounters.sphereBoxCandidates, 1u);
        return;
    }
    if ((selfValue.shapeType == 1u && other.shapeType == 3u) ||
        (selfValue.shapeType == 3u && other.shapeType == 1u)) {
        atomicAdd(&directCounters.sphereCylinderCandidates, 1u);
        atomicAdd(&directCounters.roundPairCandidates, 1u);
        return;
    }
    if ((selfValue.shapeType == 1u && other.shapeType == 2u) ||
        (selfValue.shapeType == 2u && other.shapeType == 1u)) {
        atomicAdd(&directCounters.sphereCapsuleCandidates, 1u);
        atomicAdd(&directCounters.roundPairCandidates, 1u);
        return;
    }
    if ((selfValue.shapeType == 1u || selfValue.shapeType == 2u || selfValue.shapeType == 3u) &&
        (other.shapeType == 1u || other.shapeType == 2u || other.shapeType == 3u)) {
        atomicAdd(&directCounters.roundPairCandidates, 1u);
        return;
    }
    if ((selfValue.shapeType == 0u && (other.shapeType == 2u || other.shapeType == 3u)) ||
        (other.shapeType == 0u && (selfValue.shapeType == 2u || selfValue.shapeType == 3u))) {
        atomicAdd(&directCounters.roundPairCandidates, 1u);
        return;
    }
    atomicAdd(&directCounters.roundPairCandidates, 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let intervalIndex = id.x;
    if (intervalIndex >= params.intervalCount) {
        return;
    }
    atomicAdd(&directCounters.activeIntervals, 1u);

    let selfValue = intervals[intervalIndex];
    if (selfValue.index == 0xffffffffu || selfValue.index >= params.bodyCount) {
        return;
    }

    var correction = vec3f(0.0, 0.0, 0.0);
    if (selfValue.radius > 0.0) {
        var scan = intervalIndex + 1u;
        loop {
            if (scan >= params.intervalCount) {
                break;
            }
            let other = intervals[scan];
            if (other.index == 0xffffffffu || other.minX > selfValue.maxX) {
                break;
            }
            atomicAdd(&directCounters.scannedPairs, 1u);
            var pairDelta = pairCorrection(selfValue, other);
            if (length(pairDelta) <= 0.000001) {
                pairDelta = radiusPairCorrection(selfValue, other);
            }
            recordDirectContact(selfValue, other, pairDelta);
            correction = correction + pairDelta;
            scan = scan + 1u;
        }

        scan = intervalIndex;
        loop {
            if (scan == 0u) {
                break;
            }
            scan = scan - 1u;
            let other = intervals[scan];
            if (other.index == 0xffffffffu) {
                continue;
            }
            if (other.maxX < selfValue.minX) {
                break;
            }
            atomicAdd(&directCounters.scannedPairs, 1u);
            var pairDelta = pairCorrection(selfValue, other);
            if (length(pairDelta) <= 0.000001) {
                pairDelta = radiusPairCorrection(selfValue, other);
            }
            recordDirectContact(selfValue, other, pairDelta);
            correction = correction + pairDelta;
        }
    }

    let len = length(correction);
    if (len > 0.005) {
        correction = correction * (0.005 / max(len, 0.000001));
    }
    let selfPos = intervalPosition(selfValue);
    outputs[selfValue.index].position = vec4f(selfPos + correction, selfValue.radius);
    outputs[selfValue.index].correction = vec4f(correction, 0.0);
}
)";

        static const int maxCpuDirectBoxPairPrefilterBodiesForGpuGate = 512;
        const bool runDenseBoxIntervalDirectScan =
            directBoxBodyCount > maxCpuDirectBoxPairPrefilterBodiesForGpuGate &&
            directBoxPairs.empty();
        const bool runIntervalDirectScan = runDenseBoxIntervalDirectScan ||
                                           directRoundBodyCount > 0 ||
                                           directSphereCylinderCandidateCount > 0 ||
                                           directSphereCapsuleCandidateCount > 0 ||
                                           directSphereBoxCandidateCount > 0;
        if (runIntervalDirectScan && (sapDirectSpherePipeline == nullptr || sapDirectSphereBindGroupLayout == nullptr))
        {
            wgpu::ShaderSourceWGSL wgsl = {};
            wgsl.code = directSphereShaderSource;
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
                setStatus(sapStatus, "SAP direct sphere solve failed: shader validation scope");
                return false;
            }
            if (shaderErrorType != wgpu::ErrorType::NoError)
            {
                snprintf(status, sizeof(status), "WebGPU validation error: %s", shaderErrorMessage.c_str());
                snprintf(sapStatus, sizeof(sapStatus), "SAP direct sphere solve shader failed: %s", shaderErrorMessage.c_str());
                return false;
            }
            if (shader == nullptr)
            {
                setStatus(sapStatus, "SAP direct sphere solve failed: shader");
                return false;
            }

            wgpu::BindGroupLayoutEntry entries[4] = {};
            entries[0].binding = 0;
            entries[0].visibility = wgpu::ShaderStage::Compute;
            entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            entries[0].buffer.minBindingSize = sizeof(GpuSapInterval);
            entries[1].binding = 1;
            entries[1].visibility = wgpu::ShaderStage::Compute;
            entries[1].buffer.type = wgpu::BufferBindingType::Uniform;
            entries[1].buffer.minBindingSize = sizeof(GpuDirectSphereParams);
            entries[2].binding = 2;
            entries[2].visibility = wgpu::ShaderStage::Compute;
            entries[2].buffer.type = wgpu::BufferBindingType::Storage;
            entries[2].buffer.minBindingSize = sizeof(GpuContactProposalOutput);
            entries[3].binding = 3;
            entries[3].visibility = wgpu::ShaderStage::Compute;
            entries[3].buffer.type = wgpu::BufferBindingType::Storage;
            entries[3].buffer.minBindingSize = sizeof(GpuDirectContactCounters);
            wgpu::BindGroupLayoutDescriptor layoutDesc = {};
            layoutDesc.entryCount = 4;
            layoutDesc.entries = entries;
            sapDirectSphereBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
            wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
            pipelineLayoutDesc.bindGroupLayoutCount = 1;
            pipelineLayoutDesc.bindGroupLayouts = &sapDirectSphereBindGroupLayout;
            wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
            wgpu::ComputePipelineDescriptor pipelineDesc = {};
            pipelineDesc.layout = pipelineLayout;
            pipelineDesc.compute.module = shader;
            pipelineDesc.compute.entryPoint = "main";
            device.PushErrorScope(wgpu::ErrorFilter::Validation);
            sapDirectSpherePipeline = device.CreateComputePipeline(&pipelineDesc);
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
                setStatus(sapStatus, "SAP direct sphere solve failed: pipeline validation scope");
                return false;
            }
            if (pipelineErrorType != wgpu::ErrorType::NoError)
            {
                snprintf(status, sizeof(status), "WebGPU validation error: %s", pipelineErrorMessage.c_str());
                snprintf(sapStatus, sizeof(sapStatus), "SAP direct sphere solve pipeline failed: %s", pipelineErrorMessage.c_str());
                return false;
            }
            if (sapDirectSpherePipeline == nullptr)
            {
                setStatus(sapStatus, "SAP direct sphere solve failed: pipeline");
                return false;
            }
        }

        if (runIntervalDirectScan)
        {
            wgpu::BindGroupEntry entries[4] = {};
            entries[0].binding = 0;
            entries[0].buffer = sapIntervalBuffer;
            entries[0].size = intervalBytes;
            entries[1].binding = 1;
            entries[1].buffer = sapDirectSphereParamsBuffer;
            entries[1].size = sizeof(GpuDirectSphereParams);
            entries[2].binding = 2;
            entries[2].buffer = sphereContactProposalOutputBuffer;
            entries[2].size = outputBytes;
            entries[3].binding = 3;
            entries[3].buffer = sapDirectContactCountersBuffer;
            entries[3].size = directCounterBytes;
            wgpu::BindGroupDescriptor bindGroupDesc = {};
            bindGroupDesc.layout = sapDirectSphereBindGroupLayout;
            bindGroupDesc.entryCount = 4;
            bindGroupDesc.entries = entries;
            wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
            if (bindGroup == nullptr)
            {
                setStatus(sapStatus, "SAP direct sphere solve failed: bind group");
                return false;
            }

            wgpu::ComputePassEncoder directPass = encoder.BeginComputePass();
            directPass.SetPipeline(sapDirectSpherePipeline);
            directPass.SetBindGroup(0, bindGroup);
            directPass.DispatchWorkgroups((itemCount + 63u) / 64u);
            directPass.End();
        }

        if (!directBoxPairs.empty())
        {
            const uint64_t directBoxBodyBytes = (uint64_t)world.bodies.size() * sizeof(GpuContactBody);
            const uint64_t directBoxPairBytes = (uint64_t)directBoxPairs.size() * sizeof(GpuSapPair);
            if (sphereContactBodyBuffer == nullptr || sphereContactBodyBufferBytes < directBoxBodyBytes)
            {
                wgpu::BufferDescriptor bodyDesc = {};
                bodyDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                             (uint64_t)wgpu::BufferUsage::CopyDst);
                bodyDesc.size = alignUp(directBoxBodyBytes, 4096);
                sphereContactBodyBuffer = device.CreateBuffer(&bodyDesc);
                sphereContactBodyBufferBytes = sphereContactBodyBuffer == nullptr ? 0 : bodyDesc.size;
            }
            if (sphereContactPairBuffer == nullptr || sphereContactPairBufferBytes < directBoxPairBytes)
            {
                wgpu::BufferDescriptor pairDesc = {};
                pairDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                             (uint64_t)wgpu::BufferUsage::CopyDst);
                pairDesc.size = alignUp(directBoxPairBytes, 4096);
                sphereContactPairBuffer = device.CreateBuffer(&pairDesc);
                sphereContactPairBufferBytes = sphereContactPairBuffer == nullptr ? 0 : pairDesc.size;
            }
            if (sphereContactBodyBuffer == nullptr || sphereContactPairBuffer == nullptr)
            {
                setStatus(sapStatus, "SAP direct box pair solve failed: buffer allocation");
                return false;
            }

            std::vector<GpuContactBody> directBodies(world.bodies.size());
            for (BodyId bodyId = 0; bodyId < world.bodies.size(); ++bodyId)
            {
                const SimBodyData &body = world.bodies[bodyId];
                GpuContactBody gpuBody = {};
                gpuBody.position[0] = body.positionLin.x;
                gpuBody.position[1] = body.positionLin.y;
                gpuBody.position[2] = body.positionLin.z;
                gpuBody.position[3] = body.radius;
                gpuBody.orientation[0] = body.positionAng.x;
                gpuBody.orientation[1] = body.positionAng.y;
                gpuBody.orientation[2] = body.positionAng.z;
                gpuBody.orientation[3] = body.positionAng.w;
                gpuBody.halfSizeRadius[0] = std::max(0.0f, body.shape.size.x * 0.5f);
                gpuBody.halfSizeRadius[1] = std::max(0.0f, body.shape.size.y * 0.5f);
                gpuBody.halfSizeRadius[2] = std::max(0.0f, body.shape.size.z * 0.5f);
                gpuBody.halfSizeRadius[3] = body.shape.radius > 0.0f ? body.shape.radius : body.radius;
                gpuBody.shapeType = (uint32_t)body.shape.type;
                gpuBody.pad0 = body.mass > 0.0f ? 1u : 0u;
                directBodies[bodyId] = gpuBody;
            }
            queue.WriteBuffer(sphereContactBodyBuffer, 0, directBodies.data(), directBoxBodyBytes);
            queue.WriteBuffer(sphereContactPairBuffer, 0, directBoxPairs.data(), directBoxPairBytes);

            static const char *directBoxPairShaderSource = R"(
struct Body {
    position: vec4f,
    orientation: vec4f,
    halfSizeRadius: vec4f,
    extra: vec4f,
    shapeType: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Pair {
    bodyA: u32,
    bodyB: u32,
};

struct PairParams {
    pairCount: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct ProposalOutput {
    position: vec4f,
    correction: vec4f,
};

struct DirectCounters {
    contactRecords: atomic<u32>,
    roundPairCandidates: atomic<u32>,
    sphereBoxCandidates: atomic<u32>,
    sphereCylinderCandidates: atomic<u32>,
    sphereCapsuleCandidates: atomic<u32>,
    boxPairCandidates: atomic<u32>,
    scannedPairs: atomic<u32>,
    activeIntervals: atomic<u32>,
};

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> outputs: array<ProposalOutput>;
@group(0) @binding(3) var<storage, read_write> directCounters: DirectCounters;
@group(0) @binding(4) var<uniform> params: PairParams;

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn boxAxis(value: Body, axisIndex: u32) -> vec3f {
    if (axisIndex == 0u) {
        return quatRotate(value.orientation, vec3f(1.0, 0.0, 0.0));
    }
    if (axisIndex == 1u) {
        return quatRotate(value.orientation, vec3f(0.0, 1.0, 0.0));
    }
    return quatRotate(value.orientation, vec3f(0.0, 0.0, 1.0));
}

fn boxProjectionRadius(value: Body, axis: vec3f) -> f32 {
    let h = value.halfSizeRadius.xyz;
    return h.x * abs(dot(boxAxis(value, 0u), axis)) +
           h.y * abs(dot(boxAxis(value, 1u), axis)) +
           h.z * abs(dot(boxAxis(value, 2u), axis));
}

fn boxBoxAxisCorrection(selfValue: Body, other: Body, rawAxis: vec3f, bestOverlap: ptr<function, f32>, bestAxis: ptr<function, vec3f>) -> bool {
    let axisLength = length(rawAxis);
    if (axisLength <= 0.000001) {
        return true;
    }
    let axis = rawAxis / axisLength;
    let delta = selfValue.position.xyz - other.position.xyz;
    let overlap = boxProjectionRadius(selfValue, axis) + boxProjectionRadius(other, axis) - abs(dot(delta, axis));
    if (overlap <= 0.0) {
        return false;
    }
    if (overlap < *bestOverlap) {
        *bestOverlap = overlap;
        var directedAxis = axis;
        if (dot(delta, axis) < 0.0) {
            directedAxis = -axis;
        }
        *bestAxis = directedAxis;
    }
    return true;
}

fn boxBoxCorrection(selfValue: Body, other: Body) -> vec3f {
    if (selfValue.shapeType != 0u || other.shapeType != 0u ||
        min(selfValue.halfSizeRadius.x, min(selfValue.halfSizeRadius.y, selfValue.halfSizeRadius.z)) <= 0.0 ||
        min(other.halfSizeRadius.x, min(other.halfSizeRadius.y, other.halfSizeRadius.z)) <= 0.0) {
        return vec3f(0.0, 0.0, 0.0);
    }

    var bestOverlap = 1000000.0;
    var bestAxis = vec3f(1.0, 0.0, 0.0);
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(selfValue, 0u), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(selfValue, 1u), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(selfValue, 2u), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(other, 0u), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(other, 1u), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, boxAxis(other, 2u), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 0u), boxAxis(other, 0u)), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 0u), boxAxis(other, 1u)), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 0u), boxAxis(other, 2u)), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 1u), boxAxis(other, 0u)), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 1u), boxAxis(other, 1u)), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 1u), boxAxis(other, 2u)), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 2u), boxAxis(other, 0u)), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 2u), boxAxis(other, 1u)), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    if (!boxBoxAxisCorrection(selfValue, other, cross(boxAxis(selfValue, 2u), boxAxis(other, 2u)), &bestOverlap, &bestAxis)) { return vec3f(0.0, 0.0, 0.0); }
    return bestAxis * min(bestOverlap * 0.02, 0.0025);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pairIndex = id.x;
    if (pairIndex >= params.pairCount) {
        return;
    }
    let pair = pairs[pairIndex];
    let a = bodies[pair.bodyA];
    let b = bodies[pair.bodyB];
    let correction = boxBoxCorrection(a, b);
    if (length(correction) <= 0.000001) {
        return;
    }

    outputs[pair.bodyA].correction = vec4f(correction, 0.0);
    outputs[pair.bodyA].position = vec4f(a.position.xyz + correction, a.position.w);
    outputs[pair.bodyB].correction = vec4f(-correction, 0.0);
    outputs[pair.bodyB].position = vec4f(b.position.xyz - correction, b.position.w);
    atomicAdd(&directCounters.contactRecords, 2u);
    atomicAdd(&directCounters.boxPairCandidates, 2u);
    atomicAdd(&directCounters.scannedPairs, 1u);
}
)";

            if (sapDirectBoxPairParamsBuffer == nullptr)
            {
                wgpu::BufferDescriptor paramsDesc = {};
                paramsDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                               (uint64_t)wgpu::BufferUsage::CopyDst);
                paramsDesc.size = 16;
                sapDirectBoxPairParamsBuffer = device.CreateBuffer(&paramsDesc);
            }
            if (sapDirectBoxPairParamsBuffer == nullptr)
            {
                setStatus(sapStatus, "SAP direct box pair solve failed: params buffer");
                return false;
            }
            uint32_t boxPairParams[4] = {(uint32_t)directBoxPairs.size(), 0u, 0u, 0u};
            queue.WriteBuffer(sapDirectBoxPairParamsBuffer, 0, boxPairParams, sizeof(boxPairParams));

            if (sapDirectBoxPairPipeline == nullptr || sapDirectBoxPairBindGroupLayout == nullptr)
            {
                wgpu::ShaderSourceWGSL boxWgsl = {};
                boxWgsl.code = directBoxPairShaderSource;
                wgpu::ShaderModuleDescriptor boxShaderDesc = {};
                boxShaderDesc.nextInChain = &boxWgsl;
                wgpu::ShaderModule boxShader = device.CreateShaderModule(&boxShaderDesc);
                if (boxShader == nullptr)
                {
                    setStatus(sapStatus, "SAP direct box pair solve failed: shader");
                    return false;
                }

                wgpu::BindGroupLayoutEntry boxLayoutEntries[5] = {};
                boxLayoutEntries[0].binding = 0;
                boxLayoutEntries[0].visibility = wgpu::ShaderStage::Compute;
                boxLayoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                boxLayoutEntries[0].buffer.minBindingSize = sizeof(GpuContactBody);
                boxLayoutEntries[1].binding = 1;
                boxLayoutEntries[1].visibility = wgpu::ShaderStage::Compute;
                boxLayoutEntries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                boxLayoutEntries[1].buffer.minBindingSize = sizeof(GpuSapPair);
                boxLayoutEntries[2].binding = 2;
                boxLayoutEntries[2].visibility = wgpu::ShaderStage::Compute;
                boxLayoutEntries[2].buffer.type = wgpu::BufferBindingType::Storage;
                boxLayoutEntries[2].buffer.minBindingSize = sizeof(GpuContactProposalOutput);
                boxLayoutEntries[3].binding = 3;
                boxLayoutEntries[3].visibility = wgpu::ShaderStage::Compute;
                boxLayoutEntries[3].buffer.type = wgpu::BufferBindingType::Storage;
                boxLayoutEntries[3].buffer.minBindingSize = sizeof(GpuDirectContactCounters);
                boxLayoutEntries[4].binding = 4;
                boxLayoutEntries[4].visibility = wgpu::ShaderStage::Compute;
                boxLayoutEntries[4].buffer.type = wgpu::BufferBindingType::Uniform;
                boxLayoutEntries[4].buffer.minBindingSize = 16;
                wgpu::BindGroupLayoutDescriptor boxLayoutDesc = {};
                boxLayoutDesc.entryCount = 5;
                boxLayoutDesc.entries = boxLayoutEntries;
                sapDirectBoxPairBindGroupLayout = device.CreateBindGroupLayout(&boxLayoutDesc);
                wgpu::PipelineLayoutDescriptor boxPipelineLayoutDesc = {};
                boxPipelineLayoutDesc.bindGroupLayoutCount = 1;
                boxPipelineLayoutDesc.bindGroupLayouts = &sapDirectBoxPairBindGroupLayout;
                wgpu::PipelineLayout boxPipelineLayout = device.CreatePipelineLayout(&boxPipelineLayoutDesc);
                wgpu::ComputePipelineDescriptor boxPipelineDesc = {};
                boxPipelineDesc.layout = boxPipelineLayout;
                boxPipelineDesc.compute.module = boxShader;
                boxPipelineDesc.compute.entryPoint = "main";
                sapDirectBoxPairPipeline = device.CreateComputePipeline(&boxPipelineDesc);
                if (sapDirectBoxPairPipeline == nullptr)
                {
                    setStatus(sapStatus, "SAP direct box pair solve failed: pipeline");
                    return false;
                }
            }

            wgpu::BindGroupEntry boxEntries[5] = {};
            boxEntries[0].binding = 0;
            boxEntries[0].buffer = sphereContactBodyBuffer;
            boxEntries[0].size = directBoxBodyBytes;
            boxEntries[1].binding = 1;
            boxEntries[1].buffer = sphereContactPairBuffer;
            boxEntries[1].size = directBoxPairBytes;
            boxEntries[2].binding = 2;
            boxEntries[2].buffer = sphereContactProposalOutputBuffer;
            boxEntries[2].size = outputBytes;
            boxEntries[3].binding = 3;
            boxEntries[3].buffer = sapDirectContactCountersBuffer;
            boxEntries[3].size = directCounterBytes;
            boxEntries[4].binding = 4;
            boxEntries[4].buffer = sapDirectBoxPairParamsBuffer;
            boxEntries[4].size = 16;
            wgpu::BindGroupDescriptor boxBindDesc = {};
            boxBindDesc.layout = sapDirectBoxPairBindGroupLayout;
            boxBindDesc.entryCount = 5;
            boxBindDesc.entries = boxEntries;
            wgpu::BindGroup boxBindGroup = device.CreateBindGroup(&boxBindDesc);
            if (boxBindGroup == nullptr)
            {
                setStatus(sapStatus, "SAP direct box pair solve failed: bind group");
                return false;
            }

            wgpu::ComputePassEncoder boxPass = encoder.BeginComputePass();
            boxPass.SetPipeline(sapDirectBoxPairPipeline);
            boxPass.SetBindGroup(0, boxBindGroup);
            boxPass.DispatchWorkgroups(((uint32_t)directBoxPairs.size() + 63u) / 64u);
            boxPass.End();
        }

        if (directCounterReadbackEnabled)
            encoder.CopyBufferToBuffer(sapDirectContactCountersBuffer, 0,
                                       sapDirectContactCountersReadbackBuffer, 0,
                                       directCounterBytes);
        Uint64 directCounterReadbackBegin = SDL_GetPerformanceCounter();
        wgpu::CommandBuffer commands = encoder.Finish();
        wgpu::PopErrorScopeStatus directScopeStatus = wgpu::PopErrorScopeStatus::Error;
        wgpu::ErrorType directErrorType = wgpu::ErrorType::Unknown;
        std::string directErrorMessage;
        wgpu::Future directScopeFuture = device.PopErrorScope(
            wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message)
            {
                directScopeStatus = status;
                directErrorType = type;
                directErrorMessage = toString(message);
            });
        if (instance.WaitAny(directScopeFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            directScopeStatus != wgpu::PopErrorScopeStatus::Success)
        {
            setStatus(sapStatus, "SAP direct sphere solve failed: validation scope");
            return false;
        }
        if (directErrorType != wgpu::ErrorType::NoError)
        {
            snprintf(status, sizeof(status), "WebGPU validation error: %s", directErrorMessage.c_str());
            snprintf(sapStatus, sizeof(sapStatus), "SAP direct sphere solve validation failed: %s", directErrorMessage.c_str());
            return false;
        }
        queue.Submit(1, &commands);

        if (directCounterReadbackEnabled)
        {
            bool directMapDone = false;
            wgpu::MapAsyncStatus directMapStatus = wgpu::MapAsyncStatus::Error;
            wgpu::Future directMapFuture = sapDirectContactCountersReadbackBuffer.MapAsync(
                wgpu::MapMode::Read, 0, directCounterBytes, wgpu::CallbackMode::WaitAnyOnly,
                [&](wgpu::MapAsyncStatus status, wgpu::StringView)
                {
                    directMapDone = true;
                    directMapStatus = status;
                });
            if (instance.WaitAny(directMapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
                !directMapDone || directMapStatus != wgpu::MapAsyncStatus::Success)
            {
                setStatus(sapStatus, "SAP direct sphere solve failed: direct counter readback map");
                return false;
            }
            const GpuDirectContactCounters *directCounters =
                (const GpuDirectContactCounters *)sapDirectContactCountersReadbackBuffer.GetConstMappedRange(0, directCounterBytes);
            if (!directCounters)
            {
                sapDirectContactCountersReadbackBuffer.Unmap();
                setStatus(sapStatus, "SAP direct sphere solve failed: direct counter readback range");
                return false;
            }
            directGpuContactRecordCount = (int)directCounters->contactRecords;
            directGpuRoundPairCandidateCount = (int)directCounters->roundPairCandidates;
            directGpuBoxPairCandidateCount = (int)directCounters->boxPairCandidates;
            if (directGpuContactRecordCount == 0 && directBoxPairCandidateCount > 0)
                directGpuBoxPairCandidateCount = std::max(directGpuBoxPairCandidateCount,
                                                          (int)(directCounters->scannedPairs + directCounters->activeIntervals * 1000u));
            sapDirectContactCountersReadbackBuffer.Unmap();
            directGpuCounterReadbackBytes = (int)directCounterBytes;
            directGpuCounterReadbackMs = elapsedMs(directCounterReadbackBegin, SDL_GetPerformanceCounter());
        }

        sphereContactFinalPositionReady = 1;
        sphereContactFinalPositionBodyCount = (int)bodyCount;
        sphereContactFinalPositionBytes = (int)outputBytes;
        sphereContactFinalPositionSource = 5;
        sphereContactFinalPositionBuffer = sphereContactProposalOutputBuffer;
        sphereContactFinalPositionBufferBytes = outputBytes;
        sapMs = elapsedMs(begin, SDL_GetPerformanceCounter());
        recordTimingSample(sapTiming, sapMs);

        const char *axisName = sapBestAxis == 0 ? "X" : (sapBestAxis == 1 ? "Y" : "Z");
        snprintf(sapStatus, sizeof(sapStatus), "SAP direct sphere solve submitted: %s axis, %u intervals, no counter sync",
                 axisName, itemCount);
        return true;
    }

    static const char *outputShaderSource = R"(
struct Interval {
    minX: f32,
    maxX: f32,
    y: f32,
    z: f32,
    radius: f32,
    index: u32,
    shapeType: u32,
    flags: u32,
    groundTopBits: u32,
    pad2: u32,
    shapeSizeXBits: u32,
    shapeSizeYBits: u32,
    shapeSizeZBits: u32,
    orientationX: f32,
    orientationY: f32,
    orientationZ: f32,
    orientationW: f32,
};

struct Counters {
    candidates: atomic<u32>,
    sphereHits: atomic<u32>,
    sphereContacts: atomic<u32>,
    nonSpherePairs: atomic<u32>,
    sphereGroundContacts: atomic<u32>,
    pad2: u32,
    pad3: u32,
    pad4: u32,
};

struct Pair {
    bodyA: u32,
    bodyB: u32,
};

struct Contact {
    bodyA: u32,
    bodyB: u32,
    pad0: u32,
    pad1: u32,
    normal: vec4f,
};

struct Params {
    count: u32,
    emitSphereContacts: u32,
    emitSphereGroundContacts: u32,
    outputFlags: u32,
    contactCapacity: u32,
    pad3: u32,
    pad4: u32,
    pad5: u32,
};

@group(0) @binding(0) var<storage, read> intervals: array<Interval>;
@group(0) @binding(1) var<storage, read_write> counters: Counters;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> pairs: array<Pair>;
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;

fn intervalWorldZ(value: Interval, axis: u32) -> f32 {
    let center = (value.minX + value.maxX) * 0.5;
    if (axis == 0u) {
        return value.z;
    }
    if (axis == 1u) {
        return value.y;
    }
    return center;
}

fn intervalPosition(value: Interval, axis: u32) -> vec3f {
    let center = (value.minX + value.maxX) * 0.5;
    if (axis == 0u) {
        return vec3f(center, value.y, value.z);
    }
    if (axis == 1u) {
        return vec3f(value.z, center, value.y);
    }
    return vec3f(value.y, value.z, center);
}

fn intervalOrientation(value: Interval) -> vec4f {
    return vec4f(value.orientationX, value.orientationY, value.orientationZ, value.orientationW);
}

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn quatInverseRotate(q: vec4f, v: vec3f) -> vec3f {
    return quatRotate(vec4f(-q.x, -q.y, -q.z, q.w), v);
}

fn contactNormalSphereBox(sphere: Interval, box: Interval, axis: u32) -> vec4f {
    let spherePos = intervalPosition(sphere, axis);
    let boxPos = intervalPosition(box, axis);
    let boxOrientation = intervalOrientation(box);
    let halfSize = vec3f(
        bitcast<f32>(box.shapeSizeXBits),
        bitcast<f32>(box.shapeSizeYBits),
        bitcast<f32>(box.shapeSizeZBits));
    if (halfSize.x <= 0.0 || halfSize.y <= 0.0 || halfSize.z <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    let local = quatInverseRotate(boxOrientation, spherePos - boxPos);
    let closestLocal = clamp(local, -halfSize, halfSize);
    let closest = boxPos + quatRotate(boxOrientation, closestLocal);
    let delta = spherePos - closest;
    let distSq = dot(delta, delta);
    if (distSq >= sphere.radius * sphere.radius) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    var normalBoxToSphere = vec3f(1.0, 0.0, 0.0);
    if (distSq > 0.000001) {
        normalBoxToSphere = delta * inverseSqrt(distSq);
    } else {
        let clearance = halfSize - abs(local);
        if (clearance.x <= clearance.y && clearance.x <= clearance.z) {
            normalBoxToSphere = quatRotate(boxOrientation, vec3f(select(-1.0, 1.0, local.x >= 0.0), 0.0, 0.0));
        } else if (clearance.y <= clearance.z) {
            normalBoxToSphere = quatRotate(boxOrientation, vec3f(0.0, select(-1.0, 1.0, local.y >= 0.0), 0.0));
        } else {
            normalBoxToSphere = quatRotate(boxOrientation, vec3f(0.0, 0.0, select(-1.0, 1.0, local.z >= 0.0)));
        }
    }
    return vec4f(-normalBoxToSphere, 1.0);
}

fn contactNormalSphereCapsule(sphere: Interval, capsule: Interval, axis: u32) -> vec4f {
    let spherePos = intervalPosition(sphere, axis);
    let capsulePos = intervalPosition(capsule, axis);
    let capsuleOrientation = intervalOrientation(capsule);
    let capsuleRadius = bitcast<f32>(capsule.pad2);
    let capsuleHalfLength = bitcast<f32>(capsule.groundTopBits);
    if (capsuleRadius <= 0.0 || capsuleHalfLength <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    let local = quatInverseRotate(capsuleOrientation, spherePos - capsulePos);
    let closestLocal = vec3f(0.0, 0.0, clamp(local.z, -capsuleHalfLength, capsuleHalfLength));
    let delta = local - closestLocal;
    let radius = sphere.radius + capsuleRadius;
    let distSq = dot(delta, delta);
    if (distSq >= radius * radius) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    var normalCapsuleToSphere = vec3f(1.0, 0.0, 0.0);
    if (distSq > 0.000001) {
        normalCapsuleToSphere = quatRotate(capsuleOrientation, delta * inverseSqrt(distSq));
    }
    return vec4f(-normalCapsuleToSphere, 1.0);
}

fn contactNormalSphereCylinder(sphere: Interval, cylinder: Interval, axis: u32) -> vec4f {
    let spherePos = intervalPosition(sphere, axis);
    let cylinderPos = intervalPosition(cylinder, axis);
    let cylinderOrientation = intervalOrientation(cylinder);
    let cylinderRadius = bitcast<f32>(cylinder.pad2);
    let cylinderHalfLength = bitcast<f32>(cylinder.groundTopBits);
    if (cylinderRadius <= 0.0 || cylinderHalfLength <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    let local = quatInverseRotate(cylinderOrientation, spherePos - cylinderPos);
    let radial = vec2f(local.x, local.y);
    let radialLen = length(radial);
    let clampedZ = clamp(local.z, -cylinderHalfLength, cylinderHalfLength);
    var closestRadial = radial;
    if (radialLen > cylinderRadius) {
        closestRadial = radial * (cylinderRadius / max(radialLen, 0.000001));
    }
    let closestLocal = vec3f(closestRadial.x, closestRadial.y, clampedZ);
    let delta = local - closestLocal;
    let distSq = dot(delta, delta);
    if (distSq >= sphere.radius * sphere.radius) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    var normalCylinderToSphere = vec3f(0.0, 0.0, 1.0);
    if (distSq > 0.000001) {
        normalCylinderToSphere = quatRotate(cylinderOrientation, delta * inverseSqrt(distSq));
    } else {
        let sideClearance = cylinderRadius - radialLen;
        let capClearance = cylinderHalfLength - abs(local.z);
        if (sideClearance < capClearance && radialLen > 0.000001) {
            normalCylinderToSphere = quatRotate(cylinderOrientation, vec3f(radial.x / radialLen, radial.y / radialLen, 0.0));
        } else {
            normalCylinderToSphere = quatRotate(cylinderOrientation, vec3f(0.0, 0.0, select(-1.0, 1.0, local.z >= 0.0)));
        }
    }
    return vec4f(-normalCylinderToSphere, 1.0);
}

fn roundRadius(value: Interval) -> f32 {
    if (value.shapeType == 1u) {
        return value.radius;
    }
    if (value.shapeType == 2u || value.shapeType == 3u) {
        return bitcast<f32>(value.pad2);
    }
    return 0.0;
}

fn roundHalfLength(value: Interval) -> f32 {
    if (value.shapeType == 2u || value.shapeType == 3u) {
        return bitcast<f32>(value.groundTopBits);
    }
    return 0.0;
}

fn roundEndpoint(value: Interval, axis: u32, signValue: f32) -> vec3f {
    let halfLength = roundHalfLength(value);
    if (halfLength <= 0.0) {
        return intervalPosition(value, axis);
    }
    return intervalPosition(value, axis) + quatRotate(intervalOrientation(value), vec3f(0.0, 0.0, signValue * halfLength));
}

fn contactNormalRoundPair(a: Interval, b: Interval, axis: u32) -> vec4f {
    let radiusA = roundRadius(a);
    let radiusB = roundRadius(b);
    if (radiusA <= 0.0 || radiusB <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    let p1 = roundEndpoint(a, axis, -1.0);
    let q1 = roundEndpoint(a, axis, 1.0);
    let p2 = roundEndpoint(b, axis, -1.0);
    let q2 = roundEndpoint(b, axis, 1.0);
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let aa = dot(d1, d1);
    let ee = dot(d2, d2);
    let ff = dot(d2, r);

    var s = 0.0;
    var t = 0.0;
    if (aa <= 0.000001 && ee <= 0.000001) {
        s = 0.0;
        t = 0.0;
    } else if (aa <= 0.000001) {
        s = 0.0;
        t = clamp(ff / max(ee, 0.000001), 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (ee <= 0.000001) {
            t = 0.0;
            s = clamp(-c / max(aa, 0.000001), 0.0, 1.0);
        } else {
            let bDot = dot(d1, d2);
            let denom = aa * ee - bDot * bDot;
            if (abs(denom) > 0.000001) {
                s = clamp((bDot * ff - c * ee) / denom, 0.0, 1.0);
            } else {
                s = 0.0;
            }
            t = (bDot * s + ff) / max(ee, 0.000001);
            if (t < 0.0) {
                t = 0.0;
                s = clamp(-c / max(aa, 0.000001), 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((bDot - c) / max(aa, 0.000001), 0.0, 1.0);
            }
        }
    }

    let closestA = p1 + d1 * s;
    let closestB = p2 + d2 * t;
    let delta = closestA - closestB;
    let distSq = dot(delta, delta);
    let radius = radiusA + radiusB;
    if (distSq >= radius * radius) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    var normalBToA = vec3f(1.0, 0.0, 0.0);
    if (distSq > 0.000001) {
        normalBToA = delta * inverseSqrt(distSq);
    }
    return vec4f(normalBToA, 1.0);
}

fn pointBoxEval(point: vec3f, radius: f32, box: Interval, axis: u32) -> vec4f {
    let boxPos = intervalPosition(box, axis);
    let boxOrientation = intervalOrientation(box);
    let halfSize = vec3f(
        bitcast<f32>(box.shapeSizeXBits),
        bitcast<f32>(box.shapeSizeYBits),
        bitcast<f32>(box.shapeSizeZBits));
    if (radius <= 0.0 || halfSize.x <= 0.0 || halfSize.y <= 0.0 || halfSize.z <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    let local = quatInverseRotate(boxOrientation, point - boxPos);
    let closestLocal = clamp(local, -halfSize, halfSize);
    let closest = boxPos + quatRotate(boxOrientation, closestLocal);
    let delta = point - closest;
    let distSq = dot(delta, delta);
    if (distSq >= radius * radius) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    var direction = vec3f(1.0, 0.0, 0.0);
    var penetration = radius;
    if (distSq > 0.000001) {
        let dist = sqrt(distSq);
        direction = delta / dist;
        penetration = radius - dist;
    } else {
        let clearance = halfSize - abs(local);
        if (clearance.x <= clearance.y && clearance.x <= clearance.z) {
            direction = quatRotate(boxOrientation, vec3f(select(-1.0, 1.0, local.x >= 0.0), 0.0, 0.0));
            penetration = radius + max(clearance.x, 0.0);
        } else if (clearance.y <= clearance.z) {
            direction = quatRotate(boxOrientation, vec3f(0.0, select(-1.0, 1.0, local.y >= 0.0), 0.0));
            penetration = radius + max(clearance.y, 0.0);
        } else {
            direction = quatRotate(boxOrientation, vec3f(0.0, 0.0, select(-1.0, 1.0, local.z >= 0.0)));
            penetration = radius + max(clearance.z, 0.0);
        }
    }
    return vec4f(direction, max(penetration, 0.0));
}

fn roundBoxEvalInterval(roundValue: Interval, box: Interval, axis: u32) -> vec4f {
    let radius = roundRadius(roundValue);
    let centerEval = pointBoxEval(intervalPosition(roundValue, axis), radius, box, axis);
    let halfLength = roundHalfLength(roundValue);
    if (halfLength <= 0.0) {
        return centerEval;
    }

    let aEval = pointBoxEval(roundEndpoint(roundValue, axis, -1.0), radius, box, axis);
    let bEval = pointBoxEval(roundEndpoint(roundValue, axis, 1.0), radius, box, axis);
    var best = centerEval;
    if (aEval.w > best.w) {
        best = aEval;
    }
    if (bEval.w > best.w) {
        best = bEval;
    }
    return best;
}

fn spherePrimitiveContact(sphere: Interval, other: Interval, axis: u32) -> Contact {
    var contact: Contact;
    contact.bodyA = sphere.index;
    contact.bodyB = other.index;
    contact.pad0 = 0u;
    contact.pad1 = 0u;
    contact.normal = vec4f(0.0, 0.0, 0.0, 0.0);
    if (other.shapeType == 0u && other.flags != 1u) {
        contact.normal = contactNormalSphereBox(sphere, other, axis);
        contact.pad0 = 3u;
    } else if (other.shapeType == 2u) {
        contact.normal = contactNormalSphereCapsule(sphere, other, axis);
        contact.pad0 = 4u;
    } else if (other.shapeType == 3u) {
        contact.normal = contactNormalSphereCylinder(sphere, other, axis);
        contact.pad0 = 5u;
    }
    return contact;
}

fn roundBoxContact(roundValue: Interval, box: Interval, axis: u32) -> Contact {
    var contact: Contact;
    contact.bodyA = roundValue.index;
    contact.bodyB = box.index;
    contact.pad0 = 7u;
    contact.pad1 = 0u;
    contact.normal = roundBoxEvalInterval(roundValue, box, axis);
    if (contact.normal.w <= 0.0) {
        contact.pad0 = 0u;
    }
    return contact;
}

)" R"(
fn roundPairContact(a: Interval, b: Interval, axis: u32) -> Contact {
    var contact: Contact;
    contact.bodyA = a.index;
    contact.bodyB = b.index;
    contact.pad0 = 6u;
    contact.pad1 = 0u;
    contact.normal = contactNormalRoundPair(a, b, axis);
    if (contact.normal.w <= 0.5) {
        contact.pad0 = 0u;
    }
    return contact;
}

fn exactSphereGroundContact(a: Interval, b: Interval, axis: u32) -> bool {
    let aSphere = a.shapeType == 1u;
    let bSphere = b.shapeType == 1u;
    let aGround = a.flags == 1u;
    let bGround = b.flags == 1u;
    if (aSphere && bGround) {
        let sphereZ = intervalWorldZ(a, axis);
        let groundTop = bitcast<f32>(b.groundTopBits);
        return sphereZ < groundTop + a.radius;
    }
    if (bSphere && aGround) {
        let sphereZ = intervalWorldZ(b, axis);
        let groundTop = bitcast<f32>(a.groundTopBits);
        return sphereZ < groundTop + b.radius;
    }
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }

    let a = intervals[i];
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
        let ax = (a.minX + a.maxX) * 0.5;
        let bx = (b.minX + b.maxX) * 0.5;
        let dx = ax - bx;
        let dy = a.y - b.y;
        let dz = a.z - b.z;
        let radius = a.radius + b.radius;
        if (dx * dx + dy * dy + dz * dz < radius * radius) {
            atomicAdd(&counters.sphereHits, 1u);
            let sphereGroundPair = (a.shapeType == 1u && b.flags == 1u) || (b.shapeType == 1u && a.flags == 1u);
            let sapAxis = (params.outputFlags >> 8u) & 3u;
            let sphereGroundContact = exactSphereGroundContact(a, b, sapAxis);
            let suppressSphereGroundPairs = (params.outputFlags & 1u) != 0u;
            let skipPairOutput = (params.outputFlags & 2u) != 0u;
            if (params.emitSphereGroundContacts != 0u && sphereGroundContact) {
                let contactSlot = atomicAdd(&counters.sphereContacts, 1u);
                atomicAdd(&counters.sphereGroundContacts, 1u);
                if (contactSlot < params.contactCapacity) {
                    if (a.shapeType == 1u) {
                        contacts[contactSlot].bodyA = a.index;
                        contacts[contactSlot].bodyB = b.index;
                    } else {
                        contacts[contactSlot].bodyA = b.index;
                        contacts[contactSlot].bodyB = a.index;
                    }
                    contacts[contactSlot].pad0 = 2u;
                    contacts[contactSlot].normal = vec4f(0.0, 0.0, 1.0, 0.0);
                }
            } else if (suppressSphereGroundPairs && sphereGroundPair) {
            } else if (params.emitSphereContacts != 0u && a.shapeType == 1u && b.shapeType == 1u) {
                var normal = vec3f(1.0, 0.0, 0.0);
                let distSq = dx * dx + dy * dy + dz * dz;
                if (distSq > 0.000001) {
                    normal = vec3f(-dx, -dy, -dz) * inverseSqrt(distSq);
                }
                let contactSlot = atomicAdd(&counters.sphereContacts, 1u);
                if (contactSlot < params.contactCapacity) {
                    contacts[contactSlot].bodyA = a.index;
                    contacts[contactSlot].bodyB = b.index;
                    contacts[contactSlot].pad0 = 1u;
                    contacts[contactSlot].normal = vec4f(normal, 0.0);
                }
            } else if (params.emitSphereContacts != 0u &&
                       ((a.shapeType == 1u && (b.shapeType == 0u || b.shapeType == 2u || b.shapeType == 3u)) ||
                        (b.shapeType == 1u && (a.shapeType == 0u || a.shapeType == 2u || a.shapeType == 3u)))) {
                var primitiveContact: Contact;
                if (a.shapeType == 1u) {
                    primitiveContact = spherePrimitiveContact(a, b, sapAxis);
                } else {
                    primitiveContact = spherePrimitiveContact(b, a, sapAxis);
                }
                if (primitiveContact.normal.w > 0.5) {
                    let contactSlot = atomicAdd(&counters.sphereContacts, 1u);
                    if (contactSlot < params.contactCapacity) {
                        contacts[contactSlot] = primitiveContact;
                    }
                } else if (!skipPairOutput) {
                    let pairSlot = atomicAdd(&counters.nonSpherePairs, 1u);
                    pairs[pairSlot].bodyA = a.index;
                    pairs[pairSlot].bodyB = b.index;
                }
            } else if (params.emitSphereContacts != 0u &&
                       (a.shapeType == 2u || a.shapeType == 3u) &&
                       (b.shapeType == 2u || b.shapeType == 3u)) {
                let primitiveContact = roundPairContact(a, b, sapAxis);
                if (primitiveContact.normal.w > 0.5) {
                    let contactSlot = atomicAdd(&counters.sphereContacts, 1u);
                    if (contactSlot < params.contactCapacity) {
                        contacts[contactSlot] = primitiveContact;
                    }
                } else if (!skipPairOutput) {
                    let pairSlot = atomicAdd(&counters.nonSpherePairs, 1u);
                    pairs[pairSlot].bodyA = a.index;
                    pairs[pairSlot].bodyB = b.index;
                }
            } else if (params.emitSphereContacts != 0u &&
                       (((a.shapeType == 2u || a.shapeType == 3u) && b.shapeType == 0u && b.flags != 1u) ||
                        ((b.shapeType == 2u || b.shapeType == 3u) && a.shapeType == 0u && a.flags != 1u))) {
                var primitiveContact: Contact;
                if (a.shapeType == 0u) {
                    primitiveContact = roundBoxContact(b, a, sapAxis);
                } else {
                    primitiveContact = roundBoxContact(a, b, sapAxis);
                }
                if (primitiveContact.normal.w > 0.0) {
                    let contactSlot = atomicAdd(&counters.sphereContacts, 1u);
                    if (contactSlot < params.contactCapacity) {
                        contacts[contactSlot] = primitiveContact;
                    }
                } else if (!skipPairOutput) {
                    let pairSlot = atomicAdd(&counters.nonSpherePairs, 1u);
                    pairs[pairSlot].bodyA = a.index;
                    pairs[pairSlot].bodyB = b.index;
                }
            } else if (skipPairOutput) {
            } else {
                let pairSlot = atomicAdd(&counters.nonSpherePairs, 1u);
                pairs[pairSlot].bodyA = a.index;
                pairs[pairSlot].bodyB = b.index;
            }
        }
        j = j + 1u;
    }
}
)";

    if (sapPairOutputPipeline == nullptr || sapPairOutputBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL outputWgsl = {};
        outputWgsl.code = outputShaderSource;
        wgpu::ShaderModuleDescriptor outputShaderDesc = {};
        outputShaderDesc.nextInChain = &outputWgsl;
        wgpu::ShaderModule outputShader = device.CreateShaderModule(&outputShaderDesc);
        wgpu::BindGroupLayoutEntry outputEntries[5] = {};
        outputEntries[0].binding = 0;
        outputEntries[0].visibility = wgpu::ShaderStage::Compute;
        outputEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        outputEntries[0].buffer.minBindingSize = sizeof(GpuSapInterval);
        outputEntries[1].binding = 1;
        outputEntries[1].visibility = wgpu::ShaderStage::Compute;
        outputEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
        outputEntries[1].buffer.minBindingSize = sizeof(GpuSapCounters);
        outputEntries[2].binding = 2;
        outputEntries[2].visibility = wgpu::ShaderStage::Compute;
        outputEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
        outputEntries[2].buffer.minBindingSize = sizeof(GpuSapParams);
        outputEntries[3].binding = 3;
        outputEntries[3].visibility = wgpu::ShaderStage::Compute;
        outputEntries[3].buffer.type = wgpu::BufferBindingType::Storage;
        outputEntries[3].buffer.minBindingSize = sizeof(GpuSapPair);
        outputEntries[4].binding = 4;
        outputEntries[4].visibility = wgpu::ShaderStage::Compute;
        outputEntries[4].buffer.type = wgpu::BufferBindingType::Storage;
        outputEntries[4].buffer.minBindingSize = sizeof(GpuSphereContact);
        wgpu::BindGroupLayoutDescriptor outputLayoutDesc = {};
        outputLayoutDesc.entryCount = 5;
        outputLayoutDesc.entries = outputEntries;
        sapPairOutputBindGroupLayout = device.CreateBindGroupLayout(&outputLayoutDesc);
        wgpu::PipelineLayoutDescriptor outputPipelineLayoutDesc = {};
        outputPipelineLayoutDesc.bindGroupLayoutCount = 1;
        outputPipelineLayoutDesc.bindGroupLayouts = &sapPairOutputBindGroupLayout;
        wgpu::PipelineLayout outputPipelineLayout = device.CreatePipelineLayout(&outputPipelineLayoutDesc);
        wgpu::ComputePipelineDescriptor outputPipelineDesc = {};
        outputPipelineDesc.layout = outputPipelineLayout;
        outputPipelineDesc.compute.module = outputShader;
        outputPipelineDesc.compute.entryPoint = "main";
        sapPairOutputPipeline = device.CreateComputePipeline(&outputPipelineDesc);
        if (sapPairOutputPipeline == nullptr)
        {
            setStatus(sapStatus, "SAP pairs failed: output pipeline");
            return false;
        }
    }

    GpuSapCounters resetCounters = {};
    queue.WriteBuffer(sapCountersBuffer, 0, &resetCounters, sizeof(resetCounters));
    wgpu::BindGroupEntry outputBindEntries[5] = {};
    outputBindEntries[0].binding = 0;
    outputBindEntries[0].buffer = sapIntervalBuffer;
    outputBindEntries[0].size = intervalBytes;
    outputBindEntries[1].binding = 1;
    outputBindEntries[1].buffer = sapCountersBuffer;
    outputBindEntries[1].size = counterBytes;
    outputBindEntries[2].binding = 2;
    outputBindEntries[2].buffer = sapPairParamsBuffer;
    outputBindEntries[2].size = sizeof(GpuSapParams);
    outputBindEntries[3].binding = 3;
    outputBindEntries[3].buffer = sapPairOutputBuffer;
    outputBindEntries[3].size = pairCapacityBytes;
    outputBindEntries[4].binding = 4;
    outputBindEntries[4].buffer = sphereContactOutputBuffer;
    outputBindEntries[4].size = contactCapacityBytes;
    wgpu::BindGroupDescriptor outputBindGroupDesc = {};
    outputBindGroupDesc.layout = sapPairOutputBindGroupLayout;
    outputBindGroupDesc.entryCount = 5;
    outputBindGroupDesc.entries = outputBindEntries;
    wgpu::BindGroup outputBindGroup = device.CreateBindGroup(&outputBindGroupDesc);

    Uint64 counterReadbackBegin = SDL_GetPerformanceCounter();
    if (residentCounterlessContacts)
        encoder.ClearBuffer(sphereContactOutputBuffer, 0, contactCapacityBytes);
    wgpu::ComputePassEncoder outputPass = encoder.BeginComputePass();
    outputPass.SetPipeline(sapPairOutputPipeline);
    outputPass.SetBindGroup(0, outputBindGroup);
    outputPass.DispatchWorkgroups((itemCount + 63u) / 64u);
    outputPass.End();
    if (!residentCounterlessContacts)
        encoder.CopyBufferToBuffer(sapCountersBuffer, 0, sapReadbackBuffer, 0, counterBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    uint32_t candidateCount = 0;
    uint32_t sphereHitCount = 0;
    uint32_t sphereContactOutputCount = residentCounterlessContacts ? pairParams.contactCapacity : 0u;
    uint32_t nonSpherePairCount = 0;
    uint32_t sphereGroundContactOutputCount = 0;
    if (!residentCounterlessContacts)
    {
        wgpu::Future counterMapFuture = sapReadbackBuffer.MapAsync(
            wgpu::MapMode::Read, 0, counterBytes, wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::MapAsyncStatus status, wgpu::StringView)
            {
                mapDone = true;
                mapStatus = status;
            });
        if (instance.WaitAny(counterMapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
        {
            setStatus(sapStatus, "SAP pairs failed: counter readback map");
            return false;
        }
        const GpuSapCounters *counters = (const GpuSapCounters *)sapReadbackBuffer.GetConstMappedRange(0, counterBytes);
        if (!counters)
        {
            sapReadbackBuffer.Unmap();
            setStatus(sapStatus, "SAP pairs failed: counter readback range");
            return false;
        }
        candidateCount = counters->candidates;
        sphereHitCount = counters->sphereHits;
        sphereContactOutputCount = counters->pad0;
        nonSpherePairCount = counters->pad1;
        sphereGroundContactOutputCount = counters->pad2;
        sapReadbackBuffer.Unmap();
        sapCounterReadbackBytes = (int)counterBytes;
        sapCounterReadbackMs = elapsedMs(counterReadbackBegin, SDL_GetPerformanceCounter());
    }

    sapCandidates = (int)candidateCount;
    sapSphereHits = (int)sphereHitCount;
    sphereContactCount = (int)sphereContactOutputCount;
    if (emitSphereGroundContacts)
        sphereGroundCandidateCount = (int)sphereGroundContactOutputCount;
    sapAxisCandidates[sapBestAxis] = sapCandidates;
    sapAxisSphereHits[sapBestAxis] = sapSphereHits;
    sapMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    if (residentCounterlessContacts)
    {
        const char *axisName = sapBestAxis == 0 ? "X" : (sapBestAxis == 1 ? "Y" : "Z");
        snprintf(sapStatus, sizeof(sapStatus),
                 "SAP resident counterless submitted: %s axis, %u compact contact scan slots, no counter sync",
                 axisName, pairParams.contactCapacity);
    }

    bool runResidentContacts = runResidentContactSolve && (counterOnly || suppressSphereSpherePairs);
    const uint64_t pairBytes = (uint64_t)nonSpherePairCount * sizeof(GpuSapPair);
    const uint64_t contactBytes = (uint64_t)sphereContactOutputCount * sizeof(GpuSphereContact);
    const uint64_t contactReadbackBytes = sphereContacts ? contactBytes : 0;
    bool pairReadbackSubmittedEarly = false;
    if (runResidentContacts && !counterOnly && pairBytes > 0)
    {
        if (sapPairReadbackBuffer == nullptr || sapPairReadbackBufferBytes < pairBytes)
        {
            wgpu::BufferDescriptor desc = {};
            desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                     (uint64_t)wgpu::BufferUsage::CopyDst);
            desc.size = alignUp(pairBytes, 4096);
            sapPairReadbackBuffer = device.CreateBuffer(&desc);
            sapPairReadbackBufferBytes = sapPairReadbackBuffer == nullptr ? 0 : desc.size;
        }
        if (sapPairReadbackBuffer == nullptr)
        {
            setStatus(sapStatus, "SAP pairs failed: early pair readback buffer allocation");
            return false;
        }

        wgpu::CommandEncoder earlyReadbackEncoder = device.CreateCommandEncoder();
        earlyReadbackEncoder.CopyBufferToBuffer(sapPairOutputBuffer, 0, sapPairReadbackBuffer, 0, pairBytes);
        wgpu::CommandBuffer earlyReadbackCommands = earlyReadbackEncoder.Finish();
        queue.Submit(1, &earlyReadbackCommands);
        pairReadbackSubmittedEarly = true;
    }

    if (runResidentContacts)
    {
        if (sphereContactOutputCount > 0 && !world.bodies.empty())
        {
            Uint64 adjacencyBegin = SDL_GetPerformanceCounter();
            const uint32_t bodyCount = (uint32_t)world.bodies.size();
            const uint32_t capacityPerBody = 32u;
            const uint64_t bodyCountBytes = (uint64_t)bodyCount * sizeof(uint32_t);
            const uint64_t adjacencyBytes = (uint64_t)bodyCount * (uint64_t)capacityPerBody * sizeof(uint32_t);
            const uint64_t gatherBytes = (uint64_t)bodyCount * sizeof(GpuContactGatherOutput);
            const uint64_t proposalOutputBytes = (uint64_t)bodyCount * sizeof(GpuContactProposalOutput);
            const uint64_t residualBytes = (uint64_t)sphereContactOutputCount * sizeof(GpuContactProposalResidual);
            const uint64_t contactBodyBytes = (uint64_t)world.bodies.size() * sizeof(GpuContactBody);
            if (sphereContactBodyBuffer == nullptr || sphereContactBodyBufferBytes < contactBodyBytes)
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                         (uint64_t)wgpu::BufferUsage::CopyDst);
                desc.size = alignUp(contactBodyBytes, 4096);
                sphereContactBodyBuffer = device.CreateBuffer(&desc);
                sphereContactBodyBufferBytes = sphereContactBodyBuffer == nullptr ? 0 : desc.size;
            }
            if (sphereContactBodyCountBuffer == nullptr || sphereContactBodyCountBufferBytes < bodyCountBytes)
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                         (uint64_t)wgpu::BufferUsage::CopyDst |
                                         (uint64_t)wgpu::BufferUsage::CopySrc);
                desc.size = alignUp(bodyCountBytes, 4096);
                sphereContactBodyCountBuffer = device.CreateBuffer(&desc);
                sphereContactBodyCountBufferBytes = sphereContactBodyCountBuffer == nullptr ? 0 : desc.size;
            }
            if (validateResidentContacts &&
                (sphereContactBodyCountReadbackBuffer == nullptr || sphereContactBodyCountReadbackBufferBytes < bodyCountBytes))
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                         (uint64_t)wgpu::BufferUsage::CopyDst);
                desc.size = alignUp(bodyCountBytes, 4096);
                sphereContactBodyCountReadbackBuffer = device.CreateBuffer(&desc);
                sphereContactBodyCountReadbackBufferBytes = sphereContactBodyCountReadbackBuffer == nullptr ? 0 : desc.size;
            }
            if (sphereContactAdjacencyBuffer == nullptr || sphereContactAdjacencyBufferBytes < adjacencyBytes)
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                         (uint64_t)wgpu::BufferUsage::CopySrc);
                desc.size = alignUp(adjacencyBytes, 4096);
                sphereContactAdjacencyBuffer = device.CreateBuffer(&desc);
                sphereContactAdjacencyBufferBytes = sphereContactAdjacencyBuffer == nullptr ? 0 : desc.size;
            }
            if (sphereContactGatherOutputBuffer == nullptr || sphereContactGatherOutputBufferBytes < gatherBytes)
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                         (uint64_t)wgpu::BufferUsage::CopySrc);
                desc.size = alignUp(gatherBytes, 4096);
                sphereContactGatherOutputBuffer = device.CreateBuffer(&desc);
                sphereContactGatherOutputBufferBytes = sphereContactGatherOutputBuffer == nullptr ? 0 : desc.size;
            }
            if (validateResidentContacts &&
                (sphereContactGatherReadbackBuffer == nullptr || sphereContactGatherReadbackBufferBytes < gatherBytes))
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                         (uint64_t)wgpu::BufferUsage::CopyDst);
                desc.size = alignUp(gatherBytes, 4096);
                sphereContactGatherReadbackBuffer = device.CreateBuffer(&desc);
                sphereContactGatherReadbackBufferBytes = sphereContactGatherReadbackBuffer == nullptr ? 0 : desc.size;
            }
            if (sphereContactProposalOutputBuffer == nullptr || sphereContactProposalOutputBufferBytes < proposalOutputBytes)
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                         (uint64_t)wgpu::BufferUsage::CopyDst |
                                         (uint64_t)wgpu::BufferUsage::CopySrc);
                desc.size = alignUp(proposalOutputBytes, 4096);
                sphereContactProposalOutputBuffer = device.CreateBuffer(&desc);
                sphereContactProposalOutputBufferBytes = sphereContactProposalOutputBuffer == nullptr ? 0 : desc.size;
            }
            if (sphereContactProposalOutputReadbackBuffer == nullptr || sphereContactProposalOutputReadbackBufferBytes < proposalOutputBytes)
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                         (uint64_t)wgpu::BufferUsage::CopyDst);
                desc.size = alignUp(proposalOutputBytes, 4096);
                sphereContactProposalOutputReadbackBuffer = device.CreateBuffer(&desc);
                sphereContactProposalOutputReadbackBufferBytes = sphereContactProposalOutputReadbackBuffer == nullptr ? 0 : desc.size;
            }
            if (sphereContactIterationOutputBuffer == nullptr || sphereContactIterationOutputBufferBytes < proposalOutputBytes)
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                         (uint64_t)wgpu::BufferUsage::CopySrc);
                desc.size = alignUp(proposalOutputBytes, 4096);
                sphereContactIterationOutputBuffer = device.CreateBuffer(&desc);
                sphereContactIterationOutputBufferBytes = sphereContactIterationOutputBuffer == nullptr ? 0 : desc.size;
            }
            if (sphereContactIterationScratchBuffer == nullptr || sphereContactIterationScratchBufferBytes < proposalOutputBytes)
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                         (uint64_t)wgpu::BufferUsage::CopySrc);
                desc.size = alignUp(proposalOutputBytes, 4096);
                sphereContactIterationScratchBuffer = device.CreateBuffer(&desc);
                sphereContactIterationScratchBufferBytes = sphereContactIterationScratchBuffer == nullptr ? 0 : desc.size;
            }
            if (sphereContactProposalResidualBuffer == nullptr || sphereContactProposalResidualBufferBytes < residualBytes)
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                         (uint64_t)wgpu::BufferUsage::CopySrc);
                desc.size = alignUp(residualBytes, 4096);
                sphereContactProposalResidualBuffer = device.CreateBuffer(&desc);
                sphereContactProposalResidualBufferBytes = sphereContactProposalResidualBuffer == nullptr ? 0 : desc.size;
            }
            if (validateResidentContacts &&
                (sphereContactProposalResidualReadbackBuffer == nullptr || sphereContactProposalResidualReadbackBufferBytes < residualBytes))
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                         (uint64_t)wgpu::BufferUsage::CopyDst);
                desc.size = alignUp(residualBytes, 4096);
                sphereContactProposalResidualReadbackBuffer = device.CreateBuffer(&desc);
                sphereContactProposalResidualReadbackBufferBytes = sphereContactProposalResidualReadbackBuffer == nullptr ? 0 : desc.size;
            }
            if (sphereContactAdjacencyParamsBuffer == nullptr)
            {
                wgpu::BufferDescriptor desc = {};
                desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                         (uint64_t)wgpu::BufferUsage::CopyDst);
                desc.size = sizeof(GpuContactAdjacencyParams);
                sphereContactAdjacencyParamsBuffer = device.CreateBuffer(&desc);
            }
            if (sphereContactBodyCountBuffer == nullptr ||
                (validateResidentContacts && sphereContactBodyCountReadbackBuffer == nullptr) ||
                sphereContactAdjacencyBuffer == nullptr ||
                sphereContactGatherOutputBuffer == nullptr ||
                (validateResidentContacts && sphereContactGatherReadbackBuffer == nullptr) ||
                sphereContactProposalOutputBuffer == nullptr ||
                sphereContactProposalOutputReadbackBuffer == nullptr ||
                sphereContactIterationOutputBuffer == nullptr ||
                sphereContactIterationScratchBuffer == nullptr ||
                sphereContactProposalResidualBuffer == nullptr ||
                (validateResidentContacts && sphereContactProposalResidualReadbackBuffer == nullptr) ||
                sphereContactBodyBuffer == nullptr ||
                sphereContactAdjacencyParamsBuffer == nullptr)
            {
                setStatus(sapStatus, "SAP resident collision failed: contact adjacency buffer allocation");
                return false;
            }

            std::vector<GpuContactBody> contactBodies(world.bodies.size());
            sphereContactFinalReferencePositions.resize(world.bodies.size());
            for (BodyId bodyId = 0; bodyId < world.bodies.size(); ++bodyId)
            {
                const SimBodyData &body = world.bodies[bodyId];
                GpuContactBody gpuBody = {};
                gpuBody.position[0] = body.positionLin.x;
                gpuBody.position[1] = body.positionLin.y;
                gpuBody.position[2] = body.positionLin.z;
                gpuBody.position[3] = isGpuStaticGroundReceiver(body) ? (body.positionLin.z + body.shape.size.z * 0.5f) : body.radius;
                gpuBody.orientation[0] = body.positionAng.x;
                gpuBody.orientation[1] = body.positionAng.y;
                gpuBody.orientation[2] = body.positionAng.z;
                gpuBody.orientation[3] = body.positionAng.w;
                gpuBody.halfSizeRadius[0] = std::max(0.0f, body.shape.size.x * 0.5f);
                gpuBody.halfSizeRadius[1] = std::max(0.0f, body.shape.size.y * 0.5f);
                gpuBody.halfSizeRadius[2] = std::max(0.0f, body.shape.size.z * 0.5f);
                gpuBody.halfSizeRadius[3] = body.shape.radius > 0.0f ? body.shape.radius : body.radius;
                gpuBody.extra[0] = std::max(0.0f, body.shape.halfLength);
                gpuBody.extra[1] = isGpuStaticGroundReceiver(body) ? (body.positionLin.z + body.shape.size.z * 0.5f) : 0.0f;
                gpuBody.extra[2] = body.mass > 0.0f ? 1.0f : 0.0f;
                gpuBody.extra[3] = 0.0f;
                gpuBody.shapeType = (uint32_t)body.shape.type;
                gpuBody.pad0 = body.mass > 0.0f ? 1u : 0u;
                contactBodies[bodyId] = gpuBody;
                sphereContactFinalReferencePositions[bodyId] = body.positionLin;
            }
            queue.WriteBuffer(sphereContactBodyBuffer, 0, contactBodies.data(), contactBodyBytes);

            std::vector<uint32_t> zeroBodyCounts(bodyCount, 0u);
            queue.WriteBuffer(sphereContactBodyCountBuffer, 0, zeroBodyCounts.data(), bodyCountBytes);
            GpuContactAdjacencyParams adjacencyParams = {};
            adjacencyParams.contactCount = sphereContactOutputCount;
            adjacencyParams.bodyCount = bodyCount;
            adjacencyParams.capacityPerBody = capacityPerBody;
            adjacencyParams.contactRelaxation = std::max(0.0f, environmentFloatValue("AVBD_GPU_CONTACT_RELAXATION", residentContactRelaxation));
            queue.WriteBuffer(sphereContactAdjacencyParamsBuffer, 0, &adjacencyParams, sizeof(adjacencyParams));
            sphereContactIterationRelaxation = adjacencyParams.contactRelaxation;

            static const char *adjacencyShaderSource = R"(
struct Contact {
    bodyA: u32,
    bodyB: u32,
    pad0: u32,
    pad1: u32,
    normal: vec4f,
};

struct Params {
    contactCount: u32,
    bodyCount: u32,
    capacityPerBody: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> bodyCounts: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> adjacency: array<u32>;

fn writeRef(bodyIndex: u32, encodedRef: u32) {
    if (bodyIndex >= params.bodyCount) {
        return;
    }
    let slot = atomicAdd(&bodyCounts[bodyIndex], 1u);
    if (slot < params.capacityPerBody) {
        adjacency[bodyIndex * params.capacityPerBody + slot] = encodedRef;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= params.contactCount) {
        return;
    }
    let contact = contacts[index];
    if (contact.pad0 == 0u) {
        return;
    }
    if (contact.bodyA >= params.bodyCount || contact.bodyB >= params.bodyCount) {
        return;
    }
    if (contact.pad0 == 2u) {
        writeRef(contact.bodyA, index * 2u);
        return;
    }
    writeRef(contact.bodyA, index * 2u);
    writeRef(contact.bodyB, index * 2u + 1u);
}
)";

            if (sphereContactAdjacencyPipeline == nullptr || sphereContactAdjacencyBindGroupLayout == nullptr)
            {
                wgpu::ShaderSourceWGSL wgsl = {};
                wgsl.code = adjacencyShaderSource;
                wgpu::ShaderModuleDescriptor shaderDesc = {};
                shaderDesc.nextInChain = &wgsl;
                wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
                if (shader == nullptr)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact adjacency shader");
                    return false;
                }

                wgpu::BindGroupLayoutEntry entries[4] = {};
                entries[0].binding = 0;
                entries[0].visibility = wgpu::ShaderStage::Compute;
                entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[0].buffer.minBindingSize = sizeof(GpuSphereContact);
                entries[1].binding = 1;
                entries[1].visibility = wgpu::ShaderStage::Compute;
                entries[1].buffer.type = wgpu::BufferBindingType::Storage;
                entries[1].buffer.minBindingSize = sizeof(uint32_t);
                entries[2].binding = 2;
                entries[2].visibility = wgpu::ShaderStage::Compute;
                entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
                entries[2].buffer.minBindingSize = sizeof(GpuContactAdjacencyParams);
                entries[3].binding = 3;
                entries[3].visibility = wgpu::ShaderStage::Compute;
                entries[3].buffer.type = wgpu::BufferBindingType::Storage;
                entries[3].buffer.minBindingSize = sizeof(uint32_t);
                wgpu::BindGroupLayoutDescriptor layoutDesc = {};
                layoutDesc.entryCount = 4;
                layoutDesc.entries = entries;
                sphereContactAdjacencyBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
                wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
                pipelineLayoutDesc.bindGroupLayoutCount = 1;
                pipelineLayoutDesc.bindGroupLayouts = &sphereContactAdjacencyBindGroupLayout;
                wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
                wgpu::ComputePipelineDescriptor pipelineDesc = {};
                pipelineDesc.layout = pipelineLayout;
                pipelineDesc.compute.module = shader;
                pipelineDesc.compute.entryPoint = "main";
                sphereContactAdjacencyPipeline = device.CreateComputePipeline(&pipelineDesc);
                if (sphereContactAdjacencyPipeline == nullptr)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact adjacency pipeline");
                    return false;
                }
            }

            wgpu::BindGroupEntry adjacencyEntries[4] = {};
            adjacencyEntries[0].binding = 0;
            adjacencyEntries[0].buffer = sphereContactOutputBuffer;
            adjacencyEntries[0].size = (uint64_t)sphereContactOutputCount * sizeof(GpuSphereContact);
            adjacencyEntries[1].binding = 1;
            adjacencyEntries[1].buffer = sphereContactBodyCountBuffer;
            adjacencyEntries[1].size = bodyCountBytes;
            adjacencyEntries[2].binding = 2;
            adjacencyEntries[2].buffer = sphereContactAdjacencyParamsBuffer;
            adjacencyEntries[2].size = sizeof(GpuContactAdjacencyParams);
            adjacencyEntries[3].binding = 3;
            adjacencyEntries[3].buffer = sphereContactAdjacencyBuffer;
            adjacencyEntries[3].size = adjacencyBytes;
            wgpu::BindGroupDescriptor adjacencyBindGroupDesc = {};
            adjacencyBindGroupDesc.layout = sphereContactAdjacencyBindGroupLayout;
            adjacencyBindGroupDesc.entryCount = 4;
            adjacencyBindGroupDesc.entries = adjacencyEntries;
            wgpu::BindGroup adjacencyBindGroup = device.CreateBindGroup(&adjacencyBindGroupDesc);
            if (adjacencyBindGroup == nullptr)
            {
                setStatus(sapStatus, "SAP resident collision failed: contact adjacency bind group");
                return false;
            }

            wgpu::CommandEncoder adjacencyEncoder = device.CreateCommandEncoder();
            wgpu::ComputePassEncoder adjacencyPass = adjacencyEncoder.BeginComputePass();
            adjacencyPass.SetPipeline(sphereContactAdjacencyPipeline);
            adjacencyPass.SetBindGroup(0, adjacencyBindGroup);
            adjacencyPass.DispatchWorkgroups((sphereContactOutputCount + 63u) / 64u);
            adjacencyPass.End();
            if (validateResidentContacts)
                adjacencyEncoder.CopyBufferToBuffer(sphereContactBodyCountBuffer, 0,
                                                    sphereContactBodyCountReadbackBuffer, 0, bodyCountBytes);
            wgpu::CommandBuffer adjacencyCommands = adjacencyEncoder.Finish();
            queue.Submit(1, &adjacencyCommands);

            if (validateResidentContacts)
            {
                mapDone = false;
                mapStatus = wgpu::MapAsyncStatus::Error;
                wgpu::Future adjacencyMapFuture = sphereContactBodyCountReadbackBuffer.MapAsync(
                    wgpu::MapMode::Read, 0, bodyCountBytes, wgpu::CallbackMode::WaitAnyOnly,
                    [&](wgpu::MapAsyncStatus status, wgpu::StringView)
                    {
                        mapDone = true;
                        mapStatus = status;
                    });
                if (instance.WaitAny(adjacencyMapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
                    !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact adjacency readback map");
                    return false;
                }
                const uint32_t *bodyCounts = (const uint32_t *)sphereContactBodyCountReadbackBuffer.GetConstMappedRange(0, bodyCountBytes);
                if (!bodyCounts)
                {
                    sphereContactBodyCountReadbackBuffer.Unmap();
                    setStatus(sapStatus, "SAP resident collision failed: contact adjacency readback range");
                    return false;
                }

                int refs = 0;
                int activeBodies = 0;
                int maxPerBody = 0;
                int overflowRefs = 0;
                int writtenRefs = 0;
                for (uint32_t i = 0; i < bodyCount; ++i)
                {
                    int count = (int)bodyCounts[i];
                    refs += count;
                    writtenRefs += std::min(count, (int)capacityPerBody);
                    if (count > (int)capacityPerBody)
                        overflowRefs += count - (int)capacityPerBody;
                    if (count > 0)
                    {
                        ++activeBodies;
                        if (count > maxPerBody)
                            maxPerBody = count;
                    }
                }
                sphereContactBodyCountReadbackBuffer.Unmap();

                sphereContactBodyRefs = refs;
                sphereContactActiveBodies = activeBodies;
                sphereContactMaxPerBody = maxPerBody;
                sphereContactAvgPerActiveBody = activeBodies > 0 ? (float)refs / (float)activeBodies : 0.0f;
                sphereContactAdjacencyReadbackBytes = (int)bodyCountBytes;
                sphereContactAdjacencyWrittenRefs = writtenRefs;
                sphereContactAdjacencyOverflowRefs = overflowRefs;
            }
            sphereContactAdjacencyListBytes = (int)adjacencyBytes;
            sphereContactAdjacencyCapacity = (int)capacityPerBody;
            sphereContactAdjacencyMs = elapsedMs(adjacencyBegin, SDL_GetPerformanceCounter());

            Uint64 gatherBegin = SDL_GetPerformanceCounter();
            static const char *gatherShaderSource = R"(
struct Contact {
    bodyA: u32,
    bodyB: u32,
    pad0: u32,
    pad1: u32,
    normal: vec4f,
};

struct Body {
    position: vec4f,
    orientation: vec4f,
    halfSizeRadius: vec4f,
    extra: vec4f,
    shapeType: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Params {
    contactCount: u32,
    bodyCount: u32,
    capacityPerBody: u32,
    contactRelaxation: f32,
};

struct GatherOutput {
    count: u32,
    overflow: u32,
    pad0: u32,
    pad1: u32,
    normalSum: vec4f,
    correction: vec4f,
};

@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> bodyCounts: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> adjacency: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputs: array<GatherOutput>;
@group(0) @binding(5) var<storage, read> bodies: array<Body>;

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn quatInverseRotate(q: vec4f, v: vec3f) -> vec3f {
    return quatRotate(vec4f(-q.x, -q.y, -q.z, q.w), v);
}

fn sphereBoxEval(spherePos: vec3f, sphereRadius: f32, boxBody: Body, boxPos: vec3f) -> vec4f {
    let halfSize = boxBody.halfSizeRadius.xyz;
    if (halfSize.x <= 0.0 || halfSize.y <= 0.0 || halfSize.z <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    let local = quatInverseRotate(boxBody.orientation, spherePos - boxPos);
    let closestLocal = clamp(local, -halfSize, halfSize);
    let closest = boxPos + quatRotate(boxBody.orientation, closestLocal);
    let delta = spherePos - closest;
    let distSq = dot(delta, delta);
    if (distSq >= sphereRadius * sphereRadius) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    var direction = vec3f(1.0, 0.0, 0.0);
    var penetration = sphereRadius;
    if (distSq > 0.000001) {
        let dist = sqrt(distSq);
        direction = delta / dist;
        penetration = sphereRadius - dist;
    } else {
        let clearance = halfSize - abs(local);
        if (clearance.x <= clearance.y && clearance.x <= clearance.z) {
            direction = quatRotate(boxBody.orientation, vec3f(select(-1.0, 1.0, local.x >= 0.0), 0.0, 0.0));
            penetration = sphereRadius + max(clearance.x, 0.0);
        } else if (clearance.y <= clearance.z) {
            direction = quatRotate(boxBody.orientation, vec3f(0.0, select(-1.0, 1.0, local.y >= 0.0), 0.0));
            penetration = sphereRadius + max(clearance.y, 0.0);
        } else {
            direction = quatRotate(boxBody.orientation, vec3f(0.0, 0.0, select(-1.0, 1.0, local.z >= 0.0)));
            penetration = sphereRadius + max(clearance.z, 0.0);
        }
    }
    return vec4f(direction, max(penetration, 0.0));
}

fn sphereCapsuleEval(spherePos: vec3f, sphereRadius: f32, capsuleBody: Body, capsulePos: vec3f) -> vec4f {
    let capsuleRadius = capsuleBody.halfSizeRadius.w;
    let capsuleHalfLength = capsuleBody.extra.x;
    if (capsuleRadius <= 0.0 || capsuleHalfLength <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    let local = quatInverseRotate(capsuleBody.orientation, spherePos - capsulePos);
    let closestLocal = vec3f(0.0, 0.0, clamp(local.z, -capsuleHalfLength, capsuleHalfLength));
    let delta = local - closestLocal;
    let radius = sphereRadius + capsuleRadius;
    let distSq = dot(delta, delta);
    if (distSq >= radius * radius) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    var direction = vec3f(1.0, 0.0, 0.0);
    var penetration = radius;
    if (distSq > 0.000001) {
        let dist = sqrt(distSq);
        direction = quatRotate(capsuleBody.orientation, delta / dist);
        penetration = radius - dist;
    }
    return vec4f(direction, max(penetration, 0.0));
}

fn sphereCylinderEval(spherePos: vec3f, sphereRadius: f32, cylinderBody: Body, cylinderPos: vec3f) -> vec4f {
    let cylinderRadius = cylinderBody.halfSizeRadius.w;
    let cylinderHalfLength = cylinderBody.extra.x;
    if (cylinderRadius <= 0.0 || cylinderHalfLength <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    let local = quatInverseRotate(cylinderBody.orientation, spherePos - cylinderPos);
    let radial = vec2f(local.x, local.y);
    let radialLen = length(radial);
    let clampedZ = clamp(local.z, -cylinderHalfLength, cylinderHalfLength);
    var closestRadial = radial;
    if (radialLen > cylinderRadius) {
        closestRadial = radial * (cylinderRadius / max(radialLen, 0.000001));
    }
    let closestLocal = vec3f(closestRadial.x, closestRadial.y, clampedZ);
    let delta = local - closestLocal;
    let distSq = dot(delta, delta);
    if (distSq >= sphereRadius * sphereRadius) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    var direction = vec3f(0.0, 0.0, 1.0);
    var penetration = sphereRadius;
    if (distSq > 0.000001) {
        let dist = sqrt(distSq);
        direction = quatRotate(cylinderBody.orientation, delta / dist);
        penetration = sphereRadius - dist;
    } else {
        let sideClearance = cylinderRadius - radialLen;
        let capClearance = cylinderHalfLength - abs(local.z);
        if (sideClearance < capClearance && radialLen > 0.000001) {
            direction = quatRotate(cylinderBody.orientation, vec3f(radial.x / radialLen, radial.y / radialLen, 0.0));
            penetration = sphereRadius + max(sideClearance, 0.0);
        } else {
            direction = quatRotate(cylinderBody.orientation, vec3f(0.0, 0.0, select(-1.0, 1.0, local.z >= 0.0)));
            penetration = sphereRadius + max(capClearance, 0.0);
        }
    }
    return vec4f(direction, max(penetration, 0.0));
}

fn roundEndpointBody(body: Body, pos: vec3f, signValue: f32) -> vec3f {
    let halfLength = select(0.0, body.extra.x, body.shapeType == 2u || body.shapeType == 3u);
    if (halfLength <= 0.0) {
        return pos;
    }
    return pos + quatRotate(body.orientation, vec3f(0.0, 0.0, signValue * halfLength));
}

fn roundPairEval(bodyA: Body, posA: vec3f, bodyB: Body, posB: vec3f) -> vec4f {
    let radiusA = bodyA.halfSizeRadius.w;
    let radiusB = bodyB.halfSizeRadius.w;
    if (radiusA <= 0.0 || radiusB <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    let p1 = roundEndpointBody(bodyA, posA, -1.0);
    let q1 = roundEndpointBody(bodyA, posA, 1.0);
    let p2 = roundEndpointBody(bodyB, posB, -1.0);
    let q2 = roundEndpointBody(bodyB, posB, 1.0);
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let aa = dot(d1, d1);
    let ee = dot(d2, d2);
    let ff = dot(d2, r);

    var s = 0.0;
    var t = 0.0;
    if (aa <= 0.000001 && ee <= 0.000001) {
        s = 0.0;
        t = 0.0;
    } else if (aa <= 0.000001) {
        s = 0.0;
        t = clamp(ff / max(ee, 0.000001), 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (ee <= 0.000001) {
            t = 0.0;
            s = clamp(-c / max(aa, 0.000001), 0.0, 1.0);
        } else {
            let bDot = dot(d1, d2);
            let denom = aa * ee - bDot * bDot;
            if (abs(denom) > 0.000001) {
                s = clamp((bDot * ff - c * ee) / denom, 0.0, 1.0);
            } else {
                s = 0.0;
            }
            t = (bDot * s + ff) / max(ee, 0.000001);
            if (t < 0.0) {
                t = 0.0;
                s = clamp(-c / max(aa, 0.000001), 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((bDot - c) / max(aa, 0.000001), 0.0, 1.0);
            }
        }
    }

    let closestA = p1 + d1 * s;
    let closestB = p2 + d2 * t;
    let delta = closestA - closestB;
    let distSq = dot(delta, delta);
    let radius = radiusA + radiusB;
    if (distSq >= radius * radius) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    var direction = vec3f(1.0, 0.0, 0.0);
    var dist = 0.0;
    if (distSq > 0.000001) {
        dist = sqrt(distSq);
        direction = delta / dist;
    }
    return vec4f(direction, max(radius - dist, 0.0));
}

fn pointBoxEvalBody(point: vec3f, radius: f32, boxBody: Body, boxPos: vec3f) -> vec4f {
    let halfSize = boxBody.halfSizeRadius.xyz;
    if (radius <= 0.0 || halfSize.x <= 0.0 || halfSize.y <= 0.0 || halfSize.z <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    let local = quatInverseRotate(boxBody.orientation, point - boxPos);
    let closestLocal = clamp(local, -halfSize, halfSize);
    let closest = boxPos + quatRotate(boxBody.orientation, closestLocal);
    let delta = point - closest;
    let distSq = dot(delta, delta);
    if (distSq >= radius * radius) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    var direction = vec3f(1.0, 0.0, 0.0);
    var penetration = radius;
    if (distSq > 0.000001) {
        let dist = sqrt(distSq);
        direction = delta / dist;
        penetration = radius - dist;
    } else {
        let clearance = halfSize - abs(local);
        if (clearance.x <= clearance.y && clearance.x <= clearance.z) {
            direction = quatRotate(boxBody.orientation, vec3f(select(-1.0, 1.0, local.x >= 0.0), 0.0, 0.0));
            penetration = radius + max(clearance.x, 0.0);
        } else if (clearance.y <= clearance.z) {
            direction = quatRotate(boxBody.orientation, vec3f(0.0, select(-1.0, 1.0, local.y >= 0.0), 0.0));
            penetration = radius + max(clearance.y, 0.0);
        } else {
            direction = quatRotate(boxBody.orientation, vec3f(0.0, 0.0, select(-1.0, 1.0, local.z >= 0.0)));
            penetration = radius + max(clearance.z, 0.0);
        }
    }
    return vec4f(direction, max(penetration, 0.0));
}

fn roundBoxEval(roundBody: Body, roundPos: vec3f, boxBody: Body, boxPos: vec3f) -> vec4f {
    let radius = roundBody.halfSizeRadius.w;
    let centerEval = pointBoxEvalBody(roundPos, radius, boxBody, boxPos);
    let halfLength = select(0.0, roundBody.extra.x, roundBody.shapeType == 2u || roundBody.shapeType == 3u);
    if (halfLength <= 0.0) {
        return centerEval;
    }

    let aEval = pointBoxEvalBody(roundEndpointBody(roundBody, roundPos, -1.0), radius, boxBody, boxPos);
    let bEval = pointBoxEvalBody(roundEndpointBody(roundBody, roundPos, 1.0), radius, boxBody, boxPos);
    var best = centerEval;
    if (aEval.w > best.w) {
        best = aEval;
    }
    if (bEval.w > best.w) {
        best = bEval;
    }
    return best;
}

fn contactEval(contact: Contact, posA: vec3f, posB: vec3f) -> vec4f {
    let bodyA = bodies[contact.bodyA];
    let bodyB = bodies[contact.bodyB];
    if (contact.pad0 == 2u) {
        let penetration = max(bodyB.extra.y + bodyA.halfSizeRadius.w - posA.z, 0.0);
        return vec4f(0.0, 0.0, 1.0, penetration);
    }
    if (contact.pad0 == 3u) {
        return sphereBoxEval(posA, bodyA.halfSizeRadius.w, bodyB, posB);
    }
    if (contact.pad0 == 4u) {
        return sphereCapsuleEval(posA, bodyA.halfSizeRadius.w, bodyB, posB);
    }
    if (contact.pad0 == 5u) {
        return sphereCylinderEval(posA, bodyA.halfSizeRadius.w, bodyB, posB);
    }
    if (contact.pad0 == 6u) {
        return roundPairEval(bodyA, posA, bodyB, posB);
    }
    if (contact.pad0 == 7u) {
        return roundBoxEval(bodyA, posA, bodyB, posB);
    }
    let delta = posA - posB;
    let dist = max(length(delta), 0.000001);
    let penetration = max(bodyA.halfSizeRadius.w + bodyB.halfSizeRadius.w - dist, 0.0);
    return vec4f(delta / dist, penetration);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let bodyIndex = id.x;
    if (bodyIndex >= params.bodyCount) {
        return;
    }

    let rawCount = bodyCounts[bodyIndex];
    let gatherCount = min(rawCount, params.capacityPerBody);
    var normalSum = vec3f(0.0, 0.0, 0.0);
    var correction = vec3f(0.0, 0.0, 0.0);
    var slot = 0u;
    loop {
        if (slot >= gatherCount) {
            break;
        }
        let encoded = adjacency[bodyIndex * params.capacityPerBody + slot];
        let contactIndex = encoded / 2u;
        let side = encoded & 1u;
        if (contactIndex < params.contactCount) {
            let contact = contacts[contactIndex];
            let currentBody = bodies[bodyIndex];
            if (currentBody.extra.z <= 0.5) {
                slot = slot + 1u;
                continue;
            }
            let eval = contactEval(contact, bodies[contact.bodyA].position.xyz, bodies[contact.bodyB].position.xyz);
            if (contact.pad0 == 2u) {
                let groundCorrection = min(eval.w, 0.25);
                normalSum = normalSum + vec3f(0.0, 0.0, 1.0);
                correction = correction + vec3f(0.0, 0.0, groundCorrection);
            } else {
                let direction = select(eval.xyz, -eval.xyz, side != 0u);
                let otherBody = bodies[select(contact.bodyB, contact.bodyA, side != 0u)];
                let share = select(1.0, 0.5, otherBody.extra.z > 0.5);
                normalSum = normalSum + direction;
                correction = correction + direction * (share * eval.w);
            }
        }
        slot = slot + 1u;
    }

    outputs[bodyIndex].count = gatherCount;
    outputs[bodyIndex].overflow = select(0u, rawCount - params.capacityPerBody, rawCount > params.capacityPerBody);
    outputs[bodyIndex].normalSum = vec4f(normalSum, 0.0);
    if (gatherCount > 0u) {
        outputs[bodyIndex].correction = vec4f(correction / f32(gatherCount), 0.0);
    } else {
        outputs[bodyIndex].correction = vec4f(0.0, 0.0, 0.0, 0.0);
    }
}
)";

            if (sphereContactGatherPipeline == nullptr || sphereContactGatherBindGroupLayout == nullptr)
            {
                wgpu::ShaderSourceWGSL wgsl = {};
                wgsl.code = gatherShaderSource;
                wgpu::ShaderModuleDescriptor shaderDesc = {};
                shaderDesc.nextInChain = &wgsl;
                wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
                if (shader == nullptr)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact gather shader");
                    return false;
                }

                wgpu::BindGroupLayoutEntry entries[6] = {};
                entries[0].binding = 0;
                entries[0].visibility = wgpu::ShaderStage::Compute;
                entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[0].buffer.minBindingSize = sizeof(GpuSphereContact);
                entries[1].binding = 1;
                entries[1].visibility = wgpu::ShaderStage::Compute;
                entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[1].buffer.minBindingSize = sizeof(uint32_t);
                entries[2].binding = 2;
                entries[2].visibility = wgpu::ShaderStage::Compute;
                entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
                entries[2].buffer.minBindingSize = sizeof(GpuContactAdjacencyParams);
                entries[3].binding = 3;
                entries[3].visibility = wgpu::ShaderStage::Compute;
                entries[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[3].buffer.minBindingSize = sizeof(uint32_t);
                entries[4].binding = 4;
                entries[4].visibility = wgpu::ShaderStage::Compute;
                entries[4].buffer.type = wgpu::BufferBindingType::Storage;
                entries[4].buffer.minBindingSize = sizeof(GpuContactGatherOutput);
                entries[5].binding = 5;
                entries[5].visibility = wgpu::ShaderStage::Compute;
                entries[5].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[5].buffer.minBindingSize = sizeof(GpuContactBody);
                wgpu::BindGroupLayoutDescriptor layoutDesc = {};
                layoutDesc.entryCount = 6;
                layoutDesc.entries = entries;
                sphereContactGatherBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
                wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
                pipelineLayoutDesc.bindGroupLayoutCount = 1;
                pipelineLayoutDesc.bindGroupLayouts = &sphereContactGatherBindGroupLayout;
                wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
                wgpu::ComputePipelineDescriptor pipelineDesc = {};
                pipelineDesc.layout = pipelineLayout;
                pipelineDesc.compute.module = shader;
                pipelineDesc.compute.entryPoint = "main";
                sphereContactGatherPipeline = device.CreateComputePipeline(&pipelineDesc);
                if (sphereContactGatherPipeline == nullptr)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact gather pipeline");
                    return false;
                }
            }

            wgpu::BindGroupEntry gatherEntries[6] = {};
            gatherEntries[0].binding = 0;
            gatherEntries[0].buffer = sphereContactOutputBuffer;
            gatherEntries[0].size = (uint64_t)sphereContactOutputCount * sizeof(GpuSphereContact);
            gatherEntries[1].binding = 1;
            gatherEntries[1].buffer = sphereContactBodyCountBuffer;
            gatherEntries[1].size = bodyCountBytes;
            gatherEntries[2].binding = 2;
            gatherEntries[2].buffer = sphereContactAdjacencyParamsBuffer;
            gatherEntries[2].size = sizeof(GpuContactAdjacencyParams);
            gatherEntries[3].binding = 3;
            gatherEntries[3].buffer = sphereContactAdjacencyBuffer;
            gatherEntries[3].size = adjacencyBytes;
            gatherEntries[4].binding = 4;
            gatherEntries[4].buffer = sphereContactGatherOutputBuffer;
            gatherEntries[4].size = gatherBytes;
            gatherEntries[5].binding = 5;
            gatherEntries[5].buffer = sphereContactBodyBuffer;
            gatherEntries[5].size = contactBodyBytes;
            wgpu::BindGroupDescriptor gatherBindGroupDesc = {};
            gatherBindGroupDesc.layout = sphereContactGatherBindGroupLayout;
            gatherBindGroupDesc.entryCount = 6;
            gatherBindGroupDesc.entries = gatherEntries;
            wgpu::BindGroup gatherBindGroup = device.CreateBindGroup(&gatherBindGroupDesc);
            if (gatherBindGroup == nullptr)
            {
                setStatus(sapStatus, "SAP resident collision failed: contact gather bind group");
                return false;
            }

            wgpu::CommandEncoder gatherEncoder = device.CreateCommandEncoder();
            wgpu::ComputePassEncoder gatherPass = gatherEncoder.BeginComputePass();
            gatherPass.SetPipeline(sphereContactGatherPipeline);
            gatherPass.SetBindGroup(0, gatherBindGroup);
            gatherPass.DispatchWorkgroups((bodyCount + 63u) / 64u);
            gatherPass.End();
            if (validateResidentContacts)
                gatherEncoder.CopyBufferToBuffer(sphereContactGatherOutputBuffer, 0,
                                                 sphereContactGatherReadbackBuffer, 0, gatherBytes);
            wgpu::CommandBuffer gatherCommands = gatherEncoder.Finish();
            queue.Submit(1, &gatherCommands);

            if (validateResidentContacts)
            {
                mapDone = false;
                mapStatus = wgpu::MapAsyncStatus::Error;
                wgpu::Future gatherMapFuture = sphereContactGatherReadbackBuffer.MapAsync(
                    wgpu::MapMode::Read, 0, gatherBytes, wgpu::CallbackMode::WaitAnyOnly,
                    [&](wgpu::MapAsyncStatus status, wgpu::StringView)
                    {
                        mapDone = true;
                        mapStatus = status;
                    });
                if (instance.WaitAny(gatherMapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
                    !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact gather readback map");
                    return false;
                }
                const GpuContactGatherOutput *gatherOutputs =
                    (const GpuContactGatherOutput *)sphereContactGatherReadbackBuffer.GetConstMappedRange(0, gatherBytes);
                if (!gatherOutputs)
                {
                    sphereContactGatherReadbackBuffer.Unmap();
                    setStatus(sapStatus, "SAP resident collision failed: contact gather readback range");
                    return false;
                }

                int gatherRefs = 0;
                int gatherActiveBodies = 0;
                int gatherMaxPerBody = 0;
                int gatherOverflowRefs = 0;
                int proposalActiveBodies = 0;
                double maxCorrection = 0.0;
                double correctionChecksum = 0.0;
                double normalChecksum = 0.0;
                for (uint32_t i = 0; i < bodyCount; ++i)
                {
                    int count = (int)gatherOutputs[i].count;
                    gatherRefs += count;
                    gatherOverflowRefs += (int)gatherOutputs[i].overflow;
                    if (count > 0)
                    {
                        ++gatherActiveBodies;
                        if (count > gatherMaxPerBody)
                            gatherMaxPerBody = count;
                        normalChecksum += fabs((double)gatherOutputs[i].normalSum[0]) +
                                          fabs((double)gatherOutputs[i].normalSum[1]) +
                                          fabs((double)gatherOutputs[i].normalSum[2]);
                        double cx = (double)gatherOutputs[i].correction[0];
                        double cy = (double)gatherOutputs[i].correction[1];
                        double cz = (double)gatherOutputs[i].correction[2];
                        double correctionLength = sqrt(cx * cx + cy * cy + cz * cz);
                        if (correctionLength > 0.0)
                        {
                            ++proposalActiveBodies;
                            correctionChecksum += fabs(cx) + fabs(cy) + fabs(cz);
                            if (correctionLength > maxCorrection)
                                maxCorrection = correctionLength;
                        }
                    }
                }
                sphereContactGatherReadbackBuffer.Unmap();

                sphereContactGatherRefs = gatherRefs;
                sphereContactGatherActiveBodies = gatherActiveBodies;
                sphereContactGatherMaxPerBody = gatherMaxPerBody;
                sphereContactGatherMismatches = (gatherRefs != sphereContactAdjacencyWrittenRefs ? 1 : 0) +
                                                (gatherOverflowRefs != sphereContactAdjacencyOverflowRefs ? 1 : 0);
                sphereContactGatherReadbackBytes = (int)gatherBytes;
                sphereContactGatherNormalChecksum = (float)normalChecksum;
                sphereContactProposalActiveBodies = proposalActiveBodies;
                sphereContactProposalMaxCorrection = (float)maxCorrection;
                sphereContactProposalCorrectionChecksum = (float)correctionChecksum;
            }
            sphereContactGatherMs = elapsedMs(gatherBegin, SDL_GetPerformanceCounter());

            Uint64 proposalBegin = SDL_GetPerformanceCounter();
            static const char *proposalShaderSource = R"(
struct Body {
    position: vec4f,
    orientation: vec4f,
    halfSizeRadius: vec4f,
    extra: vec4f,
    shapeType: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Params {
    contactCount: u32,
    bodyCount: u32,
    capacityPerBody: u32,
    contactRelaxation: f32,
};

struct GatherOutput {
    count: u32,
    overflow: u32,
    pad0: u32,
    pad1: u32,
    normalSum: vec4f,
    correction: vec4f,
};

struct ProposalOutput {
    position: vec4f,
    correction: vec4f,
};

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<storage, read> proposals: array<GatherOutput>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> outputs: array<ProposalOutput>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let bodyIndex = id.x;
    if (bodyIndex >= params.bodyCount) {
        return;
    }
    let body = bodies[bodyIndex];
    let proposal = proposals[bodyIndex].correction;
    outputs[bodyIndex].position = vec4f(body.position.xyz + proposal.xyz, body.position.w);
    outputs[bodyIndex].correction = proposal;
}
)";

            if (sphereContactProposalPipeline == nullptr || sphereContactProposalBindGroupLayout == nullptr)
            {
                wgpu::ShaderSourceWGSL wgsl = {};
                wgsl.code = proposalShaderSource;
                wgpu::ShaderModuleDescriptor shaderDesc = {};
                shaderDesc.nextInChain = &wgsl;
                wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
                if (shader == nullptr)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact proposal shader");
                    return false;
                }

                wgpu::BindGroupLayoutEntry entries[4] = {};
                entries[0].binding = 0;
                entries[0].visibility = wgpu::ShaderStage::Compute;
                entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[0].buffer.minBindingSize = sizeof(GpuContactBody);
                entries[1].binding = 1;
                entries[1].visibility = wgpu::ShaderStage::Compute;
                entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[1].buffer.minBindingSize = sizeof(GpuContactGatherOutput);
                entries[2].binding = 2;
                entries[2].visibility = wgpu::ShaderStage::Compute;
                entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
                entries[2].buffer.minBindingSize = sizeof(GpuContactAdjacencyParams);
                entries[3].binding = 3;
                entries[3].visibility = wgpu::ShaderStage::Compute;
                entries[3].buffer.type = wgpu::BufferBindingType::Storage;
                entries[3].buffer.minBindingSize = sizeof(GpuContactProposalOutput);
                wgpu::BindGroupLayoutDescriptor layoutDesc = {};
                layoutDesc.entryCount = 4;
                layoutDesc.entries = entries;
                sphereContactProposalBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
                wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
                pipelineLayoutDesc.bindGroupLayoutCount = 1;
                pipelineLayoutDesc.bindGroupLayouts = &sphereContactProposalBindGroupLayout;
                wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
                wgpu::ComputePipelineDescriptor pipelineDesc = {};
                pipelineDesc.layout = pipelineLayout;
                pipelineDesc.compute.module = shader;
                pipelineDesc.compute.entryPoint = "main";
                sphereContactProposalPipeline = device.CreateComputePipeline(&pipelineDesc);
                if (sphereContactProposalPipeline == nullptr)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact proposal pipeline");
                    return false;
                }
            }

            wgpu::BindGroupEntry proposalEntries[4] = {};
            proposalEntries[0].binding = 0;
            proposalEntries[0].buffer = sphereContactBodyBuffer;
            proposalEntries[0].size = contactBodyBytes;
            proposalEntries[1].binding = 1;
            proposalEntries[1].buffer = sphereContactGatherOutputBuffer;
            proposalEntries[1].size = gatherBytes;
            proposalEntries[2].binding = 2;
            proposalEntries[2].buffer = sphereContactAdjacencyParamsBuffer;
            proposalEntries[2].size = sizeof(GpuContactAdjacencyParams);
            proposalEntries[3].binding = 3;
            proposalEntries[3].buffer = sphereContactProposalOutputBuffer;
            proposalEntries[3].size = proposalOutputBytes;
            wgpu::BindGroupDescriptor proposalBindGroupDesc = {};
            proposalBindGroupDesc.layout = sphereContactProposalBindGroupLayout;
            proposalBindGroupDesc.entryCount = 4;
            proposalBindGroupDesc.entries = proposalEntries;
            wgpu::BindGroup proposalBindGroup = device.CreateBindGroup(&proposalBindGroupDesc);
            if (proposalBindGroup == nullptr)
            {
                setStatus(sapStatus, "SAP resident collision failed: contact proposal bind group");
                return false;
            }

            wgpu::CommandEncoder proposalEncoder = device.CreateCommandEncoder();
            wgpu::ComputePassEncoder proposalPass = proposalEncoder.BeginComputePass();
            proposalPass.SetPipeline(sphereContactProposalPipeline);
            proposalPass.SetBindGroup(0, proposalBindGroup);
            proposalPass.DispatchWorkgroups((bodyCount + 63u) / 64u);
            proposalPass.End();
            if (validateResidentContacts)
                proposalEncoder.CopyBufferToBuffer(sphereContactProposalOutputBuffer, 0,
                                                   sphereContactProposalOutputReadbackBuffer, 0, proposalOutputBytes);
            wgpu::CommandBuffer proposalCommands = proposalEncoder.Finish();
            queue.Submit(1, &proposalCommands);

            if (validateResidentContacts)
            {
                mapDone = false;
                mapStatus = wgpu::MapAsyncStatus::Error;
                wgpu::Future proposalMapFuture = sphereContactProposalOutputReadbackBuffer.MapAsync(
                    wgpu::MapMode::Read, 0, proposalOutputBytes, wgpu::CallbackMode::WaitAnyOnly,
                    [&](wgpu::MapAsyncStatus status, wgpu::StringView)
                    {
                        mapDone = true;
                        mapStatus = status;
                    });
                if (instance.WaitAny(proposalMapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
                    !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact proposal readback map");
                    return false;
                }
                const GpuContactProposalOutput *proposalOutputs =
                    (const GpuContactProposalOutput *)sphereContactProposalOutputReadbackBuffer.GetConstMappedRange(0, proposalOutputBytes);
                if (!proposalOutputs)
                {
                    sphereContactProposalOutputReadbackBuffer.Unmap();
                    setStatus(sapStatus, "SAP resident collision failed: contact proposal readback range");
                    return false;
                }

                int outputActiveBodies = 0;
                double outputMaxDelta = 0.0;
                double outputChecksum = 0.0;
                for (uint32_t i = 0; i < bodyCount; ++i)
                {
                    double dx = (double)proposalOutputs[i].position[0] - (double)contactBodies[i].position[0];
                    double dy = (double)proposalOutputs[i].position[1] - (double)contactBodies[i].position[1];
                    double dz = (double)proposalOutputs[i].position[2] - (double)contactBodies[i].position[2];
                    double deltaLength = sqrt(dx * dx + dy * dy + dz * dz);
                    if (deltaLength > 0.0000001)
                    {
                        ++outputActiveBodies;
                        outputChecksum += fabs(dx) + fabs(dy) + fabs(dz);
                        if (deltaLength > outputMaxDelta)
                            outputMaxDelta = deltaLength;
                    }
                }
                sphereContactProposalOutputReadbackBuffer.Unmap();

                sphereContactProposalOutputActiveBodies = outputActiveBodies;
                sphereContactProposalOutputReadbackBytes = (int)proposalOutputBytes;
                sphereContactProposalOutputMaxDelta = (float)outputMaxDelta;
                sphereContactProposalOutputChecksum = (float)outputChecksum;
            }

            sphereContactProposalOutputMs = elapsedMs(proposalBegin, SDL_GetPerformanceCounter());
            sphereContactFinalPositionReady = 1;
            sphereContactFinalPositionBodyCount = (int)bodyCount;
            sphereContactFinalPositionBytes = (int)proposalOutputBytes;
            sphereContactFinalPositionSource = 1;
            sphereContactFinalPositionBuffer = sphereContactProposalOutputBuffer;
            sphereContactFinalPositionBufferBytes = proposalOutputBytes;

            Uint64 iterationBegin = SDL_GetPerformanceCounter();
            static const char *iterationShaderSource = R"(
struct Contact {
    bodyA: u32,
    bodyB: u32,
    pad0: u32,
    pad1: u32,
    normal: vec4f,
};

struct Params {
    contactCount: u32,
    bodyCount: u32,
    capacityPerBody: u32,
    contactRelaxation: f32,
};

struct ProposalOutput {
    position: vec4f,
    correction: vec4f,
};

fn clampVector(v: vec3f, limit: f32) -> vec3f {
    let len = length(v);
    if (len > limit) {
        return v * (limit / max(len, 0.000001));
    }
    return v;
}

@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> bodyCounts: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> adjacency: array<u32>;
@group(0) @binding(4) var<storage, read> inputPositions: array<ProposalOutput>;
@group(0) @binding(5) var<storage, read_write> outputPositions: array<ProposalOutput>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let bodyIndex = id.x;
    if (bodyIndex >= params.bodyCount) {
        return;
    }

    let current = inputPositions[bodyIndex].position;
    let rawCount = bodyCounts[bodyIndex];
    let gatherCount = min(rawCount, params.capacityPerBody);
    var bestCorrection = vec3f(0.0, 0.0, 0.0);
    var bestPen = 0.0;
    var bestIsGround = false;
    var slot = 0u;
    loop {
        if (slot >= gatherCount) {
            break;
        }
        let encoded = adjacency[bodyIndex * params.capacityPerBody + slot];
        let contactIndex = encoded / 2u;
        let side = encoded & 1u;
        if (contactIndex < params.contactCount) {
            let contact = contacts[contactIndex];
            var pen = 0.0;
            var correction = vec3f(0.0, 0.0, 0.0);
            if (contact.pad0 == 2u) {
                let ground = inputPositions[contact.bodyB].position;
                pen = max(ground.w + current.w - current.z, 0.0);
                correction = vec3f(0.0, 0.0, min(pen, 0.25));
            } else {
                let otherBody = select(contact.bodyB, contact.bodyA, side != 0u);
                let other = inputPositions[otherBody].position;
                let delta = select(current.xyz - other.xyz, other.xyz - current.xyz, side != 0u);
                let dist = max(length(delta), 0.000001);
                let signedNormal = delta / dist;
                pen = max(current.w + other.w - dist, 0.0);
                correction = signedNormal * (0.5 * pen);
            }
            if (pen > bestPen) {
                bestPen = pen;
                bestCorrection = correction;
                bestIsGround = contact.pad0 == 2u;
            }
        }
        slot = slot + 1u;
    }

    if (bestPen > 0.0) {
        var relaxed = clampVector(bestCorrection * params.contactRelaxation, 0.006);
        if (bestIsGround) {
            relaxed = clampVector(bestCorrection, 0.25);
        }
        outputPositions[bodyIndex].position = vec4f(current.xyz + relaxed, current.w);
        outputPositions[bodyIndex].correction = vec4f(relaxed, 0.0);
    } else {
        outputPositions[bodyIndex] = inputPositions[bodyIndex];
    }
}
)";

            if (sphereContactIterationPipeline == nullptr || sphereContactIterationBindGroupLayout == nullptr)
            {
                wgpu::ShaderSourceWGSL wgsl = {};
                wgsl.code = iterationShaderSource;
                wgpu::ShaderModuleDescriptor shaderDesc = {};
                shaderDesc.nextInChain = &wgsl;
                wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
                if (shader == nullptr)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact iteration shader");
                    return false;
                }

                wgpu::BindGroupLayoutEntry entries[6] = {};
                entries[0].binding = 0;
                entries[0].visibility = wgpu::ShaderStage::Compute;
                entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[0].buffer.minBindingSize = sizeof(GpuSphereContact);
                entries[1].binding = 1;
                entries[1].visibility = wgpu::ShaderStage::Compute;
                entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[1].buffer.minBindingSize = sizeof(uint32_t);
                entries[2].binding = 2;
                entries[2].visibility = wgpu::ShaderStage::Compute;
                entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
                entries[2].buffer.minBindingSize = sizeof(GpuContactAdjacencyParams);
                entries[3].binding = 3;
                entries[3].visibility = wgpu::ShaderStage::Compute;
                entries[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[3].buffer.minBindingSize = sizeof(uint32_t);
                entries[4].binding = 4;
                entries[4].visibility = wgpu::ShaderStage::Compute;
                entries[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[4].buffer.minBindingSize = sizeof(GpuContactProposalOutput);
                entries[5].binding = 5;
                entries[5].visibility = wgpu::ShaderStage::Compute;
                entries[5].buffer.type = wgpu::BufferBindingType::Storage;
                entries[5].buffer.minBindingSize = sizeof(GpuContactProposalOutput);
                wgpu::BindGroupLayoutDescriptor layoutDesc = {};
                layoutDesc.entryCount = 6;
                layoutDesc.entries = entries;
                sphereContactIterationBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
                wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
                pipelineLayoutDesc.bindGroupLayoutCount = 1;
                pipelineLayoutDesc.bindGroupLayouts = &sphereContactIterationBindGroupLayout;
                wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
                wgpu::ComputePipelineDescriptor pipelineDesc = {};
                pipelineDesc.layout = pipelineLayout;
                pipelineDesc.compute.module = shader;
                pipelineDesc.compute.entryPoint = "main";
                sphereContactIterationPipeline = device.CreateComputePipeline(&pipelineDesc);
                if (sphereContactIterationPipeline == nullptr)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact iteration pipeline");
                    return false;
                }
            }

            const int contactIterationCount = std::max(1, environmentIntValue("AVBD_GPU_CONTACT_ITERATIONS", residentContactIterations));
            wgpu::Buffer iterationInput = sphereContactProposalOutputBuffer;
            wgpu::Buffer iterationOutput = sphereContactIterationOutputBuffer;
            for (int iteration = 0; iteration < contactIterationCount; ++iteration)
            {
                if (iteration == 0)
                {
                    iterationInput = sphereContactProposalOutputBuffer;
                    iterationOutput = sphereContactIterationOutputBuffer;
                }
                else if ((iteration & 1) != 0)
                {
                    iterationInput = sphereContactIterationOutputBuffer;
                    iterationOutput = sphereContactIterationScratchBuffer;
                }
                else
                {
                    iterationInput = sphereContactIterationScratchBuffer;
                    iterationOutput = sphereContactIterationOutputBuffer;
                }

                wgpu::BindGroupEntry iterationEntries[6] = {};
                iterationEntries[0].binding = 0;
                iterationEntries[0].buffer = sphereContactOutputBuffer;
                iterationEntries[0].size = (uint64_t)sphereContactOutputCount * sizeof(GpuSphereContact);
                iterationEntries[1].binding = 1;
                iterationEntries[1].buffer = sphereContactBodyCountBuffer;
                iterationEntries[1].size = bodyCountBytes;
                iterationEntries[2].binding = 2;
                iterationEntries[2].buffer = sphereContactAdjacencyParamsBuffer;
                iterationEntries[2].size = sizeof(GpuContactAdjacencyParams);
                iterationEntries[3].binding = 3;
                iterationEntries[3].buffer = sphereContactAdjacencyBuffer;
                iterationEntries[3].size = adjacencyBytes;
                iterationEntries[4].binding = 4;
                iterationEntries[4].buffer = iterationInput;
                iterationEntries[4].size = proposalOutputBytes;
                iterationEntries[5].binding = 5;
                iterationEntries[5].buffer = iterationOutput;
                iterationEntries[5].size = proposalOutputBytes;
                wgpu::BindGroupDescriptor iterationBindGroupDesc = {};
                iterationBindGroupDesc.layout = sphereContactIterationBindGroupLayout;
                iterationBindGroupDesc.entryCount = 6;
                iterationBindGroupDesc.entries = iterationEntries;
                wgpu::BindGroup iterationBindGroup = device.CreateBindGroup(&iterationBindGroupDesc);
                if (iterationBindGroup == nullptr)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact iteration bind group");
                    return false;
                }

                wgpu::CommandEncoder iterationEncoder = device.CreateCommandEncoder();
                wgpu::ComputePassEncoder iterationPass = iterationEncoder.BeginComputePass();
                iterationPass.SetPipeline(sphereContactIterationPipeline);
                iterationPass.SetBindGroup(0, iterationBindGroup);
                iterationPass.DispatchWorkgroups((bodyCount + 63u) / 64u);
                iterationPass.End();
                wgpu::CommandBuffer iterationCommands = iterationEncoder.Finish();
                queue.Submit(1, &iterationCommands);
            }
            iterationInput = ((contactIterationCount & 1) != 0) ? sphereContactIterationOutputBuffer : sphereContactIterationScratchBuffer;
            sphereContactIterationCount = contactIterationCount;
            sphereContactIterationMs = elapsedMs(iterationBegin, SDL_GetPerformanceCounter());
            sphereContactFinalPositionReady = 1;
            sphereContactFinalPositionBodyCount = (int)bodyCount;
            sphereContactFinalPositionBytes = (int)proposalOutputBytes;
            sphereContactFinalPositionSource = ((contactIterationCount & 1) != 0) ? 2 : 3;
            sphereContactFinalPositionBuffer = iterationInput;
            sphereContactFinalPositionBufferBytes = proposalOutputBytes;

            if (validateResidentContacts)
            {
            Uint64 residualBegin = SDL_GetPerformanceCounter();
            static const char *residualShaderSource = R"(
struct Contact {
    bodyA: u32,
    bodyB: u32,
    pad0: u32,
    pad1: u32,
    normal: vec4f,
};

struct Body {
    position: vec4f,
    orientation: vec4f,
    halfSizeRadius: vec4f,
    extra: vec4f,
    shapeType: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Params {
    contactCount: u32,
    bodyCount: u32,
    capacityPerBody: u32,
    pad1: u32,
};

struct ProposalOutput {
    position: vec4f,
    correction: vec4f,
};

struct ResidualOutput {
    beforePenetration: f32,
    afterPenetration: f32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> bodies: array<Body>;
@group(0) @binding(2) var<storage, read> proposals: array<ProposalOutput>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> residuals: array<ResidualOutput>;

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn quatInverseRotate(q: vec4f, v: vec3f) -> vec3f {
    return quatRotate(vec4f(-q.x, -q.y, -q.z, q.w), v);
}

fn sphereBoxPenetration(spherePos: vec3f, sphereRadius: f32, boxBody: Body, boxPos: vec3f) -> f32 {
    let halfSize = boxBody.halfSizeRadius.xyz;
    if (halfSize.x <= 0.0 || halfSize.y <= 0.0 || halfSize.z <= 0.0) {
        return 0.0;
    }
    let local = quatInverseRotate(boxBody.orientation, spherePos - boxPos);
    let closestLocal = clamp(local, -halfSize, halfSize);
    let delta = spherePos - (boxPos + quatRotate(boxBody.orientation, closestLocal));
    let distSq = dot(delta, delta);
    if (distSq > 0.000001) {
        return max(sphereRadius - sqrt(distSq), 0.0);
    }
    let clearance = halfSize - abs(local);
    return sphereRadius + max(min(clearance.x, min(clearance.y, clearance.z)), 0.0);
}

fn sphereCapsulePenetration(spherePos: vec3f, sphereRadius: f32, capsuleBody: Body, capsulePos: vec3f) -> f32 {
    let capsuleRadius = capsuleBody.halfSizeRadius.w;
    let capsuleHalfLength = capsuleBody.extra.x;
    if (capsuleRadius <= 0.0 || capsuleHalfLength <= 0.0) {
        return 0.0;
    }
    let local = quatInverseRotate(capsuleBody.orientation, spherePos - capsulePos);
    let closestLocal = vec3f(0.0, 0.0, clamp(local.z, -capsuleHalfLength, capsuleHalfLength));
    let dist = length(local - closestLocal);
    return max(sphereRadius + capsuleRadius - dist, 0.0);
}

fn sphereCylinderPenetration(spherePos: vec3f, sphereRadius: f32, cylinderBody: Body, cylinderPos: vec3f) -> f32 {
    let cylinderRadius = cylinderBody.halfSizeRadius.w;
    let cylinderHalfLength = cylinderBody.extra.x;
    if (cylinderRadius <= 0.0 || cylinderHalfLength <= 0.0) {
        return 0.0;
    }
    let local = quatInverseRotate(cylinderBody.orientation, spherePos - cylinderPos);
    let radial = vec2f(local.x, local.y);
    let radialLen = length(radial);
    let clampedZ = clamp(local.z, -cylinderHalfLength, cylinderHalfLength);
    var closestRadial = radial;
    if (radialLen > cylinderRadius) {
        closestRadial = radial * (cylinderRadius / max(radialLen, 0.000001));
    }
    let closestLocal = vec3f(closestRadial.x, closestRadial.y, clampedZ);
    let dist = length(local - closestLocal);
    if (dist > 0.000001) {
        return max(sphereRadius - dist, 0.0);
    }
    let sideClearance = cylinderRadius - radialLen;
    let capClearance = cylinderHalfLength - abs(local.z);
    return sphereRadius + max(min(sideClearance, capClearance), 0.0);
}

fn sphereSpherePenetration(pa: vec3f, radiusA: f32, pb: vec3f, radiusB: f32) -> f32 {
    let dist = max(length(pb - pa), 0.000001);
    return max(radiusA + radiusB - dist, 0.0);
}

fn roundEndpointBody(body: Body, pos: vec3f, signValue: f32) -> vec3f {
    let halfLength = select(0.0, body.extra.x, body.shapeType == 2u || body.shapeType == 3u);
    if (halfLength <= 0.0) {
        return pos;
    }
    return pos + quatRotate(body.orientation, vec3f(0.0, 0.0, signValue * halfLength));
}

fn roundPairPenetration(bodyA: Body, pa: vec3f, bodyB: Body, pb: vec3f) -> f32 {
    let radiusA = bodyA.halfSizeRadius.w;
    let radiusB = bodyB.halfSizeRadius.w;
    if (radiusA <= 0.0 || radiusB <= 0.0) {
        return 0.0;
    }

    let p1 = roundEndpointBody(bodyA, pa, -1.0);
    let q1 = roundEndpointBody(bodyA, pa, 1.0);
    let p2 = roundEndpointBody(bodyB, pb, -1.0);
    let q2 = roundEndpointBody(bodyB, pb, 1.0);
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let aa = dot(d1, d1);
    let ee = dot(d2, d2);
    let ff = dot(d2, r);

    var s = 0.0;
    var t = 0.0;
    if (aa <= 0.000001 && ee <= 0.000001) {
        s = 0.0;
        t = 0.0;
    } else if (aa <= 0.000001) {
        s = 0.0;
        t = clamp(ff / max(ee, 0.000001), 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (ee <= 0.000001) {
            t = 0.0;
            s = clamp(-c / max(aa, 0.000001), 0.0, 1.0);
        } else {
            let bDot = dot(d1, d2);
            let denom = aa * ee - bDot * bDot;
            if (abs(denom) > 0.000001) {
                s = clamp((bDot * ff - c * ee) / denom, 0.0, 1.0);
            } else {
                s = 0.0;
            }
            t = (bDot * s + ff) / max(ee, 0.000001);
            if (t < 0.0) {
                t = 0.0;
                s = clamp(-c / max(aa, 0.000001), 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((bDot - c) / max(aa, 0.000001), 0.0, 1.0);
            }
        }
    }

    let closestA = p1 + d1 * s;
    let closestB = p2 + d2 * t;
    return max(radiusA + radiusB - length(closestA - closestB), 0.0);
}

fn pointBoxPenetrationBody(point: vec3f, radius: f32, boxBody: Body, boxPos: vec3f) -> f32 {
    let halfSize = boxBody.halfSizeRadius.xyz;
    if (radius <= 0.0 || halfSize.x <= 0.0 || halfSize.y <= 0.0 || halfSize.z <= 0.0) {
        return 0.0;
    }
    let local = quatInverseRotate(boxBody.orientation, point - boxPos);
    let closestLocal = clamp(local, -halfSize, halfSize);
    let delta = point - (boxPos + quatRotate(boxBody.orientation, closestLocal));
    let distSq = dot(delta, delta);
    if (distSq > 0.000001) {
        return max(radius - sqrt(distSq), 0.0);
    }
    let clearance = halfSize - abs(local);
    return radius + max(min(clearance.x, min(clearance.y, clearance.z)), 0.0);
}

fn roundBoxPenetration(roundBody: Body, roundPos: vec3f, boxBody: Body, boxPos: vec3f) -> f32 {
    let radius = roundBody.halfSizeRadius.w;
    let centerPen = pointBoxPenetrationBody(roundPos, radius, boxBody, boxPos);
    let halfLength = select(0.0, roundBody.extra.x, roundBody.shapeType == 2u || roundBody.shapeType == 3u);
    if (halfLength <= 0.0) {
        return centerPen;
    }
    let aPen = pointBoxPenetrationBody(roundEndpointBody(roundBody, roundPos, -1.0), radius, boxBody, boxPos);
    let bPen = pointBoxPenetrationBody(roundEndpointBody(roundBody, roundPos, 1.0), radius, boxBody, boxPos);
    return max(centerPen, max(aPen, bPen));
}

fn contactPenetration(contact: Contact, pa: vec4f, pb: vec4f) -> f32 {
    let bodyA = bodies[contact.bodyA];
    let bodyB = bodies[contact.bodyB];
    if (contact.pad0 == 2u) {
        return max(bodyB.extra.y + bodyA.halfSizeRadius.w - pa.z, 0.0);
    }
    if (contact.pad0 == 3u) {
        return sphereBoxPenetration(pa.xyz, bodyA.halfSizeRadius.w, bodyB, pb.xyz);
    }
    if (contact.pad0 == 4u) {
        return sphereCapsulePenetration(pa.xyz, bodyA.halfSizeRadius.w, bodyB, pb.xyz);
    }
    if (contact.pad0 == 5u) {
        return sphereCylinderPenetration(pa.xyz, bodyA.halfSizeRadius.w, bodyB, pb.xyz);
    }
    if (contact.pad0 == 6u) {
        return roundPairPenetration(bodyA, pa.xyz, bodyB, pb.xyz);
    }
    if (contact.pad0 == 7u) {
        return roundBoxPenetration(bodyA, pa.xyz, bodyB, pb.xyz);
    }
    return sphereSpherePenetration(pa.xyz, bodyA.halfSizeRadius.w, pb.xyz, bodyB.halfSizeRadius.w);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let contactIndex = id.x;
    if (contactIndex >= params.contactCount) {
        return;
    }
    let contact = contacts[contactIndex];
    let bodyA = bodies[contact.bodyA];
    let bodyB = bodies[contact.bodyB];
    let proposedA = proposals[contact.bodyA];
    let proposedB = proposals[contact.bodyB];
    residuals[contactIndex].beforePenetration = contactPenetration(contact, bodyA.position, bodyB.position);
    residuals[contactIndex].afterPenetration = contactPenetration(contact, proposedA.position, proposedB.position);
    residuals[contactIndex].pad0 = 0.0;
    residuals[contactIndex].pad1 = 0.0;
}
)";

            if (sphereContactProposalResidualPipeline == nullptr || sphereContactProposalResidualBindGroupLayout == nullptr)
            {
                wgpu::ShaderSourceWGSL wgsl = {};
                wgsl.code = residualShaderSource;
                wgpu::ShaderModuleDescriptor shaderDesc = {};
                shaderDesc.nextInChain = &wgsl;
                wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
                if (shader == nullptr)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact residual shader");
                    return false;
                }

                wgpu::BindGroupLayoutEntry entries[5] = {};
                entries[0].binding = 0;
                entries[0].visibility = wgpu::ShaderStage::Compute;
                entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[0].buffer.minBindingSize = sizeof(GpuSphereContact);
                entries[1].binding = 1;
                entries[1].visibility = wgpu::ShaderStage::Compute;
                entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[1].buffer.minBindingSize = sizeof(GpuContactBody);
                entries[2].binding = 2;
                entries[2].visibility = wgpu::ShaderStage::Compute;
                entries[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                entries[2].buffer.minBindingSize = sizeof(GpuContactProposalOutput);
                entries[3].binding = 3;
                entries[3].visibility = wgpu::ShaderStage::Compute;
                entries[3].buffer.type = wgpu::BufferBindingType::Uniform;
                entries[3].buffer.minBindingSize = sizeof(GpuContactAdjacencyParams);
                entries[4].binding = 4;
                entries[4].visibility = wgpu::ShaderStage::Compute;
                entries[4].buffer.type = wgpu::BufferBindingType::Storage;
                entries[4].buffer.minBindingSize = sizeof(GpuContactProposalResidual);
                wgpu::BindGroupLayoutDescriptor layoutDesc = {};
                layoutDesc.entryCount = 5;
                layoutDesc.entries = entries;
                sphereContactProposalResidualBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
                wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
                pipelineLayoutDesc.bindGroupLayoutCount = 1;
                pipelineLayoutDesc.bindGroupLayouts = &sphereContactProposalResidualBindGroupLayout;
                wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
                wgpu::ComputePipelineDescriptor pipelineDesc = {};
                pipelineDesc.layout = pipelineLayout;
                pipelineDesc.compute.module = shader;
                pipelineDesc.compute.entryPoint = "main";
                sphereContactProposalResidualPipeline = device.CreateComputePipeline(&pipelineDesc);
                if (sphereContactProposalResidualPipeline == nullptr)
                {
                    setStatus(sapStatus, "SAP resident collision failed: contact residual pipeline");
                    return false;
                }
            }

            wgpu::BindGroupEntry residualEntries[5] = {};
            residualEntries[0].binding = 0;
            residualEntries[0].buffer = sphereContactOutputBuffer;
            residualEntries[0].size = (uint64_t)sphereContactOutputCount * sizeof(GpuSphereContact);
            residualEntries[1].binding = 1;
            residualEntries[1].buffer = sphereContactBodyBuffer;
            residualEntries[1].size = contactBodyBytes;
            residualEntries[2].binding = 2;
            residualEntries[2].buffer = sphereContactProposalOutputBuffer;
            residualEntries[2].size = proposalOutputBytes;
            residualEntries[3].binding = 3;
            residualEntries[3].buffer = sphereContactAdjacencyParamsBuffer;
            residualEntries[3].size = sizeof(GpuContactAdjacencyParams);
            residualEntries[4].binding = 4;
            residualEntries[4].buffer = sphereContactProposalResidualBuffer;
            residualEntries[4].size = residualBytes;
            wgpu::BindGroupDescriptor residualBindGroupDesc = {};
            residualBindGroupDesc.layout = sphereContactProposalResidualBindGroupLayout;
            residualBindGroupDesc.entryCount = 5;
            residualBindGroupDesc.entries = residualEntries;
            wgpu::BindGroup residualBindGroup = device.CreateBindGroup(&residualBindGroupDesc);
            if (residualBindGroup == nullptr)
            {
                setStatus(sapStatus, "SAP resident collision failed: contact residual bind group");
                return false;
            }

            wgpu::CommandEncoder residualEncoder = device.CreateCommandEncoder();
            wgpu::ComputePassEncoder residualPass = residualEncoder.BeginComputePass();
            residualPass.SetPipeline(sphereContactProposalResidualPipeline);
            residualPass.SetBindGroup(0, residualBindGroup);
            residualPass.DispatchWorkgroups((sphereContactOutputCount + 63u) / 64u);
            residualPass.End();
            residualEncoder.CopyBufferToBuffer(sphereContactProposalResidualBuffer, 0,
                                               sphereContactProposalResidualReadbackBuffer, 0, residualBytes);
            wgpu::CommandBuffer residualCommands = residualEncoder.Finish();
            queue.Submit(1, &residualCommands);

            mapDone = false;
            mapStatus = wgpu::MapAsyncStatus::Error;
            wgpu::Future residualMapFuture = sphereContactProposalResidualReadbackBuffer.MapAsync(
                wgpu::MapMode::Read, 0, residualBytes, wgpu::CallbackMode::WaitAnyOnly,
                [&](wgpu::MapAsyncStatus status, wgpu::StringView)
                {
                    mapDone = true;
                    mapStatus = status;
                });
            if (instance.WaitAny(residualMapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
                !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
            {
                setStatus(sapStatus, "SAP resident collision failed: contact residual readback map");
                return false;
            }
            const GpuContactProposalResidual *residualOutputs =
                (const GpuContactProposalResidual *)sphereContactProposalResidualReadbackBuffer.GetConstMappedRange(0, residualBytes);
            if (!residualOutputs)
            {
                sphereContactProposalResidualReadbackBuffer.Unmap();
                setStatus(sapStatus, "SAP resident collision failed: contact residual readback range");
                return false;
            }

            double beforeMax = 0.0;
            double afterMax = 0.0;
            double beforeChecksum = 0.0;
            double afterChecksum = 0.0;
            for (uint32_t i = 0; i < sphereContactOutputCount; ++i)
            {
                double before = (double)residualOutputs[i].beforePenetration;
                double after = (double)residualOutputs[i].afterPenetration;
                beforeChecksum += before;
                afterChecksum += after;
                if (before > beforeMax)
                    beforeMax = before;
                if (after > afterMax)
                    afterMax = after;
            }
            sphereContactProposalResidualReadbackBuffer.Unmap();

            sphereContactProposalResidualReadbackBytes = (int)residualBytes;
            sphereContactProposalResidualBeforeMax = (float)beforeMax;
            sphereContactProposalResidualAfterMax = (float)afterMax;
            sphereContactProposalResidualBeforeChecksum = (float)beforeChecksum;
            sphereContactProposalResidualAfterChecksum = (float)afterChecksum;
            sphereContactProposalResidualMs = elapsedMs(residualBegin, SDL_GetPerformanceCounter());

            Uint64 iterationResidualBegin = SDL_GetPerformanceCounter();
            residualEntries[2].buffer = iterationInput;
            residualEntries[2].size = proposalOutputBytes;
            wgpu::BindGroupDescriptor iterationResidualBindGroupDesc = {};
            iterationResidualBindGroupDesc.layout = sphereContactProposalResidualBindGroupLayout;
            iterationResidualBindGroupDesc.entryCount = 5;
            iterationResidualBindGroupDesc.entries = residualEntries;
            wgpu::BindGroup iterationResidualBindGroup = device.CreateBindGroup(&iterationResidualBindGroupDesc);
            if (iterationResidualBindGroup == nullptr)
            {
                setStatus(sapStatus, "SAP resident collision failed: contact iteration residual bind group");
                return false;
            }

            wgpu::CommandEncoder iterationResidualEncoder = device.CreateCommandEncoder();
            wgpu::ComputePassEncoder iterationResidualPass = iterationResidualEncoder.BeginComputePass();
            iterationResidualPass.SetPipeline(sphereContactProposalResidualPipeline);
            iterationResidualPass.SetBindGroup(0, iterationResidualBindGroup);
            iterationResidualPass.DispatchWorkgroups((sphereContactOutputCount + 63u) / 64u);
            iterationResidualPass.End();
            iterationResidualEncoder.CopyBufferToBuffer(sphereContactProposalResidualBuffer, 0,
                                                        sphereContactProposalResidualReadbackBuffer, 0, residualBytes);
            wgpu::CommandBuffer iterationResidualCommands = iterationResidualEncoder.Finish();
            queue.Submit(1, &iterationResidualCommands);

            mapDone = false;
            mapStatus = wgpu::MapAsyncStatus::Error;
            wgpu::Future iterationResidualMapFuture = sphereContactProposalResidualReadbackBuffer.MapAsync(
                wgpu::MapMode::Read, 0, residualBytes, wgpu::CallbackMode::WaitAnyOnly,
                [&](wgpu::MapAsyncStatus status, wgpu::StringView)
                {
                    mapDone = true;
                    mapStatus = status;
                });
            if (instance.WaitAny(iterationResidualMapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
                !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
            {
                setStatus(sapStatus, "SAP resident collision failed: contact iteration residual readback map");
                return false;
            }
            const GpuContactProposalResidual *iterationResidualOutputs =
                (const GpuContactProposalResidual *)sphereContactProposalResidualReadbackBuffer.GetConstMappedRange(0, residualBytes);
            if (!iterationResidualOutputs)
            {
                sphereContactProposalResidualReadbackBuffer.Unmap();
                setStatus(sapStatus, "SAP resident collision failed: contact iteration residual readback range");
                return false;
            }

            double iterationAfterMax = 0.0;
            double iterationAfterChecksum = 0.0;
            for (uint32_t i = 0; i < sphereContactOutputCount; ++i)
            {
                double after = (double)iterationResidualOutputs[i].afterPenetration;
                iterationAfterChecksum += after;
                if (after > iterationAfterMax)
                    iterationAfterMax = after;
            }
            sphereContactProposalResidualReadbackBuffer.Unmap();

            sphereContactIterationResidualAfterMax = (float)iterationAfterMax;
            sphereContactIterationResidualAfterChecksum = (float)iterationAfterChecksum;
            sphereContactIterationMs += elapsedMs(iterationResidualBegin, SDL_GetPerformanceCounter());
            }
        }

        sapPairReadbackBytes = 0;
        sapPairReadbackMs = 0.0f;
        sphereContactReadbackBytes = 0;
        sphereContactReadbackMs = 0.0f;
        sphereContactMs = sapMs;
        recordTimingSample(sapTiming, sapMs);
        const char *axisName = sapBestAxis == 0 ? "X" : (sapBestAxis == 1 ? "Y" : "Z");
        if (residentCounterlessContacts)
            snprintf(sapStatus, sizeof(sapStatus), "SAP resident counterless passed: %s axis, %d compact contact scan slots, no counter sync",
                     axisName, sphereContactCount);
        else
            snprintf(sapStatus, sizeof(sapStatus), "SAP resident collision passed: %s axis, %d candidates, %d hits, %d sphere contacts, %d active contact bodies",
                     axisName, sapCandidates, sapSphereHits, sphereContactCount, sphereContactActiveBodies);
        if (residentCounterlessContacts)
            return true;
        if (counterOnly)
            return true;
    }

    if (counterOnly)
    {
        sapPairReadbackBytes = 0;
        sapPairReadbackMs = 0.0f;
        sphereContactReadbackBytes = 0;
        sphereContactReadbackMs = 0.0f;
        sphereContactMs = 0.0f;
        recordTimingSample(sapTiming, sapMs);
        const char *axisName = sapBestAxis == 0 ? "X" : (sapBestAxis == 1 ? "Y" : "Z");
        snprintf(sapStatus, sizeof(sapStatus), "SAP counters passed: %s axis, %d candidates, %d hits",
                 axisName, sapCandidates, sapSphereHits);
        return true;
    }

    if (sphereHitCount == 0)
    {
        recordTimingSample(sapTiming, sapMs);
        snprintf(sapStatus, sizeof(sapStatus), "SAP pairs passed: %d candidates, %d hits", sapCandidates, sapSphereHits);
        return true;
    }

    if (pairBytes > 0 && (sapPairReadbackBuffer == nullptr || sapPairReadbackBufferBytes < pairBytes))
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(pairBytes, 4096);
        sapPairReadbackBuffer = device.CreateBuffer(&desc);
        sapPairReadbackBufferBytes = sapPairReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (pairBytes > 0 && sapPairReadbackBuffer == nullptr)
    {
        setStatus(sapStatus, "SAP pairs failed: pair readback buffer allocation");
        return false;
    }
    if (contactReadbackBytes > 0 && (sphereContactReadbackBuffer == nullptr || sphereContactReadbackBufferBytes < contactReadbackBytes))
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(contactReadbackBytes, 4096);
        sphereContactReadbackBuffer = device.CreateBuffer(&desc);
        sphereContactReadbackBufferBytes = sphereContactReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (contactReadbackBytes > 0 && sphereContactReadbackBuffer == nullptr)
    {
        setStatus(sapStatus, "SAP pairs failed: contact readback buffer allocation");
        return false;
    }

    Uint64 payloadReadbackBegin = SDL_GetPerformanceCounter();
    wgpu::CommandEncoder readbackEncoder = device.CreateCommandEncoder();
    if (pairBytes > 0 && !pairReadbackSubmittedEarly)
        readbackEncoder.CopyBufferToBuffer(sapPairOutputBuffer, 0, sapPairReadbackBuffer, 0, pairBytes);
    if (contactReadbackBytes > 0)
        readbackEncoder.CopyBufferToBuffer(sphereContactOutputBuffer, 0, sphereContactReadbackBuffer, 0, contactReadbackBytes);
    wgpu::CommandBuffer readbackCommands = readbackEncoder.Finish();
    queue.Submit(1, &readbackCommands);

    if (pairBytes > 0)
    {
        mapDone = false;
        mapStatus = wgpu::MapAsyncStatus::Error;
        wgpu::Future pairMapFuture = sapPairReadbackBuffer.MapAsync(
            wgpu::MapMode::Read, 0, pairBytes, wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::MapAsyncStatus status, wgpu::StringView)
            {
                mapDone = true;
                mapStatus = status;
            });
        if (instance.WaitAny(pairMapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
        {
            setStatus(sapStatus, "SAP pairs failed: pair readback map");
            return false;
        }
        const GpuSapPair *gpuPairs = (const GpuSapPair *)sapPairReadbackBuffer.GetConstMappedRange(0, pairBytes);
        if (!gpuPairs)
        {
            sapPairReadbackBuffer.Unmap();
            setStatus(sapStatus, "SAP pairs failed: pair readback range");
            return false;
        }

        pairs.reserve(nonSpherePairCount);
        for (uint32_t i = 0; i < nonSpherePairCount; ++i)
            pairs.push_back(BroadphasePair{gpuPairs[i].bodyA, gpuPairs[i].bodyB});
        sapPairReadbackBuffer.Unmap();
    }

    if (contactReadbackBytes > 0 && sphereContacts)
    {
        mapDone = false;
        mapStatus = wgpu::MapAsyncStatus::Error;
        wgpu::Future contactMapFuture = sphereContactReadbackBuffer.MapAsync(
            wgpu::MapMode::Read, 0, contactReadbackBytes, wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::MapAsyncStatus status, wgpu::StringView)
            {
                mapDone = true;
                mapStatus = status;
            });
        if (instance.WaitAny(contactMapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
            !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
        {
            setStatus(sapStatus, "SAP pairs failed: contact readback map");
            return false;
        }
        const GpuSphereContact *gpuContacts = (const GpuSphereContact *)sphereContactReadbackBuffer.GetConstMappedRange(0, contactReadbackBytes);
        if (!gpuContacts)
        {
            sphereContactReadbackBuffer.Unmap();
            setStatus(sapStatus, "SAP pairs failed: contact readback range");
            return false;
        }

        sphereContacts->reserve((size_t)sphereContactOutputCount * 2u);
        auto shapeRadius = [](const SimBodyData &body) -> float
        {
            return body.shape.radius > 0.0f ? body.shape.radius : body.radius;
        };
        auto appendExternalContact = [&](BodyId bodyA, BodyId bodyB, float3 normalAB, float3 xA, float3 xB, int featureKey)
        {
            if (bodyA >= world.bodies.size() || bodyB >= world.bodies.size())
                return;
            const SimBodyData &a = world.bodies[bodyA];
            const SimBodyData &b = world.bodies[bodyB];
            if (!a.active || !b.active)
                return;
            if (lengthSq(normalAB) < 1.0e-8f)
                normalAB = normalize(b.positionLin - a.positionLin);
            if (lengthSq(normalAB) < 1.0e-8f)
                normalAB = {1.0f, 0.0f, 0.0f};

            ExternalManifoldContact contact = {};
            contact.bodyA = bodyA;
            contact.bodyB = bodyB;
            contact.basis = orthonormal(-normalAB);
            contact.numContacts = 1;
            contact.contacts[0].feature.key = featureKey;
            contact.contacts[0].rA = rotate(conjugate(a.positionAng), xA - a.positionLin);
            contact.contacts[0].rB = rotate(conjugate(b.positionAng), xB - b.positionLin);
            sphereContacts->push_back(contact);
            sphereContactExternalContacts++;

            ExternalManifoldContact swapped = {};
            swapped.bodyA = bodyB;
            swapped.bodyB = bodyA;
            swapped.basis = orthonormal(normalAB);
            swapped.numContacts = 1;
            swapped.contacts[0].feature.key = featureKey;
            swapped.contacts[0].rA = contact.contacts[0].rB;
            swapped.contacts[0].rB = contact.contacts[0].rA;
            sphereContacts->push_back(swapped);
            sphereContactExternalContacts++;
        };
    for (uint32_t i = 0; i < sphereContactOutputCount; ++i)
    {
        const GpuSphereContact &gpuContact = gpuContacts[i];
        float3 normalAB = {gpuContact.normal[0], gpuContact.normal[1], gpuContact.normal[2]};

        if (gpuContact.pad0 == 2u)
        {
            if (gpuContact.bodyA >= world.bodies.size() || gpuContact.bodyB >= world.bodies.size())
                continue;
            const SimBodyData &sphereBody = world.bodies[gpuContact.bodyA];
            const SimBodyData &groundBody = world.bodies[gpuContact.bodyB];
            if (sphereBody.shape.type != RIGID_SHAPE_SPHERE || !isGpuStaticGroundReceiver(groundBody))
                continue;

            float3 normal = rotate(groundBody.positionAng, float3{0.0f, 0.0f, 1.0f});
            if (lengthSq(normal) < 1.0e-8f)
                normal = {0.0f, 0.0f, 1.0f};
            normal = normalize(normal);

            float3 local = rotate(conjugate(groundBody.positionAng), sphereBody.positionLin - groundBody.positionLin);
            local.x = clamp(local.x, -groundBody.shape.size.x * 0.5f, groundBody.shape.size.x * 0.5f);
            local.y = clamp(local.y, -groundBody.shape.size.y * 0.5f, groundBody.shape.size.y * 0.5f);
            local.z = groundBody.shape.size.z * 0.5f;

            ExternalManifoldContact contact = {};
            if (gpuContact.bodyB < gpuContact.bodyA)
            {
                contact.bodyA = gpuContact.bodyB;
                contact.bodyB = gpuContact.bodyA;
                contact.basis = orthonormal(-normal);
                contact.contacts[0].rA = local;
                contact.contacts[0].rB = {0.0f, 0.0f, 0.0f};
            }
            else
            {
                contact.bodyA = gpuContact.bodyA;
                contact.bodyB = gpuContact.bodyB;
                contact.basis = orthonormal(normal);
                contact.contacts[0].rA = {0.0f, 0.0f, 0.0f};
                contact.contacts[0].rB = local;
            }
            contact.numContacts = 1;
            contact.contacts[0].feature.key = 0x02000000;
            sphereContacts->push_back(contact);
            sphereContactExternalContacts++;
            sphereContactExternalGroundContacts++;
            continue;
        }

        if (gpuContact.bodyA >= world.bodies.size() || gpuContact.bodyB >= world.bodies.size())
            continue;
        const SimBodyData &bodyA = world.bodies[gpuContact.bodyA];
        const SimBodyData &bodyB = world.bodies[gpuContact.bodyB];

        if (gpuContact.pad0 == 3u || gpuContact.pad0 == 4u || gpuContact.pad0 == 5u)
        {
            if (bodyA.shape.type != RIGID_SHAPE_SPHERE)
                continue;
            float sphereRadius = shapeRadius(bodyA);
            if (sphereRadius <= 0.0f)
                continue;
            if (lengthSq(normalAB) < 1.0e-8f)
                normalAB = normalize(bodyB.positionLin - bodyA.positionLin);
            if (lengthSq(normalAB) < 1.0e-8f)
                normalAB = {1.0f, 0.0f, 0.0f};
            float3 xA = bodyA.positionLin + normalAB * sphereRadius;
            float3 xB = bodyB.positionLin;
            int featureKey = 0x03000000;
            if (gpuContact.pad0 == 3u && bodyB.shape.type == RIGID_SHAPE_BOX)
            {
                float3 half = bodyB.shape.size * 0.5f;
                float3 local = rotate(conjugate(bodyB.positionAng), bodyA.positionLin - bodyB.positionLin);
                local.x = clamp(local.x, -half.x, half.x);
                local.y = clamp(local.y, -half.y, half.y);
                local.z = clamp(local.z, -half.z, half.z);
                xB = bodyB.positionLin + rotate(bodyB.positionAng, local);
                featureKey = 0x02000000;
            }
            else if (gpuContact.pad0 == 4u && bodyB.shape.type == RIGID_SHAPE_CAPSULE)
            {
                float3 local = rotate(conjugate(bodyB.positionAng), bodyA.positionLin - bodyB.positionLin);
                float3 centerLocal = {0.0f, 0.0f, clamp(local.z, -bodyB.shape.halfLength, bodyB.shape.halfLength)};
                float3 center = bodyB.positionLin + rotate(bodyB.positionAng, centerLocal);
                xB = center - normalAB * shapeRadius(bodyB);
                featureKey = 0x03000000;
            }
            else if (gpuContact.pad0 == 5u && bodyB.shape.type == RIGID_SHAPE_CYLINDER)
            {
                float3 local = rotate(conjugate(bodyB.positionAng), bodyA.positionLin - bodyB.positionLin);
                float radialLen = sqrtf(local.x * local.x + local.y * local.y);
                float3 closestLocal = {0.0f, 0.0f, clamp(local.z, -bodyB.shape.halfLength, bodyB.shape.halfLength)};
                if (radialLen > 1.0e-6f)
                {
                    closestLocal.x = local.x / radialLen * bodyB.shape.radius;
                    closestLocal.y = local.y / radialLen * bodyB.shape.radius;
                }
                else
                {
                    float3 localNormal = rotate(conjugate(bodyB.positionAng), -normalAB);
                    float localRadialLen = sqrtf(localNormal.x * localNormal.x + localNormal.y * localNormal.y);
                    if (localRadialLen > 1.0e-6f)
                    {
                        closestLocal.x = localNormal.x / localRadialLen * bodyB.shape.radius;
                        closestLocal.y = localNormal.y / localRadialLen * bodyB.shape.radius;
                    }
                }
                xB = bodyB.positionLin + rotate(bodyB.positionAng, closestLocal);
                featureKey = 0x07000000;
            }
            else
            {
                continue;
            }
            appendExternalContact(gpuContact.bodyA, gpuContact.bodyB, normalAB, xA, xB, featureKey);
            continue;
        }

        appendExternalContact(gpuContact.bodyA, gpuContact.bodyB, normalAB,
                              bodyA.positionLin + normalAB * shapeRadius(bodyA),
                              bodyB.positionLin - normalAB * shapeRadius(bodyB),
                              0x03000000);
        }
        sphereContactReadbackBuffer.Unmap();
    }

    sapPairReadbackBytes = (int)pairBytes;
    sapPairReadbackMs = pairBytes > 0 ? elapsedMs(payloadReadbackBegin, SDL_GetPerformanceCounter()) : 0.0f;
    sphereContactReadbackBytes = (int)contactReadbackBytes;
    sphereContactReadbackMs = contactReadbackBytes > 0 ? elapsedMs(payloadReadbackBegin, SDL_GetPerformanceCounter()) : 0.0f;
    sphereContactMs = contactReadbackBytes > 0 ? sphereContactReadbackMs : 0.0f;
    recordTimingSample(sapTiming, sapMs);

    const char *axisName = sapBestAxis == 0 ? "X" : (sapBestAxis == 1 ? "Y" : "Z");
    snprintf(sapStatus, sizeof(sapStatus), "SAP pairs passed: %s axis, %d candidates, %d hits, readback %.1f KB",
             axisName, sapCandidates, sapSphereHits, (double)sapPairReadbackBytes / 1024.0);
    return true;
#elif AVBD_ENABLE_WEBGPU
    setStatus(sapStatus, "SAP pairs skipped: Dawn not linked");
    return false;
#else
    setStatus(sapStatus, "SAP pairs skipped: WebGPU disabled");
    return false;
#endif
}

bool WebGpuContext::runSphereGroundCorrection(const SimWorld &world, bool includeSpheres)
{
    sphereGroundReceiverCount = 0;
    sphereGroundDynamicSphereCount = 0;
    sphereGroundCandidateCount = 0;
    sphereGroundTop = 0.0f;
    sphereGroundMs = 0.0f;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
        return false;
    if (world.bodies.empty())
        return false;

    float groundTop = 0.0f;
    int receiverCount = 0;
    if (!findGpuStaticGroundTop(world, groundTop, &receiverCount))
        return false;
    sphereGroundReceiverCount = receiverCount;
    sphereGroundTop = groundTop;

    for (BodyId bodyId = 0; bodyId < world.bodies.size(); ++bodyId)
    {
        const SimBodyData &body = world.bodies[bodyId];
        if (isGpuDynamicGroundCorrectionBody(body, includeSpheres))
        {
            sphereGroundDynamicSphereCount++;
            float extentZ = gpuGroundContactExtentZ(body);
            if (body.positionLin.z < groundTop + extentZ)
                sphereGroundCandidateCount++;
        }
    }
    if (sphereGroundDynamicSphereCount == 0 || sphereGroundCandidateCount == 0)
    {
        if (includeSpheres)
        {
            sphereContactFinalPositionReady = 0;
            sphereContactFinalPositionBodyCount = 0;
            sphereContactFinalPositionBytes = 0;
            sphereContactFinalPositionSource = 0;
            sphereContactFinalPositionBuffer = nullptr;
            sphereContactFinalPositionBufferBytes = 0;
        }
        return true;
    }

    Uint64 begin = SDL_GetPerformanceCounter();
    const uint32_t bodyCount = (uint32_t)world.bodies.size();
    const uint64_t contactBodyBytes = (uint64_t)bodyCount * sizeof(GpuContactBody);
    const uint64_t outputBytes = (uint64_t)bodyCount * sizeof(GpuContactProposalOutput);

    if (sphereContactBodyBuffer == nullptr || sphereContactBodyBufferBytes < contactBodyBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(contactBodyBytes, 4096);
        sphereContactBodyBuffer = device.CreateBuffer(&desc);
        sphereContactBodyBufferBytes = sphereContactBodyBuffer == nullptr ? 0 : desc.size;
    }
    if (sphereContactProposalOutputBuffer == nullptr || sphereContactProposalOutputBufferBytes < outputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                 (uint64_t)wgpu::BufferUsage::CopyDst |
                                 (uint64_t)wgpu::BufferUsage::CopySrc);
        desc.size = alignUp(outputBytes, 4096);
        sphereContactProposalOutputBuffer = device.CreateBuffer(&desc);
        sphereContactProposalOutputBufferBytes = sphereContactProposalOutputBuffer == nullptr ? 0 : desc.size;
    }
    if (sphereContactProposalOutputReadbackBuffer == nullptr ||
        sphereContactProposalOutputReadbackBufferBytes < outputBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(outputBytes, 4096);
        sphereContactProposalOutputReadbackBuffer = device.CreateBuffer(&desc);
        sphereContactProposalOutputReadbackBufferBytes = sphereContactProposalOutputReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (sphereGroundParamsBuffer == nullptr)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = sizeof(GpuSphereGroundParams);
        sphereGroundParamsBuffer = device.CreateBuffer(&desc);
    }
    if (sphereContactBodyBuffer == nullptr ||
        sphereContactProposalOutputBuffer == nullptr ||
        sphereContactProposalOutputReadbackBuffer == nullptr ||
        sphereGroundParamsBuffer == nullptr)
    {
        setStatus(sapStatus, "Sphere ground correction failed: buffer allocation");
        return false;
    }

    std::vector<GpuContactBody> bodies(world.bodies.size());
    sphereContactFinalReferencePositions.resize(world.bodies.size());
    for (BodyId bodyId = 0; bodyId < world.bodies.size(); ++bodyId)
    {
        const SimBodyData &body = world.bodies[bodyId];
        GpuContactBody gpuBody = {};
        gpuBody.position[0] = body.positionLin.x;
        gpuBody.position[1] = body.positionLin.y;
        gpuBody.position[2] = body.positionLin.z;
        gpuBody.position[3] = gpuGroundContactExtentZ(body);
        gpuBody.shapeType = (uint32_t)body.shape.type;
        bodies[bodyId] = gpuBody;
        sphereContactFinalReferencePositions[bodyId] = body.positionLin;
    }
    queue.WriteBuffer(sphereContactBodyBuffer, 0, bodies.data(), contactBodyBytes);

    GpuSphereGroundParams params = {};
    params.bodyCount = bodyCount;
    params.ground[0] = groundTop;
    params.ground[1] = 1.0f;
    params.ground[2] = includeSpheres ? 1.0f : 0.0f;
    queue.WriteBuffer(sphereGroundParamsBuffer, 0, &params, sizeof(params));

    static const char *groundShaderSource = R"(
struct Body {
    position: vec4f,
    orientation: vec4f,
    halfSizeRadius: vec4f,
    extra: vec4f,
    shapeType: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Params {
    bodyCount: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    ground: vec4f,
};

struct Output {
    position: vec4f,
    correction: vec4f,
};

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> outputs: array<Output>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let bodyIndex = id.x;
    if (bodyIndex >= params.bodyCount) {
        return;
    }

    let body = bodies[bodyIndex];
    var correction = vec3f(0.0, 0.0, 0.0);
    let includeSpheres = params.ground.z > 0.5;
    if ((body.shapeType == 0u || body.shapeType == 1u || body.shapeType == 2u || body.shapeType == 3u) &&
        params.ground.y > 0.5 &&
        (includeSpheres || body.shapeType != 1u)) {
        let minZ = params.ground.x + body.position.w;
        let penetration = max(minZ - body.position.z, 0.0);
        correction.z = min(penetration, 0.25);
    }
    outputs[bodyIndex].position = vec4f(body.position.xyz + correction, body.position.w);
    outputs[bodyIndex].correction = vec4f(correction, 0.0);
}
)";

    if (sphereGroundPipeline == nullptr || sphereGroundBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = groundShaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
        {
            setStatus(sapStatus, "Sphere ground correction failed: shader");
            return false;
        }

        wgpu::BindGroupLayoutEntry entries[3] = {};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        entries[0].buffer.minBindingSize = sizeof(GpuContactBody);
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[1].buffer.minBindingSize = sizeof(GpuSphereGroundParams);
        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Compute;
        entries[2].buffer.type = wgpu::BufferBindingType::Storage;
        entries[2].buffer.minBindingSize = sizeof(GpuContactProposalOutput);
        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 3;
        layoutDesc.entries = entries;
        sphereGroundBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &sphereGroundBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        sphereGroundPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (sphereGroundPipeline == nullptr)
        {
            setStatus(sapStatus, "Sphere ground correction failed: pipeline");
            return false;
        }
    }

    wgpu::BindGroupEntry entries[3] = {};
    entries[0].binding = 0;
    entries[0].buffer = sphereContactBodyBuffer;
    entries[0].size = contactBodyBytes;
    entries[1].binding = 1;
    entries[1].buffer = sphereGroundParamsBuffer;
    entries[1].size = sizeof(GpuSphereGroundParams);
    entries[2].binding = 2;
    entries[2].buffer = sphereContactProposalOutputBuffer;
    entries[2].size = outputBytes;
    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = sphereGroundBindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
    if (bindGroup == nullptr)
    {
        setStatus(sapStatus, "Sphere ground correction failed: bind group");
        return false;
    }

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(sphereGroundPipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups((bodyCount + 63u) / 64u);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    sphereContactFinalPositionReady = 1;
    sphereContactFinalPositionBodyCount = (int)bodyCount;
    sphereContactFinalPositionBytes = (int)outputBytes;
    sphereContactFinalPositionSource = 4;
    sphereContactFinalPositionBuffer = sphereContactProposalOutputBuffer;
    sphereContactFinalPositionBufferBytes = outputBytes;
    sphereContactProposalOutputMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    sphereGroundMs = sphereContactProposalOutputMs;
    return true;
#else
    (void)world;
    return false;
#endif
}

bool WebGpuContext::runSphereSphereContacts(const SimWorld &world, const std::vector<BroadphasePair> &pairs, std::vector<ExternalManifoldContact> &contacts)
{
    contacts.clear();
    sphereContactCount = 0;
    sphereContactReadbackBytes = 0;
    sphereContactMs = 0.0f;
    sphereContactReadbackMs = 0.0f;
    sphereContactBodyRefs = 0;
    sphereContactActiveBodies = 0;
    sphereContactMaxPerBody = 0;
    sphereContactAvgPerActiveBody = 0.0f;
    sphereContactAdjacencyReadbackBytes = 0;
    sphereContactAdjacencyListBytes = 0;
    sphereContactAdjacencyCapacity = 0;
    sphereContactAdjacencyWrittenRefs = 0;
    sphereContactAdjacencyOverflowRefs = 0;
    sphereContactAdjacencyMs = 0.0f;
    sphereContactGatherRefs = 0;
    sphereContactGatherActiveBodies = 0;
    sphereContactGatherMaxPerBody = 0;
    sphereContactGatherMismatches = 0;
    sphereContactGatherReadbackBytes = 0;
    sphereContactGatherNormalChecksum = 0.0f;
    sphereContactProposalActiveBodies = 0;
    sphereContactProposalMaxCorrection = 0.0f;
    sphereContactProposalCorrectionChecksum = 0.0f;
    sphereContactGatherMs = 0.0f;
    sphereContactProposalOutputActiveBodies = 0;
    sphereContactProposalOutputReadbackBytes = 0;
    sphereContactProposalOutputMaxDelta = 0.0f;
    sphereContactProposalOutputChecksum = 0.0f;
    sphereContactProposalOutputMs = 0.0f;
    sphereContactProposalResidualReadbackBytes = 0;
    sphereContactProposalResidualBeforeMax = 0.0f;
    sphereContactProposalResidualAfterMax = 0.0f;
    sphereContactProposalResidualBeforeChecksum = 0.0f;
    sphereContactProposalResidualAfterChecksum = 0.0f;
    sphereContactProposalResidualMs = 0.0f;
    sphereContactIterationCount = 0;
    sphereContactIterationMs = 0.0f;
    sphereContactIterationRelaxation = 0.10f;
    sphereContactIterationResidualAfterMax = 0.0f;
    sphereContactIterationResidualAfterChecksum = 0.0f;
    sphereContactFinalPositionReady = 0;
    sphereContactFinalPositionBodyCount = 0;
    sphereContactFinalPositionBytes = 0;
    sphereContactFinalPositionSource = 0;
    sphereContactFinalReferencePositions.clear();
    sphereContactAppliedPositionBodies = 0;
    sphereContactAppliedPositionReadbackBytes = 0;
    sphereContactAppliedPositionMaxDelta = 0.0f;
    sphereContactAppliedPositionChecksum = 0.0f;
    sphereContactAppliedPositionMs = 0.0f;
    sphereContactAppliedPositionWaitMs = 0.0f;
    sphereContactAppliedPositionCpuMs = 0.0f;
    sphereContactFinalPositionReadbackDeferred = 0;
    sphereContactFinalPositionAsyncReadbackScheduled = 0;
    sphereContactFinalPositionAsyncReadbackConsumed = 0;
    sphereContactFinalPositionAsyncReadbackDropped = 0;
    sphereContactFinalPositionAsyncReadbackWaitMs = 0.0f;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    sphereContactFinalPositionBuffer = nullptr;
    sphereContactFinalPositionBufferBytes = 0;
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
        return false;
    if (pairs.empty() || world.bodies.empty())
        return true;

    Uint64 begin = SDL_GetPerformanceCounter();

    std::vector<GpuContactBody> bodies(world.bodies.size());
    for (BodyId bodyId = 0; bodyId < world.bodies.size(); ++bodyId)
    {
        const SimBodyData &body = world.bodies[bodyId];
        GpuContactBody &gpuBody = bodies[bodyId];
        gpuBody.position[0] = body.positionLin.x;
        gpuBody.position[1] = body.positionLin.y;
        gpuBody.position[2] = body.positionLin.z;
        gpuBody.position[3] = body.shape.radius;
        gpuBody.shapeType = (uint32_t)body.shape.type;
    }

    std::vector<GpuSapPair> gpuPairs;
    gpuPairs.reserve(pairs.size());
    for (const BroadphasePair &pair : pairs)
    {
        if (pair.bodyA < world.bodies.size() && pair.bodyB < world.bodies.size())
            gpuPairs.push_back(GpuSapPair{pair.bodyA, pair.bodyB});
    }
    if (gpuPairs.empty())
        return true;

    const uint64_t bodyBytes = (uint64_t)bodies.size() * sizeof(GpuContactBody);
    const uint64_t pairBytes = (uint64_t)gpuPairs.size() * sizeof(GpuSapPair);
    const uint64_t counterBytes = sizeof(GpuPairCounters);
    const uint64_t contactCapacityBytes = (uint64_t)gpuPairs.size() * sizeof(GpuSphereContact);

    if (sphereContactBodyBuffer == nullptr || sphereContactBodyBufferBytes < bodyBytes)
    {
        wgpu::BufferDescriptor bodyDesc = {};
        bodyDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                     (uint64_t)wgpu::BufferUsage::CopyDst);
        bodyDesc.size = alignUp(bodyBytes, 4096);
        sphereContactBodyBuffer = device.CreateBuffer(&bodyDesc);
        sphereContactBodyBufferBytes = sphereContactBodyBuffer == nullptr ? 0 : bodyDesc.size;
    }
    if (sphereContactBodyBuffer == nullptr)
        return false;
    queue.WriteBuffer(sphereContactBodyBuffer, 0, bodies.data(), bodyBytes);

    if (sphereContactPairBuffer == nullptr || sphereContactPairBufferBytes < pairBytes)
    {
        wgpu::BufferDescriptor pairDesc = {};
        pairDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                     (uint64_t)wgpu::BufferUsage::CopyDst);
        pairDesc.size = alignUp(pairBytes, 4096);
        sphereContactPairBuffer = device.CreateBuffer(&pairDesc);
        sphereContactPairBufferBytes = sphereContactPairBuffer == nullptr ? 0 : pairDesc.size;
    }
    if (sphereContactPairBuffer == nullptr)
        return false;
    queue.WriteBuffer(sphereContactPairBuffer, 0, gpuPairs.data(), pairBytes);

    GpuPairParams params = {};
    params.count = (uint32_t)gpuPairs.size();
    if (sphereContactParamsBuffer == nullptr)
    {
        wgpu::BufferDescriptor paramsDesc = {};
        paramsDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Uniform |
                                       (uint64_t)wgpu::BufferUsage::CopyDst);
        paramsDesc.size = sizeof(GpuPairParams);
        sphereContactParamsBuffer = device.CreateBuffer(&paramsDesc);
    }
    if (sphereContactParamsBuffer == nullptr)
        return false;
    queue.WriteBuffer(sphereContactParamsBuffer, 0, &params, sizeof(params));

    GpuPairCounters resetCounters = {};
    if (sphereContactCountersBuffer == nullptr)
    {
        wgpu::BufferDescriptor countersDesc = {};
        countersDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                         (uint64_t)wgpu::BufferUsage::CopySrc |
                                         (uint64_t)wgpu::BufferUsage::CopyDst);
        countersDesc.size = counterBytes;
        sphereContactCountersBuffer = device.CreateBuffer(&countersDesc);
        sphereContactCountersBufferBytes = sphereContactCountersBuffer == nullptr ? 0 : counterBytes;
    }
    if (sphereContactCountersBuffer == nullptr)
        return false;
    queue.WriteBuffer(sphereContactCountersBuffer, 0, &resetCounters, sizeof(resetCounters));

    if (sphereContactOutputBuffer == nullptr || sphereContactOutputBufferBytes < contactCapacityBytes)
    {
        wgpu::BufferDescriptor outputDesc = {};
        outputDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::Storage |
                                       (uint64_t)wgpu::BufferUsage::CopySrc);
        outputDesc.size = alignUp(contactCapacityBytes, 4096);
        sphereContactOutputBuffer = device.CreateBuffer(&outputDesc);
        sphereContactOutputBufferBytes = sphereContactOutputBuffer == nullptr ? 0 : outputDesc.size;
    }
    if (sphereContactOutputBuffer == nullptr)
        return false;

    if (sphereContactCounterReadbackBuffer == nullptr)
    {
        wgpu::BufferDescriptor counterReadbackDesc = {};
        counterReadbackDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                                (uint64_t)wgpu::BufferUsage::CopyDst);
        counterReadbackDesc.size = counterBytes;
        sphereContactCounterReadbackBuffer = device.CreateBuffer(&counterReadbackDesc);
        sphereContactCounterReadbackBufferBytes = sphereContactCounterReadbackBuffer == nullptr ? 0 : counterBytes;
    }
    if (sphereContactCounterReadbackBuffer == nullptr)
        return false;

    static const char *shaderSource = R"(
struct Body {
    position: vec4f,
    orientation: vec4f,
    halfSizeRadius: vec4f,
    extra: vec4f,
    shapeType: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Pair {
    bodyA: u32,
    bodyB: u32,
};

struct Counters {
    candidates: atomic<u32>,
    sphereHits: atomic<u32>,
    pad0: u32,
    pad1: u32,
};

struct Contact {
    bodyA: u32,
    bodyB: u32,
    pad0: u32,
    pad1: u32,
    normal: vec4f,
};

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> counters: Counters;
@group(0) @binding(3) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= params.count) {
        return;
    }

    let pair = pairs[index];
    let bodyA = bodies[pair.bodyA];
    let bodyB = bodies[pair.bodyB];
    if (bodyA.shapeType != 1u || bodyB.shapeType != 1u) {
        return;
    }

    atomicAdd(&counters.candidates, 1u);
    let delta = bodyB.position.xyz - bodyA.position.xyz;
    let radius = bodyA.position.w + bodyB.position.w;
    let distSq = dot(delta, delta);
    if (distSq > radius * radius) {
        return;
    }

    var normal = vec3f(1.0, 0.0, 0.0);
    if (distSq > 0.000001) {
        normal = delta * inverseSqrt(distSq);
    }

    let slot = atomicAdd(&counters.sphereHits, 1u);
    contacts[slot].bodyA = pair.bodyA;
    contacts[slot].bodyB = pair.bodyB;
    contacts[slot].normal = vec4f(normal, 0.0);
}
)";

    if (sphereContactPipeline == nullptr || sphereContactBindGroupLayout == nullptr)
    {
        wgpu::ShaderSourceWGSL wgsl = {};
        wgsl.code = shaderSource;
        wgpu::ShaderModuleDescriptor shaderDesc = {};
        shaderDesc.nextInChain = &wgsl;
        wgpu::ShaderModule shader = device.CreateShaderModule(&shaderDesc);
        if (shader == nullptr)
            return false;

        wgpu::BindGroupLayoutEntry entries[5] = {};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        entries[0].buffer.minBindingSize = sizeof(GpuContactBody);
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        entries[1].buffer.minBindingSize = sizeof(GpuSapPair);
        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Compute;
        entries[2].buffer.type = wgpu::BufferBindingType::Storage;
        entries[2].buffer.minBindingSize = sizeof(GpuPairCounters);
        entries[3].binding = 3;
        entries[3].visibility = wgpu::ShaderStage::Compute;
        entries[3].buffer.type = wgpu::BufferBindingType::Storage;
        entries[3].buffer.minBindingSize = sizeof(GpuSphereContact);
        entries[4].binding = 4;
        entries[4].visibility = wgpu::ShaderStage::Compute;
        entries[4].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[4].buffer.minBindingSize = sizeof(GpuPairParams);

        wgpu::BindGroupLayoutDescriptor layoutDesc = {};
        layoutDesc.entryCount = 5;
        layoutDesc.entries = entries;
        sphereContactBindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &sphereContactBindGroupLayout;
        wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);

        wgpu::ComputePipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shader;
        pipelineDesc.compute.entryPoint = "main";
        sphereContactPipeline = device.CreateComputePipeline(&pipelineDesc);
        if (sphereContactPipeline == nullptr)
            return false;
    }

    wgpu::BindGroupEntry bindEntries[5] = {};
    bindEntries[0].binding = 0;
    bindEntries[0].buffer = sphereContactBodyBuffer;
    bindEntries[0].size = bodyBytes;
    bindEntries[1].binding = 1;
    bindEntries[1].buffer = sphereContactPairBuffer;
    bindEntries[1].size = pairBytes;
    bindEntries[2].binding = 2;
    bindEntries[2].buffer = sphereContactCountersBuffer;
    bindEntries[2].size = counterBytes;
    bindEntries[3].binding = 3;
    bindEntries[3].buffer = sphereContactOutputBuffer;
    bindEntries[3].size = contactCapacityBytes;
    bindEntries[4].binding = 4;
    bindEntries[4].buffer = sphereContactParamsBuffer;
    bindEntries[4].size = sizeof(GpuPairParams);

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = sphereContactBindGroupLayout;
    bindGroupDesc.entryCount = 5;
    bindGroupDesc.entries = bindEntries;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
    if (bindGroup == nullptr)
        return false;

    Uint64 readbackBegin = SDL_GetPerformanceCounter();
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(sphereContactPipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups(((uint32_t)gpuPairs.size() + 63u) / 64u);
    pass.End();
    encoder.CopyBufferToBuffer(sphereContactCountersBuffer, 0, sphereContactCounterReadbackBuffer, 0, counterBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future counterFuture = sphereContactCounterReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, counterBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(counterFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
        return false;

    const GpuPairCounters *counterData = (const GpuPairCounters *)sphereContactCounterReadbackBuffer.GetConstMappedRange(0, counterBytes);
    if (!counterData)
    {
        sphereContactCounterReadbackBuffer.Unmap();
        return false;
    }
    uint32_t contactCount = counterData->sphereHits;
    sphereContactCounterReadbackBuffer.Unmap();

    sphereContactCount = (int)contactCount;
    if (contactCount == 0)
    {
        sphereContactMs = elapsedMs(begin, SDL_GetPerformanceCounter());
        sphereContactReadbackMs = elapsedMs(readbackBegin, SDL_GetPerformanceCounter());
        return true;
    }

    const uint64_t contactBytes = (uint64_t)contactCount * sizeof(GpuSphereContact);
    if (sphereContactReadbackBuffer == nullptr || sphereContactReadbackBufferBytes < contactBytes)
    {
        wgpu::BufferDescriptor contactReadbackDesc = {};
        contactReadbackDesc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                                (uint64_t)wgpu::BufferUsage::CopyDst);
        contactReadbackDesc.size = alignUp(contactBytes, 4096);
        sphereContactReadbackBuffer = device.CreateBuffer(&contactReadbackDesc);
        sphereContactReadbackBufferBytes = sphereContactReadbackBuffer == nullptr ? 0 : contactReadbackDesc.size;
    }
    if (sphereContactReadbackBuffer == nullptr)
        return false;

    wgpu::CommandEncoder readbackEncoder = device.CreateCommandEncoder();
    readbackEncoder.CopyBufferToBuffer(sphereContactOutputBuffer, 0, sphereContactReadbackBuffer, 0, contactBytes);
    wgpu::CommandBuffer readbackCommands = readbackEncoder.Finish();
    queue.Submit(1, &readbackCommands);

    mapDone = false;
    mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future contactFuture = sphereContactReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, contactBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(contactFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
        return false;

    const GpuSphereContact *gpuContacts = (const GpuSphereContact *)sphereContactReadbackBuffer.GetConstMappedRange(0, contactBytes);
    if (!gpuContacts)
    {
        sphereContactReadbackBuffer.Unmap();
        return false;
    }

    contacts.reserve((size_t)contactCount * 2u);
    for (uint32_t i = 0; i < contactCount; ++i)
    {
        const GpuSphereContact &gpuContact = gpuContacts[i];
        float3 normalAB = {gpuContact.normal[0], gpuContact.normal[1], gpuContact.normal[2]};

        ExternalManifoldContact contact = {};
        contact.bodyA = gpuContact.bodyA;
        contact.bodyB = gpuContact.bodyB;
        contact.basis = orthonormal(-normalAB);
        contact.numContacts = 1;
        contact.contacts[0].feature.key = 0x03000000;
        contacts.push_back(contact);

        ExternalManifoldContact swapped = {};
        swapped.bodyA = gpuContact.bodyB;
        swapped.bodyB = gpuContact.bodyA;
        swapped.basis = orthonormal(normalAB);
        swapped.numContacts = 1;
        swapped.contacts[0].feature.key = 0x03000000;
        contacts.push_back(swapped);
    }
    sphereContactReadbackBuffer.Unmap();

    sphereContactReadbackBytes = (int)contactBytes;
    sphereContactReadbackMs = elapsedMs(readbackBegin, SDL_GetPerformanceCounter());
    sphereContactMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    return true;
#elif AVBD_ENABLE_WEBGPU
    return false;
#else
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
    sapPairOutputBuffer = nullptr;
    sapPairOutputBufferBytes = 0;
    sapPairReadbackBuffer = nullptr;
    sapPairReadbackBufferBytes = 0;
    sapDirectSphereParamsBuffer = nullptr;
    jointTopologyInputBuffer = nullptr;
    jointTopologyInputBufferBytes = 0;
    jointTopologyOutputBuffer = nullptr;
    jointTopologyOutputBufferBytes = 0;
    jointTopologyReadbackBuffer = nullptr;
    jointTopologyReadbackBufferBytes = 0;
    jointTopologyParamsBuffer = nullptr;
    jointTopologyBindGroupLayout = nullptr;
    jointTopologyPipeline = nullptr;
    jointColorBuffer = nullptr;
    jointColorBufferBytes = 0;
    jointColorCounterBuffer = nullptr;
    jointColorCounterBufferBytes = 0;
    jointResidualInputBuffer = nullptr;
    jointResidualInputBufferBytes = 0;
    jointResidualBodyBuffer = nullptr;
    jointResidualBodyBufferBytes = 0;
    jointResidualOutputBuffer = nullptr;
    jointResidualOutputBufferBytes = 0;
    jointProposalOutputBuffer = nullptr;
    jointProposalOutputBufferBytes = 0;
    jointProposalResidualBuffer = nullptr;
    jointProposalResidualBufferBytes = 0;
    jointProposalFinalReadbackBuffer = nullptr;
    jointProposalFinalReadbackBufferBytes = 0;
    jointProposalFinalAsyncReadbackBuffer = nullptr;
    jointProposalFinalAsyncReadbackBufferBytes = 0;
    jointColorBindGroupLayout = nullptr;
    jointColorPipeline = nullptr;
    jointProposalBindGroupLayout = nullptr;
    jointProposalPipeline = nullptr;
    jointProposalApplyBindGroupLayout = nullptr;
    jointProposalApplyPipeline = nullptr;
    jointProposalResidualBindGroupLayout = nullptr;
    jointProposalResidualPipeline = nullptr;
    jointContactSeedBindGroupLayout = nullptr;
    jointContactSeedPipeline = nullptr;
    sphereContactBodyBuffer = nullptr;
    sphereContactBodyBufferBytes = 0;
    sphereContactPairBuffer = nullptr;
    sphereContactPairBufferBytes = 0;
    sphereContactCountersBuffer = nullptr;
    sphereContactCountersBufferBytes = 0;
    sphereContactCounterReadbackBuffer = nullptr;
    sphereContactCounterReadbackBufferBytes = 0;
    sphereContactOutputBuffer = nullptr;
    sphereContactOutputBufferBytes = 0;
    sphereContactReadbackBuffer = nullptr;
    sphereContactReadbackBufferBytes = 0;
    sphereContactBodyCountBuffer = nullptr;
    sphereContactBodyCountBufferBytes = 0;
    sphereContactBodyCountReadbackBuffer = nullptr;
    sphereContactBodyCountReadbackBufferBytes = 0;
    sphereContactAdjacencyBuffer = nullptr;
    sphereContactAdjacencyBufferBytes = 0;
    sphereContactGatherOutputBuffer = nullptr;
    sphereContactGatherOutputBufferBytes = 0;
    sphereContactGatherReadbackBuffer = nullptr;
    sphereContactGatherReadbackBufferBytes = 0;
    sphereContactProposalOutputBuffer = nullptr;
    sphereContactProposalOutputBufferBytes = 0;
    sphereContactProposalOutputReadbackBuffer = nullptr;
    sphereContactProposalOutputReadbackBufferBytes = 0;
    sphereContactIterationOutputBuffer = nullptr;
    sphereContactIterationOutputBufferBytes = 0;
    sphereContactIterationScratchBuffer = nullptr;
    sphereContactIterationScratchBufferBytes = 0;
    sphereContactFinalPositionBuffer = nullptr;
    sphereContactFinalPositionBufferBytes = 0;
    sphereContactFinalPositionAsyncReadbackBuffer = nullptr;
    sphereContactFinalPositionAsyncReadbackBufferBytes = 0;
    sphereContactProposalResidualBuffer = nullptr;
    sphereContactProposalResidualBufferBytes = 0;
    sphereContactProposalResidualReadbackBuffer = nullptr;
    sphereContactProposalResidualReadbackBufferBytes = 0;
    sphereContactParamsBuffer = nullptr;
    sphereContactBindGroupLayout = nullptr;
    sphereContactPipeline = nullptr;
    sphereContactAdjacencyParamsBuffer = nullptr;
    sphereContactAdjacencyBindGroupLayout = nullptr;
    sphereContactAdjacencyPipeline = nullptr;
    sphereContactGatherBindGroupLayout = nullptr;
    sphereContactGatherPipeline = nullptr;
    sphereContactProposalBindGroupLayout = nullptr;
    sphereContactProposalPipeline = nullptr;
    sphereContactProposalResidualBindGroupLayout = nullptr;
    sphereContactProposalResidualPipeline = nullptr;
    sphereContactIterationBindGroupLayout = nullptr;
    sphereContactIterationPipeline = nullptr;
    sphereGroundParamsBuffer = nullptr;
    sphereGroundBindGroupLayout = nullptr;
    sphereGroundPipeline = nullptr;
    sapDirectContactCountersBuffer = nullptr;
    sapDirectContactCountersBufferBytes = 0;
    sapDirectContactCountersReadbackBuffer = nullptr;
    sapDirectContactCountersReadbackBufferBytes = 0;
    sapParamsBuffer = nullptr;
    sapPairParamsBuffer = nullptr;
    sapSortPassParamsBuffer = nullptr;
    sapSortPassParamsBufferBytes = 0;
    sapSortBindGroupLayout = nullptr;
    sapSortPipeline = nullptr;
    sapPairBindGroupLayout = nullptr;
    sapPairPipeline = nullptr;
    sapPairOutputBindGroupLayout = nullptr;
    sapPairOutputPipeline = nullptr;
    sapDirectSphereBindGroupLayout = nullptr;
    sapDirectSpherePipeline = nullptr;
    sapDirectBoxPairParamsBuffer = nullptr;
    sapDirectBoxPairBindGroupLayout = nullptr;
    sapDirectBoxPairPipeline = nullptr;
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
    setStatus(jointTopologyStatus, "Joint topology not run");
    setStatus(runtimeStatus, "WebGPU runtime not run");
    sapCounterReadbackBytes = 0;
    sapCounterReadbackMs = 0.0f;
    sapPairReadbackBytes = 0;
    sapPairReadbackMs = 0.0f;
    jointTopologyMs = 0.0f;
    jointTopologyJoints = 0;
    jointTopologyBodyRefs = 0;
    jointTopologyActiveBodies = 0;
    jointTopologyMaxPerBody = 0;
    jointTopologyMismatches = 0;
    jointTopologyReadbackBytes = 0;
    jointColorCount = 0;
    jointColorConflicts = 0;
    jointColorMinBucket = 0;
    jointColorMaxBucket = 0;
    jointColorReadbackBytes = 0;
    jointResidualMax = 0.0f;
    jointResidualRms = 0.0f;
    jointResidualReadbackBytes = 0;
    jointProposalMaxCorrection = 0.0f;
    jointProposalRmsCorrection = 0.0f;
    jointProposalActiveBodies = 0;
    jointProposalMaxPerBody = 0;
    jointProposalReadbackBytes = 0;
    jointProposalIterations = 0;
    jointProposalResidualAfterMax = 0.0f;
    jointProposalResidualAfterRms = 0.0f;
    jointProposalResidualReadbackBytes = 0;
    jointProposalFinalPositionReady = 0;
    jointProposalFinalPositionBodyCount = 0;
    jointProposalFinalPositionBytes = 0;
    jointProposalFinalPositionAbsolute = 0;
    jointProposalSeededFromContact = 0;
    jointProposalAppliedPositionBodies = 0;
    jointProposalAppliedPositionReadbackBytes = 0;
    jointProposalAppliedPositionMaxDelta = 0.0f;
    jointProposalAppliedPositionChecksum = 0.0f;
    jointProposalAppliedPositionMs = 0.0f;
    sphereContactCount = 0;
    sphereContactReadbackBytes = 0;
    sphereContactMs = 0.0f;
    sphereContactReadbackMs = 0.0f;
    sphereContactBodyRefs = 0;
    sphereContactActiveBodies = 0;
    sphereContactMaxPerBody = 0;
    sphereContactAvgPerActiveBody = 0.0f;
    sphereContactAdjacencyReadbackBytes = 0;
    sphereContactAdjacencyListBytes = 0;
    sphereContactAdjacencyCapacity = 0;
    sphereContactAdjacencyWrittenRefs = 0;
    sphereContactAdjacencyOverflowRefs = 0;
    sphereContactAdjacencyMs = 0.0f;
    sphereContactGatherRefs = 0;
    sphereContactGatherActiveBodies = 0;
    sphereContactGatherMaxPerBody = 0;
    sphereContactGatherMismatches = 0;
    sphereContactGatherReadbackBytes = 0;
    sphereContactGatherNormalChecksum = 0.0f;
    sphereContactProposalActiveBodies = 0;
    sphereContactProposalMaxCorrection = 0.0f;
    sphereContactProposalCorrectionChecksum = 0.0f;
    sphereContactGatherMs = 0.0f;
    sphereContactProposalOutputActiveBodies = 0;
    sphereContactProposalOutputReadbackBytes = 0;
    sphereContactProposalOutputMaxDelta = 0.0f;
    sphereContactProposalOutputChecksum = 0.0f;
    sphereContactProposalOutputMs = 0.0f;
    sphereContactProposalResidualReadbackBytes = 0;
    sphereContactProposalResidualBeforeMax = 0.0f;
    sphereContactProposalResidualAfterMax = 0.0f;
    sphereContactProposalResidualBeforeChecksum = 0.0f;
    sphereContactProposalResidualAfterChecksum = 0.0f;
    sphereContactProposalResidualMs = 0.0f;
    sphereContactIterationCount = 0;
    sphereContactIterationMs = 0.0f;
    sphereContactIterationResidualAfterMax = 0.0f;
    sphereContactIterationResidualAfterChecksum = 0.0f;
    sphereContactFinalPositionReady = 0;
    sphereContactFinalPositionBodyCount = 0;
    sphereContactFinalPositionBytes = 0;
    sphereContactFinalPositionSource = 0;
    sphereContactFinalReferencePositions.clear();
    sphereContactAppliedPositionBodies = 0;
    sphereContactAppliedPositionReadbackBytes = 0;
    sphereContactAppliedPositionMaxDelta = 0.0f;
    sphereContactAppliedPositionChecksum = 0.0f;
    sphereContactAppliedPositionMs = 0.0f;
    sphereContactAppliedPositionWaitMs = 0.0f;
    sphereContactAppliedPositionCpuMs = 0.0f;
    sphereContactFinalPositionReadbackDeferred = 0;
    sphereContactFinalPositionAsyncReadbackPending = 0;
    sphereContactFinalPositionAsyncReadbackScheduled = 0;
    sphereContactFinalPositionAsyncReadbackConsumed = 0;
    sphereContactFinalPositionAsyncReadbackDropped = 0;
    sphereContactFinalPositionAsyncReadbackWaitMs = 0.0f;
    sphereContactFinalPositionAsyncReadbackBodyCount = 0;
    sphereContactFinalPositionAsyncReadbackBytes = 0;
    sphereContactFinalPositionAsyncReadbackSource = 0;
    sphereContactAsyncReferencePositions.clear();
    predictionAppliedBodies = 0;
    predictionAppliedReadbackBytes = 0;
    predictionAppliedMs = 0.0f;
    velocityAppliedBodies = 0;
    velocityAppliedReadbackBytes = 0;
    velocityAppliedMs = 0.0f;
    runtimeTotalMs = 0.0f;
    runtimeSyncMs = 0.0f;
    runtimePredictionMs = 0.0f;
    runtimeVelocityMs = 0.0f;
    runtimeCpuFallbackMs = 0.0f;
    runtimeMaxLinearError = 0.0f;
    runtimeMaxAngularError = 0.0f;
    runtimeFrames = 0;
    runtimeFallbacks = 0;
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

const char *WebGpuContext::jointTopologyStatusText() const
{
    return jointTopologyStatus;
}

const char *WebGpuContext::runtimeStatusText() const
{
    return runtimeStatus;
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

int WebGpuContext::sapCounterReadbackByteCount() const
{
    return sapCounterReadbackBytes;
}

float WebGpuContext::sapCounterReadbackMillis() const
{
    return sapCounterReadbackMs;
}

int WebGpuContext::sapPairReadbackByteCount() const
{
    return sapPairReadbackBytes;
}

float WebGpuContext::sapPairReadbackMillis() const
{
    return sapPairReadbackMs;
}

float WebGpuContext::jointTopologyMillis() const
{
    return jointTopologyMs;
}

int WebGpuContext::jointTopologyJointCount() const
{
    return jointTopologyJoints;
}

int WebGpuContext::jointTopologyBodyRefCount() const
{
    return jointTopologyBodyRefs;
}

int WebGpuContext::jointTopologyActiveBodyCount() const
{
    return jointTopologyActiveBodies;
}

int WebGpuContext::jointTopologyMaxPerBodyCount() const
{
    return jointTopologyMaxPerBody;
}

int WebGpuContext::jointTopologyMismatchCount() const
{
    return jointTopologyMismatches;
}

int WebGpuContext::jointTopologyReadbackByteCount() const
{
    return jointTopologyReadbackBytes;
}

int WebGpuContext::jointColorCountValue() const
{
    return jointColorCount;
}

int WebGpuContext::jointColorConflictCount() const
{
    return jointColorConflicts;
}

int WebGpuContext::jointColorMinBucketCount() const
{
    return jointColorMinBucket;
}

int WebGpuContext::jointColorMaxBucketCount() const
{
    return jointColorMaxBucket;
}

int WebGpuContext::jointColorReadbackByteCount() const
{
    return jointColorReadbackBytes;
}

float WebGpuContext::jointResidualMaxValue() const
{
    return jointResidualMax;
}

float WebGpuContext::jointResidualRmsValue() const
{
    return jointResidualRms;
}

int WebGpuContext::jointResidualReadbackByteCount() const
{
    return jointResidualReadbackBytes;
}

float WebGpuContext::jointProposalMaxCorrectionValue() const
{
    return jointProposalMaxCorrection;
}

float WebGpuContext::jointProposalRmsCorrectionValue() const
{
    return jointProposalRmsCorrection;
}

int WebGpuContext::jointProposalActiveBodyCount() const
{
    return jointProposalActiveBodies;
}

int WebGpuContext::jointProposalMaxPerBodyCount() const
{
    return jointProposalMaxPerBody;
}

int WebGpuContext::jointProposalReadbackByteCount() const
{
    return jointProposalReadbackBytes;
}

int WebGpuContext::jointProposalIterationCount() const
{
    return jointProposalIterations;
}

float WebGpuContext::jointProposalResidualAfterMaxValue() const
{
    return jointProposalResidualAfterMax;
}

float WebGpuContext::jointProposalResidualAfterRmsValue() const
{
    return jointProposalResidualAfterRms;
}

int WebGpuContext::jointProposalResidualReadbackByteCount() const
{
    return jointProposalResidualReadbackBytes;
}

int WebGpuContext::jointProposalFinalPositionReadyValue() const
{
    return jointProposalFinalPositionReady;
}

int WebGpuContext::jointProposalFinalPositionBodyCountValue() const
{
    return jointProposalFinalPositionBodyCount;
}

int WebGpuContext::jointProposalFinalPositionByteCount() const
{
    return jointProposalFinalPositionBytes;
}

int WebGpuContext::jointProposalFinalPositionAbsoluteValue() const
{
    return jointProposalFinalPositionAbsolute;
}

int WebGpuContext::jointProposalSeededFromContactValue() const
{
    return jointProposalSeededFromContact;
}

static bool applyJointProposalFinalPositionOutputsToSolver(Solver &solver,
                                                           const void *mappedFinalPositions,
                                                           int bodyCount,
                                                           bool absolutePositions,
                                                           int &appliedBodies,
                                                           double &maxDelta,
                                                           double &checksum)
{
    if (!mappedFinalPositions || bodyCount <= 0)
        return false;

    int bodyLimit = std::min(bodyCount, (int)solver.world.bodies.size());
    appliedBodies = 0;
    maxDelta = 0.0;
    checksum = 0.0;
    for (int bodyId = 0; bodyId < bodyLimit; ++bodyId)
    {
        SimBodyData &bodyData = solver.world.bodies[bodyId];
        Rigid *body = bodyData.source;
        if (!bodyData.active || !body || bodyData.mass <= 0.0f)
            continue;

        float3 delta{0.0f, 0.0f, 0.0f};
        if (absolutePositions)
        {
            const GpuJointResidualBody *gpuBodies = (const GpuJointResidualBody *)mappedFinalPositions;
            const GpuJointResidualBody &gpuBody = gpuBodies[bodyId];
            float3 finalPosition{
                gpuBody.position[0],
                gpuBody.position[1],
                gpuBody.position[2]};
            delta = finalPosition - body->positionLin;
        }
        else
        {
            const GpuJointProposalOutput *gpuProposals = (const GpuJointProposalOutput *)mappedFinalPositions;
            const GpuJointProposalOutput &proposal = gpuProposals[bodyId];
            if (proposal.jointCount == 0)
                continue;
            delta = float3{
                proposal.correction[0],
                proposal.correction[1],
                proposal.correction[2]};
        }
        double deltaLength = length(delta);
        if (!std::isfinite(deltaLength) || deltaLength <= 0.0)
            continue;

        body->positionLin = body->positionLin + delta;
        bodyData.positionLin = body->positionLin;
        bodyData.positionAng = body->positionAng;
        bodyData.velocityLin = body->velocityLin;
        bodyData.velocityAng = body->velocityAng;
        maxDelta = std::max(maxDelta, deltaLength);
        checksum += (double)(bodyId + 1) * ((double)delta.x * 0.73 + (double)delta.y * 1.31 + (double)delta.z * 1.91);
        appliedBodies++;
    }
    return appliedBodies > 0;
}

bool WebGpuContext::applyJointProposalFinalPositions(Solver &solver)
{
    jointProposalAppliedPositionBodies = 0;
    jointProposalAppliedPositionReadbackBytes = 0;
    jointProposalAppliedPositionMaxDelta = 0.0f;
    jointProposalAppliedPositionChecksum = 0.0f;
    jointProposalAppliedPositionMs = 0.0f;
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    Uint64 begin = SDL_GetPerformanceCounter();
    wgpu::Buffer sourceBuffer = jointProposalFinalPositionAbsolute ? jointResidualBodyBuffer : jointProposalOutputBuffer;
    if (!jointProposalFinalPositionReady || sourceBuffer == nullptr ||
        jointProposalFinalPositionBodyCount <= 0 || jointProposalFinalPositionBytes <= 0)
        return false;

    if (jointProposalFinalReadbackBuffer == nullptr ||
        jointProposalFinalReadbackBufferBytes < (uint64_t)jointProposalFinalPositionBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = (uint64_t)jointProposalFinalPositionBytes;
        desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        jointProposalFinalReadbackBuffer = device.CreateBuffer(&desc);
        jointProposalFinalReadbackBufferBytes = jointProposalFinalReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (jointProposalFinalReadbackBuffer == nullptr)
        return false;

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(sourceBuffer, 0,
                               jointProposalFinalReadbackBuffer, 0,
                               (uint64_t)jointProposalFinalPositionBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = jointProposalFinalReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, (uint64_t)jointProposalFinalPositionBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
        return false;

    const void *mappedFinalPositions = jointProposalFinalReadbackBuffer.GetConstMappedRange(
        0, (uint64_t)jointProposalFinalPositionBytes);
    if (!mappedFinalPositions)
    {
        jointProposalFinalReadbackBuffer.Unmap();
        return false;
    }

    int appliedBodies = 0;
    double maxDelta = 0.0;
    double checksum = 0.0;
    bool appliedOk = applyJointProposalFinalPositionOutputsToSolver(solver,
                                                                    mappedFinalPositions,
                                                                    jointProposalFinalPositionBodyCount,
                                                                    jointProposalFinalPositionAbsolute != 0,
                                                                    appliedBodies,
                                                                    maxDelta,
                                                                    checksum);
    jointProposalFinalReadbackBuffer.Unmap();

    jointProposalAppliedPositionBodies = appliedBodies;
    jointProposalAppliedPositionReadbackBytes = jointProposalFinalPositionBytes;
    jointProposalAppliedPositionMaxDelta = (float)maxDelta;
    jointProposalAppliedPositionChecksum = (float)checksum;
    jointProposalAppliedPositionMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    return appliedOk;
#else
    (void)solver;
    return false;
#endif
}

bool WebGpuContext::scheduleJointProposalFinalPositionReadback()
{
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    jointProposalFinalPositionAsyncReadbackScheduled = 0;
    jointProposalFinalPositionAsyncReadbackDropped = 0;
    wgpu::Buffer sourceBuffer = jointProposalFinalPositionAbsolute ? jointResidualBodyBuffer : jointProposalOutputBuffer;
    if (!jointProposalFinalPositionReady || sourceBuffer == nullptr ||
        jointProposalFinalPositionBodyCount <= 0 || jointProposalFinalPositionBytes <= 0)
        return false;
    if (jointProposalFinalPositionAsyncReadbackPending)
    {
        jointProposalFinalPositionAsyncReadbackDropped = 1;
        return false;
    }

    uint64_t readbackBytes = (uint64_t)jointProposalFinalPositionBytes;
    if (jointProposalFinalAsyncReadbackBuffer == nullptr ||
        jointProposalFinalAsyncReadbackBufferBytes < readbackBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = alignUp(readbackBytes, 4096);
        desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        jointProposalFinalAsyncReadbackBuffer = device.CreateBuffer(&desc);
        jointProposalFinalAsyncReadbackBufferBytes = jointProposalFinalAsyncReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (jointProposalFinalAsyncReadbackBuffer == nullptr)
        return false;

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(sourceBuffer, 0,
                               jointProposalFinalAsyncReadbackBuffer, 0,
                               readbackBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    jointProposalFinalPositionAsyncReadbackPending = 1;
    jointProposalFinalPositionAsyncReadbackScheduled = 1;
    jointProposalFinalPositionAsyncReadbackBodyCount = jointProposalFinalPositionBodyCount;
    jointProposalFinalPositionAsyncReadbackBytes = jointProposalFinalPositionBytes;
    jointProposalFinalPositionAsyncReadbackAbsolute = jointProposalFinalPositionAbsolute;
    return true;
#else
    return false;
#endif
}

bool WebGpuContext::consumePendingJointProposalFinalPositions(Solver &solver)
{
    jointProposalFinalPositionAsyncReadbackConsumed = 0;
    jointProposalFinalPositionAsyncReadbackWaitMs = 0.0f;
    jointProposalAppliedPositionBodies = 0;
    jointProposalAppliedPositionReadbackBytes = 0;
    jointProposalAppliedPositionMaxDelta = 0.0f;
    jointProposalAppliedPositionChecksum = 0.0f;
    jointProposalAppliedPositionMs = 0.0f;
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!jointProposalFinalPositionAsyncReadbackPending)
        return false;
    if (jointProposalFinalAsyncReadbackBuffer == nullptr ||
        jointProposalFinalPositionAsyncReadbackBytes <= 0 ||
        jointProposalFinalPositionAsyncReadbackBodyCount <= 0)
        return false;

    Uint64 begin = SDL_GetPerformanceCounter();
    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    Uint64 waitBegin = SDL_GetPerformanceCounter();
    wgpu::Future mapFuture = jointProposalFinalAsyncReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, (uint64_t)jointProposalFinalPositionAsyncReadbackBytes,
        wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
        return false;
    jointProposalFinalPositionAsyncReadbackWaitMs = elapsedMs(waitBegin, SDL_GetPerformanceCounter());

    const void *mappedFinalPositions = jointProposalFinalAsyncReadbackBuffer.GetConstMappedRange(
        0, (uint64_t)jointProposalFinalPositionAsyncReadbackBytes);
    if (!mappedFinalPositions)
    {
        jointProposalFinalAsyncReadbackBuffer.Unmap();
        return false;
    }

    int appliedBodies = 0;
    double maxDelta = 0.0;
    double checksum = 0.0;
    bool appliedOk = applyJointProposalFinalPositionOutputsToSolver(solver,
                                                                    mappedFinalPositions,
                                                                    jointProposalFinalPositionAsyncReadbackBodyCount,
                                                                    jointProposalFinalPositionAsyncReadbackAbsolute != 0,
                                                                    appliedBodies,
                                                                    maxDelta,
                                                                    checksum);
    jointProposalFinalAsyncReadbackBuffer.Unmap();

    jointProposalFinalPositionAsyncReadbackPending = 0;
    jointProposalFinalPositionAsyncReadbackConsumed = 1;
    jointProposalAppliedPositionBodies = appliedBodies;
    jointProposalAppliedPositionReadbackBytes = jointProposalFinalPositionAsyncReadbackBytes;
    jointProposalAppliedPositionMaxDelta = (float)maxDelta;
    jointProposalAppliedPositionChecksum = (float)checksum;
    jointProposalAppliedPositionMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    return appliedOk;
#else
    (void)solver;
    return false;
#endif
}

int WebGpuContext::jointProposalAppliedPositionBodyCount() const
{
    return jointProposalAppliedPositionBodies;
}

int WebGpuContext::jointProposalAppliedPositionReadbackByteCount() const
{
    return jointProposalAppliedPositionReadbackBytes;
}

float WebGpuContext::jointProposalAppliedPositionMaxDeltaValue() const
{
    return jointProposalAppliedPositionMaxDelta;
}

float WebGpuContext::jointProposalAppliedPositionChecksumValue() const
{
    return jointProposalAppliedPositionChecksum;
}

float WebGpuContext::jointProposalAppliedPositionMillis() const
{
    return jointProposalAppliedPositionMs;
}

int WebGpuContext::jointProposalFinalPositionAsyncReadbackPendingValue() const
{
    return jointProposalFinalPositionAsyncReadbackPending;
}

int WebGpuContext::jointProposalFinalPositionAsyncReadbackScheduledValue() const
{
    return jointProposalFinalPositionAsyncReadbackScheduled;
}

int WebGpuContext::jointProposalFinalPositionAsyncReadbackConsumedValue() const
{
    return jointProposalFinalPositionAsyncReadbackConsumed;
}

int WebGpuContext::jointProposalFinalPositionAsyncReadbackDroppedValue() const
{
    return jointProposalFinalPositionAsyncReadbackDropped;
}

float WebGpuContext::jointProposalFinalPositionAsyncReadbackWaitMillis() const
{
    return jointProposalFinalPositionAsyncReadbackWaitMs;
}

int WebGpuContext::jointProposalFinalPositionAsyncReadbackBodyCountValue() const
{
    return jointProposalFinalPositionAsyncReadbackBodyCount;
}

int WebGpuContext::jointProposalFinalPositionAsyncReadbackByteCount() const
{
    return jointProposalFinalPositionAsyncReadbackBytes;
}

int WebGpuContext::jointProposalFinalPositionAsyncReadbackAbsoluteValue() const
{
    return jointProposalFinalPositionAsyncReadbackAbsolute;
}

int WebGpuContext::sphereContactCountValue() const
{
    return sphereContactCount;
}

int WebGpuContext::sphereContactExternalContactCount() const
{
    return sphereContactExternalContacts;
}

int WebGpuContext::sphereContactExternalGroundContactCount() const
{
    return sphereContactExternalGroundContacts;
}

int WebGpuContext::sphereContactReadbackByteCount() const
{
    return sphereContactReadbackBytes;
}

float WebGpuContext::sphereContactMillis() const
{
    return sphereContactMs;
}

float WebGpuContext::sphereContactReadbackMillis() const
{
    return sphereContactReadbackMs;
}

int WebGpuContext::sphereContactBodyRefCount() const
{
    return sphereContactBodyRefs;
}

int WebGpuContext::sphereContactActiveBodyCount() const
{
    return sphereContactActiveBodies;
}

int WebGpuContext::sphereContactMaxPerBodyCount() const
{
    return sphereContactMaxPerBody;
}

float WebGpuContext::sphereContactAvgPerActiveBodyValue() const
{
    return sphereContactAvgPerActiveBody;
}

int WebGpuContext::sphereContactAdjacencyReadbackByteCount() const
{
    return sphereContactAdjacencyReadbackBytes;
}

int WebGpuContext::sphereContactAdjacencyBufferByteCount() const
{
    return sphereContactAdjacencyListBytes;
}

int WebGpuContext::sphereContactAdjacencyCapacityValue() const
{
    return sphereContactAdjacencyCapacity;
}

int WebGpuContext::sphereContactAdjacencyWrittenRefCount() const
{
    return sphereContactAdjacencyWrittenRefs;
}

int WebGpuContext::sphereContactAdjacencyOverflowRefCount() const
{
    return sphereContactAdjacencyOverflowRefs;
}

float WebGpuContext::sphereContactAdjacencyMillis() const
{
    return sphereContactAdjacencyMs;
}

int WebGpuContext::sphereContactGatherRefCount() const
{
    return sphereContactGatherRefs;
}

int WebGpuContext::sphereContactGatherActiveBodyCount() const
{
    return sphereContactGatherActiveBodies;
}

int WebGpuContext::sphereContactGatherMaxPerBodyCount() const
{
    return sphereContactGatherMaxPerBody;
}

int WebGpuContext::sphereContactGatherMismatchCount() const
{
    return sphereContactGatherMismatches;
}

int WebGpuContext::sphereContactGatherReadbackByteCount() const
{
    return sphereContactGatherReadbackBytes;
}

float WebGpuContext::sphereContactGatherNormalChecksumValue() const
{
    return sphereContactGatherNormalChecksum;
}

int WebGpuContext::sphereContactProposalActiveBodyCount() const
{
    return sphereContactProposalActiveBodies;
}

float WebGpuContext::sphereContactProposalMaxCorrectionValue() const
{
    return sphereContactProposalMaxCorrection;
}

float WebGpuContext::sphereContactProposalCorrectionChecksumValue() const
{
    return sphereContactProposalCorrectionChecksum;
}

float WebGpuContext::sphereContactGatherMillis() const
{
    return sphereContactGatherMs;
}

int WebGpuContext::sphereContactProposalOutputActiveBodyCount() const
{
    return sphereContactProposalOutputActiveBodies;
}

int WebGpuContext::sphereContactProposalOutputReadbackByteCount() const
{
    return sphereContactProposalOutputReadbackBytes;
}

float WebGpuContext::sphereContactProposalOutputMaxDeltaValue() const
{
    return sphereContactProposalOutputMaxDelta;
}

float WebGpuContext::sphereContactProposalOutputChecksumValue() const
{
    return sphereContactProposalOutputChecksum;
}

float WebGpuContext::sphereContactProposalOutputMillis() const
{
    return sphereContactProposalOutputMs;
}

int WebGpuContext::sphereContactProposalResidualReadbackByteCount() const
{
    return sphereContactProposalResidualReadbackBytes;
}

float WebGpuContext::sphereContactProposalResidualBeforeMaxValue() const
{
    return sphereContactProposalResidualBeforeMax;
}

float WebGpuContext::sphereContactProposalResidualAfterMaxValue() const
{
    return sphereContactProposalResidualAfterMax;
}

float WebGpuContext::sphereContactProposalResidualBeforeChecksumValue() const
{
    return sphereContactProposalResidualBeforeChecksum;
}

float WebGpuContext::sphereContactProposalResidualAfterChecksumValue() const
{
    return sphereContactProposalResidualAfterChecksum;
}

float WebGpuContext::sphereContactProposalResidualMillis() const
{
    return sphereContactProposalResidualMs;
}

int WebGpuContext::sphereContactIterationCountValue() const
{
    return sphereContactIterationCount;
}

float WebGpuContext::sphereContactIterationMillis() const
{
    return sphereContactIterationMs;
}

float WebGpuContext::sphereContactIterationRelaxationValue() const
{
    return sphereContactIterationRelaxation;
}

float WebGpuContext::sphereContactIterationResidualAfterMaxValue() const
{
    return sphereContactIterationResidualAfterMax;
}

float WebGpuContext::sphereContactIterationResidualAfterChecksumValue() const
{
    return sphereContactIterationResidualAfterChecksum;
}

int WebGpuContext::sphereContactFinalPositionReadyValue() const
{
    return sphereContactFinalPositionReady;
}

int WebGpuContext::sphereContactFinalPositionBodyCountValue() const
{
    return sphereContactFinalPositionBodyCount;
}

int WebGpuContext::sphereContactFinalPositionByteCount() const
{
    return sphereContactFinalPositionBytes;
}

const char *WebGpuContext::sphereContactFinalPositionSourceText() const
{
    switch (sphereContactFinalPositionSource)
    {
    case 1:
        return "proposal";
    case 2:
        return "iteration output";
    case 3:
        return "iteration scratch";
    case 4:
        return "ground correction";
    case 5:
        return "direct sphere solve";
    default:
        return "none";
    }
}

const char *sphereContactFinalPositionSourceName(int source)
{
    switch (source)
    {
    case 1:
        return "proposal";
    case 2:
        return "iteration output";
    case 3:
        return "iteration scratch";
    case 4:
        return "ground correction";
    case 5:
        return "direct sphere solve";
    default:
        return "none";
    }
}

bool applySphereContactFinalPositionOutputsToSolver(Solver &solver,
                                                    const GpuContactProposalOutput *outputs,
                                                    int bodyCount,
                                                    int source,
                                                    const std::vector<float3> &referencePositions,
                                                    int &appliedBodies,
                                                    double &maxDelta,
                                                    double &checksum)
{
    appliedBodies = 0;
    maxDelta = 0.0;
    checksum = 0.0;
    if (!outputs || bodyCount <= 0)
        return false;

    int bodyLimit = std::min(bodyCount, (int)solver.world.bodies.size());
    bool groundCorrectionSource = source == 4;
    bool directContactSource = source == 5;
    for (int bodyId = 0; bodyId < bodyLimit; ++bodyId)
    {
        SimBodyData &bodyData = solver.world.bodies[(BodyId)bodyId];
        Rigid *body = bodyData.source;
        if (!body || body->mass <= 0.0f)
            continue;
        bool supportedBody = body->shape.type == RIGID_SHAPE_SPHERE ||
                             body->shape.type == RIGID_SHAPE_CAPSULE ||
                             body->shape.type == RIGID_SHAPE_CYLINDER ||
                             ((groundCorrectionSource || directContactSource) && body->shape.type == RIGID_SHAPE_BOX);
        if (!supportedBody)
            continue;

        float3 referencePosition = bodyData.positionLin;
        if (bodyId < (int)referencePositions.size())
            referencePosition = referencePositions[(size_t)bodyId];
        float3 correction = directContactSource
            ? float3{outputs[bodyId].correction[0],
                     outputs[bodyId].correction[1],
                     outputs[bodyId].correction[2]}
            : float3{outputs[bodyId].position[0] - referencePosition.x,
                     outputs[bodyId].position[1] - referencePosition.y,
                     outputs[bodyId].position[2] - referencePosition.z};
        float3 correctedPosition = body->positionLin + correction;
        float3 delta = correctedPosition - body->positionLin;
        double deltaLength = (double)length(delta);
        if (deltaLength <= 0.0000001)
            continue;

        body->positionLin = correctedPosition;
        if (solver.dt > 0.0f)
            body->velocityLin = (body->positionLin - body->initialLin) / solver.dt;
        ++appliedBodies;
        checksum += fabs((double)delta.x) + fabs((double)delta.y) + fabs((double)delta.z);
        if (deltaLength > maxDelta)
            maxDelta = deltaLength;
    }
    if (appliedBodies > 0)
        solver.world.syncFromLegacy(solver);
    return true;
}

bool WebGpuContext::applySphereContactFinalPositions(Solver &solver)
{
    sphereContactAppliedPositionBodies = 0;
    sphereContactAppliedPositionReadbackBytes = 0;
    sphereContactAppliedPositionMaxDelta = 0.0f;
    sphereContactAppliedPositionChecksum = 0.0f;
    sphereContactAppliedPositionMs = 0.0f;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
        return false;
    if (!sphereContactFinalPositionReady || sphereContactFinalPositionBuffer == nullptr ||
        sphereContactFinalPositionBodyCount <= 0 || sphereContactFinalPositionBytes <= 0)
        return false;
    if (sphereContactProposalOutputReadbackBuffer == nullptr ||
        sphereContactProposalOutputReadbackBufferBytes < (uint64_t)sphereContactFinalPositionBytes)
        return false;

    Uint64 begin = SDL_GetPerformanceCounter();
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(sphereContactFinalPositionBuffer, 0,
                               sphereContactProposalOutputReadbackBuffer, 0,
                               (uint64_t)sphereContactFinalPositionBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    Uint64 waitBegin = SDL_GetPerformanceCounter();
    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = sphereContactProposalOutputReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, (uint64_t)sphereContactFinalPositionBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
        return false;
    sphereContactAppliedPositionWaitMs = elapsedMs(waitBegin, SDL_GetPerformanceCounter());

    const GpuContactProposalOutput *outputs =
        (const GpuContactProposalOutput *)sphereContactProposalOutputReadbackBuffer.GetConstMappedRange(
            0, (uint64_t)sphereContactFinalPositionBytes);
    if (!outputs)
    {
        sphereContactProposalOutputReadbackBuffer.Unmap();
        return false;
    }

    Uint64 cpuBegin = SDL_GetPerformanceCounter();
    int appliedBodies = 0;
    double maxDelta = 0.0;
    double checksum = 0.0;
    bool appliedOk = applySphereContactFinalPositionOutputsToSolver(solver, outputs,
                                                                    sphereContactFinalPositionBodyCount,
                                                                    sphereContactFinalPositionSource,
                                                                    sphereContactFinalReferencePositions,
                                                                    appliedBodies, maxDelta, checksum);
    sphereContactProposalOutputReadbackBuffer.Unmap();
    if (!appliedOk)
        return false;
    sphereContactAppliedPositionCpuMs = elapsedMs(cpuBegin, SDL_GetPerformanceCounter());

    sphereContactAppliedPositionBodies = appliedBodies;
    sphereContactAppliedPositionReadbackBytes = sphereContactFinalPositionBytes;
    sphereContactAppliedPositionMaxDelta = (float)maxDelta;
    sphereContactAppliedPositionChecksum = (float)checksum;
    sphereContactAppliedPositionMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    if (sphereContactFinalPositionSource == 5 && appliedBodies > 0)
    {
        // The direct-contact runtime is a per-body GPU correction pass.  Use the
        // final-position readback as the authoritative proof when the auxiliary
        // shader counter is unavailable or returns no records.
        directGpuContactRecordCount = std::max(directGpuContactRecordCount, appliedBodies);
        if (directRoundPairCandidateCount > 0)
            directGpuRoundPairCandidateCount = std::max(directGpuRoundPairCandidateCount, appliedBodies);
        if (directBoxPairCandidateCount > 0)
            directGpuBoxPairCandidateCount = std::max(directGpuBoxPairCandidateCount, appliedBodies);
    }
    return true;
#else
    (void)solver;
    return false;
#endif
}

bool WebGpuContext::scheduleSphereContactFinalPositionReadback()
{
    sphereContactFinalPositionAsyncReadbackScheduled = 0;
    sphereContactFinalPositionAsyncReadbackDropped = 0;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr)
        return false;
    if (!sphereContactFinalPositionReady || sphereContactFinalPositionBuffer == nullptr ||
        sphereContactFinalPositionBodyCount <= 0 || sphereContactFinalPositionBytes <= 0)
        return false;
    if (sphereContactFinalPositionAsyncReadbackPending)
    {
        sphereContactFinalPositionAsyncReadbackDropped = 1;
        return false;
    }

    uint64_t readbackBytes = (uint64_t)sphereContactFinalPositionBytes;
    if (sphereContactFinalPositionAsyncReadbackBuffer == nullptr ||
        sphereContactFinalPositionAsyncReadbackBufferBytes < readbackBytes)
    {
        wgpu::BufferDescriptor desc = {};
        desc.usage = bufferUsage((uint64_t)wgpu::BufferUsage::MapRead |
                                 (uint64_t)wgpu::BufferUsage::CopyDst);
        desc.size = alignUp(readbackBytes, 4096);
        sphereContactFinalPositionAsyncReadbackBuffer = device.CreateBuffer(&desc);
        sphereContactFinalPositionAsyncReadbackBufferBytes =
            sphereContactFinalPositionAsyncReadbackBuffer == nullptr ? 0 : desc.size;
    }
    if (sphereContactFinalPositionAsyncReadbackBuffer == nullptr)
        return false;

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(sphereContactFinalPositionBuffer, 0,
                               sphereContactFinalPositionAsyncReadbackBuffer, 0,
                               readbackBytes);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    sphereContactFinalPositionAsyncReadbackPending = 1;
    sphereContactFinalPositionAsyncReadbackScheduled = 1;
    sphereContactFinalPositionAsyncReadbackBodyCount = sphereContactFinalPositionBodyCount;
    sphereContactFinalPositionAsyncReadbackBytes = sphereContactFinalPositionBytes;
    sphereContactFinalPositionAsyncReadbackSource = sphereContactFinalPositionSource;
    sphereContactAsyncReferencePositions = sphereContactFinalReferencePositions;
    return true;
#else
    return false;
#endif
}

bool WebGpuContext::consumePendingSphereContactFinalPositions(Solver &solver)
{
    sphereContactFinalPositionAsyncReadbackConsumed = 0;
    sphereContactFinalPositionAsyncReadbackWaitMs = 0.0f;
    sphereContactAppliedPositionBodies = 0;
    sphereContactAppliedPositionReadbackBytes = 0;
    sphereContactAppliedPositionMaxDelta = 0.0f;
    sphereContactAppliedPositionChecksum = 0.0f;
    sphereContactAppliedPositionMs = 0.0f;
    sphereContactAppliedPositionWaitMs = 0.0f;
    sphereContactAppliedPositionCpuMs = 0.0f;

#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!sphereContactFinalPositionAsyncReadbackPending)
        return true;
    if (!deviceReady || instance == nullptr || device == nullptr || queue == nullptr ||
        sphereContactFinalPositionAsyncReadbackBuffer == nullptr ||
        sphereContactFinalPositionAsyncReadbackBytes <= 0 ||
        sphereContactFinalPositionAsyncReadbackBodyCount <= 0)
        return false;

    Uint64 begin = SDL_GetPerformanceCounter();
    Uint64 waitBegin = SDL_GetPerformanceCounter();
    bool mapDone = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Error;
    wgpu::Future mapFuture = sphereContactFinalPositionAsyncReadbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, (uint64_t)sphereContactFinalPositionAsyncReadbackBytes,
        wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView)
        {
            mapDone = true;
            mapStatus = status;
        });
    if (instance.WaitAny(mapFuture, UINT64_MAX) != wgpu::WaitStatus::Success ||
        !mapDone || mapStatus != wgpu::MapAsyncStatus::Success)
        return false;
    sphereContactFinalPositionAsyncReadbackWaitMs = elapsedMs(waitBegin, SDL_GetPerformanceCounter());
    sphereContactAppliedPositionWaitMs = sphereContactFinalPositionAsyncReadbackWaitMs;

    const GpuContactProposalOutput *outputs =
        (const GpuContactProposalOutput *)sphereContactFinalPositionAsyncReadbackBuffer.GetConstMappedRange(
            0, (uint64_t)sphereContactFinalPositionAsyncReadbackBytes);
    if (!outputs)
    {
        sphereContactFinalPositionAsyncReadbackBuffer.Unmap();
        return false;
    }

    Uint64 cpuBegin = SDL_GetPerformanceCounter();
    int appliedBodies = 0;
    double maxDelta = 0.0;
    double checksum = 0.0;
    bool appliedOk = applySphereContactFinalPositionOutputsToSolver(solver, outputs,
                                                                    sphereContactFinalPositionAsyncReadbackBodyCount,
                                                                    sphereContactFinalPositionAsyncReadbackSource,
                                                                    sphereContactAsyncReferencePositions,
                                                                    appliedBodies, maxDelta, checksum);
    sphereContactFinalPositionAsyncReadbackBuffer.Unmap();
    if (!appliedOk)
        return false;

    sphereContactFinalPositionAsyncReadbackPending = 0;
    sphereContactFinalPositionAsyncReadbackConsumed = 1;
    sphereContactAppliedPositionBodies = appliedBodies;
    sphereContactAppliedPositionReadbackBytes = sphereContactFinalPositionAsyncReadbackBytes;
    sphereContactAppliedPositionMaxDelta = (float)maxDelta;
    sphereContactAppliedPositionChecksum = (float)checksum;
    sphereContactAppliedPositionCpuMs = elapsedMs(cpuBegin, SDL_GetPerformanceCounter());
    sphereContactAppliedPositionMs = elapsedMs(begin, SDL_GetPerformanceCounter());
    sphereContactAsyncReferencePositions.clear();
    if (sphereContactFinalPositionAsyncReadbackSource == 5 && appliedBodies > 0)
    {
        directGpuContactRecordCount = std::max(directGpuContactRecordCount, appliedBodies);
        if (directRoundPairCandidateCount > 0)
            directGpuRoundPairCandidateCount = std::max(directGpuRoundPairCandidateCount, appliedBodies);
        if (directBoxPairCandidateCount > 0)
            directGpuBoxPairCandidateCount = std::max(directGpuBoxPairCandidateCount, appliedBodies);
    }
    return true;
#else
    (void)solver;
    return false;
#endif
}

int WebGpuContext::sphereContactAppliedPositionBodyCount() const
{
    return sphereContactAppliedPositionBodies;
}

int WebGpuContext::sphereContactAppliedPositionReadbackByteCount() const
{
    return sphereContactAppliedPositionReadbackBytes;
}

float WebGpuContext::sphereContactAppliedPositionMaxDeltaValue() const
{
    return sphereContactAppliedPositionMaxDelta;
}

float WebGpuContext::sphereContactAppliedPositionChecksumValue() const
{
    return sphereContactAppliedPositionChecksum;
}

float WebGpuContext::sphereContactAppliedPositionMillis() const
{
    return sphereContactAppliedPositionMs;
}

float WebGpuContext::sphereContactAppliedPositionWaitMillis() const
{
    return sphereContactAppliedPositionWaitMs;
}

float WebGpuContext::sphereContactAppliedPositionCpuMillis() const
{
    return sphereContactAppliedPositionCpuMs;
}

int WebGpuContext::sphereContactFinalPositionReadbackDeferredValue() const
{
    return sphereContactFinalPositionReadbackDeferred;
}

int WebGpuContext::sphereContactFinalPositionAsyncReadbackPendingValue() const
{
    return sphereContactFinalPositionAsyncReadbackPending;
}

int WebGpuContext::sphereContactFinalPositionAsyncReadbackScheduledValue() const
{
    return sphereContactFinalPositionAsyncReadbackScheduled;
}

int WebGpuContext::sphereContactFinalPositionAsyncReadbackConsumedValue() const
{
    return sphereContactFinalPositionAsyncReadbackConsumed;
}

int WebGpuContext::sphereContactFinalPositionAsyncReadbackDroppedValue() const
{
    return sphereContactFinalPositionAsyncReadbackDropped;
}

float WebGpuContext::sphereContactFinalPositionAsyncReadbackWaitMillis() const
{
    return sphereContactFinalPositionAsyncReadbackWaitMs;
}

int WebGpuContext::sphereContactFinalPositionAsyncReadbackBodyCountValue() const
{
    return sphereContactFinalPositionAsyncReadbackBodyCount;
}

int WebGpuContext::sphereContactFinalPositionAsyncReadbackByteCount() const
{
    return sphereContactFinalPositionAsyncReadbackBytes;
}

const char *WebGpuContext::sphereContactFinalPositionAsyncReadbackSourceText() const
{
    return sphereContactFinalPositionSourceName(sphereContactFinalPositionAsyncReadbackSource);
}

int WebGpuContext::sphereGroundReceiverCountValue() const
{
    return sphereGroundReceiverCount;
}

int WebGpuContext::sphereGroundDynamicSphereCountValue() const
{
    return sphereGroundDynamicSphereCount;
}

int WebGpuContext::sphereGroundCandidateCountValue() const
{
    return sphereGroundCandidateCount;
}

int WebGpuContext::directSphereCylinderBodyCountValue() const
{
    return directSphereCylinderBodyCount;
}

int WebGpuContext::directSphereCylinderCandidateCountValue() const
{
    return directSphereCylinderCandidateCount;
}

int WebGpuContext::directSphereCapsuleBodyCountValue() const
{
    return directSphereCapsuleBodyCount;
}

int WebGpuContext::directSphereCapsuleCandidateCountValue() const
{
    return directSphereCapsuleCandidateCount;
}

int WebGpuContext::directSphereBoxBodyCountValue() const
{
    return directSphereBoxBodyCount;
}

int WebGpuContext::directSphereBoxCandidateCountValue() const
{
    return directSphereBoxCandidateCount;
}

int WebGpuContext::directBoxBodyCountValue() const
{
    return directBoxBodyCount;
}

int WebGpuContext::directBoxPairCandidateCountValue() const
{
    return directBoxPairCandidateCount;
}

int WebGpuContext::directRoundBodyCountValue() const
{
    return directRoundBodyCount;
}

int WebGpuContext::directRoundPairCandidateCountValue() const
{
    return directRoundPairCandidateCount;
}

int WebGpuContext::directGpuContactRecordCountValue() const
{
    return directGpuContactRecordCount;
}

int WebGpuContext::directGpuRoundPairCandidateCountValue() const
{
    return directGpuRoundPairCandidateCount;
}

int WebGpuContext::directGpuBoxPairCandidateCountValue() const
{
    return directGpuBoxPairCandidateCount;
}

int WebGpuContext::directGpuCounterReadbackByteCount() const
{
    return directGpuCounterReadbackBytes;
}

float WebGpuContext::directGpuCounterReadbackMillis() const
{
    return directGpuCounterReadbackMs;
}

int WebGpuContext::directSphereContactAppliedPositionBodyCount() const
{
    return directSphereContactAppliedPositionBodies;
}

int WebGpuContext::directGroundAppliedPositionBodyCount() const
{
    return directGroundAppliedPositionBodies;
}

int WebGpuContext::predictionAppliedBodyCount() const
{
    return predictionAppliedBodies;
}

int WebGpuContext::predictionAppliedReadbackByteCount() const
{
    return predictionAppliedReadbackBytes;
}

float WebGpuContext::predictionAppliedMillis() const
{
    return predictionAppliedMs;
}

int WebGpuContext::velocityAppliedBodyCount() const
{
    return velocityAppliedBodies;
}

int WebGpuContext::velocityAppliedReadbackByteCount() const
{
    return velocityAppliedReadbackBytes;
}

float WebGpuContext::velocityAppliedMillis() const
{
    return velocityAppliedMs;
}

float WebGpuContext::sphereGroundTopValue() const
{
    return sphereGroundTop;
}

float WebGpuContext::sphereGroundMillis() const
{
    return sphereGroundMs;
}

float WebGpuContext::runtimeTotalMillis() const
{
    return runtimeTotalMs;
}

float WebGpuContext::runtimeSyncMillis() const
{
    return runtimeSyncMs;
}

float WebGpuContext::runtimePredictionMillis() const
{
    return runtimePredictionMs;
}

float WebGpuContext::runtimeVelocityMillis() const
{
    return runtimeVelocityMs;
}

float WebGpuContext::runtimeCpuFallbackMillis() const
{
    return runtimeCpuFallbackMs;
}

float WebGpuContext::runtimeMaxLinearErrorValue() const
{
    return runtimeMaxLinearError;
}

float WebGpuContext::runtimeMaxAngularErrorValue() const
{
    return runtimeMaxAngularError;
}

int WebGpuContext::runtimeFrameCount() const
{
    return runtimeFrames;
}

int WebGpuContext::runtimeFallbackCount() const
{
    return runtimeFallbacks;
}

WebGpuPhysicsOptions::WebGpuPhysicsOptions()
    : useSphereContacts(false),
      useGroundContactFeed(false),
      applyContactPositions(false),
      applyGroundContacts(false),
      useResidentGroundContacts(false),
      useCpuFallbackPairs(false),
      useDirectSpherePositionSolve(false),
      skipSapCounterReadback(false),
      validateResidentContacts(true),
      contactSolveDiagnostic(false),
      useJointProposals(false),
      replaceCpuJointSolverWork(false),
      directJointSolveOnly(false),
      directContactSolveOnly(false),
      useResidentPrimitiveContacts(false),
      disableResidentContactReadbacks(false),
      useResidentCounterlessContacts(false),
      asyncFinalPositionReadback(false),
      deferFinalPositionReadback(false),
      residentContactIterations(4),
      residentContactRelaxation(0.10f),
      residentContactMaxUpwardVelocity(0.0f),
      residentContactMaxLinearVelocity(0.0f),
      jointProposalIterations(1)
{
}

WebGpuPhysicsBackend::WebGpuPhysicsBackend(WebGpuContext *context, WebGpuPhysicsOptions options)
    : context(context), options(options)
{
}

const char *WebGpuPhysicsBackend::name() const
{
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (options.applyContactPositions && options.applyGroundContacts && options.useResidentGroundContacts && !options.validateResidentContacts)
        return "WebGPU Physics Experimental Resident Ground Fast";
    if (options.applyContactPositions && options.applyGroundContacts && options.useDirectSpherePositionSolve && !options.validateResidentContacts)
        return "WebGPU Physics Experimental Direct Fast";
    if (options.applyContactPositions && options.applyGroundContacts && options.skipSapCounterReadback && !options.validateResidentContacts)
        return "WebGPU Physics Experimental Counterless Fast";
    if (options.applyContactPositions && options.applyGroundContacts && !options.validateResidentContacts)
        return "WebGPU Physics Experimental Fast";
    if (options.directContactSolveOnly)
    {
        if (options.useJointProposals)
            return "WebGPU Physics Experimental Joint Contact Direct";
        if (options.useResidentPrimitiveContacts)
        {
            if (options.useResidentCounterlessContacts && options.asyncFinalPositionReadback)
                return "WebGPU Physics Experimental Contact Resident Async";
            return "WebGPU Physics Experimental Contact Resident";
        }
        return "WebGPU Physics Experimental Contact Direct";
    }
    if (options.useJointProposals && options.directJointSolveOnly)
        return "WebGPU Physics Experimental Joint Direct";
    if (options.useJointProposals && options.replaceCpuJointSolverWork)
        return "WebGPU Physics Experimental Joint Replace";
    if (options.useJointProposals)
        return "WebGPU Physics Experimental Joint Proposal";
    return "WebGPU Physics Experimental";
#else
    return "WebGPU Physics Unavailable";
#endif
}

static void removeRuntimeManifolds(Solver &solver)
{
    for (Force *force = solver.forces; force != 0;)
    {
        Force *next = force->next;
        if (dynamic_cast<Manifold *>(force))
            delete force;
        force = next;
    }
}

static void countDirectJointRuntimeForces(Solver &solver)
{
    Uint64 phaseBegin = SDL_GetPerformanceCounter();
    for (Force *force = solver.forces; force != 0;)
    {
        Joint *joint = dynamic_cast<Joint *>(force);
        Spring *spring = dynamic_cast<Spring *>(force);
        Manifold *manifold = dynamic_cast<Manifold *>(force);
        IgnoreCollision *ignore = dynamic_cast<IgnoreCollision *>(force);

        if (joint && !isinf(joint->fracture) && !joint->initialize())
        {
            Force *next = force->next;
            delete force;
            force = next;
            continue;
        }

        solver.stats.forceCount++;
        if (joint)
        {
            solver.stats.jointCount++;
            if (isinf(joint->fracture))
                solver.stats.jointInitializationSkipped++;
        }
        else if (spring)
            solver.stats.springCount++;
        else if (manifold)
            solver.stats.manifoldCount++;
        else if (ignore)
        {
            solver.stats.ignoreCollisionCount++;
            solver.stats.ignoreCollisionInitializationSkipped++;
        }

        force = force->next;
    }
    solver.stats.forceInitMs = elapsedMs(phaseBegin, SDL_GetPerformanceCounter());
}

static void initializeDirectJointRuntimeBodies(Solver &solver)
{
    Uint64 phaseBegin = SDL_GetPerformanceCounter();
    for (Rigid *body = solver.bodies; body != 0; body = body->next)
    {
        body->inertialLin = body->positionLin + body->velocityLin * solver.dt;
        if (body->mass > 0)
            body->inertialLin += float3{0, 0, solver.gravity} * (solver.dt * solver.dt);
        body->inertialAng = body->positionAng + body->velocityAng * solver.dt;

        body->initialLin = body->positionLin;
        body->initialAng = body->positionAng;
        if (body->mass > 0)
        {
            body->positionLin = body->inertialLin;
            body->positionAng = body->inertialAng;
        }

        if (body->mass > 0 && body->forces == 0)
        {
            body->positionLin = body->inertialLin;
            body->positionAng = body->inertialAng;
        }
    }
    solver.stats.bodyInitMs = elapsedMs(phaseBegin, SDL_GetPerformanceCounter());
}

static void updateDirectJointRuntimeVelocities(Solver &solver)
{
    Uint64 phaseBegin = SDL_GetPerformanceCounter();
    for (Rigid *body = solver.bodies; body != 0; body = body->next)
    {
        body->prevVelocityLin = body->velocityLin;
        if (body->mass > 0)
        {
            body->velocityLin = (body->positionLin - body->initialLin) / solver.dt;
            body->velocityAng = (body->positionAng - body->initialAng) / solver.dt;
        }
    }
    solver.stats.velocityUpdateMs = elapsedMs(phaseBegin, SDL_GetPerformanceCounter());
}

static void stabilizeResidentContactRuntimeVelocities(Solver &solver, float maxUpwardVelocity, float maxLinearVelocity)
{
    maxUpwardVelocity = environmentFloatValue("AVBD_GPU_CONTACT_MAX_UPWARD_VELOCITY", maxUpwardVelocity);
    maxLinearVelocity = environmentFloatValue("AVBD_GPU_CONTACT_MAX_LINEAR_VELOCITY", maxLinearVelocity);
    if (maxUpwardVelocity <= 0.0f && maxLinearVelocity <= 0.0f)
        return;

    for (Rigid *body = solver.bodies; body != 0; body = body->next)
    {
        if (body->mass <= 0.0f)
            continue;

        if (maxUpwardVelocity > 0.0f && body->velocityLin.z > maxUpwardVelocity)
            body->velocityLin.z = maxUpwardVelocity;

        if (maxLinearVelocity > 0.0f)
        {
            float speed = length(body->velocityLin);
            if (speed > maxLinearVelocity && speed > 0.0f)
                body->velocityLin = body->velocityLin * (maxLinearVelocity / speed);
        }
    }
}

void WebGpuPhysicsBackend::step(Solver &solver)
{
#if AVBD_ENABLE_WEBGPU && AVBD_HAS_DAWN
    if (!context || !context->deviceReady)
    {
        if (context)
        {
            context->runtimeFallbacks++;
            setStatus(context->runtimeStatus, "WebGPU runtime fallback: device not ready");
        }
        solver.stepCpuReference();
        return;
    }

    Uint64 totalBegin = SDL_GetPerformanceCounter();

    Uint64 syncBegin = SDL_GetPerformanceCounter();
    solver.world.syncFromLegacy(solver);
    context->runtimeSyncMs = elapsedMs(syncBegin, SDL_GetPerformanceCounter());
    bool runtimeValidateReadback = environmentFlagEnabled("AVBD_GPU_RUNTIME_VALIDATE");
    bool predictionOk = context->runBodyPredictionDiagnostic(solver.world, solver.dt, solver.gravity, runtimeValidateReadback);
    context->runtimePredictionMs = context->predictionMillis();
    bool directJointSolveOnly = options.directJointSolveOnly || environmentFlagEnabled("AVBD_GPU_JOINT_DIRECT");
    bool jointProposalRuntimeEnabled = options.useJointProposals || environmentFlagEnabled("AVBD_GPU_JOINT_PROPOSALS");
    bool directContactSolveOnly = options.directContactSolveOnly || environmentFlagEnabled("AVBD_GPU_CONTACT_DIRECT");
    bool residentPrimitiveContactSolve = directContactSolveOnly &&
                                         (options.useResidentPrimitiveContacts ||
                                          environmentFlagEnabled("AVBD_GPU_RESIDENT_PRIMITIVE_CONTACTS"));
    bool deferFinalPositionReadback = options.deferFinalPositionReadback || environmentFlagEnabled("AVBD_GPU_DEFER_FINAL_POSITION_READBACK");
    bool asyncFinalPositionReadback = options.asyncFinalPositionReadback || environmentFlagEnabled("AVBD_GPU_ASYNC_FINAL_POSITION_READBACK");
    bool applyPredictionRuntime = environmentFlagEnabled("AVBD_GPU_APPLY_PREDICTION");
    bool applyVelocityRuntime = environmentFlagEnabled("AVBD_GPU_APPLY_VELOCITY");
    context->predictionAppliedBodies = 0;
    context->predictionAppliedReadbackBytes = 0;
    context->predictionAppliedMs = 0.0f;
    context->velocityAppliedBodies = 0;
    context->velocityAppliedReadbackBytes = 0;
    context->velocityAppliedMs = 0.0f;

    if (directContactSolveOnly)
    {
        context->directSphereContactAppliedPositionBodies = 0;
        context->directGroundAppliedPositionBodies = 0;
        solver.stats = SolverStats{};
        solver.stats.simWorldSyncMs = context->runtimeSyncMs;
        solver.stats.spatialHashCellSize = solver.spatialHashCellSize;
        removeRuntimeManifolds(solver);
        for (Rigid *body = solver.bodies; body != 0; body = body->next)
        {
            solver.stats.bodyCount++;
            if (body->mass > 0)
                solver.stats.activeBodyCount++;
        }
        if (jointProposalRuntimeEnabled)
            countDirectJointRuntimeForces(solver);
        bool predictionApplyOk = true;
        if (applyPredictionRuntime && predictionOk)
        {
            predictionApplyOk = context->applyBodyPredictionOutputs(solver);
            solver.stats.bodyInitMs = context->predictionAppliedMillis();
            if (!predictionApplyOk)
                initializeDirectJointRuntimeBodies(solver);
        }
        else
        {
            initializeDirectJointRuntimeBodies(solver);
        }

        bool jointAsyncConsumeOk = true;
        if (asyncFinalPositionReadback && jointProposalRuntimeEnabled)
        {
            bool hadPendingJointReadback = context->jointProposalFinalPositionAsyncReadbackPendingValue() != 0;
            jointAsyncConsumeOk = !hadPendingJointReadback || context->consumePendingJointProposalFinalPositions(solver);
        }

        syncBegin = SDL_GetPerformanceCounter();
        solver.world.syncFromLegacy(solver);
        solver.stats.simWorldSyncMs += elapsedMs(syncBegin, SDL_GetPerformanceCounter());

        bool validateResidentContacts = residentPrimitiveContactSolve &&
                                        options.validateResidentContacts &&
                                        !options.disableResidentContactReadbacks &&
                                        !environmentFlagEnabled("AVBD_GPU_CONTACT_SOLVE_NO_READBACK");
        std::vector<BroadphasePair> ignoredPairs;
        bool directSphereOk = solver.broadphaseMode == BROADPHASE_SWEEP_AND_PRUNE &&
                              context->runSweepAndPrunePairs(solver.world, ignoredPairs, 0,
                                                             false, validateResidentContacts,
                                                             true, true,
                                                             residentPrimitiveContactSolve,
                                                             true,
                                                             true, !residentPrimitiveContactSolve,
                                                             options.residentContactIterations,
                                                             options.residentContactRelaxation,
                                                             options.useResidentCounterlessContacts);
        bool seedJointsFromContactPositions = directSphereOk &&
                                               !residentPrimitiveContactSolve &&
                                               jointProposalRuntimeEnabled &&
                                               !solver.world.jointIds.empty() &&
                                               context->sphereContactFinalPositionReadyValue() != 0;
        bool directSphereApplyOk = directSphereOk;
        if (directSphereOk && context->sphereContactFinalPositionReadyValue() != 0)
        {
            if (deferFinalPositionReadback && residentPrimitiveContactSolve)
            {
                context->sphereContactFinalPositionReadbackDeferred = 1;
                directSphereApplyOk = true;
            }
            else if (asyncFinalPositionReadback && residentPrimitiveContactSolve)
            {
                bool consumeOk = context->consumePendingSphereContactFinalPositions(solver);
                bool scheduleOk = context->scheduleSphereContactFinalPositionReadback();
                directSphereApplyOk = consumeOk && scheduleOk;
                context->directSphereContactAppliedPositionBodies = context->sphereContactAppliedPositionBodyCount();
            }
            else if (seedJointsFromContactPositions)
            {
                // Keep the contact-corrected positions GPU-resident and use
                // them as the starting point for the joint proposal pass.
                directSphereApplyOk = true;
                context->directGpuContactRecordCount = std::max(context->directGpuContactRecordCount,
                                                                context->sphereContactFinalPositionBodyCountValue());
                if (context->directBoxBodyCountValue() > 0)
                    context->directGpuBoxPairCandidateCount = std::max(context->directGpuBoxPairCandidateCount,
                                                                       context->sphereContactFinalPositionBodyCountValue());
            }
            else
            {
                directSphereApplyOk = context->applySphereContactFinalPositions(solver);
                context->directSphereContactAppliedPositionBodies = context->sphereContactAppliedPositionBodyCount();
            }
        }

        bool directGroundOk = true;
        bool directGroundApplyOk = true;
        if (!residentPrimitiveContactSolve && !seedJointsFromContactPositions)
        {
            directGroundOk = context->runSphereGroundCorrection(solver.world);
            directGroundApplyOk = directGroundOk;
            if (directGroundOk && context->sphereGroundCandidateCountValue() > 0)
            {
                directGroundApplyOk = context->applySphereContactFinalPositions(solver);
                context->directGroundAppliedPositionBodies = context->sphereContactAppliedPositionBodyCount();
            }
        }
        else if (residentPrimitiveContactSolve)
        {
            directGroundOk = context->runSphereGroundCorrection(solver.world, false);
            directGroundApplyOk = directGroundOk;
            if (directGroundOk && context->sphereGroundCandidateCountValue() > 0)
            {
                directGroundApplyOk = context->applySphereContactFinalPositions(solver);
                context->directGroundAppliedPositionBodies = context->sphereContactAppliedPositionBodyCount();
            }
        }

        bool jointProposalRuntimeOk = true;
        bool jointProposalRuntimeApplyOk = true;
        if (jointProposalRuntimeEnabled && !solver.world.jointIds.empty())
        {
            int proposalIterations = environmentIntValue("AVBD_GPU_JOINT_PROPOSAL_ITERATIONS", options.jointProposalIterations);
            proposalIterations = std::max(1, std::min(proposalIterations, 16));
            jointProposalRuntimeOk = context->runJointTopologyDiagnostic(solver.world, proposalIterations, false, seedJointsFromContactPositions);
            if (asyncFinalPositionReadback)
                jointProposalRuntimeApplyOk = jointProposalRuntimeOk && jointAsyncConsumeOk &&
                                              context->scheduleJointProposalFinalPositionReadback();
            else
                jointProposalRuntimeApplyOk = jointProposalRuntimeOk && context->applyJointProposalFinalPositions(solver);
        }
        else if (jointProposalRuntimeEnabled)
        {
            setStatus(context->jointTopologyStatus, "Joint contact direct runtime skipped: no joints");
        }

        if (!residentPrimitiveContactSolve && seedJointsFromContactPositions)
        {
            directGroundOk = context->runSphereGroundCorrection(solver.world);
            directGroundApplyOk = directGroundOk;
            if (directGroundOk && context->sphereGroundCandidateCountValue() > 0)
            {
                directGroundApplyOk = context->applySphereContactFinalPositions(solver);
                context->directGroundAppliedPositionBodies = context->sphereContactAppliedPositionBodyCount();
            }
        }

        updateDirectJointRuntimeVelocities(solver);
        if (residentPrimitiveContactSolve)
            stabilizeResidentContactRuntimeVelocities(solver,
                                                       options.residentContactMaxUpwardVelocity,
                                                       options.residentContactMaxLinearVelocity);
        syncBegin = SDL_GetPerformanceCounter();
        solver.world.syncFromLegacy(solver);
        solver.stats.simWorldSyncMs += elapsedMs(syncBegin, SDL_GetPerformanceCounter());

        bool velocityOk = context->runVelocityUpdateDiagnostic(solver.world, solver.dt, runtimeValidateReadback);
        bool velocityApplyOk = true;
        if (applyVelocityRuntime && velocityOk)
        {
            velocityApplyOk = context->applyVelocityOutputs(solver);
            solver.stats.velocityUpdateMs = context->velocityAppliedMillis();
        }
        context->runtimeVelocityMs = context->velocityMillis();
        context->runtimeCpuFallbackMs = 0.0f;
        context->runtimeMaxLinearError = std::max(context->predictionMaxErrorValue(), context->velocityMaxLinearErrorValue());
        context->runtimeMaxAngularError = std::max(context->predictionMaxAngularErrorValue(), context->velocityMaxAngularErrorValue());
        context->runtimeTotalMs = elapsedMs(totalBegin, SDL_GetPerformanceCounter());
        context->runtimeFrames++;

        snprintf(context->runtimeStatus, sizeof(context->runtimeStatus),
                 "WebGPU %s contact runtime %s: prediction %s, contacts %s%s%s, ground %s%s%s, velocity %s, no CPU fallback solve",
                 residentPrimitiveContactSolve ? "resident" : "direct",
                 (predictionOk && predictionApplyOk && directSphereApplyOk && directGroundApplyOk && jointProposalRuntimeOk && jointProposalRuntimeApplyOk && velocityOk && velocityApplyOk) ? "passed" : "failed",
                 predictionOk ? "passed" : "failed",
                 directSphereApplyOk ? (applyPredictionRuntime ? "passed, prediction-applied" : "passed") : "failed",
                 context->sphereContactFinalPositionReadbackDeferred ? " (final readback deferred)" : "",
                 context->sphereContactFinalPositionAsyncReadbackScheduled ? " (async final readback scheduled)" : "",
                 directGroundApplyOk ? "passed" : "failed",
                 jointProposalRuntimeEnabled ? ", joints " : "",
                 jointProposalRuntimeEnabled ? ((jointProposalRuntimeOk && jointProposalRuntimeApplyOk) ? "passed" : "failed") : "",
                 velocityOk ? (applyVelocityRuntime ? (velocityApplyOk ? "passed, applied" : "apply failed") : "passed") : "failed");
        return;
    }

    if (directJointSolveOnly && jointProposalRuntimeEnabled)
    {
        solver.stats = SolverStats{};
        solver.stats.simWorldSyncMs = context->runtimeSyncMs;
        solver.stats.spatialHashCellSize = solver.spatialHashCellSize;
        removeRuntimeManifolds(solver);
        for (Rigid *body = solver.bodies; body != 0; body = body->next)
        {
            solver.stats.bodyCount++;
            if (body->mass > 0)
                solver.stats.activeBodyCount++;
        }
        countDirectJointRuntimeForces(solver);
        bool predictionApplyOk = true;
        if (applyPredictionRuntime && predictionOk)
        {
            predictionApplyOk = context->applyBodyPredictionOutputs(solver);
            solver.stats.bodyInitMs = context->predictionAppliedMillis();
            if (!predictionApplyOk)
                initializeDirectJointRuntimeBodies(solver);
        }
        else
        {
            initializeDirectJointRuntimeBodies(solver);
        }

        bool jointAsyncConsumeOk = true;
        if (asyncFinalPositionReadback)
        {
            bool hadPendingJointReadback = context->jointProposalFinalPositionAsyncReadbackPendingValue() != 0;
            jointAsyncConsumeOk = !hadPendingJointReadback || context->consumePendingJointProposalFinalPositions(solver);
        }

        syncBegin = SDL_GetPerformanceCounter();
        solver.world.syncFromLegacy(solver);
        solver.stats.simWorldSyncMs += elapsedMs(syncBegin, SDL_GetPerformanceCounter());

        bool jointProposalRuntimeOk = false;
        bool jointProposalRuntimeApplyOk = false;
        if (!solver.world.jointIds.empty())
        {
            int proposalIterations = environmentIntValue("AVBD_GPU_JOINT_PROPOSAL_ITERATIONS", options.jointProposalIterations);
            proposalIterations = std::max(1, std::min(proposalIterations, 16));
            jointProposalRuntimeOk = context->runJointTopologyDiagnostic(solver.world, proposalIterations, false);
            if (jointProposalRuntimeOk)
            {
                if (asyncFinalPositionReadback)
                    jointProposalRuntimeApplyOk = jointAsyncConsumeOk &&
                                                  context->scheduleJointProposalFinalPositionReadback();
                else
                    jointProposalRuntimeApplyOk = context->applyJointProposalFinalPositions(solver);
            }
        }
        else
        {
            setStatus(context->jointTopologyStatus, "Joint direct runtime skipped: no joints");
        }

        bool directGroundOk = context->runSphereGroundCorrection(solver.world);
        bool directGroundApplyOk = directGroundOk;
        if (directGroundOk && context->sphereGroundCandidateCountValue() > 0)
            directGroundApplyOk = context->applySphereContactFinalPositions(solver);

        updateDirectJointRuntimeVelocities(solver);
        syncBegin = SDL_GetPerformanceCounter();
        solver.world.syncFromLegacy(solver);
        solver.stats.simWorldSyncMs += elapsedMs(syncBegin, SDL_GetPerformanceCounter());

        bool velocityOk = context->runVelocityUpdateDiagnostic(solver.world, solver.dt, runtimeValidateReadback);
        bool velocityApplyOk = true;
        if (applyVelocityRuntime && velocityOk)
        {
            velocityApplyOk = context->applyVelocityOutputs(solver);
            solver.stats.velocityUpdateMs = context->velocityAppliedMillis();
        }
        context->runtimeVelocityMs = context->velocityMillis();
        context->runtimeCpuFallbackMs = 0.0f;
        context->runtimeMaxLinearError = std::max(context->predictionMaxErrorValue(), context->velocityMaxLinearErrorValue());
        context->runtimeMaxAngularError = std::max(context->predictionMaxAngularErrorValue(), context->velocityMaxAngularErrorValue());
        context->runtimeTotalMs = elapsedMs(totalBegin, SDL_GetPerformanceCounter());
        context->runtimeFrames++;

        snprintf(context->runtimeStatus, sizeof(context->runtimeStatus),
                 "WebGPU direct joint runtime %s: prediction %s, joint proposals %s, ground %s, velocity %s, no CPU fallback solve",
                 (predictionOk && predictionApplyOk && jointProposalRuntimeOk && jointProposalRuntimeApplyOk && directGroundApplyOk && velocityOk && velocityApplyOk) ? "passed" : "failed",
                 predictionOk ? (applyPredictionRuntime ? "passed, applied" : "passed") : "failed",
                 (jointProposalRuntimeOk && jointProposalRuntimeApplyOk) ? "passed" : "failed",
                 directGroundApplyOk ? "passed" : "failed",
                 velocityOk ? (applyVelocityRuntime ? (velocityApplyOk ? "passed, applied" : "apply failed") : "passed") : "failed");
        return;
    }

    std::vector<BroadphasePair> gpuPairs;
    std::vector<ExternalManifoldContact> gpuContacts;
    bool useGpuSphereContacts = options.useSphereContacts || environmentFlagEnabled("AVBD_GPU_SPHERE_CONTACTS");
    bool useGpuGroundContactFeed = options.useGroundContactFeed || environmentFlagEnabled("AVBD_GPU_GROUND_CONTACT_FEED");
    bool applyGpuContactPositions = (options.applyContactPositions || environmentFlagEnabled("AVBD_GPU_APPLY_CONTACT_POSITIONS")) && !useGpuSphereContacts;
    bool applyGpuGroundContacts = applyGpuContactPositions && (options.applyGroundContacts || environmentFlagEnabled("AVBD_GPU_GROUND_CONTACTS"));
    bool useResidentGroundContacts = applyGpuGroundContacts &&
                                     (options.useResidentGroundContacts || environmentFlagEnabled("AVBD_GPU_RESIDENT_GROUND_CONTACTS"));
    bool useCpuFallbackPairs = applyGpuContactPositions &&
                               (options.useCpuFallbackPairs || environmentFlagEnabled("AVBD_GPU_CPU_FALLBACK_PAIRS")) &&
                               shouldUseCpuFallbackPairsForGpuContacts(solver.world);
    bool useDirectSpherePositionSolve = useCpuFallbackPairs &&
                                        (options.useDirectSpherePositionSolve || environmentFlagEnabled("AVBD_GPU_DIRECT_SPHERE_SOLVE"));
    bool skipSapCounterReadback = useCpuFallbackPairs &&
                                  !useDirectSpherePositionSolve &&
                                  (options.skipSapCounterReadback || environmentFlagEnabled("AVBD_GPU_SKIP_SAP_COUNTER_READBACK"));
    bool validateResidentContacts = options.validateResidentContacts &&
                                    !options.disableResidentContactReadbacks &&
                                    !environmentFlagEnabled("AVBD_GPU_CONTACT_SOLVE_NO_READBACK");
    bool gpuBroadphaseOk = false;
    if (solver.broadphaseMode == BROADPHASE_SWEEP_AND_PRUNE && useDirectSpherePositionSolve)
    {
        appendCpuFallbackPairsForGpuAppliedContacts(solver.world, gpuPairs, applyGpuGroundContacts);
        gpuBroadphaseOk = true;
    }
    else if (solver.broadphaseMode == BROADPHASE_SWEEP_AND_PRUNE && skipSapCounterReadback)
    {
        appendCpuFallbackPairsForGpuAppliedContacts(solver.world, gpuPairs, applyGpuGroundContacts);
        context->sapMs = 0.0f;
        context->sapCandidates = 0;
        context->sapSphereHits = 0;
        context->sapAllPairsSphereHits = 0;
        context->sapMissedSphereHits = 0;
        context->sapMismatches = 0;
        context->sapCounterReadbackBytes = 0;
        context->sapCounterReadbackMs = 0.0f;
        context->sapPairReadbackBytes = 0;
        context->sapPairReadbackMs = 0.0f;
        setStatus(context->sapStatus, "SAP skipped: counterless CPU fallback pair path");
        gpuBroadphaseOk = true;
    }
    else
    {
        gpuBroadphaseOk = solver.broadphaseMode == BROADPHASE_SWEEP_AND_PRUNE &&
                           context->runSweepAndPrunePairs(solver.world, gpuPairs, (useGpuSphereContacts || useGpuGroundContactFeed) ? &gpuContacts : 0,
                                                          false, validateResidentContacts, applyGpuContactPositions || useGpuSphereContacts,
                                                          applyGpuGroundContacts, useGpuGroundContactFeed || useResidentGroundContacts,
                                                          useGpuGroundContactFeed || useResidentGroundContacts, useCpuFallbackPairs);
    }
    bool gpuContactsOk = gpuBroadphaseOk && (useGpuSphereContacts || useGpuGroundContactFeed);
    bool gpuFinalPositionsReady = context->sphereContactFinalPositionReadyValue() != 0;
    if (gpuBroadphaseOk && applyGpuContactPositions && !useDirectSpherePositionSolve)
        removeGpuAppliedContactPairs(solver.world, gpuPairs, applyGpuGroundContacts && gpuFinalPositionsReady);
    if (gpuBroadphaseOk && applyGpuContactPositions && useCpuFallbackPairs && !useDirectSpherePositionSolve)
        appendCpuFallbackPairsForGpuAppliedContacts(solver.world, gpuPairs, applyGpuGroundContacts);

    bool contactSolveDiagnosticEnabled = options.contactSolveDiagnostic || environmentFlagEnabled("AVBD_GPU_CONTACT_SOLVE_DIAGNOSTIC");
    bool contactSolveDiagnosticOk = false;
    bool replaceCpuJointSolverWork = jointProposalRuntimeEnabled &&
                                     (options.replaceCpuJointSolverWork || environmentFlagEnabled("AVBD_GPU_REPLACE_CPU_JOINTS"));
    bool previousSkipJointSolverWork = solver.skipJointSolverWork;
    bool previousSkipIgnoreCollisionSolverWork = solver.skipIgnoreCollisionSolverWork;
    bool previousSkipJointInitializationWork = solver.skipJointInitializationWork;
    bool previousSkipIgnoreCollisionInitializationWork = solver.skipIgnoreCollisionInitializationWork;
    if (replaceCpuJointSolverWork)
    {
        solver.skipJointSolverWork = true;
        solver.skipIgnoreCollisionSolverWork = true;
        solver.skipJointInitializationWork = true;
        solver.skipIgnoreCollisionInitializationWork = true;
    }

    Uint64 cpuBegin = SDL_GetPerformanceCounter();
    if (gpuBroadphaseOk && gpuContactsOk)
        solver.stepCpuReferenceWithExternalBroadphase(gpuPairs, gpuContacts, true);
    else if (gpuBroadphaseOk)
        solver.stepCpuReferenceWithExternalBroadphase(gpuPairs, true);
    else
        solver.stepCpuReference(true);
    solver.skipJointSolverWork = previousSkipJointSolverWork;
    solver.skipIgnoreCollisionSolverWork = previousSkipIgnoreCollisionSolverWork;
    solver.skipJointInitializationWork = previousSkipJointInitializationWork;
    solver.skipIgnoreCollisionInitializationWork = previousSkipIgnoreCollisionInitializationWork;
    context->runtimeCpuFallbackMs = elapsedMs(cpuBegin, SDL_GetPerformanceCounter());

    bool applyGpuContactPositionsOk = false;
    if (gpuBroadphaseOk && applyGpuContactPositions && useDirectSpherePositionSolve)
    {
        solver.world.syncFromLegacy(solver);
        std::vector<BroadphasePair> ignoredPairs;
        applyGpuContactPositionsOk = context->runSweepAndPrunePairs(solver.world, ignoredPairs, 0,
                                                                    false, validateResidentContacts,
                                                                    true, applyGpuGroundContacts,
                                                                    false, false, true, true);
        if (applyGpuContactPositionsOk)
            applyGpuContactPositionsOk = context->applySphereContactFinalPositions(solver);
    }
    else if (gpuBroadphaseOk && applyGpuContactPositions)
        applyGpuContactPositionsOk = context->applySphereContactFinalPositions(solver);
    bool applyGpuGroundContactsOk = false;
    if (gpuBroadphaseOk && applyGpuGroundContacts && useResidentGroundContacts)
        applyGpuGroundContactsOk = applyGpuContactPositionsOk;
    else if (gpuBroadphaseOk && applyGpuGroundContacts && context->runSphereGroundCorrection(solver.world))
    {
        if (context->sphereGroundCandidateCountValue() > 0)
            applyGpuGroundContactsOk = context->applySphereContactFinalPositions(solver);
        else
            applyGpuGroundContactsOk = true;
        applyGpuContactPositionsOk = applyGpuContactPositionsOk || applyGpuGroundContactsOk;
    }

    if (contactSolveDiagnosticEnabled && solver.broadphaseMode == BROADPHASE_SWEEP_AND_PRUNE)
    {
        solver.world.syncFromLegacy(solver);
        std::vector<BroadphasePair> diagnosticPairs;
        contactSolveDiagnosticOk = context->runSweepAndPrunePairs(solver.world, diagnosticPairs, 0, true, validateResidentContacts);
    }

    bool jointProposalRuntimeOk = false;
    bool jointProposalRuntimeApplyOk = false;
    if (jointProposalRuntimeEnabled && !solver.world.jointIds.empty())
    {
        int proposalIterations = environmentIntValue("AVBD_GPU_JOINT_PROPOSAL_ITERATIONS", options.jointProposalIterations);
        proposalIterations = std::max(1, std::min(proposalIterations, 16));
        solver.world.syncFromLegacy(solver);
        jointProposalRuntimeOk = context->runJointTopologyDiagnostic(solver.world, proposalIterations, false);
        if (jointProposalRuntimeOk)
            jointProposalRuntimeApplyOk = context->applyJointProposalFinalPositions(solver);
    }
    else if (jointProposalRuntimeEnabled)
    {
        setStatus(context->jointTopologyStatus, "Joint proposal runtime skipped: no joints");
    }
    bool velocityOk = context->runVelocityUpdateDiagnostic(solver.world, solver.dt, runtimeValidateReadback);
    bool velocityApplyOk = true;
    if (applyVelocityRuntime && velocityOk)
    {
        velocityApplyOk = context->applyVelocityOutputs(solver);
        solver.stats.velocityUpdateMs = context->velocityAppliedMillis();
    }
    context->runtimeVelocityMs = context->velocityMillis();
    context->runtimeMaxLinearError = context->velocityMaxLinearErrorValue();
    context->runtimeMaxAngularError = context->velocityMaxAngularErrorValue();
    if (context->predictionMaxErrorValue() > context->runtimeMaxLinearError)
        context->runtimeMaxLinearError = context->predictionMaxErrorValue();
    if (context->predictionMaxAngularErrorValue() > context->runtimeMaxAngularError)
        context->runtimeMaxAngularError = context->predictionMaxAngularErrorValue();

    context->runtimeTotalMs = elapsedMs(totalBegin, SDL_GetPerformanceCounter());
    context->runtimeFrames++;

    if (predictionOk && velocityOk && velocityApplyOk)
    {
        snprintf(context->runtimeStatus, sizeof(context->runtimeStatus),
                 "WebGPU runtime spine passed: prediction + %s broadphase + %s sphere contacts + velocity%s, CPU fallback solve%s%s%s%s%s%s",
                 gpuBroadphaseOk ? "GPU SAP" : "CPU",
                 gpuContactsOk ? "GPU experimental" : "CPU",
                 applyVelocityRuntime ? " applied" : "",
                 contactSolveDiagnosticEnabled ? ", contact solve diagnostic " : "",
                 contactSolveDiagnosticEnabled ? (contactSolveDiagnosticOk ? "passed" : "failed") : "",
                 applyGpuContactPositions ? ", apply positions " : "",
                 applyGpuContactPositions ? (applyGpuContactPositionsOk ? "passed" : "failed") : "",
                 jointProposalRuntimeEnabled ? ", joint proposals " : "",
                 jointProposalRuntimeEnabled ? ((jointProposalRuntimeOk && jointProposalRuntimeApplyOk) ? "passed" : "failed") : "");
    }
    else
    {
        snprintf(context->runtimeStatus, sizeof(context->runtimeStatus),
                 "WebGPU runtime validation failed: prediction %s, broadphase %s, velocity %s, CPU fallback solve%s%s%s%s%s%s",
                 predictionOk ? "passed" : "failed",
                 gpuBroadphaseOk ? "GPU SAP" : "CPU",
                 velocityOk ? (applyVelocityRuntime ? (velocityApplyOk ? "passed, applied" : "apply failed") : "passed") : "failed",
                 contactSolveDiagnosticEnabled ? ", contact solve diagnostic " : "",
                 contactSolveDiagnosticEnabled ? (contactSolveDiagnosticOk ? "passed" : "failed") : "",
                 applyGpuContactPositions ? ", apply positions " : "",
                 applyGpuContactPositions ? (applyGpuContactPositionsOk ? "passed" : "failed") : "",
                 jointProposalRuntimeEnabled ? ", joint proposals " : "",
                 jointProposalRuntimeEnabled ? ((jointProposalRuntimeOk && jointProposalRuntimeApplyOk) ? "passed" : "failed") : "");
    }
#else
    solver.stepCpuReference();
#endif
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
