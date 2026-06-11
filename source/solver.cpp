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

#include "parallel_for.h"
#include "solver.h"

#include <algorithm>
#include <atomic>
#include <execution>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cfloat>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{
const float SPATIAL_HASH_CELL_SIZE = 2.0f;

using Clock = std::chrono::high_resolution_clock;

struct SapInterval
{
    float minAxis;
    float maxAxis;
    int bodyIndex;
    float3 center;
    float radius;
};

uint64_t exactBodyPairKey(BodyId a, BodyId b)
{
    return ((uint64_t)a << 32) | (uint64_t)b;
}

float elapsedMs(Clock::time_point begin, Clock::time_point end)
{
    return (float)std::chrono::duration<double, std::milli>(end - begin).count();
}

void accumulatePrimalForceStats(Solver *solver, Force *force, float elapsed)
{
    if (force->type == SIM_CONSTRAINT_JOINT)
    {
        solver->stats.primalJointVisits++;
        solver->stats.primalJointMs += elapsed;
    }
    else if (force->type == SIM_CONSTRAINT_SPRING)
    {
        solver->stats.primalSpringVisits++;
        solver->stats.primalSpringMs += elapsed;
    }
    else if (force->type == SIM_CONSTRAINT_MANIFOLD)
    {
        solver->stats.primalManifoldVisits++;
        solver->stats.primalManifoldMs += elapsed;
    }
    else if (force->type == SIM_CONSTRAINT_IGNORE_COLLISION)
    {
        solver->stats.primalIgnoreCollisionVisits++;
        solver->stats.primalIgnoreCollisionMs += elapsed;
    }
}

void accumulateDualForceStats(Solver *solver, Force *force, float elapsed)
{
    if (force->type == SIM_CONSTRAINT_JOINT)
    {
        solver->stats.dualJointVisits++;
        solver->stats.dualJointMs += elapsed;
    }
    else if (force->type == SIM_CONSTRAINT_SPRING)
    {
        solver->stats.dualSpringVisits++;
        solver->stats.dualSpringMs += elapsed;
    }
    else if (force->type == SIM_CONSTRAINT_MANIFOLD)
    {
        solver->stats.dualManifoldVisits++;
        solver->stats.dualManifoldMs += elapsed;
    }
    else if (force->type == SIM_CONSTRAINT_IGNORE_COLLISION)
    {
        solver->stats.dualIgnoreCollisionVisits++;
        solver->stats.dualIgnoreCollisionMs += elapsed;
    }
}

void countForceForStats(Solver *solver, Force *force)
{
    solver->stats.forceCount++;
    if (force->type == SIM_CONSTRAINT_JOINT)
        solver->stats.jointCount++;
    else if (force->type == SIM_CONSTRAINT_SPRING)
        solver->stats.springCount++;
    else if (force->type == SIM_CONSTRAINT_MANIFOLD)
        solver->stats.manifoldCount++;
    else if (force->type == SIM_CONSTRAINT_IGNORE_COLLISION)
        solver->stats.ignoreCollisionCount++;
}

// Pairs the GPU narrowphase understands: sphere-sphere, and a dynamic
// sphere against a static box or cylinder.
bool isGpuNarrowphasePair(Rigid *bodyA, Rigid *bodyB)
{
    bool sphereA = bodyA->shape.type == RIGID_SHAPE_SPHERE;
    bool sphereB = bodyB->shape.type == RIGID_SHAPE_SPHERE;
    if (sphereA && sphereB)
        return true;
    Rigid *sphere = sphereA ? bodyA : (sphereB ? bodyB : 0);
    Rigid *other = sphereA ? bodyB : bodyA;
    if (!sphere || sphere->mass <= 0.0f || other->mass > 0.0f)
        return false;
    return other->shape.type == RIGID_SHAPE_BOX || other->shape.type == RIGID_SHAPE_CYLINDER;
}

// Handles a pair whose bounding spheres already overlap: checks existing
// constraints and allocates a manifold. Mutates solver state, so callers
// must invoke it serially.
void processSphereHit(Solver *solver, Rigid *bodyA, Rigid *bodyB)
{
    solver->stats.sphereHits++;
    solver->stats.constrainedChecks++;
    Clock::time_point constrainedBegin;
    if (solver->deepProfiling)
        constrainedBegin = Clock::now();
    bool constrained = bodyA->attachedForceCount <= bodyB->attachedForceCount
                           ? bodyA->constrainedTo(bodyB)
                           : bodyB->constrainedTo(bodyA);
    if (solver->deepProfiling)
        solver->stats.constrainedMs += elapsedMs(constrainedBegin, Clock::now());

    if (constrained)
        solver->stats.constrainedHits++;
    else if (solver->spherePairSink && isGpuNarrowphasePair(bodyA, bodyB))
    {
        // Narrowphase and contact warmstarting for these pairs run on the
        // GPU; no CPU manifold is allocated.
        solver->spherePairSink->push_back({bodyA, bodyB});
        bodyA->gpuPairCount++;
        bodyB->gpuPairCount++;
    }
    else
    {
        Clock::time_point manifoldBegin;
        if (solver->deepProfiling)
            manifoldBegin = Clock::now();
        new Manifold(solver, bodyA, bodyB);
        if (solver->deepProfiling)
            solver->stats.manifoldAllocMs += elapsedMs(manifoldBegin, Clock::now());
        solver->stats.manifoldsCreated++;
    }
}

void generatePair(Solver *solver, Rigid *bodyA, Rigid *bodyB)
{
    solver->stats.pairChecks++;

    float3 dp = bodyA->positionLin - bodyB->positionLin;
    float r = bodyA->radius + bodyB->radius;
    if (dot(dp, dp) <= r * r)
        processSphereHit(solver, bodyA, bodyB);
}

void broadphaseAllPairs(Solver *solver)
{
    Clock::time_point candidateBegin = Clock::now();
    for (Rigid *bodyA = solver->bodies; bodyA != 0; bodyA = bodyA->next)
    {
        for (Rigid *bodyB = bodyA->next; bodyB != 0; bodyB = bodyB->next)
            generatePair(solver, bodyA, bodyB);
    }
    solver->stats.spatialHashCandidateMs = elapsedMs(candidateBegin, Clock::now());
}

// Per-chunk output for parallel broadphase pair generation. Pair tests and
// routing run lock-free per chunk; solver-state mutation happens in a serial
// pass over the chunk outputs so creation order stays deterministic.
struct BroadphaseChunkOut
{
    std::vector<std::pair<int, int>> sinkPairs;     // GPU narrowphase pairs
    std::vector<std::pair<int, int>> manifoldPairs; // need CPU manifolds
    int checks = 0;
    int hits = 0;
    int constrainedHits = 0;
};

// Classify a bounding-sphere hit inside a parallel chunk. Mirrors
// processSphereHit but only reads solver state; mutation is deferred to
// applyBroadphaseChunks.
inline void classifyHit(Solver *solver, BroadphaseChunkOut &out, Rigid *bodyA, Rigid *bodyB, int indexA, int indexB)
{
    out.hits++;
    bool constrained = bodyA->attachedForceCount <= bodyB->attachedForceCount
                           ? bodyA->constrainedTo(bodyB)
                           : bodyB->constrainedTo(bodyA);
    if (constrained)
        out.constrainedHits++;
    else if (solver->spherePairSink && isGpuNarrowphasePair(bodyA, bodyB))
        out.sinkPairs.push_back({indexA, indexB});
    else
        out.manifoldPairs.push_back({indexA, indexB});
}

// Serial pass: solver-state mutation (sink bookkeeping, manifold allocation).
// Chunk-ordered iteration keeps creation order deterministic. When the chunks
// did not classify (deep profiling), every recorded pair goes through
// processSphereHit instead so the per-phase counters stay meaningful.
void applyBroadphaseChunks(Solver *solver, const std::vector<Rigid *> &bodies, std::vector<BroadphaseChunkOut> &chunkOut, bool classified)
{
    for (BroadphaseChunkOut &out : chunkOut)
    {
        solver->stats.pairChecks += out.checks;
        if (!classified)
        {
            for (const std::pair<int, int> &pair : out.manifoldPairs)
                processSphereHit(solver, bodies[pair.first], bodies[pair.second]);
            continue;
        }
        solver->stats.sphereHits += out.hits;
        solver->stats.constrainedChecks += out.hits;
        solver->stats.constrainedHits += out.constrainedHits;
        for (const std::pair<int, int> &pair : out.sinkPairs)
        {
            Rigid *bodyA = bodies[pair.first];
            Rigid *bodyB = bodies[pair.second];
            solver->spherePairSink->push_back({bodyA, bodyB});
            bodyA->gpuPairCount++;
            bodyB->gpuPairCount++;
        }
        for (const std::pair<int, int> &pair : out.manifoldPairs)
        {
            new Manifold(solver, bodies[pair.first], bodies[pair.second]);
            solver->stats.manifoldsCreated++;
        }
    }
}

struct GridEntry
{
    uint32_t key;
    int bodyIndex;
    float3 center;
    float radius;
    float slack; // per-step travel allowance, |v| * dt
};

// A body too big for the grid, tested brute-force against everything. Boxes
// carry a tight world AABB so a huge ground slab does not pair with every
// body in the scene the way its bounding sphere would; other shapes fall
// back to bounding-sphere extents.
struct LargeBody
{
    int bodyIndex;
    float3 aabbMin;
    float3 aabbMax;
    float slack;
};

// Persistent cell-start table for the uniform-grid broadphase. Every used
// entry is restored to UINT32_MAX at the end of the frame, so the grid origin
// and dimensions are free to change between frames. Sharing the scratch
// across solver instances is fine: steps never run concurrently and nothing
// here carries meaning across calls.
std::vector<uint32_t> gridCellStart;

// Uniform-grid broadphase. Bodies whose bounding radius fits in half a cell
// bin by center, so any overlapping binned pair (rA + rB <= cellSize) is
// found within the 27-cell neighborhood; sorting by cell key makes each
// cell's occupants contiguous and the forward half-neighborhood visits each
// unordered pair exactly once, with no dedup set. The few bodies larger than
// that (ground, container statics) are tested brute-force against everything.
void broadphaseUniformGrid(Solver *solver)
{
    Clock::time_point buildBegin = Clock::now();

    std::vector<Rigid *> bodies;
    for (Rigid *body = solver->bodies; body != 0; body = body->next)
        bodies.push_back(body);
    size_t n = bodies.size();
    if (n < 2)
        return;

    // Cell size: twice the 95th-percentile bounding radius, so at least ~95%
    // of bodies bin into the grid.
    std::vector<float> radii(n);
    for (size_t i = 0; i < n; ++i)
        radii[i] = bodies[i]->radius;
    size_t pct = min(n - 1, n * 95 / 100);
    std::nth_element(radii.begin(), radii.begin() + pct, radii.end());
    float binnedRadius = max(radii[pct], 1e-3f);
    float cellSize = binnedRadius * 2.0f;

    std::vector<LargeBody> largeBodies;
    std::vector<GridEntry> entries;
    entries.reserve(n);
    float3 boundsMin = {FLT_MAX, FLT_MAX, FLT_MAX};
    float3 boundsMax = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (int i = 0; i < (int)n; ++i)
    {
        Rigid *body = bodies[i];
        float slack = length(body->velocityLin) * solver->dt;
        if (body->radius > binnedRadius)
        {
            float3 h = {body->radius, body->radius, body->radius};
            if (body->shape.type == RIGID_SHAPE_BOX)
            {
                float3 e = body->shape.size * 0.5f;
                float3 cx = rotate(body->positionAng, float3{1, 0, 0});
                float3 cy = rotate(body->positionAng, float3{0, 1, 0});
                float3 cz = rotate(body->positionAng, float3{0, 0, 1});
                h = {fabsf(cx.x) * e.x + fabsf(cy.x) * e.y + fabsf(cz.x) * e.z,
                     fabsf(cx.y) * e.x + fabsf(cy.y) * e.y + fabsf(cz.y) * e.z,
                     fabsf(cx.z) * e.x + fabsf(cy.z) * e.y + fabsf(cz.z) * e.z};
            }
            largeBodies.push_back(LargeBody{i, body->positionLin - h, body->positionLin + h, slack});
            continue;
        }
        float3 c = body->positionLin;
        boundsMin = {min(boundsMin.x, c.x), min(boundsMin.y, c.y), min(boundsMin.z, c.z)};
        boundsMax = {max(boundsMax.x, c.x), max(boundsMax.y, c.y), max(boundsMax.z, c.z)};
        entries.push_back(GridEntry{0, i, c, body->radius, slack});
    }
    size_t binned = entries.size();
    solver->stats.spatialHashGlobalBodies = (int)largeBodies.size();

    int nx = 1, ny = 1, nz = 1;
    if (binned > 0)
    {
        // Cap the cell count by enlarging cells; correctness only needs
        // rA + rB <= cellSize for binned bodies, which growing preserves.
        const uint64_t maxCells = 1ull << 22;
        float3 extent = boundsMax - boundsMin;
        auto dimsFor = [&](float cs) -> uint64_t
        {
            nx = (int)(extent.x / cs) + 1;
            ny = (int)(extent.y / cs) + 1;
            nz = (int)(extent.z / cs) + 1;
            return (uint64_t)nx * (uint64_t)ny * (uint64_t)nz;
        };
        uint64_t numCells = dimsFor(cellSize);
        while (numCells > maxCells)
        {
            cellSize *= max(cbrtf((float)numCells / (float)maxCells), 1.1f);
            numCells = dimsFor(cellSize);
        }

        for (GridEntry &entry : entries)
        {
            int ix = (int)clamp((entry.center.x - boundsMin.x) / cellSize, 0.0f, (float)(nx - 1));
            int iy = (int)clamp((entry.center.y - boundsMin.y) / cellSize, 0.0f, (float)(ny - 1));
            int iz = (int)clamp((entry.center.z - boundsMin.z) / cellSize, 0.0f, (float)(nz - 1));
            entry.key = (uint32_t)(ix + nx * (iy + ny * iz));
        }
        std::sort(std::execution::par_unseq, entries.begin(), entries.end(),
                  [](const GridEntry &a, const GridEntry &b)
                  {
                      if (a.key == b.key)
                          return a.bodyIndex < b.bodyIndex;
                      return a.key < b.key;
                  });

        if (gridCellStart.size() < numCells)
            gridCellStart.resize(numCells, UINT32_MAX);
        int occupied = 0;
        for (size_t i = 0; i < binned; ++i)
        {
            if (i == 0 || entries[i].key != entries[i - 1].key)
            {
                gridCellStart[entries[i].key] = (uint32_t)i;
                occupied++;
            }
        }
        solver->stats.spatialHashOccupiedCells = occupied;
    }
    solver->stats.spatialHashCellSize = cellSize;
    solver->stats.spatialHashBuildMs = elapsedMs(buildBegin, Clock::now());

    Clock::time_point candidateBegin = Clock::now();
    const size_t chunkSize = 512;
    size_t numChunks = (binned + chunkSize - 1) / chunkSize;
    std::vector<BroadphaseChunkOut> chunkOut(numChunks);
    bool classifyInChunk = !solver->deepProfiling;

    // Forward half of the 27-cell neighborhood: each unordered pair of
    // adjacent cells is visited from exactly one side.
    static const int forward[13][3] = {
        {1, 0, 0},
        {-1, 1, 0}, {0, 1, 0}, {1, 1, 0},
        {-1, -1, 1}, {0, -1, 1}, {1, -1, 1},
        {-1, 0, 1}, {0, 0, 1}, {1, 0, 1},
        {-1, 1, 1}, {0, 1, 1}, {1, 1, 1}};

    auto pairChunk = [&](size_t c)
    {
        size_t begin = c * chunkSize;
        size_t end = begin + chunkSize < binned ? begin + chunkSize : binned;
        BroadphaseChunkOut &out = chunkOut[c];
        for (size_t i = begin; i < end; ++i)
        {
            const GridEntry &a = entries[i];
            auto testEntry = [&](const GridEntry &b)
            {
                out.checks++;
                float3 dp = a.center - b.center;
                float r = a.radius + b.radius;
                if (dot(dp, dp) > r * r)
                    return;
                if (!classifyInChunk)
                    out.manifoldPairs.push_back({a.bodyIndex, b.bodyIndex});
                else
                    classifyHit(solver, out, bodies[a.bodyIndex], bodies[b.bodyIndex], a.bodyIndex, b.bodyIndex);
            };

            // Rest of own cell (occupants are contiguous after the sort).
            for (size_t j = i + 1; j < binned && entries[j].key == a.key; ++j)
                testEntry(entries[j]);

            int ix = (int)(a.key % (uint32_t)nx);
            uint32_t t = a.key / (uint32_t)nx;
            int iy = (int)(t % (uint32_t)ny);
            int iz = (int)(t / (uint32_t)ny);
            for (const int *d : forward)
            {
                int cx = ix + d[0];
                int cy = iy + d[1];
                int cz = iz + d[2];
                if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz)
                    continue;
                uint32_t nkey = (uint32_t)(cx + nx * (cy + ny * cz));
                uint32_t s = gridCellStart[nkey];
                if (s == UINT32_MAX)
                    continue;
                for (size_t j = s; j < binned && entries[j].key == nkey; ++j)
                    testEntry(entries[j]);
            }

            for (const LargeBody &large : largeBodies)
            {
                out.checks++;
                float3 q = {clamp(a.center.x, large.aabbMin.x, large.aabbMax.x),
                            clamp(a.center.y, large.aabbMin.y, large.aabbMax.y),
                            clamp(a.center.z, large.aabbMin.z, large.aabbMax.z)};
                float3 dp = a.center - q;
                float reach = a.radius + a.slack + large.slack + 4.0f * COLLISION_MARGIN;
                if (dot(dp, dp) > reach * reach)
                    continue;
                if (!classifyInChunk)
                    out.manifoldPairs.push_back({a.bodyIndex, large.bodyIndex});
                else
                    classifyHit(solver, out, bodies[a.bodyIndex], bodies[large.bodyIndex], a.bodyIndex, large.bodyIndex);
            }
        }
    };

    if (numChunks > 0)
        WorkerPool::instance().parallelFor(numChunks, 1, [&](size_t begin, size_t end)
        {
            for (size_t c = begin; c < end; ++c)
                pairChunk(c);
        });

    applyBroadphaseChunks(solver, bodies, chunkOut, classifyInChunk);

    // Large vs large: a handful of statics, serial.
    for (size_t i = 0; i < largeBodies.size(); ++i)
        for (size_t j = i + 1; j < largeBodies.size(); ++j)
            generatePair(solver, bodies[largeBodies[i].bodyIndex], bodies[largeBodies[j].bodyIndex]);

    // Restore the cell table to empty for the next frame.
    for (size_t i = 0; i < binned; ++i)
        gridCellStart[entries[i].key] = UINT32_MAX;

    solver->stats.spatialHashCandidateMs = elapsedMs(candidateBegin, Clock::now());
}

void broadphaseSweepAndPrune(Solver *solver)
{
    std::vector<Rigid *> bodies;
    for (Rigid *body = solver->bodies; body != 0; body = body->next)
        bodies.push_back(body);

    size_t n = bodies.size();
    if (n < 2)
        return;

    // Pick the sweep axis by center variance (largest spread) instead of
    // counting candidates on all three axes, which cost three extra sorts
    // and full sweeps.
    float3 mean = {0, 0, 0};
    for (Rigid *body : bodies)
        mean += body->positionLin;
    mean = mean / (float)n;
    float3 variance = {0, 0, 0};
    for (Rigid *body : bodies)
    {
        float3 d = body->positionLin - mean;
        variance += float3{d.x * d.x, d.y * d.y, d.z * d.z};
    }
    int bestAxis = 0;
    if (variance.y > variance[bestAxis])
        bestAxis = 1;
    if (variance.z > variance[bestAxis])
        bestAxis = 2;

    std::vector<SapInterval> intervals;
    intervals.reserve(n);
    for (int i = 0; i < (int)n; ++i)
    {
        Rigid *body = bodies[i];
        float center = body->positionLin[bestAxis];
        intervals.push_back(SapInterval{center - body->radius, center + body->radius, i, body->positionLin, body->radius});
    }
    std::sort(std::execution::par_unseq, intervals.begin(), intervals.end(),
              [](const SapInterval &a, const SapInterval &b)
              {
                  if (a.minAxis == b.minAxis)
                      return a.bodyIndex < b.bodyIndex;
                  return a.minAxis < b.minAxis;
              });

    // Sweep in parallel: each chunk scans its interval range and emits pairs
    // whose bounding spheres actually overlap. The sphere test only reads
    // interval data, so this is race-free. Results are stored per chunk slot
    // to keep pair order deterministic regardless of thread scheduling.
    Clock::time_point candidateBegin = Clock::now();
    const size_t chunkSize = 512;
    size_t numChunks = (n + chunkSize - 1) / chunkSize;
    std::vector<BroadphaseChunkOut> chunkOut(numChunks);
    bool classifyInSweep = !solver->deepProfiling;

    auto sweepChunk = [&](size_t c)
    {
        size_t begin = c * chunkSize;
        size_t end = begin + chunkSize < n ? begin + chunkSize : n;
        BroadphaseChunkOut &out = chunkOut[c];
        for (size_t i = begin; i < end; ++i)
        {
            const SapInterval &a = intervals[i];
            for (size_t j = i + 1; j < n; ++j)
            {
                const SapInterval &b = intervals[j];
                if (b.minAxis > a.maxAxis)
                    break;
                out.checks++;
                float3 dp = a.center - b.center;
                float r = a.radius + b.radius;
                if (dot(dp, dp) > r * r)
                    continue;
                if (!classifyInSweep)
                    out.manifoldPairs.push_back({a.bodyIndex, b.bodyIndex});
                else
                    classifyHit(solver, out, bodies[a.bodyIndex], bodies[b.bodyIndex], a.bodyIndex, b.bodyIndex);
            }
        }
    };

    WorkerPool::instance().parallelFor(numChunks, 1, [&](size_t begin, size_t end)
    {
        for (size_t c = begin; c < end; ++c)
            sweepChunk(c);
    });

    applyBroadphaseChunks(solver, bodies, chunkOut, classifyInSweep);
    solver->stats.spatialHashCandidateMs = elapsedMs(candidateBegin, Clock::now());
}

void broadphaseExternalPairs(Solver *solver)
{
    Clock::time_point candidateBegin = Clock::now();
    for (const BroadphasePair &pair : solver->externalBroadphasePairs)
    {
        if (pair.bodyA == INVALID_BODY_ID || pair.bodyB == INVALID_BODY_ID ||
            pair.bodyA == pair.bodyB ||
            pair.bodyA >= solver->world.bodies.size() ||
            pair.bodyB >= solver->world.bodies.size())
            continue;

        Rigid *bodyA = solver->world.bodies[pair.bodyA].source;
        Rigid *bodyB = solver->world.bodies[pair.bodyB].source;
        if (!bodyA || !bodyB)
            continue;

        generatePair(solver, bodyA, bodyB);
    }
    for (const ExternalManifoldContact &contact : solver->externalManifoldContacts)
    {
        if (contact.bodyA >= contact.bodyB ||
            contact.bodyA >= solver->world.bodies.size() ||
            contact.bodyB >= solver->world.bodies.size())
            continue;

        Rigid *bodyA = solver->world.bodies[contact.bodyA].source;
        Rigid *bodyB = solver->world.bodies[contact.bodyB].source;
        if (!bodyA || !bodyB)
            continue;

        solver->stats.pairChecks++;
        solver->stats.sphereHits++;
        solver->stats.constrainedChecks++;
        bool constrained = bodyA->attachedForceCount <= bodyB->attachedForceCount
                               ? bodyA->constrainedTo(bodyB)
                               : bodyB->constrainedTo(bodyA);
        if (constrained)
        {
            solver->stats.constrainedHits++;
        }
        else
        {
            new Manifold(solver, bodyA, bodyB);
            solver->stats.manifoldsCreated++;
        }
    }
    solver->stats.spatialHashCandidateMs = elapsedMs(candidateBegin, Clock::now());
}

void applySphereRollingFriction(Manifold *manifold)
{
    Rigid *sphere = 0;
    Rigid *other = 0;
    if (manifold->bodyA->shape.type == RIGID_SHAPE_SPHERE)
    {
        sphere = manifold->bodyA;
        other = manifold->bodyB;
    }
    else if (manifold->bodyB->shape.type == RIGID_SHAPE_SPHERE)
    {
        sphere = manifold->bodyB;
        other = manifold->bodyA;
    }

    if (!sphere || sphere->mass <= 0.0f || other->mass > 0.0f || manifold->numContacts <= 0)
        return;

    float materialFriction = sqrtf(sphere->friction * other->friction);
    if (materialFriction <= 0.0f)
        return;

    float blend = clamp(materialFriction * 0.08f, 0.0f, 0.18f);
    for (int i = 0; i < manifold->numContacts; ++i)
    {
        float3 normal = sphere == manifold->bodyA ? manifold->basis[0] : -manifold->basis[0];
        float3 rWorld = -normal * sphere->shape.radius;
        float3 contactVelocity = sphere->velocityLin + cross(sphere->velocityAng, rWorld);
        float3 tangentVelocity = contactVelocity - normal * dot(contactVelocity, normal);
        sphere->velocityLin -= tangentVelocity * blend;

        float3 desiredOmega = cross(normal, sphere->velocityLin) / sphere->shape.radius;
        sphere->velocityAng += (desiredOmega - sphere->velocityAng) * blend;
    }
}

struct CpuReferenceBackend : PhysicsBackend
{
    const char *name() const override { return "CPU Reference"; }
    void step(Solver &solver) override { solver.stepCpuReference(); }
};
} // namespace

std::unique_ptr<PhysicsBackend> makeCpuReferencePhysicsBackend()
{
    return std::unique_ptr<PhysicsBackend>(new CpuReferenceBackend());
}

Solver::Solver()
    : bodies(0), forces(0), physicsBackend(makeCpuReferencePhysicsBackend()), useExternalBroadphasePairs(false), useExternalManifoldContacts(false), broadphaseMode(BROADPHASE_UNIFORM_GRID), spatialHashCellSize(SPATIAL_HASH_CELL_SIZE), skipIgnoreCollisionSolverWork(false), skipJointSolverWork(false), skipIgnoreCollisionInitializationWork(false), skipJointInitializationWork(false), deepProfiling(false), stats{}, spherePairSink(0)
{
    defaultParams();
}

Solver::~Solver()
{
    clear();
}

Rigid *Solver::pick(float3 origin, float3 dir, float3 &local)
{
    const float epsilon = 1.0e-6f;
    float bestT = INFINITY;
    Rigid *bestBody = 0;
    float3 bestLocal = {0, 0, 0};

    // Ray-cast against each OBB by transforming the ray into body local space.
    for (Rigid *body = bodies; body != 0; body = body->next)
    {
        if (body->mass <= 0.0f)
            continue;

        quat invRot = conjugate(body->positionAng);
        float3 o = rotate(invRot, origin - body->positionLin);
        float3 d = rotate(invRot, dir);

        if (body->shape.type == RIGID_SHAPE_SPHERE)
        {
            float b = dot(o, d);
            float c = dot(o, o) - body->shape.radius * body->shape.radius;
            float discriminant = b * b - c;
            if (discriminant < 0.0f)
                continue;

            float sqrtDiscriminant = sqrtf(discriminant);
            float t0 = -b - sqrtDiscriminant;
            float t1 = -b + sqrtDiscriminant;
            float tHit = t0 >= 0.0f ? t0 : t1;
            if (tHit < 0.0f)
                continue;

            if (tHit < bestT)
            {
                bestT = tHit;
                bestBody = body;
                bestLocal = o + d * tHit;
            }
            continue;
        }

        if (body->shape.type == RIGID_SHAPE_CAPSULE)
        {
            float bestCapsuleT = INFINITY;
            float3 bestCapsuleLocal = {0, 0, 0};

            float h = body->shape.halfLength;
            float r = body->shape.radius;
            float a = d.x * d.x + d.y * d.y;
            float b = o.x * d.x + o.y * d.y;
            float c = o.x * o.x + o.y * o.y - r * r;
            float discriminant = b * b - a * c;
            if (a > epsilon && discriminant >= 0.0f)
            {
                float sqrtDiscriminant = sqrtf(discriminant);
                float t0 = (-b - sqrtDiscriminant) / a;
                float t1 = (-b + sqrtDiscriminant) / a;
                float candidates[2] = {t0, t1};
                for (int i = 0; i < 2; ++i)
                {
                    float t = candidates[i];
                    float z = o.z + d.z * t;
                    if (t >= 0.0f && z >= -h && z <= h && t < bestCapsuleT)
                    {
                        bestCapsuleT = t;
                        bestCapsuleLocal = o + d * t;
                    }
                }
            }

            for (int cap = 0; cap < 2; ++cap)
            {
                float3 capCenter = {0.0f, 0.0f, cap == 0 ? -h : h};
                float3 oc = o - capCenter;
                float qb = dot(oc, d);
                float qc = dot(oc, oc) - r * r;
                float qd = qb * qb - qc;
                if (qd < 0.0f)
                    continue;
                float sqrtQd = sqrtf(qd);
                float t0 = -qb - sqrtQd;
                float t1 = -qb + sqrtQd;
                float t = t0 >= 0.0f ? t0 : t1;
                if (t >= 0.0f && t < bestCapsuleT)
                {
                    bestCapsuleT = t;
                    bestCapsuleLocal = o + d * t;
                }
            }

            if (bestCapsuleT < bestT)
            {
                bestT = bestCapsuleT;
                bestBody = body;
                bestLocal = bestCapsuleLocal;
            }
            continue;
        }

        if (body->shape.type == RIGID_SHAPE_CYLINDER)
        {
            float bestCylinderT = INFINITY;
            float3 bestCylinderLocal = {0, 0, 0};
            float h = body->shape.halfLength;
            float r = body->shape.radius;

            float a = d.x * d.x + d.y * d.y;
            float b = o.x * d.x + o.y * d.y;
            float c = o.x * o.x + o.y * o.y - r * r;
            float discriminant = b * b - a * c;
            if (a > epsilon && discriminant >= 0.0f)
            {
                float sqrtDiscriminant = sqrtf(discriminant);
                float t0 = (-b - sqrtDiscriminant) / a;
                float t1 = (-b + sqrtDiscriminant) / a;
                float candidates[2] = {t0, t1};
                for (int i = 0; i < 2; ++i)
                {
                    float t = candidates[i];
                    float z = o.z + d.z * t;
                    if (t >= 0.0f && z >= -h && z <= h && t < bestCylinderT)
                    {
                        bestCylinderT = t;
                        bestCylinderLocal = o + d * t;
                    }
                }
            }

            if (fabsf(d.z) > epsilon)
            {
                float caps[2] = {-h, h};
                for (int i = 0; i < 2; ++i)
                {
                    float t = (caps[i] - o.z) / d.z;
                    float3 p = o + d * t;
                    if (t >= 0.0f && p.x * p.x + p.y * p.y <= r * r && t < bestCylinderT)
                    {
                        bestCylinderT = t;
                        bestCylinderLocal = p;
                    }
                }
            }

            if (bestCylinderT < bestT)
            {
                bestT = bestCylinderT;
                bestBody = body;
                bestLocal = bestCylinderLocal;
            }
            continue;
        }

        float3 half = body->size * 0.5f;

        float tEnter = 0.0f;
        float tExit = INFINITY;
        bool hit = true;

        for (int i = 0; i < 3; ++i)
        {
            if (fabsf(d[i]) < epsilon)
            {
                if (o[i] < -half[i] || o[i] > half[i])
                {
                    hit = false;
                    break;
                }
                continue;
            }

            float invD = 1.0f / d[i];
            float t0 = (-half[i] - o[i]) * invD;
            float t1 = (half[i] - o[i]) * invD;
            if (t0 > t1)
            {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }

            tEnter = max(tEnter, t0);
            tExit = min(tExit, t1);
            if (tEnter > tExit)
            {
                hit = false;
                break;
            }
        }

        if (!hit)
            continue;

        float tHit = tEnter >= 0.0f ? tEnter : tExit;
        if (tHit < 0.0f)
            continue;

        if (tHit < bestT)
        {
            bestT = tHit;
            bestBody = body;
            bestLocal = o + d * tHit;
        }
    }

    if (!bestBody)
        return 0;

    local = bestLocal;
    return bestBody;
}

void Solver::clear()
{
    while (forces)
        delete forces;

    while (bodies)
        delete bodies;

    world.clear();
}

void Solver::defaultParams()
{
    dt = 1.0f / 60.0f;
    gravity = -10.0f;
    iterations = 10;

    // Note: in the paper, beta is suggested to be [1, 1000]. Technically, the best choice will
    // depend on the length, mass, and constraint function scales (ie units) of your simulation,
    // along with your strategy for incrementing the penalty parameters.
    // If the value is not in the right range, you may see slower convergance for complex scenes.
    // A minor upgrade from the paper is using separate betas for constraints of different units (eg linear vs angular).
    betaLin = 10000.0f;
    betaAng = 100.0f;

    // Alpha controls how much stabilization is applied. Higher values give slower and smoother
    // error correction, and lower values are more responsive and energetic. Tune this depending
    // on your desired constraint error response.
    alpha = 0.99f;

    // Gamma controls how much the penalty and lambda values are decayed each step during warmstarting.
    // This should always be < 1 so that the penalty values can decrease (unless you use a different
    // penalty parameter strategy which does not require decay).
    gamma = 0.999f;
}

void Solver::step()
{
    physicsBackend->step(*this);
}

void Solver::stepCpuReferenceWithExternalBroadphase(const std::vector<BroadphasePair> &pairs, bool worldAlreadySynced)
{
    externalBroadphasePairs = pairs;
    useExternalBroadphasePairs = true;
    stepCpuReference(worldAlreadySynced);
    useExternalBroadphasePairs = false;
    externalBroadphasePairs.clear();
}

void Solver::stepCpuReferenceWithExternalBroadphase(const std::vector<BroadphasePair> &pairs, const std::vector<ExternalManifoldContact> &contacts, bool worldAlreadySynced)
{
    setExternalManifoldContacts(contacts);
    externalBroadphasePairs = pairs;
    useExternalBroadphasePairs = true;
    stepCpuReference(worldAlreadySynced);
    useExternalBroadphasePairs = false;
    externalBroadphasePairs.clear();
    clearExternalManifoldContacts();
}

void Solver::setExternalManifoldContacts(const std::vector<ExternalManifoldContact> &contacts)
{
    externalManifoldContacts = contacts;
    externalManifoldContactMap.clear();
    externalManifoldContactMap.reserve(externalManifoldContacts.size());
    for (size_t i = 0; i < externalManifoldContacts.size(); ++i)
        externalManifoldContactMap[exactBodyPairKey(externalManifoldContacts[i].bodyA, externalManifoldContacts[i].bodyB)] = i;
    useExternalManifoldContacts = !externalManifoldContacts.empty();
}

void Solver::clearExternalManifoldContacts()
{
    externalManifoldContacts.clear();
    externalManifoldContactMap.clear();
    useExternalManifoldContacts = false;
}

const ExternalManifoldContact *Solver::findExternalManifoldContact(BodyId bodyA, BodyId bodyB) const
{
    if (!useExternalManifoldContacts)
        return 0;

    auto it = externalManifoldContactMap.find(exactBodyPairKey(bodyA, bodyB));
    if (it == externalManifoldContactMap.end())
        it = externalManifoldContactMap.find(exactBodyPairKey(bodyB, bodyA));
    if (it == externalManifoldContactMap.end())
        return 0;
    return &externalManifoldContacts[it->second];
}

void Solver::benchmarkBroadphaseOnly()
{
    for (Force *force = forces; force != 0;)
    {
        Force *next = force->next;
        if (force->type == SIM_CONSTRAINT_MANIFOLD)
            delete force;
        force = next;
    }

    Clock::time_point syncBegin = Clock::now();
    world.syncFromLegacy(*this);
    stats = SolverStats{};
    stats.simWorldSyncMs = elapsedMs(syncBegin, Clock::now());
    stats.spatialHashCellSize = spatialHashCellSize;
    for (Rigid *body = bodies; body != 0; body = body->next)
    {
        stats.bodyCount++;
        if (body->mass > 0)
            stats.activeBodyCount++;
    }

    Clock::time_point phaseBegin = Clock::now();
    if (broadphaseMode == BROADPHASE_UNIFORM_GRID)
        broadphaseUniformGrid(this);
    else if (broadphaseMode == BROADPHASE_SWEEP_AND_PRUNE)
        broadphaseSweepAndPrune(this);
    else
        broadphaseAllPairs(this);
    stats.broadphaseMs = elapsedMs(phaseBegin, Clock::now());

    for (Force *force = forces; force != 0; force = force->next)
    {
        stats.forceCount++;
        if (force->type == SIM_CONSTRAINT_JOINT)
            stats.jointCount++;
        else if (force->type == SIM_CONSTRAINT_SPRING)
            stats.springCount++;
        else if (force->type == SIM_CONSTRAINT_MANIFOLD)
            stats.manifoldCount++;
        else if (force->type == SIM_CONSTRAINT_IGNORE_COLLISION)
            stats.ignoreCollisionCount++;
    }

    world.syncFromLegacy(*this);
}

void Solver::stepCpuReference(bool worldAlreadySynced)
{
    prepareStep(worldAlreadySynced);
    iteratePrimalDualCpu();
    finishStep();
}

void Solver::prepareStep(bool worldAlreadySynced)
{
    Clock::time_point syncBegin = Clock::now();
    if (!worldAlreadySynced)
        world.syncFromLegacy(*this);
    stats = SolverStats{};
    stats.simWorldSyncMs = worldAlreadySynced ? 0.0f : elapsedMs(syncBegin, Clock::now());
    stats.spatialHashCellSize = spatialHashCellSize;
    for (Rigid *body = bodies; body != 0; body = body->next)
    {
        stats.bodyCount++;
        if (body->mass > 0)
        {
            stats.activeBodyCount++;
            if (deepProfiling)
            {
                stats.avgAttachedForces += (float)body->attachedForceCount;
                stats.maxAttachedForces = max(stats.maxAttachedForces, body->attachedForceCount);
            }
        }
    }
    if (deepProfiling && stats.activeBodyCount > 0)
        stats.avgAttachedForces /= (float)stats.activeBodyCount;

    if (spherePairSink)
    {
        spherePairSink->clear();
        for (Rigid *body = bodies; body != 0; body = body->next)
            body->gpuPairCount = 0;
    }

    Clock::time_point phaseBegin = Clock::now();
    if (useExternalBroadphasePairs)
        broadphaseExternalPairs(this);
    else if (broadphaseMode == BROADPHASE_UNIFORM_GRID)
        broadphaseUniformGrid(this);
    else if (broadphaseMode == BROADPHASE_SWEEP_AND_PRUNE)
        broadphaseSweepAndPrune(this);
    else
        broadphaseAllPairs(this);
    stats.broadphaseMs = elapsedMs(phaseBegin, Clock::now());

    // Initialize and warmstart forces
    phaseBegin = Clock::now();
    // Initialization can include caching anything that is constant over the
    // step. Each force's initialize() touches only its own state and reads
    // body/solver state, so the calls run in parallel; removal of inactive
    // forces (which mutates the linked lists and SimWorld) stays serial.
    initScratchForces.clear();
    Clock::time_point gatherBegin = Clock::now();
    for (Force *force = forces; force != 0; force = force->next)
    {
        bool skipInitialization = false;
        if (force->type == SIM_CONSTRAINT_JOINT && skipJointInitializationWork && isinf(((Joint *)force)->fracture))
        {
            stats.jointInitializationSkipped++;
            skipInitialization = true;
        }
        else if (force->type == SIM_CONSTRAINT_IGNORE_COLLISION && skipIgnoreCollisionInitializationWork)
        {
            stats.ignoreCollisionInitializationSkipped++;
            skipInitialization = true;
        }
        if (!skipInitialization)
            initScratchForces.push_back(force);
        else
            countForceForStats(this, force);
    }

    stats.forceInitGatherMs = elapsedMs(gatherBegin, Clock::now());
    Clock::time_point parallelBegin = Clock::now();
    initScratchKeep.assign(initScratchForces.size(), 1);
    size_t initCount = initScratchForces.size();
    WorkerPool::instance().parallelFor(initCount, 256, [&](size_t begin, size_t end)
    {
        for (size_t i = begin; i < end; ++i)
            initScratchKeep[i] = initScratchForces[i]->initialize() ? 1 : 0;
    });

    stats.forceInitParallelMs = elapsedMs(parallelBegin, Clock::now());
    Clock::time_point cleanupBegin = Clock::now();
    for (size_t i = 0; i < initCount; ++i)
    {
        if (!initScratchKeep[i])
            delete initScratchForces[i]; // inactive: remove from the solver
        else
            countForceForStats(this, initScratchForces[i]);
    }
    stats.forceInitCleanupMs = elapsedMs(cleanupBegin, Clock::now());
    stats.forceInitMs = elapsedMs(phaseBegin, Clock::now());

    // Initialize and warmstart bodies (ie primal variables)
    phaseBegin = Clock::now();
    for (Rigid *body = bodies; body != 0; body = body->next)
    {
        // Compute inertial position (Eq 2)
        body->inertialLin = body->positionLin + body->velocityLin * dt;
        if (body->mass > 0)
            body->inertialLin += float3{0, 0, gravity} * (dt * dt);
        body->inertialAng = body->positionAng + body->velocityAng * dt;

        // Adaptive warmstart (See original VBD paper)
        float3 accel = (body->velocityLin - body->prevVelocityLin) / dt;
        float accelExt = accel.z * sign(gravity);
        float accelWeight = clamp(accelExt / abs(gravity), 0.0f, 1.0f);
        if (!isfinite(accelWeight))
            accelWeight = 0.0f;

        // Save initial position (x-) and compute warmstarted position (See original VBD paper)
        body->initialLin = body->positionLin;
        body->initialAng = body->positionAng;
        if (body->mass > 0)
        {
            body->positionLin = body->positionLin + body->velocityLin * dt + float3{0, 0, gravity} * (accelWeight * dt * dt);
            body->positionAng = body->positionAng + body->velocityAng * dt;
        }

        // Unconstrained bodies solve exactly to their inertial prediction. Do it once
        // instead of revisiting them in every primal iteration. Bodies with
        // GPU-resident sphere pairs are constrained even without CPU forces.
        if (body->mass > 0 && body->forces == 0 && body->gpuPairCount == 0)
        {
            body->positionLin = body->inertialLin;
            body->positionAng = body->inertialAng;
        }
    }
    stats.bodyInitMs = elapsedMs(phaseBegin, Clock::now());
}

void Solver::iteratePrimalDualCpu()
{
    Clock::time_point phaseBegin;

    // Main solver loop
    for (int it = 0; it < iterations; it++)
    {
        // Primal update
        phaseBegin = Clock::now();
        for (Rigid *body = bodies; body != 0; body = body->next)
        {
            // Skip static / kinematic bodies
            if (body->mass <= 0)
                continue;
            if (body->forces == 0)
                continue;

            // Initialize left and right hand sides of the linear system (Eqs. 5, 6)
            float3x3 MLin = diagonal(body->mass, body->mass, body->mass);
            float3x3 MAng = diagonal(body->moment.x, body->moment.y, body->moment.z);

            float3x3 lhsLin = MLin / (dt * dt);
            float3x3 lhsAng = MAng / (dt * dt);
            float3x3 lhsCross = float3x3{0, 0, 0, 0, 0, 0, 0, 0, 0};

            float3 rhsLin = MLin / (dt * dt) * (body->positionLin - body->inertialLin);
            float3 rhsAng = MAng / (dt * dt) * (body->positionAng - body->inertialAng);

            // Iterate over all forces acting on the body
            for (Force *force = body->forces; force != 0; force = (force->bodyA == body) ? force->nextA : force->nextB)
            {
                if (skipJointSolverWork && force->type == SIM_CONSTRAINT_JOINT)
                {
                    stats.primalJointSkipped++;
                    continue;
                }
                if (skipIgnoreCollisionSolverWork && force->type == SIM_CONSTRAINT_IGNORE_COLLISION)
                {
                    stats.primalIgnoreCollisionSkipped++;
                    continue;
                }

                // Stamp the force and hessian into the linear system
                stats.primalForceVisits++;
                if (deepProfiling)
                {
                    Clock::time_point forceBegin = Clock::now();
                    force->updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
                    accumulatePrimalForceStats(this, force, elapsedMs(forceBegin, Clock::now()));
                }
                else
                {
                    force->updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
                }
            }

            // Solve the SPD linear system using LDL and apply the update (Eq. 4)
            float3 dxLin, dxAng;
            Clock::time_point solveBegin;
            if (deepProfiling)
                solveBegin = Clock::now();
            solve(lhsLin, lhsAng, lhsCross, -rhsLin, -rhsAng, dxLin, dxAng);
            if (deepProfiling)
            {
                stats.bodySolveMs += elapsedMs(solveBegin, Clock::now());
                stats.bodySolveCount++;
            }
            body->positionLin = body->positionLin + dxLin;
            body->positionAng = body->positionAng + dxAng;
        }
        stats.primalSolveMs += elapsedMs(phaseBegin, Clock::now());

        // Dual update
        phaseBegin = Clock::now();
        for (Force *force = forces; force != 0; force = force->next)
        {
            if (skipJointSolverWork && force->type == SIM_CONSTRAINT_JOINT)
            {
                stats.dualJointSkipped++;
                continue;
            }
            if (skipIgnoreCollisionSolverWork && force->type == SIM_CONSTRAINT_IGNORE_COLLISION)
            {
                stats.dualIgnoreCollisionSkipped++;
                continue;
            }

            stats.dualForceVisits++;
            if (deepProfiling)
            {
                Clock::time_point forceBegin = Clock::now();
                force->updateDual(alpha);
                accumulateDualForceStats(this, force, elapsedMs(forceBegin, Clock::now()));
            }
            else
            {
                force->updateDual(alpha);
            }
        }
        stats.dualUpdateMs += elapsedMs(phaseBegin, Clock::now());
    }
}

void Solver::finishStep()
{
    // Compute velocities (BDF1) after the final iteration
    Clock::time_point phaseBegin = Clock::now();
    for (Rigid* body = bodies; body != 0; body = body->next)
    {
        body->prevVelocityLin = body->velocityLin;
        if (body->mass > 0)
        {
            body->velocityLin = (body->positionLin - body->initialLin) / dt;
            body->velocityAng = (body->positionAng - body->initialAng) / dt;
        }
    }

    for (Force *force = forces; force != 0; force = force->next)
    {
        if (force->type == SIM_CONSTRAINT_MANIFOLD)
            applySphereRollingFriction((Manifold *)force);
    }
    stats.velocityUpdateMs = elapsedMs(phaseBegin, Clock::now());
    Clock::time_point syncBegin = Clock::now();
    world.syncFromLegacy(*this);
    stats.simWorldSyncMs += elapsedMs(syncBegin, Clock::now());
}
