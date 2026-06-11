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

#include "solver.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
const float SPATIAL_HASH_CELL_SIZE = 2.0f;

using Clock = std::chrono::high_resolution_clock;

struct SpatialCell
{
    int x, y, z;

    bool operator==(const SpatialCell &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct SpatialCellHash
{
    size_t operator()(const SpatialCell &cell) const
    {
        uint32_t hx = (uint32_t)cell.x * 73856093u;
        uint32_t hy = (uint32_t)cell.y * 19349663u;
        uint32_t hz = (uint32_t)cell.z * 83492791u;
        return (size_t)(hx ^ hy ^ hz);
    }
};

struct SapInterval
{
    float minAxis;
    float maxAxis;
    int bodyIndex;
};

uint64_t pairKey(int a, int b)
{
    uint32_t lo = (uint32_t)min(a, b);
    uint32_t hi = (uint32_t)max(a, b);
    return ((uint64_t)lo << 32) | hi;
}

uint64_t exactBodyPairKey(BodyId a, BodyId b)
{
    return ((uint64_t)a << 32) | (uint64_t)b;
}

int cellCoord(float x, float cellSize)
{
    return (int)floorf(x / cellSize);
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

void generatePair(Solver *solver, Rigid *bodyA, Rigid *bodyB)
{
    solver->stats.pairChecks++;

    float3 dp = bodyA->positionLin - bodyB->positionLin;
    float r = bodyA->radius + bodyB->radius;
    if (dot(dp, dp) <= r * r)
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

void broadphaseSpatialHash(Solver *solver)
{
    std::vector<Rigid *> bodies;
    std::vector<int> globalBodies;
    std::unordered_map<SpatialCell, std::vector<int>, SpatialCellHash> cells;
    std::unordered_set<uint64_t> seenPairs;
    float cellSize = solver->spatialHashCellSize;
    float globalRadius = cellSize * 8.0f;

    solver->stats.spatialHashCellSize = cellSize;
    Clock::time_point buildBegin = Clock::now();
    for (Rigid *body = solver->bodies; body != 0; body = body->next)
        bodies.push_back(body);

    for (int i = 0; i < (int)bodies.size(); ++i)
    {
        Rigid *body = bodies[i];
        if (body->radius > globalRadius)
        {
            globalBodies.push_back(i);
            if (solver->deepProfiling)
                solver->stats.spatialHashGlobalBodies++;
            continue;
        }

        int minX = cellCoord(body->positionLin.x - body->radius, cellSize);
        int minY = cellCoord(body->positionLin.y - body->radius, cellSize);
        int minZ = cellCoord(body->positionLin.z - body->radius, cellSize);
        int maxX = cellCoord(body->positionLin.x + body->radius, cellSize);
        int maxY = cellCoord(body->positionLin.y + body->radius, cellSize);
        int maxZ = cellCoord(body->positionLin.z + body->radius, cellSize);

        for (int x = minX; x <= maxX; ++x)
            for (int y = minY; y <= maxY; ++y)
                for (int z = minZ; z <= maxZ; ++z)
                {
                    cells[SpatialCell{x, y, z}].push_back(i);
                    if (solver->deepProfiling)
                        solver->stats.spatialHashCellInsertions++;
                }
    }
    if (solver->deepProfiling)
    {
        solver->stats.spatialHashOccupiedCells = (int)cells.size();
        int totalOccupancy = 0;
        for (const auto &entry : cells)
        {
            int occupancy = (int)entry.second.size();
            totalOccupancy += occupancy;
            solver->stats.spatialHashMaxCellOccupancy = max(solver->stats.spatialHashMaxCellOccupancy, occupancy);
        }
        if (solver->stats.spatialHashOccupiedCells > 0)
            solver->stats.spatialHashAvgCellOccupancy = (float)totalOccupancy / (float)solver->stats.spatialHashOccupiedCells;
    }
    solver->stats.spatialHashBuildMs = elapsedMs(buildBegin, Clock::now());

    auto generateUniquePair = [&](int a, int b, bool globalPair)
    {
        if (a == b)
            return;

        if (solver->deepProfiling)
        {
            solver->stats.spatialHashPairAttempts++;
            if (globalPair)
                solver->stats.spatialHashGlobalPairAttempts++;
        }

        uint64_t key = pairKey(a, b);
        bool inserted;
        if (solver->deepProfiling)
        {
            Clock::time_point dedupBegin = Clock::now();
            inserted = seenPairs.insert(key).second;
            solver->stats.spatialHashDedupMs += elapsedMs(dedupBegin, Clock::now());
        }
        else
        {
            inserted = seenPairs.insert(key).second;
        }

        if (inserted)
            generatePair(solver, bodies[a], bodies[b]);
        else if (solver->deepProfiling)
        {
            solver->stats.spatialHashDuplicatePairs++;
        }
    };

    Clock::time_point candidateBegin = Clock::now();
    for (const auto &entry : cells)
    {
        const std::vector<int> &indices = entry.second;
        for (int i = 0; i < (int)indices.size(); ++i)
            for (int j = i + 1; j < (int)indices.size(); ++j)
                generateUniquePair(indices[i], indices[j], false);
    }

    for (int global : globalBodies)
        for (int i = 0; i < (int)bodies.size(); ++i)
            generateUniquePair(global, i, true);
    solver->stats.spatialHashCandidateMs = elapsedMs(candidateBegin, Clock::now());
}

void broadphaseSweepAndPrune(Solver *solver)
{
    std::vector<Rigid *> bodies;
    for (Rigid *body = solver->bodies; body != 0; body = body->next)
        bodies.push_back(body);

    if (bodies.size() < 2)
        return;

    std::vector<SapInterval> axisIntervals[3];
    for (int axis = 0; axis < 3; ++axis)
        axisIntervals[axis].reserve(bodies.size());

    for (int i = 0; i < (int)bodies.size(); ++i)
    {
        Rigid *body = bodies[i];
        for (int axis = 0; axis < 3; ++axis)
        {
            float center = body->positionLin[axis];
            axisIntervals[axis].push_back(SapInterval{center - body->radius, center + body->radius, i});
        }
    }

    int bestAxis = 0;
    int bestCandidateCount = INT_MAX;
    for (int axis = 0; axis < 3; ++axis)
    {
        std::vector<SapInterval> &intervals = axisIntervals[axis];
        std::sort(intervals.begin(), intervals.end(), [](const SapInterval &a, const SapInterval &b)
                  {
                      if (a.minAxis == b.minAxis)
                          return a.bodyIndex < b.bodyIndex;
                      return a.minAxis < b.minAxis;
                  });

        int candidateCount = 0;
        for (int i = 0; i < (int)intervals.size(); ++i)
        {
            const SapInterval &a = intervals[i];
            for (int j = i + 1; j < (int)intervals.size(); ++j)
            {
                const SapInterval &b = intervals[j];
                if (b.minAxis > a.maxAxis)
                    break;
                candidateCount++;
            }
        }

        if (candidateCount < bestCandidateCount)
        {
            bestCandidateCount = candidateCount;
            bestAxis = axis;
        }
    }

    Clock::time_point candidateBegin = Clock::now();
    const std::vector<SapInterval> &intervals = axisIntervals[bestAxis];
    for (int i = 0; i < (int)intervals.size(); ++i)
    {
        const SapInterval &a = intervals[i];
        for (int j = i + 1; j < (int)intervals.size(); ++j)
        {
            const SapInterval &b = intervals[j];
            if (b.minAxis > a.maxAxis)
                break;
            generatePair(solver, bodies[a.bodyIndex], bodies[b.bodyIndex]);
        }
    }
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
    : bodies(0), forces(0), physicsBackend(makeCpuReferencePhysicsBackend()), useExternalBroadphasePairs(false), useExternalManifoldContacts(false), broadphaseMode(BROADPHASE_SWEEP_AND_PRUNE), spatialHashCellSize(SPATIAL_HASH_CELL_SIZE), skipIgnoreCollisionSolverWork(false), skipJointSolverWork(false), skipIgnoreCollisionInitializationWork(false), skipJointInitializationWork(false), deepProfiling(false), stats{}
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
    if (broadphaseMode == BROADPHASE_SPATIAL_HASH)
        broadphaseSpatialHash(this);
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

    Clock::time_point phaseBegin = Clock::now();
    if (useExternalBroadphasePairs)
        broadphaseExternalPairs(this);
    else if (broadphaseMode == BROADPHASE_SPATIAL_HASH)
        broadphaseSpatialHash(this);
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
    unsigned int workerCount = std::thread::hardware_concurrency();
    if (workerCount > 8)
        workerCount = 8;
    if (workerCount < 2 || initCount < 512)
    {
        for (size_t i = 0; i < initCount; ++i)
            initScratchKeep[i] = initScratchForces[i]->initialize() ? 1 : 0;
    }
    else
    {
        std::atomic<size_t> nextChunk(0);
        const size_t chunkSize = 256;
        auto worker = [&]()
        {
            for (;;)
            {
                size_t begin = nextChunk.fetch_add(chunkSize);
                if (begin >= initCount)
                    return;
                size_t end = begin + chunkSize < initCount ? begin + chunkSize : initCount;
                for (size_t i = begin; i < end; ++i)
                    initScratchKeep[i] = initScratchForces[i]->initialize() ? 1 : 0;
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(workerCount - 1);
        for (unsigned int t = 0; t + 1 < workerCount; ++t)
            workers.emplace_back(worker);
        worker();
        for (std::thread &t : workers)
            t.join();
    }

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
        // instead of revisiting them in every primal iteration.
        if (body->mass > 0 && body->forces == 0)
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
