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

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "maths.h"

#define PENALTY_MIN 1.0f           // Minimum penalty parameter
#define PENALTY_MAX 10000000000.0f // Maximum penalty parameter
#define COLLISION_MARGIN 0.01f     // Margin for collision detection to avoid flickering contacts
#define STICK_THRESH 0.00001f      // Position threshold for sticking contacts (ie static friction)
#define SHOW_CONTACTS true         // Whether to show contacts in the debug draw

struct Rigid;
struct Force;
struct Manifold;
struct Solver;
struct PhysicsBackend;

using BodyId = uint32_t;
using ForceId = uint32_t;

static const BodyId INVALID_BODY_ID = UINT32_MAX;
static const ForceId INVALID_FORCE_ID = UINT32_MAX;

enum BroadphaseMode
{
    BROADPHASE_ALL_PAIRS,
    BROADPHASE_SPATIAL_HASH,
    BROADPHASE_SWEEP_AND_PRUNE,
    BROADPHASE_COUNT
};

enum RigidShapeType
{
    RIGID_SHAPE_BOX,
    RIGID_SHAPE_SPHERE,
    RIGID_SHAPE_CAPSULE,
    RIGID_SHAPE_CYLINDER
};

struct RigidShape
{
    RigidShapeType type;
    float3 size; // Full box widths, or bounding dimensions for rounded primitives
    float radius;
    float halfLength; // Capsule center-segment half length; zero for boxes/spheres
};

enum SimConstraintType
{
    SIM_CONSTRAINT_UNKNOWN,
    SIM_CONSTRAINT_JOINT,
    SIM_CONSTRAINT_SPRING,
    SIM_CONSTRAINT_IGNORE_COLLISION,
    SIM_CONSTRAINT_MANIFOLD
};

struct SimBodyData
{
    bool active;
    Rigid *source;
    RigidShape shape;
    float3 size;
    float3 positionLin;
    quat positionAng;
    float3 velocityLin;
    float3 velocityAng;
    float mass;
    float3 moment;
    float friction;
    float radius;
    int attachedForceCount;
};

struct SimConstraintData
{
    bool active;
    Force *source;
    SimConstraintType type;
    BodyId bodyA;
    BodyId bodyB;
};

struct SimWorld
{
    std::vector<SimBodyData> bodies;
    std::vector<SimConstraintData> constraints;
    std::vector<BodyId> activeBodyIds;
    std::vector<ForceId> jointIds;
    std::vector<ForceId> springIds;
    std::vector<ForceId> ignoreCollisionIds;
    std::vector<ForceId> manifoldIds;
    std::vector<std::vector<ForceId>> bodyConstraintIds;
    std::vector<BodyId> freeBodyIds;
    std::vector<ForceId> freeConstraintIds;

    void clear();
    BodyId registerBody(Rigid *body);
    void updateBodyFromRigid(Rigid *body);
    void unregisterBody(BodyId id);
    ForceId registerForce(Force *force, SimConstraintType type = SIM_CONSTRAINT_UNKNOWN);
    void unregisterForce(ForceId id);
    void setForceType(ForceId id, SimConstraintType type);
    void syncFromLegacy(Solver &solver);
};

struct SolverStats
{
    int bodyCount;
    int activeBodyCount;
    int pairChecks;
    int sphereHits;
    int manifoldsCreated;
    int constrainedChecks;
    int constrainedHits;
    int constrainedForceVisits;
    int forceCount;
    int jointCount;
    int springCount;
    int manifoldCount;
    int ignoreCollisionCount;
    int jointInitializationSkipped;
    int ignoreCollisionInitializationSkipped;
    int primalForceVisits;
    int dualForceVisits;
    int primalJointVisits;
    int primalSpringVisits;
    int primalManifoldVisits;
    int primalIgnoreCollisionVisits;
    int primalIgnoreCollisionSkipped;
    int primalJointSkipped;
    int dualJointVisits;
    int dualSpringVisits;
    int dualManifoldVisits;
    int dualIgnoreCollisionVisits;
    int dualIgnoreCollisionSkipped;
    int dualJointSkipped;
    int bodySolveCount;
    int maxAttachedForces;
    float avgAttachedForces;
    float broadphaseMs;
    float simWorldSyncMs;
    float spatialHashBuildMs;
    float spatialHashCandidateMs;
    float spatialHashCellSize;
    int spatialHashOccupiedCells;
    int spatialHashCellInsertions;
    int spatialHashMaxCellOccupancy;
    float spatialHashAvgCellOccupancy;
    int spatialHashPairAttempts;
    int spatialHashDuplicatePairs;
    int spatialHashGlobalBodies;
    int spatialHashGlobalPairAttempts;
    float spatialHashDedupMs;
    float constrainedMs;
    float manifoldAllocMs;
    float forceInitMs;
    float forceInitGatherMs;
    float forceInitParallelMs;
    float forceInitCleanupMs;
    float bodyInitMs;
    float primalSolveMs;
    float dualUpdateMs;
    float velocityUpdateMs;
    float primalJointMs;
    float primalSpringMs;
    float primalManifoldMs;
    float primalIgnoreCollisionMs;
    float dualJointMs;
    float dualSpringMs;
    float dualManifoldMs;
    float dualIgnoreCollisionMs;
    float bodySolveMs;
};

struct BroadphasePair
{
    BodyId bodyA;
    BodyId bodyB;
};

// Holds all the state for a single rigid body that is needed by AVBD
struct Rigid
{
    Solver *solver;
    BodyId denseId;
    Force *forces;
    Rigid *next;
    float3 positionLin;
    quat positionAng;
    float3 initialLin;
    quat initialAng;
    float3 inertialLin;
    quat inertialAng;
    float3 velocityLin;
    float3 velocityAng;
    float3 prevVelocityLin;
    RigidShape shape;
    float3 size; // Full widths in each dimension, retained for existing joint/debug helpers
    float mass;
    float3 moment;
    float friction;
    float radius;
    int attachedForceCount;

    Rigid(Solver *solver, float3 size, float density, float friction, float3 position, float3 velocity = float3{0, 0, 0});
    static Rigid *makeSphere(Solver *solver, float radius, float density, float friction, float3 position, float3 velocity = float3{0, 0, 0});
    static Rigid *makeCapsule(Solver *solver, float radius, float halfLength, float density, float friction, float3 position, float3 velocity = float3{0, 0, 0});
    static Rigid *makeCylinder(Solver *solver, float radius, float halfLength, float density, float friction, float3 position, float3 velocity = float3{0, 0, 0});
    ~Rigid();

    bool constrainedTo(Rigid *other) const;
};

// Holds all user defined and derived constraint parameters, and provides a common interface for all forces.
struct Force
{
    Solver *solver;
    ForceId denseId;
    SimConstraintType type; // set by each concrete force; avoids dynamic_cast in hot loops
    Rigid *bodyA;
    Rigid *bodyB;
    Force *nextA;
    Force *nextB;
    Force *next;
    Force *prev; // doubly-linked solver list so destruction unlinks in O(1)

    Force(Solver *solver, Rigid *bodyA, Rigid *bodyB);
    virtual ~Force();

    virtual bool initialize() = 0;
    virtual void updatePrimal(Rigid *body, float alpha, float3x3 &lhsLin, float3x3 &lhsAng, float3x3 &lhsCross, float3 &rhsLin, float3 &rhsAng) = 0;
    virtual void updateDual(float alpha) = 0;
};

// Revolute joint + angle constraint between two rigid bodies, with optional fracture
struct Joint : Force
{
    float3 rA, rB;
    float3 C0Lin, C0Ang;
    float3 penaltyLin, penaltyAng;
    float3 lambdaLin, lambdaAng;
    float stiffnessLin, stiffnessAng, fracture;
    float torqueArm;
    bool broken;

    Joint(Solver *solver, Rigid *bodyA, Rigid *bodyB, float3 rA, float3 rB, float stiffnessLin = INFINITY, float stiffnessAng = 0.0f, float fracture = INFINITY);

    bool initialize() override;
    void updatePrimal(Rigid *body, float alpha, float3x3 &lhsLin, float3x3 &lhsAng, float3x3 &lhsCross, float3 &rhsLin, float3 &rhsAng) override;
    void updateDual(float alpha) override;
};

// Standard spring force
struct Spring : Force
{
    float3 rA, rB;
    float rest;
    float stiffness;

    Spring(Solver *solver, Rigid *bodyA, Rigid *bodyB, float3 rA, float3 rB, float stiffness, float rest = -1);

    bool initialize() override { return true; }
    void updatePrimal(Rigid *body, float alpha, float3x3 &lhsLin, float3x3 &lhsAng, float3x3 &lhsCross, float3 &rhsLin, float3 &rhsAng) override;
    void updateDual(float alpha) override;
};

// Force which has no physical effect, but is used to ignore collisions between two bodies
struct IgnoreCollision : Force
{
    IgnoreCollision(Solver *solver, Rigid *bodyA, Rigid *bodyB);

    bool initialize() override { return true; }
    void updatePrimal(Rigid *body, float alpha, float3x3 &lhsLin, float3x3 &lhsAng, float3x3 &lhsCross, float3 &rhsLin, float3 &rhsAng) override {}
    void updateDual(float alpha) override {}
};

// Collision manifold between two rigid bodies, which contains up to eight frictional contact points
struct Manifold : Force
{
    // Used to track contact features between frames
    union FeaturePair
    {
        struct
        {
            char inR;
            char outR;
            char inI;
            char outI;
        };

        int key;
    };

    // Contact point information for a single contact
    struct Contact
    {
        FeaturePair feature;
        float3 rA; // contact offset in A's local space (relative to center)
        float3 rB; // contact offset in B's local space (relative to center)
        float3 C0;
        float3 penalty;
        float3 lambda;
        bool stick;
    };

    Contact contacts[8];
    float3x3 basis; // Normal in the first row (pointing from B to A), and tangents in the second and third rows
    int numContacts;
    float friction;

    Manifold(Solver *solver, Rigid *bodyA, Rigid *bodyB);

    bool initialize() override;
    void updatePrimal(Rigid *body, float alpha, float3x3 &lhsLin, float3x3 &lhsAng, float3x3 &lhsCross, float3 &rhsLin, float3 &rhsAng) override;
    void updateDual(float alpha) override;

    static int collide(Rigid *bodyA, Rigid *bodyB, Contact *contacts, float3x3 &basis);
};

struct ExternalManifoldContact
{
    BodyId bodyA;
    BodyId bodyB;
    Manifold::Contact contacts[8];
    float3x3 basis;
    int numContacts;
};

// Core solver class which holds all the rigid bodies and forces, and has logic to step the simulation forward in time
struct Solver
{
    float dt;       // Timestep
    float gravity;  // Gravity
    int iterations; // Solver iterations

    float alpha; // Stabilization parameter
    float betaLin;  // Penalty ramping parameter for linear constraints
    float betaAng;  // Penalty ramping parameter for angular constraints
    float gamma; // Warmstarting decay parameter

    Rigid *bodies;
    Force *forces;
    SimWorld world;
    std::unique_ptr<PhysicsBackend> physicsBackend;
    std::vector<BroadphasePair> externalBroadphasePairs;
    std::vector<ExternalManifoldContact> externalManifoldContacts;
    std::unordered_map<uint64_t, size_t> externalManifoldContactMap;
    bool useExternalBroadphasePairs;
    bool useExternalManifoldContacts;
    BroadphaseMode broadphaseMode;
    float spatialHashCellSize;
    bool skipIgnoreCollisionSolverWork;
    bool skipJointSolverWork;
    bool skipIgnoreCollisionInitializationWork;
    bool skipJointInitializationWork;
    bool deepProfiling;
    SolverStats stats;

    // Scratch reused by prepareStep's parallel force initialization.
    std::vector<Force *> initScratchForces;
    std::vector<uint8_t> initScratchKeep;

    Solver();
    ~Solver();

    Rigid *pick(float3 origin, float3 dir, float3 &local);
    void clear();
    void defaultParams();
    void step();
    void stepCpuReference(bool worldAlreadySynced = false);

    // Phased stepping used by backends that replace only the iteration loop.
    // prepareStep runs broadphase, force initialization/warmstarting, and body
    // warmstarting. iteratePrimalDualCpu runs the reference AVBD iteration
    // loop. finishStep derives velocities and re-syncs the dense world.
    // stepCpuReference is exactly prepareStep + iteratePrimalDualCpu + finishStep.
    void prepareStep(bool worldAlreadySynced = false);
    void iteratePrimalDualCpu();
    void finishStep();
    void benchmarkBroadphaseOnly();
    void stepCpuReferenceWithExternalBroadphase(const std::vector<BroadphasePair> &pairs, bool worldAlreadySynced = false);
    void stepCpuReferenceWithExternalBroadphase(const std::vector<BroadphasePair> &pairs, const std::vector<ExternalManifoldContact> &contacts, bool worldAlreadySynced = false);
    void setExternalManifoldContacts(const std::vector<ExternalManifoldContact> &contacts);
    void clearExternalManifoldContacts();
    const ExternalManifoldContact *findExternalManifoldContact(BodyId bodyA, BodyId bodyB) const;
};

struct PhysicsBackend
{
    virtual ~PhysicsBackend() {}
    virtual const char *name() const = 0;
    virtual void step(Solver &solver) = 0;
};

std::unique_ptr<PhysicsBackend> makeCpuReferencePhysicsBackend();
