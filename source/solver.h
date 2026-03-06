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

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifdef TARGET_OS_MAC
#include <OpenGL/GL.h>
#else
#include <GL/gl.h>
#endif

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

// Holds all the state for a single rigid body that is needed by AVBD
struct Rigid
{
    Solver *solver;
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
    float3 size; // Full widths in each dimension
    float mass;
    float3 moment;
    float friction;
    float radius;

    Rigid(Solver *solver, float3 size, float density, float friction, float3 position, float3 velocity = float3{0, 0, 0});
    ~Rigid();

    bool constrainedTo(Rigid *other) const;
};

// Holds all user defined and derived constraint parameters, and provides a common interface for all forces.
struct Force
{
    Solver *solver;
    Rigid *bodyA;
    Rigid *bodyB;
    Force *nextA;
    Force *nextB;
    Force *next;

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
    IgnoreCollision(Solver *solver, Rigid *bodyA, Rigid *bodyB)
        : Force(solver, bodyA, bodyB) {}

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

    Solver();
    ~Solver();

    Rigid *pick(float3 origin, float3 dir, float3 &local);
    void clear();
    void defaultParams();
    void step();
};
