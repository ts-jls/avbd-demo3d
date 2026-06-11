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

#include "maths.h"
#include "solver.h"

static void sceneEmpty(Solver *solver)
{
    solver->clear();
}

static void sceneGround(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0}, {0, 0, 0});
    new Rigid(solver, {1, 1, 1}, 1.0f, 0.5f, {0, 0, 4});
}

static void sceneDynamicFriction(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0}, {0, 0, 0});
    for (int x = 0; x <= 10; x++)
        new Rigid(solver, {1, 1, 0.5f}, 1.0f, 5.0f - (x / 10.0f * 5.0f), {0, -30.0f + x * 2.0f, 0.75f}, {10.0f, 0, 0});
}

static void sceneStaticFriction(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});

    const float angle = rad(30.0f);
    Rigid *ramp = new Rigid(solver, {40, 24, 1}, 0.0f, 1.0f, {0, 0, 3});
    ramp->positionAng = {0, sinf(angle * 0.5f), 0, cosf(angle * 0.5f)};

    float3 rampTangent = normalize(rotate(ramp->positionAng, float3{1, 0, 0}));
    float3 rampNormal = normalize(rotate(ramp->positionAng, float3{0, 0, 1}));

    for (int i = 0; i <= 10; i++)
    {
        float friction = i / 10.0f * 0.25f + 0.25f;
        float y = -10.0f + i * 2.0f;
        float3 pos = ramp->positionLin + rampTangent * -12.0f + float3{0, y, 0} + rampNormal * 1.05f;
        new Rigid(solver, {1, 1, 1}, 1.0f, friction, pos);
    }
}

static void scenePyramid(Solver *solver)
{
    const int SIZE = 16;
    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0.0f, 0.0f, -0.5f});

    for (int y = 0; y < SIZE; y++)
        for (int x = 0; x < SIZE - y; x++)
            new Rigid(solver, {1, 0.5f, 0.5f}, 1.0f, 0.5f, {x * 1.01f + y * 0.5f - SIZE / 2.0f, 0.0f, y * 0.85f + 0.5f});
}

static void sceneRope(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, -20});

    Rigid *prev = 0;
    for (int i = 0; i < 20; i++)
    {
        Rigid *curr = new Rigid(solver, {1, 0.5f, 0.5f}, i == 0 ? 0.0f : 1.0f, 0.5f, {(float)i, 0.0f, 10.0f});
        if (prev)
            new Joint(solver, prev, curr, {0.5f, 0, 0}, {-0.5f, 0, 0});
        prev = curr;
    }
}

static void sceneHeavyRope(Solver *solver)
{
    const int N = 20;
    const float SIZE = 5;
    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, -20});

    Rigid *prev = 0;
    for (int i = 0; i < N; i++)
    {
        Rigid *curr = new Rigid(solver, i == N - 1 ? float3{SIZE, SIZE, SIZE} : float3{1, 0.5f, 0.5f},
                                i == 0 ? 0.0f : 1.0f, 0.5f, {(float)i + (i == N - 1 ? SIZE / 2 : 0), 0.0f, 10.0f});
        if (prev)
            new Joint(solver, prev, curr, {0.5f, 0, 0}, i == N - 1 ? float3{-SIZE / 2, 0, 0} : float3{-0.5f, 0, 0});
        prev = curr;
    }
}

static void sceneSpring(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});

    Rigid *anchor = new Rigid(solver, {1, 1, 1}, 0.0f, 0.5f, {0, 0, 14.0f});
    Rigid *block = new Rigid(solver, {2, 2, 2}, 1.0f, 0.5f, {0, 0, 8.0f});
    new Spring(solver, anchor, block, {0, 0, 0}, {0, 0, 0}, 100.0f, 4.0f);
}

static void sceneSpringsRatio(Solver *solver)
{
    const int N = 8;
    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, -10});

    Rigid *prev = 0;
    for (int i = 0; i < N; i++)
    {
        float x = (i - (N - 1) * 0.5f) * 3.0f;
        Rigid *curr = new Rigid(solver, {1, 0.75f, 0.75f}, i == 0 || i == N - 1 ? 0.0f : 1.0f, 0.5f, {x, 0.0f, 12.0f});
        if (prev)
            new Spring(solver, prev, curr, {0.5f, 0, 0}, {-0.5f, 0, 0}, i % 2 == 0 ? 10.0f : 10000.0f, 3.0f);
        prev = curr;
    }
}

static void sceneStack(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});
    for (int i = 0; i < 10; i++)
        new Rigid(solver, {1, 1, 1}, 1.0f, 0.5f, {0, 0, i * 1.5f + 1.0f});
}

static void sceneStackRatio(Solver *solver)
{
    solver->clear();
    const float groundThickness = 1.0f;
    new Rigid(solver, {100, 100, groundThickness}, 0.0f, 0.5f, {0, 0, 0});

    float topZ = groundThickness * 0.5f;
    float s = 1.0f;
    for (int i = 0; i < 4; i++)
    {
        float half = s * 0.5f;
        float centerZ = topZ + half;
        new Rigid(solver, {s, s, s}, 1.0f, 0.5f, {0, 0, centerZ});
        topZ = centerZ + half;
        s *= 2.0f;
    }
}

static void sceneSphereStack(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {1000, 1000, 1}, 0.0f, 0.5f, {0, 0, 0});

    const float radius = 0.5f;
    for (int i = 0; i < 10; i++)
        Rigid::makeSphere(solver, radius, 1.0f, 0.5f, {0, 0, 0.5f + radius + i * radius * 2.05f});
}

static void sceneSphereRamp(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {1000, 1000, 1}, 0.0f, 0.5f, {0, 0, 0});

    const float angle = rad(20.0f);
    Rigid *ramp = new Rigid(solver, {36, 16, 1}, 0.0f, 0.8f, {0, 0, 3});
    ramp->positionAng = {0, sinf(angle * 0.5f), 0, cosf(angle * 0.5f)};

    float3 rampTangent = normalize(rotate(ramp->positionAng, float3{1, 0, 0}));
    float3 rampNormal = normalize(rotate(ramp->positionAng, float3{0, 0, 1}));
    const float radius = 0.55f;

    for (int i = 0; i < 8; i++)
    {
        float friction = 0.1f + i * 0.2f;
        float3 pos = ramp->positionLin + rampTangent * (-12.0f + i * 1.4f) + float3{0, -3.5f + i, 0} + rampNormal * (0.5f + radius + 0.02f);
        Rigid::makeSphere(solver, radius, 1.0f, friction, pos);
    }
}

static void sceneCapsuleStack(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {1000, 1000, 1}, 0.0f, 0.5f, {0, 0, 0});

    const float radius = 0.35f;
    const float halfLength = 0.9f;
    for (int i = 0; i < 10; i++)
    {
        Rigid *capsule = Rigid::makeCapsule(solver, radius, halfLength, 1.0f, 0.5f, {0, 0, 0.5f + radius + i * radius * 2.4f});
        float angle = i % 2 == 0 ? rad(90.0f) : rad(90.0f);
        capsule->positionAng = i % 2 == 0
            ? quat{0.0f, sinf(angle * 0.5f), 0.0f, cosf(angle * 0.5f)}
            : quat{sinf(angle * 0.5f), 0.0f, 0.0f, cosf(angle * 0.5f)};
    }
}

static void sceneCapsuleRamp(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {1000, 1000, 1}, 0.0f, 0.5f, {0, 0, 0});

    const float angle = rad(20.0f);
    Rigid *ramp = new Rigid(solver, {36, 16, 1}, 0.0f, 0.8f, {0, 0, 3});
    ramp->positionAng = {0, sinf(angle * 0.5f), 0, cosf(angle * 0.5f)};

    float3 rampTangent = normalize(rotate(ramp->positionAng, float3{1, 0, 0}));
    float3 rampNormal = normalize(rotate(ramp->positionAng, float3{0, 0, 1}));
    const float radius = 0.35f;
    const float halfLength = 0.9f;

    for (int i = 0; i < 6; i++)
    {
        float friction = 0.2f + i * 0.15f;
        float3 pos = ramp->positionLin + rampTangent * (-12.0f + i * 2.0f) + float3{0, -2.5f + i, 0} + rampNormal * (0.5f + radius + 0.03f);
        Rigid *capsule = Rigid::makeCapsule(solver, radius, halfLength, 1.0f, friction, pos);
        float layDown = rad(90.0f);
        capsule->positionAng = {0.0f, sinf(layDown * 0.5f), 0.0f, cosf(layDown * 0.5f)};
    }
}

static void sceneCylinderStack(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {1000, 1000, 1}, 0.0f, 0.6f, {0, 0, 0});

    const float radius = 0.48f;
    const float halfLength = 0.35f;
    for (int i = 0; i < 10; i++)
    {
        float z = 0.5f + halfLength + i * (halfLength * 2.0f + 0.025f);
        Rigid::makeCylinder(solver, radius, halfLength, 1.0f, 0.6f, {0, 0, z});
    }
}

static void sceneCylinderRamp(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {1000, 1000, 1}, 0.0f, 0.5f, {0, 0, 0});

    const float angle = rad(20.0f);
    Rigid *ramp = new Rigid(solver, {36, 16, 1}, 0.0f, 0.8f, {0, 0, 3});
    ramp->positionAng = {0, sinf(angle * 0.5f), 0, cosf(angle * 0.5f)};

    float3 rampTangent = normalize(rotate(ramp->positionAng, float3{1, 0, 0}));
    float3 rampNormal = normalize(rotate(ramp->positionAng, float3{0, 0, 1}));
    const float radius = 0.48f;
    const float halfLength = 0.65f;
    const float layAcross = rad(-90.0f);

    for (int i = 0; i < 7; i++)
    {
        float friction = 0.2f + i * 0.12f;
        float3 pos = ramp->positionLin + rampTangent * (-12.0f + i * 2.1f) + float3{0, -3.0f + i, 0} + rampNormal * (0.5f + radius + 0.03f);
        Rigid *cylinder = Rigid::makeCylinder(solver, radius, halfLength, 1.0f, friction, pos);
        cylinder->positionAng = {sinf(layAcross * 0.5f), 0.0f, 0.0f, cosf(layAcross * 0.5f)};
    }
}

static float sceneJitter(int value)
{
    unsigned int x = (unsigned int)value * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737u;
    x = (x >> 22) ^ x;
    return (float)(x & 0xffffu) / 32767.5f - 1.0f;
}

static void sceneSpherePourOnCylindersVariant(Solver *solver, int sphereCount, int width, int depth, float startZ, float layerSpacingScale, float streamDrift, float forwardVelocity = 1.1f, float downwardVelocity = -1.5f, bool taperColumn = true)
{
    solver->clear();
    new Rigid(solver, {1000, 1000, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float cylinderRadius = 0.45f;
    const float cylinderHalfLength = 0.55f;
    const float cylinderZ = 0.5f + cylinderHalfLength;
    const float cylinderFriction = 0.85f;
    const float cylinderSpacing = 1.7f;
    const float cylinderPositions[7][2] = {
        {0.0f, 0.0f},
        {-cylinderSpacing, 0.0f},
        {cylinderSpacing, 0.0f},
        {-cylinderSpacing * 0.5f, cylinderSpacing * 0.9f},
        {cylinderSpacing * 0.5f, cylinderSpacing * 0.9f},
        {-cylinderSpacing * 0.5f, -cylinderSpacing * 0.9f},
        {cylinderSpacing * 0.5f, -cylinderSpacing * 0.9f}};

    for (int i = 0; i < 7; ++i)
    {
        Rigid::makeCylinder(solver, cylinderRadius, cylinderHalfLength, 1.0f, cylinderFriction,
            {cylinderPositions[i][0], cylinderPositions[i][1], cylinderZ});
    }

    const float sphereRadius = 0.12f;
    const float spacing = sphereRadius * 2.35f;
    const float layerSpacing = spacing * layerSpacingScale;
    int count = 0;

    for (int layer = 0; count < sphereCount; ++layer)
    {
        for (int y = 0; y < depth && count < sphereCount; ++y)
        {
            for (int x = 0; x < width && count < sphereCount; ++x)
            {
                float jitterX = sceneJitter(count * 3 + 0) * sphereRadius * 0.35f;
                float jitterY = sceneJitter(count * 3 + 1) * sphereRadius * 0.35f;
                float jitterZ = sceneJitter(count * 3 + 2) * sphereRadius * 0.20f;
                float columnTaper = taperColumn ? 1.0f - min(layer / 18.0f, 0.45f) : 1.0f;
                float px = ((float)x - (width - 1) * 0.5f) * spacing * columnTaper + jitterX - 1.2f + layer * streamDrift;
                float py = ((float)y - (depth - 1) * 0.5f) * spacing * columnTaper + jitterY - layer * streamDrift * 0.35f;
                float pz = startZ + layer * layerSpacing + jitterZ;
                float3 velocity = {
                    forwardVelocity + sceneJitter(count * 5 + 0) * 0.18f,
                    sceneJitter(count * 5 + 1) * 0.35f,
                    downwardVelocity + sceneJitter(count * 5 + 2) * 0.18f};

                Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.45f, {px, py, pz}, velocity);
                ++count;
            }
        }
    }
}

static void sceneSpherePourOnCylinders(Solver *solver)
{
    sceneSpherePourOnCylindersVariant(solver, 1000, 8, 8, 10.0f, 1.0f, 0.0f);
}

static void sceneSpherePour5000OnCylinders(Solver *solver)
{
    sceneSpherePourOnCylindersVariant(solver, 5000, 14, 8, 9.0f, 1.45f, 0.0f, 0.35f, -1.0f, false);
}

static void sceneSpherePour20000OnCylinders(Solver *solver)
{
    sceneSpherePourOnCylindersVariant(solver, 20000, 20, 14, 9.0f, 1.45f, 0.0f, 0.35f, -1.0f, false);
}

static void sceneSphereSphereContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float sphereRadius = 0.35f;
    const float centerZ = 4.0f;
    Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, {-sphereRadius * 0.45f, 0.0f, centerZ});
    Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, {sphereRadius * 0.45f, 0.0f, centerZ});
}

static void sceneSphereCylinderContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float cylinderRadius = 0.45f;
    const float cylinderHalfLength = 0.55f;
    const float sphereRadius = 0.2f;
    const float centerZ = 4.0f;
    Rigid::makeCylinder(solver, cylinderRadius, cylinderHalfLength, 1.0f, 0.8f, {0.0f, 0.0f, centerZ});
    Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, {cylinderRadius + sphereRadius * 0.45f, 0.0f, centerZ});
}

static void sceneSphereBoxContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float sphereRadius = 0.2f;
    const float boxTop = 4.0f;
    new Rigid(solver, {0.8f, 0.8f, 1.0f}, 1.0f, 0.7f, {0.0f, 0.0f, boxTop});
    Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, {0.4f + sphereRadius * 0.45f, 0.0f, boxTop});
}

static void sceneSphereCapsuleContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float capsuleRadius = 0.35f;
    const float capsuleHalfLength = 0.55f;
    const float sphereRadius = 0.2f;
    const float centerZ = 4.0f;
    Rigid::makeCapsule(solver, capsuleRadius, capsuleHalfLength, 1.0f, 0.8f, {0.0f, 0.0f, centerZ});
    Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, {capsuleRadius + sphereRadius * 0.45f, 0.0f, centerZ});
}

static void sceneSphereRotatedCylinderContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float cylinderRadius = 0.45f;
    const float cylinderHalfLength = 0.55f;
    const float sphereRadius = 0.2f;
    const float angle = 0.65f;
    Rigid *cylinder = Rigid::makeCylinder(solver, cylinderRadius, cylinderHalfLength, 0.0f, 0.8f, {0.0f, 0.0f, 2.0f});
    cylinder->positionAng = {0.0f, sinf(angle * 0.5f), 0.0f, cosf(angle * 0.5f)};
    float3 spherePos = cylinder->positionLin + rotate(cylinder->positionAng, float3{cylinderRadius + sphereRadius * 0.45f, 0.0f, 0.0f});
    Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, spherePos);
}

static void sceneSphereRotatedCapsuleContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float capsuleRadius = 0.35f;
    const float capsuleHalfLength = 0.55f;
    const float sphereRadius = 0.2f;
    const float angle = 0.65f;
    Rigid *capsule = Rigid::makeCapsule(solver, capsuleRadius, capsuleHalfLength, 0.0f, 0.8f, {0.0f, 0.0f, 2.0f});
    capsule->positionAng = {0.0f, sinf(angle * 0.5f), 0.0f, cosf(angle * 0.5f)};
    float3 spherePos = capsule->positionLin + rotate(capsule->positionAng, float3{capsuleRadius + sphereRadius * 0.45f, 0.0f, 0.0f});
    Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, spherePos);
}

static void sceneSphereRotatedBoxContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float sphereRadius = 0.2f;
    const float angle = 0.7f;
    Rigid *box = new Rigid(solver, {0.8f, 0.6f, 1.0f}, 0.0f, 0.7f, {0.0f, 0.0f, 2.0f});
    box->positionAng = {0.0f, 0.0f, sinf(angle * 0.5f), cosf(angle * 0.5f)};
    float3 spherePos = box->positionLin + rotate(box->positionAng, float3{0.4f + sphereRadius * 0.45f, 0.0f, 0.0f});
    Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, spherePos);
}

static void sceneSphereDynamicRotatedCylinderContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float cylinderRadius = 0.45f;
    const float cylinderHalfLength = 0.55f;
    const float sphereRadius = 0.2f;
    const float angle = 0.65f;
    Rigid *cylinder = Rigid::makeCylinder(solver, cylinderRadius, cylinderHalfLength, 1.0f, 0.8f, {0.0f, 0.0f, 2.0f});
    cylinder->positionAng = {0.0f, sinf(angle * 0.5f), 0.0f, cosf(angle * 0.5f)};
    float3 spherePos = cylinder->positionLin + rotate(cylinder->positionAng, float3{cylinderRadius + sphereRadius * 0.45f, 0.0f, 0.0f});
    Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, spherePos);
}

static void sceneSphereDynamicRotatedCapsuleContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float capsuleRadius = 0.35f;
    const float capsuleHalfLength = 0.55f;
    const float sphereRadius = 0.2f;
    const float angle = 0.65f;
    Rigid *capsule = Rigid::makeCapsule(solver, capsuleRadius, capsuleHalfLength, 1.0f, 0.8f, {0.0f, 0.0f, 2.0f});
    capsule->positionAng = {0.0f, sinf(angle * 0.5f), 0.0f, cosf(angle * 0.5f)};
    float3 spherePos = capsule->positionLin + rotate(capsule->positionAng, float3{capsuleRadius + sphereRadius * 0.45f, 0.0f, 0.0f});
    Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, spherePos);
}

static void sceneSphereDynamicRotatedBoxContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float sphereRadius = 0.2f;
    const float angle = 0.7f;
    Rigid *box = new Rigid(solver, {0.8f, 0.6f, 1.0f}, 1.0f, 0.7f, {0.0f, 0.0f, 2.0f});
    box->positionAng = {0.0f, 0.0f, sinf(angle * 0.5f), cosf(angle * 0.5f)};
    float3 spherePos = box->positionLin + rotate(box->positionAng, float3{0.4f + sphereRadius * 0.45f, 0.0f, 0.0f});
    Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, spherePos);
}

static void sceneCapsuleCapsuleContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float radius = 0.35f;
    const float halfLength = 0.55f;
    const float centerZ = 4.0f;
    Rigid::makeCapsule(solver, radius, halfLength, 1.0f, 0.8f, {0.0f, 0.0f, centerZ});
    Rigid::makeCapsule(solver, radius, halfLength, 1.0f, 0.8f, {radius * 1.2f, 0.0f, centerZ});
}

static void sceneCylinderCylinderContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float radius = 0.45f;
    const float halfLength = 0.55f;
    const float centerZ = 4.0f;
    Rigid::makeCylinder(solver, radius, halfLength, 1.0f, 0.8f, {0.0f, 0.0f, centerZ});
    Rigid::makeCylinder(solver, radius, halfLength, 1.0f, 0.8f, {radius * 1.2f, 0.0f, centerZ});
}

static void sceneCapsuleCylinderContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float capsuleRadius = 0.35f;
    const float cylinderRadius = 0.45f;
    const float halfLength = 0.55f;
    const float centerZ = 4.0f;
    Rigid::makeCapsule(solver, capsuleRadius, halfLength, 1.0f, 0.8f, {0.0f, 0.0f, centerZ});
    Rigid::makeCylinder(solver, cylinderRadius, halfLength, 1.0f, 0.8f, {capsuleRadius + cylinderRadius * 0.45f, 0.0f, centerZ});
}

static void sceneRotatedCapsuleCylinderContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float capsuleRadius = 0.35f;
    const float cylinderRadius = 0.45f;
    const float halfLength = 0.55f;
    const float centerZ = 4.0f;
    const float capsuleAngle = 0.65f;
    const float cylinderAngle = -0.55f;
    Rigid *capsule = Rigid::makeCapsule(solver, capsuleRadius, halfLength, 1.0f, 0.8f, {0.0f, 0.0f, centerZ});
    capsule->positionAng = {0.0f, sinf(capsuleAngle * 0.5f), 0.0f, cosf(capsuleAngle * 0.5f)};
    float3 offset = rotate(capsule->positionAng, float3{capsuleRadius + cylinderRadius * 0.45f, 0.0f, 0.0f});
    Rigid *cylinder = Rigid::makeCylinder(solver, cylinderRadius, halfLength, 1.0f, 0.8f, capsule->positionLin + offset);
    cylinder->positionAng = {sinf(cylinderAngle * 0.5f), 0.0f, 0.0f, cosf(cylinderAngle * 0.5f)};
}

static void sceneMixedPrimitiveContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {32, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float z = 2.0f;
    const float sphereRadius = 0.2f;
    const float capsuleRadius = 0.35f;
    const float cylinderRadius = 0.45f;
    const float halfLength = 0.55f;

    {
        Rigid *cylinder = Rigid::makeCylinder(solver, cylinderRadius, halfLength, 1.0f, 0.8f, {-6.0f, 0.0f, z});
        const float angle = 0.5f;
        cylinder->positionAng = {0.0f, sinf(angle * 0.5f), 0.0f, cosf(angle * 0.5f)};
        float3 spherePos = cylinder->positionLin + rotate(cylinder->positionAng, float3{cylinderRadius + sphereRadius * 0.45f, 0.0f, 0.0f});
        Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, spherePos);
    }

    {
        Rigid *capsule = Rigid::makeCapsule(solver, capsuleRadius, halfLength, 1.0f, 0.8f, {-3.5f, 0.0f, z});
        const float angle = 0.55f;
        capsule->positionAng = {0.0f, sinf(angle * 0.5f), 0.0f, cosf(angle * 0.5f)};
        float3 spherePos = capsule->positionLin + rotate(capsule->positionAng, float3{capsuleRadius + sphereRadius * 0.45f, 0.0f, 0.0f});
        Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, spherePos);
    }

    {
        Rigid *box = new Rigid(solver, {0.8f, 0.6f, 1.0f}, 1.0f, 0.7f, {-1.0f, 0.0f, z});
        const float angle = 0.65f;
        box->positionAng = {0.0f, 0.0f, sinf(angle * 0.5f), cosf(angle * 0.5f)};
        float3 spherePos = box->positionLin + rotate(box->positionAng, float3{0.4f + sphereRadius * 0.45f, 0.0f, 0.0f});
        Rigid::makeSphere(solver, sphereRadius, 1.0f, 0.5f, spherePos);
    }

    Rigid::makeCapsule(solver, capsuleRadius, halfLength, 1.0f, 0.8f, {1.5f, 0.0f, z});
    Rigid::makeCapsule(solver, capsuleRadius, halfLength, 1.0f, 0.8f, {1.5f + capsuleRadius * 1.2f, 0.0f, z});

    Rigid::makeCylinder(solver, cylinderRadius, halfLength, 1.0f, 0.8f, {4.0f, 0.0f, z});
    Rigid::makeCylinder(solver, cylinderRadius, halfLength, 1.0f, 0.8f, {4.0f + cylinderRadius * 1.2f, 0.0f, z});

    {
        Rigid *capsule = Rigid::makeCapsule(solver, capsuleRadius, halfLength, 1.0f, 0.8f, {6.5f, 0.0f, z});
        Rigid *cylinder = Rigid::makeCylinder(solver, cylinderRadius, halfLength, 1.0f, 0.8f, {6.5f + capsuleRadius + cylinderRadius * 0.45f, 0.0f, z});
        const float capsuleAngle = 0.45f;
        const float cylinderAngle = -0.35f;
        capsule->positionAng = {0.0f, sinf(capsuleAngle * 0.5f), 0.0f, cosf(capsuleAngle * 0.5f)};
        cylinder->positionAng = {sinf(cylinderAngle * 0.5f), 0.0f, 0.0f, cosf(cylinderAngle * 0.5f)};
    }

    {
        new Rigid(solver, {0.8f, 0.6f, 1.0f}, 1.0f, 0.7f, {-2.0f, 3.0f, z});
        Rigid::makeCapsule(solver, capsuleRadius, halfLength, 1.0f, 0.8f, {-2.0f + 0.4f + capsuleRadius * 0.45f, 3.0f, z});
    }

    {
        new Rigid(solver, {0.8f, 0.6f, 1.0f}, 1.0f, 0.7f, {2.0f, 3.0f, z});
        Rigid::makeCylinder(solver, cylinderRadius, halfLength, 1.0f, 0.8f, {2.0f + 0.4f + cylinderRadius * 0.45f, 3.0f, z});
    }
}

static void sceneCapsuleBoxContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float capsuleRadius = 0.35f;
    const float halfLength = 0.55f;
    const float centerZ = 4.0f;
    new Rigid(solver, {0.8f, 0.6f, 1.0f}, 1.0f, 0.7f, {0.0f, 0.0f, centerZ});
    Rigid::makeCapsule(solver, capsuleRadius, halfLength, 1.0f, 0.8f, {0.4f + capsuleRadius * 0.45f, 0.0f, centerZ});
}

static void sceneCylinderBoxContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float cylinderRadius = 0.45f;
    const float halfLength = 0.55f;
    const float centerZ = 2.0f;
    new Rigid(solver, {0.8f, 0.6f, 1.0f}, 1.0f, 0.7f, {0.0f, 0.0f, centerZ});
    Rigid::makeCylinder(solver, cylinderRadius, halfLength, 1.0f, 0.8f, {0.4f + cylinderRadius * 0.45f, 0.0f, centerZ});
}

static void sceneBoxBoxContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float centerZ = 4.0f;
    new Rigid(solver, {0.9f, 0.65f, 1.0f}, 1.0f, 0.7f, {-0.2f, 0.0f, centerZ});
    new Rigid(solver, {0.9f, 0.65f, 1.0f}, 1.0f, 0.7f, {0.45f, 0.0f, centerZ});
}

static void sceneRotatedBoxBoxContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float centerZ = 4.0f;
    Rigid *a = new Rigid(solver, {1.1f, 0.65f, 1.0f}, 1.0f, 0.7f, {-0.18f, 0.0f, centerZ});
    Rigid *b = new Rigid(solver, {1.1f, 0.65f, 1.0f}, 1.0f, 0.7f, {0.42f, 0.03f, centerZ});
    const float angleA = 0.55f;
    const float angleB = -0.35f;
    a->positionAng = {0.0f, 0.0f, sinf(angleA * 0.5f), cosf(angleA * 0.5f)};
    b->positionAng = {0.0f, 0.0f, sinf(angleB * 0.5f), cosf(angleB * 0.5f)};
}

static void sceneBoxClusterContactProbe(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {24, 24, 1}, 0.0f, 0.7f, {0, 0, 0});

    const float centerZ = 4.0f;
    const float spacing = 0.78f;
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x)
        {
            float3 position = {
                (float)(x - 1) * spacing + ((y & 1) ? 0.10f : 0.0f),
                (float)(y - 1) * spacing * 0.85f,
                centerZ};
            Rigid *body = new Rigid(solver, {0.9f, 0.75f, 1.0f}, 1.0f, 0.7f, position);
            float angle = 0.10f * (float)(x - y);
            body->positionAng = {0.0f, 0.0f, sinf(angle * 0.5f), cosf(angle * 0.5f)};
        }
}

static quat alignZToVector(float3 dir)
{
    dir = normalize(dir);
    float d = dot(float3{0.0f, 0.0f, 1.0f}, dir);
    if (d > 0.9999f)
        return {0.0f, 0.0f, 0.0f, 1.0f};
    if (d < -0.9999f)
        return {1.0f, 0.0f, 0.0f, 0.0f};

    float3 axis = cross(float3{0.0f, 0.0f, 1.0f}, dir);
    float s = sqrtf((1.0f + d) * 2.0f);
    float invS = 1.0f / s;
    return normalize(quat{axis.x * invS, axis.y * invS, axis.z * invS, s * 0.5f});
}

static Rigid *makeStringCapsule(Solver *solver, float3 a, float3 b, float radius, float density, float friction, float &halfLength)
{
    float3 span = b - a;
    float length = max(::length(span), radius * 2.0f);
    halfLength = length * 0.5f;

    Rigid *segment = Rigid::makeCapsule(solver, radius, halfLength, density, friction, (a + b) * 0.5f);
    segment->positionAng = alignZToVector(span);
    return segment;
}

static void addCapsuleString(Solver *solver, Rigid *ball, float3 anchor, float3 ballAnchorLocal, int segmentCount, float radius)
{
    float3 ballAnchorWorld = transform(ball->positionLin, ball->positionAng, ballAnchorLocal);
    Rigid *prev = 0;
    float prevHalf = 0.0f;

    for (int i = 0; i < segmentCount; ++i)
    {
        float t0 = (float)i / (float)segmentCount;
        float t1 = (float)(i + 1) / (float)segmentCount;
        float3 a = anchor + (ballAnchorWorld - anchor) * t0;
        float3 b = anchor + (ballAnchorWorld - anchor) * t1;

        float halfLength = 0.0f;
        Rigid *segment = makeStringCapsule(solver, a, b, radius, 0.08f, 0.6f, halfLength);
        if (prev)
            new Joint(solver, prev, segment, {0.0f, 0.0f, prevHalf}, {0.0f, 0.0f, -halfLength}, INFINITY, 0.0f);
        else
            new Joint(solver, 0, segment, anchor, {0.0f, 0.0f, -halfLength}, INFINITY, 0.0f);

        prev = segment;
        prevHalf = halfLength;
    }

    if (prev)
        new Joint(solver, prev, ball, {0.0f, 0.0f, prevHalf}, ballAnchorLocal, INFINITY, 0.0f);
}

static void sceneNewtonsCradle(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {20, 16, 1}, 0.0f, 0.5f, {0, 0, -1.0f});

    new Rigid(solver, {7.0f, 0.16f, 0.16f}, 0.0f, 0.5f, {0, -0.65f, 8.0f});
    new Rigid(solver, {7.0f, 0.16f, 0.16f}, 0.0f, 0.5f, {0, 0.65f, 8.0f});
    new Rigid(solver, {0.16f, 0.16f, 7.5f}, 0.0f, 0.5f, {-3.5f, -0.65f, 4.2f});
    new Rigid(solver, {0.16f, 0.16f, 7.5f}, 0.0f, 0.5f, {3.5f, -0.65f, 4.2f});
    new Rigid(solver, {0.16f, 0.16f, 7.5f}, 0.0f, 0.5f, {-3.5f, 0.65f, 4.2f});
    new Rigid(solver, {0.16f, 0.16f, 7.5f}, 0.0f, 0.5f, {3.5f, 0.65f, 4.2f});

    const int ballCount = 5;
    const int stringSegments = 5;
    const float ballRadius = 0.45f;
    const float spacing = ballRadius * 2.02f;
    const float anchorZ = 8.0f;
    const float restBallZ = 3.25f;
    const float stringY = 0.48f;
    const float stringRadius = 0.045f;
    const float pullX = -1.65f;

    Rigid *balls[ballCount];
    for (int i = 0; i < ballCount; ++i)
    {
        float restX = (i - (ballCount - 1) * 0.5f) * spacing;
        float x = restX;
        float z = restBallZ;
        float vx = 0.0f;
        if (i == 0)
        {
            float stringLength = anchorZ - (restBallZ + ballRadius * 0.7f);
            x += pullX;
            z = anchorZ - sqrtf(max(stringLength * stringLength - pullX * pullX, 0.0f)) - ballRadius * 0.7f;
            vx = 1.0f;
        }

        balls[i] = Rigid::makeSphere(solver, ballRadius, 1.0f, 0.65f, {x, 0.0f, z}, {vx, 0.0f, 0.0f});
        float3 frontAnchor = {restX, -stringY, anchorZ};
        float3 backAnchor = {restX, stringY, anchorZ};
        addCapsuleString(solver, balls[i], frontAnchor, {0.0f, -stringY * 0.55f, ballRadius * 0.7f}, stringSegments, stringRadius);
        addCapsuleString(solver, balls[i], backAnchor, {0.0f, stringY * 0.55f, ballRadius * 0.7f}, stringSegments, stringRadius);
    }

    for (int i = 0; i < ballCount; ++i)
    {
        for (Force *force = solver->forces; force != 0; force = force->next)
        {
            Rigid *other = 0;
            if (force->bodyA == balls[i])
                other = force->bodyB;
            else if (force->bodyB == balls[i])
                other = force->bodyA;

            if (other && other->shape.type == RIGID_SHAPE_CAPSULE)
                new IgnoreCollision(solver, balls[i], other);
        }
    }
}

static void sceneSoftBody(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});

    const float Klin = 5000.0f;
    const float Kang = 1000.0f;
    const int W = 4;
    const int D = 4;
    const int H = 4;
    const int N = 3;
    const float size = 0.8f;
    const float half = size * 0.5f;
    const float baseZ = 8.0f;
    const float stackGap = 2.0f;

    for (int i = 0; i < N; i++)
    {
        Rigid *grid[W][D][H];
        float stackX = (i - (N - 1) * 0.5f) * (W * size + stackGap);
        float stackZ = 0.0f;

        for (int x = 0; x < W; x++)
        {
            for (int y = 0; y < D; y++)
            {
                for (int z = 0; z < H; z++)
                {
                    float px = stackX + (x - (W - 1) * 0.5f) * size;
                    float py = (y - (D - 1) * 0.5f) * size;
                    float pz = baseZ + stackZ + z * size;
                    grid[x][y][z] = new Rigid(solver, {size, size, size}, 1.0f, 0.5f, {px, py, pz});
                }
            }
        }

        for (int x = 1; x < W; x++)
        {
            for (int y = 0; y < D; y++)
            {
                for (int z = 0; z < H; z++)
                {
                    new Joint(solver, grid[x - 1][y][z], grid[x][y][z], {half, 0, 0}, {-half, 0, 0}, Klin, Kang);
                }
            }
        }

        for (int x = 0; x < W; x++)
        {
            for (int y = 1; y < D; y++)
            {
                for (int z = 0; z < H; z++)
                {
                    new Joint(solver, grid[x][y - 1][z], grid[x][y][z], {0, half, 0}, {0, -half, 0}, Klin, Kang);
                }
            }
        }

        for (int x = 0; x < W; x++)
        {
            for (int y = 0; y < D; y++)
            {
                for (int z = 1; z < H; z++)
                {
                    new Joint(solver, grid[x][y][z - 1], grid[x][y][z], {0, 0, half}, {0, 0, -half}, Klin, Kang);
                }
            }
        }

        for (int x = 1; x < W; x++)
        {
            for (int y = 0; y < D; y++)
            {
                for (int z = 1; z < H; z++)
                {
                    new IgnoreCollision(solver, grid[x - 1][y][z - 1], grid[x][y][z]);
                    new IgnoreCollision(solver, grid[x][y][z - 1], grid[x - 1][y][z]);
                }
            }
        }

        for (int x = 0; x < W; x++)
        {
            for (int y = 1; y < D; y++)
            {
                for (int z = 1; z < H; z++)
                {
                    new IgnoreCollision(solver, grid[x][y - 1][z - 1], grid[x][y][z]);
                    new IgnoreCollision(solver, grid[x][y][z - 1], grid[x][y - 1][z]);
                }
            }
        }

        for (int x = 1; x < W; x++)
        {
            for (int y = 1; y < D; y++)
            {
                for (int z = 0; z < H; z++)
                {
                    new IgnoreCollision(solver, grid[x - 1][y - 1][z], grid[x][y][z]);
                    new IgnoreCollision(solver, grid[x][y - 1][z], grid[x - 1][y][z]);
                }
            }
        }
    }
}

static void sceneSoftBodyFine(Solver *solver)
{
    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});

    const float Klin = 4000.0f;
    const float Kang = 800.0f;
    const int W = 8;
    const int D = 8;
    const int H = 8;
    const int N = 3;
    const float size = 0.4f;
    const float half = size * 0.5f;
    const float baseZ = 8.0f;
    const float stackGap = 2.0f;

    for (int i = 0; i < N; i++)
    {
        Rigid *grid[W][D][H];
        float stackX = (i - (N - 1) * 0.5f) * (W * size + stackGap);
        float stackZ = 0.0f;

        for (int x = 0; x < W; x++)
        {
            for (int y = 0; y < D; y++)
            {
                for (int z = 0; z < H; z++)
                {
                    float px = stackX + (x - (W - 1) * 0.5f) * size;
                    float py = (y - (D - 1) * 0.5f) * size;
                    float pz = baseZ + stackZ + z * size;
                    grid[x][y][z] = new Rigid(solver, {size, size, size}, 1.0f, 0.5f, {px, py, pz});
                }
            }
        }

        for (int x = 1; x < W; x++)
        {
            for (int y = 0; y < D; y++)
            {
                for (int z = 0; z < H; z++)
                {
                    new Joint(solver, grid[x - 1][y][z], grid[x][y][z], {half, 0, 0}, {-half, 0, 0}, Klin, Kang);
                }
            }
        }

        for (int x = 0; x < W; x++)
        {
            for (int y = 1; y < D; y++)
            {
                for (int z = 0; z < H; z++)
                {
                    new Joint(solver, grid[x][y - 1][z], grid[x][y][z], {0, half, 0}, {0, -half, 0}, Klin, Kang);
                }
            }
        }

        for (int x = 0; x < W; x++)
        {
            for (int y = 0; y < D; y++)
            {
                for (int z = 1; z < H; z++)
                {
                    new Joint(solver, grid[x][y][z - 1], grid[x][y][z], {0, 0, half}, {0, 0, -half}, Klin, Kang);
                }
            }
        }

        for (int x = 1; x < W; x++)
        {
            for (int y = 0; y < D; y++)
            {
                for (int z = 1; z < H; z++)
                {
                    new IgnoreCollision(solver, grid[x - 1][y][z - 1], grid[x][y][z]);
                    new IgnoreCollision(solver, grid[x][y][z - 1], grid[x - 1][y][z]);
                }
            }
        }

        for (int x = 0; x < W; x++)
        {
            for (int y = 1; y < D; y++)
            {
                for (int z = 1; z < H; z++)
                {
                    new IgnoreCollision(solver, grid[x][y - 1][z - 1], grid[x][y][z]);
                    new IgnoreCollision(solver, grid[x][y][z - 1], grid[x][y - 1][z]);
                }
            }
        }

        for (int x = 1; x < W; x++)
        {
            for (int y = 1; y < D; y++)
            {
                for (int z = 0; z < H; z++)
                {
                    new IgnoreCollision(solver, grid[x - 1][y - 1][z], grid[x][y][z]);
                    new IgnoreCollision(solver, grid[x][y - 1][z], grid[x - 1][y][z]);
                }
            }
        }
    }
}

static void sceneBridge(Solver *solver)
{
    const int N = 40;
    const float plankLength = 1.0f;
    const float plankWidth = 4.0f;
    const float plankHeight = 0.5f;
    const float halfLength = plankLength * 0.5f;
    const float halfWidth = plankWidth * 0.5f;

    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});

    Rigid *prev = 0;
    for (int i = 0; i < N; i++)
    {
        Rigid *curr = new Rigid(solver, {plankLength, plankWidth, plankHeight}, i == 0 || i == N - 1 ? 0.0f : 1.0f, 0.5f, {(float)i - N / 2.0f, 0.0f, 10.0f});
        if (prev)
        {
            new Joint(solver, prev, curr, {halfLength, halfWidth, 0}, {-halfLength, halfWidth, 0}, INFINITY, 0.0f);
            new Joint(solver, prev, curr, {halfLength, -halfWidth, 0}, {-halfLength, -halfWidth, 0}, INFINITY, 0.0f);
        }
        prev = curr;
    }

    for (int x = 0; x < N / 4; x++)
    {
        for (int y = 0; y < N / 8; y++)
        {
            new Rigid(solver, {1, 1, 1}, 1.0f, 0.5f, {(float)x - N / 8.0f, 0.0f, (float)y + 12.0f});
        }
    }
}

static void sceneBreakable(Solver *solver)
{
    const int N = 10;
    const int M = 5;
    const float breakForce = 90.0f;

    solver->clear();
    new Rigid(solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});

    Rigid *prev = 0;
    for (int i = 0; i <= N; i++)
    {
        Rigid *curr = new Rigid(solver, {1, 1, 0.5f}, 1.0f, 0.5f, {(float)i - N / 2.0f, 0.0f, 6.0f});
        if (prev)
            new Joint(solver, prev, curr, {0.5f, 0, 0}, {-0.5f, 0, 0}, INFINITY, INFINITY, breakForce);
        prev = curr;
    }

    new Rigid(solver, {1, 1, 5}, 0.0f, 0.5f, {-N / 2.0f, 0, 2.5f});
    new Rigid(solver, {1, 1, 5}, 0.0f, 0.5f, {N / 2.0f, 0, 2.5f});

    for (int i = 0; i < M; i++)
        new Rigid(solver, {2, 1, 1}, 1.0f, 0.5f, {0, 0, i * 2.0f + 8.0f});
}

static void (*scenes[])(Solver *) = {
    sceneEmpty,
    sceneGround,
    sceneDynamicFriction,
    sceneStaticFriction,
    scenePyramid,
    sceneRope,
    sceneHeavyRope,
    sceneSpring,
    sceneSpringsRatio,
    sceneStack,
    sceneStackRatio,
    sceneSphereStack,
    sceneSphereRamp,
    sceneCapsuleStack,
    sceneCapsuleRamp,
    sceneCylinderStack,
    sceneCylinderRamp,
    sceneSpherePourOnCylinders,
    sceneSpherePour5000OnCylinders,
    sceneSpherePour20000OnCylinders,
    sceneSphereSphereContactProbe,
    sceneSphereCylinderContactProbe,
    sceneSphereCapsuleContactProbe,
    sceneSphereBoxContactProbe,
    sceneSphereRotatedCylinderContactProbe,
    sceneSphereRotatedCapsuleContactProbe,
    sceneSphereRotatedBoxContactProbe,
    sceneSphereDynamicRotatedCylinderContactProbe,
    sceneSphereDynamicRotatedCapsuleContactProbe,
    sceneSphereDynamicRotatedBoxContactProbe,
    sceneCapsuleCapsuleContactProbe,
    sceneCylinderCylinderContactProbe,
    sceneCapsuleCylinderContactProbe,
    sceneRotatedCapsuleCylinderContactProbe,
    sceneMixedPrimitiveContactProbe,
    sceneCapsuleBoxContactProbe,
    sceneCylinderBoxContactProbe,
    sceneBoxBoxContactProbe,
    sceneRotatedBoxBoxContactProbe,
    sceneBoxClusterContactProbe,
    sceneNewtonsCradle,
    sceneSoftBody,
    sceneSoftBodyFine,
    sceneBridge,
    sceneBreakable};

static const char *sceneNames[] = {
    "Empty",
    "Ground",
    "Dynamic Friction",
    "Static Friction",
    "Pyramid",
    "Rope",
    "Heavy Rope",
    "Spring",
    "Spring Ratio",
    "Stack",
    "Stack Ratio",
    "Sphere Stack",
    "Sphere Ramp",
    "Capsule Stack",
    "Capsule Ramp",
    "Cylinder Stack",
    "Cylinder Ramp",
    "Sphere Pour on Cylinders",
    "Sphere Pour 5000 on Cylinders",
    "Sphere Pour 20000 on Cylinders",
    "Sphere Sphere Contact Probe",
    "Sphere Cylinder Contact Probe",
    "Sphere Capsule Contact Probe",
    "Sphere Box Contact Probe",
    "Sphere Rotated Cylinder Contact Probe",
    "Sphere Rotated Capsule Contact Probe",
    "Sphere Rotated Box Contact Probe",
    "Sphere Dynamic Rotated Cylinder Contact Probe",
    "Sphere Dynamic Rotated Capsule Contact Probe",
    "Sphere Dynamic Rotated Box Contact Probe",
    "Capsule Capsule Contact Probe",
    "Cylinder Cylinder Contact Probe",
    "Capsule Cylinder Contact Probe",
    "Rotated Capsule Cylinder Contact Probe",
    "Mixed Primitive Contact Probe",
    "Capsule Box Contact Probe",
    "Cylinder Box Contact Probe",
    "Box Box Contact Probe",
    "Rotated Box Box Contact Probe",
    "Box Cluster Contact Probe",
    "Newton's Cradle",
    "Soft Body",
    "Soft Body 8x8x8",
    "Bridge",
    "Breakable"};

static const int sceneCount = 45;
