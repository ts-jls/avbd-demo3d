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

namespace
{
bool isRoundShape(const Rigid *body)
{
    return body->shape.type == RIGID_SHAPE_SPHERE
        || body->shape.type == RIGID_SHAPE_CAPSULE
        || body->shape.type == RIGID_SHAPE_CYLINDER;
}

bool isCylinderCapContact(int featureKey)
{
    return (featureKey & 0xFFFFFF00) == 0x05000000
        || (featureKey & 0xFFFFFF00) == 0x06000000;
}

float3 contactOffsetWorld(Rigid *body, float3 localOffset, const float3x3 &basis, bool bodyA, int featureKey)
{
    if (body->shape.type == RIGID_SHAPE_SPHERE)
        return basis[0] * (bodyA ? -body->shape.radius : body->shape.radius);
    if (body->shape.type == RIGID_SHAPE_CAPSULE)
    {
        float3 axis = rotate(body->positionAng, float3{0.0f, 0.0f, 1.0f});
        float3 normal = basis[0];
        float3 normalLocal = rotate(conjugate(body->positionAng), normal);
        float centerZ = bodyA
            ? localOffset.z + normalLocal.z * body->shape.radius
            : localOffset.z - normalLocal.z * body->shape.radius;
        centerZ = clamp(centerZ, -body->shape.halfLength, body->shape.halfLength);
        float3 centerOffset = axis * centerZ;
        return centerOffset + normal * (bodyA ? -body->shape.radius : body->shape.radius);
    }
    if (body->shape.type == RIGID_SHAPE_CYLINDER)
    {
        float3 supportDirWorld = basis[0] * (bodyA ? -1.0f : 1.0f);
        float3 supportDirLocal = rotate(conjugate(body->positionAng), supportDirWorld);
        float storedRadialLen = sqrtf(localOffset.x * localOffset.x + localOffset.y * localOffset.y);

        if (isCylinderCapContact(featureKey))
        {
            float3 local = {0.0f, 0.0f, 0.0f};
            if (storedRadialLen > 1.0e-6f)
            {
                local.x = localOffset.x / storedRadialLen * body->shape.radius;
                local.y = localOffset.y / storedRadialLen * body->shape.radius;
            }

            float capSign = fabsf(localOffset.z) > 1.0e-6f
                ? (localOffset.z >= 0.0f ? 1.0f : -1.0f)
                : (supportDirLocal.z >= 0.0f ? 1.0f : -1.0f);
            local.z = body->shape.halfLength * capSign;
            return rotate(body->positionAng, local);
        }

        float radialLen = sqrtf(supportDirLocal.x * supportDirLocal.x + supportDirLocal.y * supportDirLocal.y);

        float z = fabsf(supportDirLocal.z) > 0.15f
            ? (supportDirLocal.z >= 0.0f ? body->shape.halfLength : -body->shape.halfLength)
            : clamp(localOffset.z, -body->shape.halfLength, body->shape.halfLength);

        float3 local = {0.0f, 0.0f, z};
        if (radialLen > 1.0e-6f)
        {
            local.x = supportDirLocal.x / radialLen * body->shape.radius;
            local.y = supportDirLocal.y / radialLen * body->shape.radius;
        }
        else
        {
            if (storedRadialLen > 1.0e-6f)
            {
                local.x = localOffset.x / storedRadialLen * body->shape.radius;
                local.y = localOffset.y / storedRadialLen * body->shape.radius;
            }
        }
        return rotate(body->positionAng, local);
    }
    return rotate(body->positionAng, localOffset);
}
}

Manifold::Manifold(Solver *solver, Rigid *bodyA, Rigid *bodyB)
    : Force(solver, bodyA, bodyB), numContacts(0)
{
    solver->world.setForceType(denseId, SIM_CONSTRAINT_MANIFOLD);
}

bool Manifold::initialize()
{
    // Compute friction
    friction = sqrtf(bodyA->friction * bodyB->friction);
    if (bodyA->shape.type == RIGID_SHAPE_SPHERE || bodyB->shape.type == RIGID_SHAPE_SPHERE)
        friction = 0.0f;

    // Compute new contacts
    Contact newContacts[8] = {0};
    int newNumContacts = 0;
    const ExternalManifoldContact *externalContact = solver->findExternalManifoldContact(bodyA->denseId, bodyB->denseId);
    if (externalContact)
    {
        basis = externalContact->basis;
        newNumContacts = min(externalContact->numContacts, 8);
        for (int i = 0; i < newNumContacts; ++i)
            newContacts[i] = externalContact->contacts[i];
    }
    else
    {
        newNumContacts = collide(bodyA, bodyB, newContacts, basis);
    }

    // Merge old contact data with new contacts
    bool refreshContactLocations = isRoundShape(bodyA) || isRoundShape(bodyB);
    for (int i = 0; i < newNumContacts; i++)
    {
        for (int j = 0; j < numContacts; j++)
        {
            if (newContacts[i].feature.key == contacts[j].feature.key)
            {
                float3 newRA = newContacts[i].rA;
                float3 newRB = newContacts[i].rB;
                newContacts[i] = contacts[j];

                // If no static friction in last frame, use the new contact point locations
                if (refreshContactLocations || !contacts[j].stick)
                {
                    newContacts[i].rA = newRA;
                    newContacts[i].rB = newRB;
                }
                if (refreshContactLocations)
                {
                    newContacts[i].lambda[1] = 0.0f;
                    newContacts[i].lambda[2] = 0.0f;
                    newContacts[i].penalty[1] = PENALTY_MIN;
                    newContacts[i].penalty[2] = PENALTY_MIN;
                    newContacts[i].stick = false;
                }
                break;
            }
        }
    }

    // Copy new contacts to the manifold
    numContacts = newNumContacts;
    for (int i = 0; i < numContacts; i++)
        contacts[i] = newContacts[i];

    // Compute error at q- and update penalty and lambdas
    for (int i = 0; i < numContacts; i++)
    {
        // Error at q-
        float3 xA = bodyA->positionLin + contactOffsetWorld(bodyA, contacts[i].rA, basis, true, contacts[i].feature.key);
        float3 xB = bodyB->positionLin + contactOffsetWorld(bodyB, contacts[i].rB, basis, false, contacts[i].feature.key);
        contacts[i].C0 = basis * (xA - xB) + float3{COLLISION_MARGIN, 0, 0};

        // Warmstart the dual variables and penalty parameters (Eq. 19)
        // Penalty is safely clamped to a minimum and maximum value
        contacts[i].lambda = contacts[i].lambda * solver->alpha * solver->gamma;
        contacts[i].penalty = clamp(contacts[i].penalty * solver->gamma, PENALTY_MIN, PENALTY_MAX);
    }

    return numContacts > 0;
}

void Manifold::updatePrimal(Rigid *body, float alpha, float3x3 &lhsLin, float3x3 &lhsAng, float3x3 &lhsCross, float3 &rhsLin, float3 &rhsAng)
{
    float3 dqALin = bodyA->positionLin - bodyA->initialLin;
    float3 dqAAng = bodyA->positionAng - bodyA->initialAng;
    float3 dqBLin = bodyB->positionLin - bodyB->initialLin;
    float3 dqBAng = bodyB->positionAng - bodyB->initialAng;

    for (int i = 0; i < numContacts; i++)
    {
        float3 rAWorld = contactOffsetWorld(bodyA, contacts[i].rA, basis, true, contacts[i].feature.key);
        float3 rBWorld = contactOffsetWorld(bodyB, contacts[i].rB, basis, false, contacts[i].feature.key);

        // Compute the Taylor series approximation of the constraint function C(x) (Sec 4)
        float3x3 jALin = basis;
        float3x3 jBLin = -basis;
        float3x3 jAAng = float3x3{cross(rAWorld, jALin[0]), cross(rAWorld, jALin[1]), cross(rAWorld, jALin[2])};
        float3x3 jBAng = float3x3{cross(rBWorld, jBLin[0]), cross(rBWorld, jBLin[1]), cross(rBWorld, jBLin[2])};

        float3x3 K = diagonal(contacts[i].penalty.x, contacts[i].penalty.y, contacts[i].penalty.z);
        float3 C = contacts[i].C0 * (1 - alpha) + jALin * dqALin + jBLin * dqBLin + jAAng * dqAAng + jBAng * dqBAng;

        // Compute force
        float3 F = K * C + contacts[i].lambda;

        // Clamp normal force
        F[0] = min(F[0], 0.0f);

        // Clamp norm of friction forces to achieve a friction cone
        float bounds = fabsf(F[0]) * friction;
        float frictionScale = length(float2{F[1], F[2]});
        if (frictionScale > bounds && frictionScale > 0)
        {
            F[1] *= bounds / frictionScale;
            F[2] *= bounds / frictionScale;
        }

        // Choose jacobian depending on input body
        float3x3 jLin = body == bodyA ? jALin : jBLin;
        float3x3 jAng = body == bodyA ? jAAng : jBAng;

        // Stamp into LHS
        float3x3 jLinT = transpose(jLin);
        float3x3 jAngT = transpose(jAng);
        float3x3 jAngTk = jAngT * K;

        lhsLin += jLinT * K * jLin;
        lhsAng += jAngTk * jAng;
        lhsCross += jAngTk * jLin;

        // Stamp into RHS
        rhsLin += jLinT * F;
        rhsAng += jAngT * F;
    }
}

void Manifold::updateDual(float alpha)
{
    float3 dqALin = bodyA->positionLin - bodyA->initialLin;
    float3 dqAAng = bodyA->positionAng - bodyA->initialAng;
    float3 dqBLin = bodyB->positionLin - bodyB->initialLin;
    float3 dqBAng = bodyB->positionAng - bodyB->initialAng;

    for (int i = 0; i < numContacts; i++)
    {
        float3 rAWorld = contactOffsetWorld(bodyA, contacts[i].rA, basis, true, contacts[i].feature.key);
        float3 rBWorld = contactOffsetWorld(bodyB, contacts[i].rB, basis, false, contacts[i].feature.key);

        // Compute the Taylor series approximation of the constraint function C(x) (Sec 4)
        float3x3 jALin = basis;
        float3x3 jBLin = -basis;
        float3x3 jAAng = float3x3{cross(rAWorld, jALin[0]), cross(rAWorld, jALin[1]), cross(rAWorld, jALin[2])};
        float3x3 jBAng = float3x3{cross(rBWorld, jBLin[0]), cross(rBWorld, jBLin[1]), cross(rBWorld, jBLin[2])};

        float3x3 K = diagonal(contacts[i].penalty.x, contacts[i].penalty.y, contacts[i].penalty.z);
        float3 C = contacts[i].C0 * (1 - alpha) + jALin * dqALin + jBLin * dqBLin + jAAng * dqAAng + jBAng * dqBAng;

        // Compute force
        float3 F = K * C + contacts[i].lambda;

        // Clamp normal force
        F[0] = min(F[0], 0.0f);

        // Clamp norm of friction forces to achieve a friction cone
        float bounds = fabsf(F[0]) * friction;
        float frictionScale = length(float2{F[1], F[2]});
        if (frictionScale > bounds && frictionScale > 0)
        {
            F[1] *= bounds / frictionScale;
            F[2] *= bounds / frictionScale;
        }

        // Store updated force
        contacts[i].lambda = F;

        // Update the penalty parameter and clamp to material stiffness if we are within the force bounds (Eq. 16)
        if (F[0] < 0)
            contacts[i].penalty[0] = min(contacts[i].penalty[0] + solver->betaLin * fabsf(C[0]), PENALTY_MAX);
        if (frictionScale <= bounds)
        {
            contacts[i].penalty[1] = min(contacts[i].penalty[1] + solver->betaLin * fabsf(C[1]), PENALTY_MAX);
            contacts[i].penalty[2] = min(contacts[i].penalty[2] + solver->betaLin * fabsf(C[2]), PENALTY_MAX);
            contacts[i].stick = length(float2{C[1], C[2]}) < STICK_THRESH;
        }
    }
}
