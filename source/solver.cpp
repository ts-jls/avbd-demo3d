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

Solver::Solver()
    : bodies(0), forces(0)
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
    // Perform broadphase collision detection
    // This is a naive O(n^2) approach, but it is sufficient for small numbers of bodies in this sample.
    for (Rigid *bodyA = bodies; bodyA != 0; bodyA = bodyA->next)
    {
        for (Rigid *bodyB = bodyA->next; bodyB != 0; bodyB = bodyB->next)
        {
            float3 dp = bodyA->positionLin - bodyB->positionLin;
            float r = bodyA->radius + bodyB->radius;
            if (dot(dp, dp) <= r * r && !bodyA->constrainedTo(bodyB))
                new Manifold(this, bodyA, bodyB);
        }
    }

    // Initialize and warmstart forces
    for (Force *force = forces; force != 0;)
    {
        // Initialization can including caching anything that is constant over the step
        if (!force->initialize())
        {
            // Force has returned false meaning it is inactive, so remove it from the solver
            Force *next = force->next;
            delete force;
            force = next;
        }
        else
            force = force->next;
    }

    // Initialize and warmstart bodies (ie primal variables)
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
    }

    // Main solver loop
    for (int it = 0; it < iterations; it++)
    {
        // Primal update
        for (Rigid *body = bodies; body != 0; body = body->next)
        {
            // Skip static / kinematic bodies
            if (body->mass <= 0)
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
                // Stamp the force and hessian into the linear system
                force->updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
            }

            // Solve the SPD linear system using LDL and apply the update (Eq. 4)
            float3 dxLin, dxAng;
            solve(lhsLin, lhsAng, lhsCross, -rhsLin, -rhsAng, dxLin, dxAng);
            body->positionLin = body->positionLin + dxLin;
            body->positionAng = body->positionAng + dxAng;
        }

        // Dual update
        for (Force *force = forces; force != 0; force = force->next)
        {
            force->updateDual(alpha);
        }
    }

    // Compute velocities (BDF1) after the final iteration
    for (Rigid* body = bodies; body != 0; body = body->next)
    {
        body->prevVelocityLin = body->velocityLin;
        if (body->mass > 0)
        {
            body->velocityLin = (body->positionLin - body->initialLin) / dt;
            body->velocityAng = (body->positionAng - body->initialAng) / dt;
        }
    }
}
