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
const float PI = 3.14159265358979323846f;

float sphereMass(float radius, float density)
{
    return 4.0f / 3.0f * PI * radius * radius * radius * density;
}
}

Rigid::Rigid(Solver* solver, float3 size, float density, float friction, float3 position, float3 velocity)
    : solver(solver), denseId(INVALID_BODY_ID), forces(0), next(0), positionLin(position), positionAng({ 0, 0, 0, 1 }), 
    velocityLin(velocity), velocityAng({ 0, 0, 0 }), prevVelocityLin(velocity),
    shape{RIGID_SHAPE_BOX, size, length(size * 0.5f), 0.0f}, size(size), friction(friction), attachedForceCount(0), gpuPairCount(0)
{
    // Add to linked list
    next = solver->bodies;
    solver->bodies = this;

    // Compute mass properties and bounding radius
    mass = size.x * size.y * size.z * density;
    moment = float3 {
        (size.y * size.y + size.z * size.z) / 12.0f * mass,
        (size.x * size.x + size.z * size.z) / 12.0f * mass,
        (size.x * size.x + size.y * size.y) / 12.0f * mass
    };
    radius = length(size * 0.5f);
    denseId = solver->world.registerBody(this);
}

Rigid *Rigid::makeSphere(Solver *solver, float radius, float density, float friction, float3 position, float3 velocity)
{
    float diameter = radius * 2.0f;
    Rigid *body = new Rigid(solver, {diameter, diameter, diameter}, density, friction, position, velocity);
    body->shape = RigidShape{RIGID_SHAPE_SPHERE, {diameter, diameter, diameter}, radius, 0.0f};
    body->mass = sphereMass(radius, density);
    float inertia = 2.0f / 5.0f * body->mass * radius * radius;
    body->moment = {inertia, inertia, inertia};
    body->radius = radius;
    solver->world.updateBodyFromRigid(body);
    return body;
}

Rigid *Rigid::makeCapsule(Solver *solver, float radius, float halfLength, float density, float friction, float3 position, float3 velocity)
{
    float diameter = radius * 2.0f;
    float cylinderLength = halfLength * 2.0f;
    Rigid *body = new Rigid(solver, {diameter, diameter, cylinderLength + diameter}, density, friction, position, velocity);
    body->shape = RigidShape{RIGID_SHAPE_CAPSULE, {diameter, diameter, cylinderLength + diameter}, radius, halfLength};

    float cylinderMass = PI * radius * radius * cylinderLength * density;
    float capMass = sphereMass(radius, density);
    body->mass = cylinderMass + capMass;

    float axialInertia = 0.5f * cylinderMass * radius * radius + 2.0f / 5.0f * capMass * radius * radius;
    float transverseInertia = (3.0f * radius * radius + cylinderLength * cylinderLength) / 12.0f * cylinderMass
        + 2.0f / 5.0f * capMass * radius * radius
        + capMass * halfLength * halfLength;
    body->moment = {transverseInertia, transverseInertia, axialInertia};
    body->radius = halfLength + radius;
    solver->world.updateBodyFromRigid(body);
    return body;
}

Rigid *Rigid::makeCylinder(Solver *solver, float radius, float halfLength, float density, float friction, float3 position, float3 velocity)
{
    float diameter = radius * 2.0f;
    float height = halfLength * 2.0f;
    Rigid *body = new Rigid(solver, {diameter, diameter, height}, density, friction, position, velocity);
    body->shape = RigidShape{RIGID_SHAPE_CYLINDER, {diameter, diameter, height}, radius, halfLength};

    body->mass = PI * radius * radius * height * density;
    float axialInertia = 0.5f * body->mass * radius * radius;
    float transverseInertia = (3.0f * radius * radius + height * height) / 12.0f * body->mass;
    body->moment = {transverseInertia, transverseInertia, axialInertia};
    body->radius = sqrtf(radius * radius + halfLength * halfLength);
    solver->world.updateBodyFromRigid(body);
    return body;
}

Rigid::~Rigid()
{
    solver->world.unregisterBody(denseId);

    // Remove from linked list
    Rigid** p = &solver->bodies;
    while (*p != this)
        p = &(*p)->next;
    *p = next;
}

bool Rigid::constrainedTo(Rigid* other) const
{
    // Check if this body is constrained to the other body
    for (Force* f = forces; f != 0; f = (f->bodyA == this) ? f->nextA : f->nextB)
    {
        if (solver->deepProfiling)
            solver->stats.constrainedForceVisits++;
        if ((f->bodyA == this && f->bodyB == other) || (f->bodyA == other && f->bodyB == this))
            return true;
    }
    return false;
}
