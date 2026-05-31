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
SimBodyData makeBodyData(Rigid *body)
{
    return SimBodyData{
        true,
        body,
        body->shape,
        body->size,
        body->positionLin,
        body->positionAng,
        body->velocityLin,
        body->velocityAng,
        body->mass,
        body->moment,
        body->friction,
        body->radius,
        body->attachedForceCount};
}

BodyId bodyIdFor(Rigid *body)
{
    return body ? body->denseId : INVALID_BODY_ID;
}
}

void SimWorld::clear()
{
    bodies.clear();
    constraints.clear();
    activeBodyIds.clear();
    jointIds.clear();
    springIds.clear();
    ignoreCollisionIds.clear();
    manifoldIds.clear();
    bodyConstraintIds.clear();
    freeBodyIds.clear();
    freeConstraintIds.clear();
}

BodyId SimWorld::registerBody(Rigid *body)
{
    BodyId id;
    if (!freeBodyIds.empty())
    {
        id = freeBodyIds.back();
        freeBodyIds.pop_back();
        bodies[id] = makeBodyData(body);
    }
    else
    {
        id = (BodyId)bodies.size();
        bodies.push_back(makeBodyData(body));
    }
    return id;
}

void SimWorld::updateBodyFromRigid(Rigid *body)
{
    if (!body || body->denseId == INVALID_BODY_ID || body->denseId >= bodies.size())
        return;

    bodies[body->denseId] = makeBodyData(body);
}

void SimWorld::unregisterBody(BodyId id)
{
    if (id == INVALID_BODY_ID || id >= bodies.size())
        return;

    bodies[id].active = false;
    bodies[id].source = 0;
    freeBodyIds.push_back(id);
}

ForceId SimWorld::registerForce(Force *force, SimConstraintType type)
{
    ForceId id;
    SimConstraintData data{
        true,
        force,
        type,
        bodyIdFor(force->bodyA),
        bodyIdFor(force->bodyB)};
    if (!freeConstraintIds.empty())
    {
        id = freeConstraintIds.back();
        freeConstraintIds.pop_back();
        constraints[id] = data;
    }
    else
    {
        id = (ForceId)constraints.size();
        constraints.push_back(data);
    }
    return id;
}

void SimWorld::unregisterForce(ForceId id)
{
    if (id == INVALID_FORCE_ID || id >= constraints.size())
        return;

    constraints[id].active = false;
    constraints[id].source = 0;
    freeConstraintIds.push_back(id);
}

void SimWorld::setForceType(ForceId id, SimConstraintType type)
{
    if (id == INVALID_FORCE_ID || id >= constraints.size())
        return;

    constraints[id].type = type;
}

void SimWorld::syncFromLegacy(Solver &solver)
{
    activeBodyIds.clear();
    jointIds.clear();
    springIds.clear();
    ignoreCollisionIds.clear();
    manifoldIds.clear();
    bodyConstraintIds.clear();
    bodyConstraintIds.resize(bodies.size());

    for (Rigid *body = solver.bodies; body != 0; body = body->next)
    {
        if (body->denseId == INVALID_BODY_ID || body->denseId >= bodies.size())
            continue;

        bodies[body->denseId] = makeBodyData(body);
        if (body->mass > 0.0f)
            activeBodyIds.push_back(body->denseId);
    }

    for (Force *force = solver.forces; force != 0; force = force->next)
    {
        if (force->denseId == INVALID_FORCE_ID || force->denseId >= constraints.size())
            continue;

        SimConstraintData &constraint = constraints[force->denseId];
        constraint.active = true;
        constraint.source = force;
        constraint.bodyA = bodyIdFor(force->bodyA);
        constraint.bodyB = bodyIdFor(force->bodyB);

        if (constraint.bodyA != INVALID_BODY_ID && constraint.bodyA < bodyConstraintIds.size())
            bodyConstraintIds[constraint.bodyA].push_back(force->denseId);
        if (constraint.bodyB != INVALID_BODY_ID && constraint.bodyB < bodyConstraintIds.size())
            bodyConstraintIds[constraint.bodyB].push_back(force->denseId);

        switch (constraint.type)
        {
        case SIM_CONSTRAINT_JOINT:
            jointIds.push_back(force->denseId);
            break;
        case SIM_CONSTRAINT_SPRING:
            springIds.push_back(force->denseId);
            break;
        case SIM_CONSTRAINT_IGNORE_COLLISION:
            ignoreCollisionIds.push_back(force->denseId);
            break;
        case SIM_CONSTRAINT_MANIFOLD:
            manifoldIds.push_back(force->denseId);
            break;
        default:
            break;
        }
    }
}
