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

Force::Force(Solver* solver, Rigid* bodyA, Rigid* bodyB)
    : solver(solver), denseId(INVALID_FORCE_ID), type(SIM_CONSTRAINT_UNKNOWN), bodyA(bodyA), bodyB(bodyB), nextA(0), nextB(0)
{
    // Add to solver linked list
    next = solver->forces;
    solver->forces = this;

    // Add to body linked lists
    if (bodyA)
    {
        nextA = bodyA->forces;
        bodyA->forces = this;
        bodyA->attachedForceCount++;
    }
    if (bodyB)
    {
        nextB = bodyB->forces;
        bodyB->forces = this;
        bodyB->attachedForceCount++;
    }

    denseId = solver->world.registerForce(this);
}


Force::~Force()
{
    solver->world.unregisterForce(denseId);

    // Remove from solver linked list
    Force** p = &solver->forces;
    while (*p != this)
        p = &(*p)->next;
    *p = next;

    // Remove from body linked lists
    if (bodyA)
    {
        p = &bodyA->forces;
        while (*p != this)
            p = (*p)->bodyA == bodyA ? &(*p)->nextA : &(*p)->nextB;
        *p = nextA;
        bodyA->attachedForceCount--;
    }

    if (bodyB)
    {
        p = &bodyB->forces;
        while (*p != this)
            p = (*p)->bodyA == bodyB ? &(*p)->nextA : &(*p)->nextB;
        *p = nextB;
        bodyB->attachedForceCount--;
    }
}

IgnoreCollision::IgnoreCollision(Solver *solver, Rigid *bodyA, Rigid *bodyB)
    : Force(solver, bodyA, bodyB)
{
    type = SIM_CONSTRAINT_IGNORE_COLLISION;
    solver->world.setForceType(denseId, SIM_CONSTRAINT_IGNORE_COLLISION);
}
