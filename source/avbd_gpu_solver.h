/*
 * WebGPU AVBD backend: runs the paper-faithful AVBD primal/dual iteration
 * loop on the GPU. Broadphase, narrowphase, and warmstarting stay on the CPU
 * (Solver::prepareStep); the per-iteration work — graph-colored per-body 6x6
 * primal solves and per-constraint dual updates — runs fully GPU-resident
 * with a single readback at the end of the step.
 */

#pragma once

#include "solver.h"

struct WebGpuContext;

std::unique_ptr<PhysicsBackend> makeWebGpuAvbdPhysicsBackend(WebGpuContext *context);
