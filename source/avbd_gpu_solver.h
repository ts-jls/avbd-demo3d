/*
 * WebGPU AVBD backend: runs the paper-faithful AVBD primal/dual iteration
 * loop on the GPU. Broadphase, narrowphase, and warmstarting stay on the CPU
 * (Solver::prepareStep); the per-iteration work — graph-colored per-body 6x6
 * primal solves and per-constraint dual updates — runs fully GPU-resident
 * with a single readback at the end of the step.
 */

#pragma once

#include "solver.h"

struct WebGpuDevice;

std::unique_ptr<PhysicsBackend> makeWebGpuAvbdPhysicsBackend(WebGpuDevice *device);

// Lightweight per-process stats from the most recent WebGPU AVBD step, for
// server metrics/benchmark reporting.
struct AvbdGpuSolverStats
{
    bool active;            // a WebGPU AVBD backend has stepped
    float gpuIterateMs;     // GPU upload + iterate + readback time, last step
    int bodies;             // solved (dynamic, constrained) bodies
    int contacts;
    int joints;
    int springs;
    int colors;             // graph colors used by the primal solve
    int cpuIterateFallbacks; // steps that fell back to the CPU iteration loop
};

const AvbdGpuSolverStats &avbdGpuSolverStats();
