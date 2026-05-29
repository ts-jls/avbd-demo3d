#include "solver.h"
#include "scenes.h"
#include <cstdio>

int main(int argc, char **argv)
{
    // probe_spherestack <dt> <iterations>
    float dt = argc > 1 ? (float)atof(argv[1]) : 1.0f / 60.0f;
    int iters = argc > 2 ? atoi(argv[2]) : 10;

    Solver solver;
    sceneSphereStack(&solver);
    solver.dt = dt;
    solver.iterations = iters;

    float minZ = 1e9f;
    bool fellThrough = false;
    for (int step = 1; step <= 600; ++step)
    {
        solver.step();
        for (Rigid *b = solver.bodies; b; b = b->next)
        {
            if (b->mass <= 0.0f)
                continue;
            minZ = min(minZ, b->positionLin.z);
            if (b->positionLin.z < -0.5f) // clearly below the ground slab
                fellThrough = true;
        }
    }
    printf("dt=%.4f iters=%2d -> minSphereZ=%8.3f  %s\n",
           dt, iters, minZ, fellThrough ? "FELL THROUGH" : "ok");
    return 0;
}
