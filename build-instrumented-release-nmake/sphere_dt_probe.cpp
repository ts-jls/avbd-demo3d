#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"

void run(float dt, int iterations) {
    Solver solver;
    solver.dt = dt;
    solver.iterations = iterations;
    new Rigid(&solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});
    const float radius = 0.5f;
    for (int i = 0; i < 10; i++)
        Rigid::makeSphere(&solver, radius, 1.0f, 0.5f, {0, 0, 0.5f + radius + i * radius * 2.05f});
    for (int i = 0; i < 120; ++i) {
        solver.step();
        float minZ = 999.0f;
        int below = 0;
        for (Rigid* b = solver.bodies; b; b = b->next) if (b->shape.type == RIGID_SHAPE_SPHERE) {
            if (b->positionLin.z < minZ) minZ = b->positionLin.z;
            if (b->positionLin.z < 0.0f) below++;
        }
        if (below || i == 119) { printf("dt=%.3f it=%d step=%d minZ=%.4f below=%d\n", dt, iterations, i, minZ, below); break; }
    }
}
int main() { run(1.0f/60.0f, 10); run(0.05f, 10); run(0.1f, 10); run(0.1f, 1); return 0; }
