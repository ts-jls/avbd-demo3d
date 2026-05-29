#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"
int main() {
    Solver solver;
    new Rigid(&solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});
    const float radius = 0.5f;
    for (int i = 0; i < 10; i++)
        Rigid::makeSphere(&solver, radius, 1.0f, 0.5f, {0, 0, 0.5f + radius + i * radius * 2.05f});
    for (int i = 0; i < 1800; ++i) {
        solver.step();
        float minZ = 999.0f;
        int underGroundTop = 0;
        for (Rigid* b = solver.bodies; b; b = b->next) if (b->shape.type == RIGID_SHAPE_SPHERE) {
            if (b->positionLin.z < minZ) minZ = b->positionLin.z;
            if (b->positionLin.z < 0.5f) underGroundTop++;
        }
        if (i % 120 == 0 || underGroundTop) printf("%d minZ=%.4f manifolds=%d hits=%d under=%d\n", i, minZ, solver.stats.manifoldCount, solver.stats.sphereHits, underGroundTop);
        if (underGroundTop) break;
    }
}
