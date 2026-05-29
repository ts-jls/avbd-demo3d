#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"

int main() {
    Solver solver;
    new Rigid(&solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});
    Rigid* first = 0;
    const float radius = 0.5f;
    for (int i = 0; i < 10; i++) {
        Rigid* s = Rigid::makeSphere(&solver, radius, 1.0f, 0.5f, {0, 0, 0.5f + radius + i * radius * 2.05f});
        if (i == 0) first = s;
    }
    for (int i = 0; i < 360; ++i) {
        solver.step();
        float minZ = 999.0f;
        float maxZ = -999.0f;
        int spheresBelow = 0;
        for (Rigid* b = solver.bodies; b; b = b->next) {
            if (b->shape.type == RIGID_SHAPE_SPHERE) {
                if (b->positionLin.z < minZ) minZ = b->positionLin.z;
                if (b->positionLin.z > maxZ) maxZ = b->positionLin.z;
                if (b->positionLin.z < 0.0f) spheresBelow++;
            }
        }
        if (i % 10 == 0 || spheresBelow) {
            printf("%d minZ=%.4f maxZ=%.4f firstZ=%.4f manifolds=%d hits=%d created=%d below=%d\n", i, minZ, maxZ, first->positionLin.z, solver.stats.manifoldCount, solver.stats.sphereHits, solver.stats.manifoldsCreated, spheresBelow);
        }
        if (spheresBelow) break;
    }
    return 0;
}
