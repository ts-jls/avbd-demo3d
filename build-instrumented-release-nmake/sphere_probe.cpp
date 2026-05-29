#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"

int main() {
    Solver solver;
    new Rigid(&solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});
    Rigid* sphere = Rigid::makeSphere(&solver, 0.5f, 1.0f, 0.5f, {0, 0, 5});
    for (int i = 0; i < 240; ++i) {
        solver.step();
        if (i % 10 == 0 || sphere->positionLin.z < 0.7f) {
            printf("%d z=%.4f vz=%.4f manifolds=%d hits=%d created=%d\n", i, sphere->positionLin.z, sphere->velocityLin.z, solver.stats.manifoldCount, solver.stats.sphereHits, solver.stats.manifoldsCreated);
        }
        if (sphere->positionLin.z < -1.0f) break;
    }
    return 0;
}
