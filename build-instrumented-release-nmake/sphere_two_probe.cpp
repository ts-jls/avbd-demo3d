#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"
int main()
{
    Solver solver;
    new Rigid(&solver, {1000, 1000, 1}, 0, 0.5f, {0, 0, 0});
    Rigid *s0 = Rigid::makeSphere(&solver, 0.35f, 0.2f, 0.5f, {0, 0, 0.85f});
    Rigid *s1 = Rigid::makeSphere(&solver, 0.35f, 0.2f, 0.5f, {0, 0, 1.55f});
    for (int step = 0; step < 600; ++step)
    {
        solver.step();
        float minZ = min(s0->positionLin.z, s1->positionLin.z);
        if (step % 60 == 0 || minZ < 0.45f)
            printf("%d s0z=%.4f s1z=%.4f manifolds=%d\n", step, s0->positionLin.z, s1->positionLin.z, solver.stats.manifoldCount);
        if (minZ < 0.45f)
            break;
    }
}
