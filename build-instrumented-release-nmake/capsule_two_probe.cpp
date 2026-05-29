#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"
int main()
{
    Solver solver;
    new Rigid(&solver, {1000, 1000, 1}, 0, 0.5f, {0, 0, 0});
    const float r = 0.35f;
    const float h = 0.9f;
    const float a = rad(90.0f);
    Rigid *c0 = Rigid::makeCapsule(&solver, r, h, 0.2f, 0.5f, {0, 0, 0.85f});
    Rigid *c1 = Rigid::makeCapsule(&solver, r, h, 0.2f, 0.5f, {0, 0, 1.55f});
    c0->positionAng = {0, sinf(a * 0.5f), 0, cosf(a * 0.5f)};
    c1->positionAng = {sinf(a * 0.5f), 0, 0, cosf(a * 0.5f)};
    for (int step = 0; step < 600; ++step)
    {
        solver.step();
        float minZ = min(c0->positionLin.z, c1->positionLin.z);
        if (step % 60 == 0 || minZ < 0.45f)
            printf("%d c0z=%.4f c1z=%.4f manifolds=%d\n", step, c0->positionLin.z, c1->positionLin.z, solver.stats.manifoldCount);
        if (minZ < 0.45f)
            break;
    }
}
