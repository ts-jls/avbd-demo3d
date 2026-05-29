#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"

int main()
{
    Solver solver;
    new Rigid(&solver, {1000, 1000, 1}, 0, 0.5f, {0, 0, 0});

    const float r = 0.35f;
    const float h = 0.9f;
    const float a = rad(90.0f);
    const quat layX = {0, sinf(a * 0.5f), 0, cosf(a * 0.5f)};
    const quat layY = {sinf(a * 0.5f), 0, 0, cosf(a * 0.5f)};
    Rigid *capsules[5];
    capsules[0] = Rigid::makeCapsule(&solver, r, h, 0.35f, 0.5f, {-1.6f, 0, 0.5f + r + 0.05f});
    capsules[1] = Rigid::makeCapsule(&solver, r, h, 0.35f, 0.5f, {0.0f, 0, 0.5f + r + 0.05f});
    capsules[2] = Rigid::makeCapsule(&solver, r, h, 0.35f, 0.5f, {1.6f, 0, 0.5f + r + 0.05f});
    capsules[3] = Rigid::makeCapsule(&solver, r, h, 0.35f, 0.5f, {-0.8f, 0, 1.35f});
    capsules[4] = Rigid::makeCapsule(&solver, r, h, 0.35f, 0.5f, {0.8f, 0, 1.35f});
    for (int i = 0; i < 5; ++i)
        capsules[i]->positionAng = i < 3 ? layX : layY;

    for (int step = 0; step < 600; ++step)
    {
        solver.step();
        float minZ = 999.0f;
        int under = 0;
        for (int i = 0; i < 5; ++i)
        {
            minZ = min(minZ, capsules[i]->positionLin.z);
            if (capsules[i]->positionLin.z < 0.45f)
                under++;
        }
        if (step % 60 == 0 || under)
            printf("%d minZ=%.4f manifolds=%d under=%d\n", step, minZ, solver.stats.manifoldCount, under);
        if (under)
            break;
    }
}
