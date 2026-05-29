#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"

int main()
{
    Solver solver;
    new Rigid(&solver, {1000, 1000, 1}, 0, 0.5f, {0, 0, 0});

    Rigid *capsules[4];
    for (int i = 0; i < 4; ++i)
    {
        capsules[i] = Rigid::makeCapsule(&solver, 0.35f, 0.9f, 1.0f, 0.5f, {0, 0, 0.85f + i * 0.9f});
        float a = rad(90.0f);
        capsules[i]->positionAng = i % 2 == 0
            ? quat{0, sinf(a * 0.5f), 0, cosf(a * 0.5f)}
            : quat{sinf(a * 0.5f), 0, 0, cosf(a * 0.5f)};
    }

    for (int step = 0; step < 240; ++step)
    {
        solver.step();
        if (step % 10 != 0)
            continue;

        printf("step %d manifolds=%d\n", step, solver.stats.manifoldCount);
        for (int i = 0; i < 4; ++i)
        {
            Rigid *c = capsules[i];
            printf("  c%d pos=(%.3f %.3f %.3f) vel=(%.3f %.3f %.3f) ang=(%.3f %.3f %.3f %.3f)\n",
                i,
                c->positionLin.x, c->positionLin.y, c->positionLin.z,
                c->velocityLin.x, c->velocityLin.y, c->velocityLin.z,
                c->positionAng.x, c->positionAng.y, c->positionAng.z, c->positionAng.w);
        }
    }
}
