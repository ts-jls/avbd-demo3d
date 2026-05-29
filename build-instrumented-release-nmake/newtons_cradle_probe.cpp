#include <stdio.h>
#include "C:/code/avbd-demo3d/source/scenes.h"

int main()
{
    Solver solver;
    sceneNewtonsCradle(&solver);

    for (int step = 0; step < 300; ++step)
    {
        solver.step();
        if (step % 60 == 0)
            printf("%d bodies=%d forces=%d manifolds=%d pairChecks=%d\n", step, solver.stats.bodyCount, solver.stats.forceCount, solver.stats.manifoldCount, solver.stats.pairChecks);
    }
}
