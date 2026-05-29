#include <stdio.h>
#include "C:/code/avbd-demo3d/source/scenes.h"

static void runScene(const char *name, void (*scene)(Solver *))
{
    Solver solver;
    scene(&solver);
    int under = 0;
    for (int step = 0; step < 600; ++step)
    {
        solver.step();
        under = 0;
        for (Rigid *body = solver.bodies; body; body = body->next)
        {
            if (body->shape.type == RIGID_SHAPE_CYLINDER && body->positionLin.z < -0.25f)
                under++;
        }
        if (under)
            break;
    }
    printf("%s bodies=%d forces=%d manifolds=%d under=%d\n", name, solver.stats.bodyCount, solver.stats.forceCount, solver.stats.manifoldCount, under);
}

int main()
{
    runScene("Cylinder Stack", sceneCylinderStack);
    runScene("Cylinder Ramp", sceneCylinderRamp);
}
