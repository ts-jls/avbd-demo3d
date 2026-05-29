#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"

static float minCylinderZ(Rigid *body)
{
    float minZ = 1.0e9f;
    const int slices = 64;
    for (int cap = 0; cap < 2; ++cap)
    {
        float z = cap == 0 ? -body->shape.halfLength : body->shape.halfLength;
        for (int i = 0; i < slices; ++i)
        {
            float theta = 6.28318530718f * (float)i / (float)slices;
            float3 p = transform(body->positionLin, body->positionAng, {
                body->shape.radius * cosf(theta),
                body->shape.radius * sinf(theta),
                z});
            minZ = min(minZ, p.z);
        }
    }
    return minZ;
}

int main()
{
    Solver solver;
    new Rigid(&solver, {1000, 1000, 1}, 0, 0.8f, {0, 0, 0});
    Rigid *cylinder = Rigid::makeCylinder(&solver, 0.48f, 0.65f, 1.0f, 0.8f, {0, 0, 1.4f}, {2.0f, 0, 0});
    float angle = rad(55.0f);
    cylinder->positionAng = {sinf(angle * 0.5f), 0.0f, 0.0f, cosf(angle * 0.5f)};
    cylinder->velocityAng = {0, 6.0f, 0};

    for (int step = 0; step < 360; ++step)
    {
        solver.step();
        float minZ = minCylinderZ(cylinder);
        if (step % 30 == 0 || minZ < 0.48f)
        {
            float3 axis = rotate(cylinder->positionAng, {0, 0, 1});
            printf("%d pos=(%.3f %.3f %.3f) minZ=%.4f axisZ=%.3f manifolds=%d contacts=",
                step, cylinder->positionLin.x, cylinder->positionLin.y, cylinder->positionLin.z, minZ, axis.z, solver.stats.manifoldCount);
            for (Force *force = solver.forces; force; force = force->next)
            {
                Manifold *m = dynamic_cast<Manifold *>(force);
                if (m)
                    printf("%d ", m->numContacts);
            }
            printf("\n");
        }
    }
}
