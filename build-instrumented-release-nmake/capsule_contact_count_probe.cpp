#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"

int main()
{
    Solver solver;
    Rigid *ground = new Rigid(&solver, {1000, 1000, 1}, 0, 0.5f, {0, 0, 0});
    const float r = 0.35f;
    const float h = 0.9f;
    const float a = rad(90.0f);
    Rigid *c0 = Rigid::makeCapsule(&solver, r, h, 0.2f, 0.5f, {0, 0, 0.85f});
    Rigid *c1 = Rigid::makeCapsule(&solver, r, h, 0.2f, 0.5f, {0, 0, 1.55f});
    c0->positionAng = {0, sinf(a * 0.5f), 0, cosf(a * 0.5f)};
    c1->positionAng = {sinf(a * 0.5f), 0, 0, cosf(a * 0.5f)};
    for (int step = 0; step < 80; ++step)
    {
        solver.step();
        if (step % 20 != 0)
            continue;
        printf("step %d c0z=%.3f c1z=%.3f\n", step, c0->positionLin.z, c1->positionLin.z);
        for (Force *f = solver.forces; f; f = f->next)
        {
            Manifold *m = dynamic_cast<Manifold *>(f);
            if (!m)
                continue;
            const char *aName = m->bodyA == ground ? "ground" : (m->bodyA == c0 ? "c0" : "c1");
            const char *bName = m->bodyB == ground ? "ground" : (m->bodyB == c0 ? "c0" : "c1");
            printf("  %s-%s contacts=%d basis0=(%.2f %.2f %.2f)\n", aName, bName, m->numContacts, m->basis[0].x, m->basis[0].y, m->basis[0].z);
        }
    }
}
