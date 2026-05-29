#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"

void sceneRamp(Solver& solver) {
    solver.clear();
    new Rigid(&solver, {100, 100, 1}, 0.0f, 0.5f, {0, 0, 0});
    const float angle = rad(20.0f);
    Rigid* ramp = new Rigid(&solver, {36, 16, 1}, 0.0f, 0.8f, {0, 0, 3});
    ramp->positionAng = {0, sinf(angle * 0.5f), 0, cosf(angle * 0.5f)};
    float3 rampTangent = normalize(rotate(ramp->positionAng, float3{1, 0, 0}));
    float3 rampNormal = normalize(rotate(ramp->positionAng, float3{0, 0, 1}));
    const float radius = 0.55f;
    for (int i = 0; i < 8; i++) {
        float friction = 0.1f + i * 0.2f;
        float3 pos = ramp->positionLin + rampTangent * (-12.0f + i * 1.4f) + float3{0, -3.5f + i, 0} + rampNormal * (0.5f + radius + 0.02f);
        Rigid::makeSphere(&solver, radius, 1.0f, friction, pos);
    }
}

int main() {
    Solver solver;
    sceneRamp(solver);
    for (int i = 0; i < 900; ++i) {
        solver.step();
        int below = 0;
        float minZ = 999.0f;
        for (Rigid* b = solver.bodies; b; b = b->next) if (b->shape.type == RIGID_SHAPE_SPHERE) {
            if (b->positionLin.z < minZ) minZ = b->positionLin.z;
            if (b->positionLin.z < -1.0f) below++;
        }
        if (i % 30 == 0 || below) printf("%d minZ=%.4f manifolds=%d hits=%d below=%d\n", i, minZ, solver.stats.manifoldCount, solver.stats.sphereHits, below);
        if (below) break;
    }
}
