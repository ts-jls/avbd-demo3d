#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"
int main(){
  Solver solver;
  new Rigid(&solver,{1000,1000,1},0,0.5f,{0,0,0});
  for(int i=0;i<8;i++){
    Rigid* c=Rigid::makeCapsule(&solver,0.35f,0.9f,1,0.5f,{0,0,0.85f+i*0.84f});
    float a=rad(90.0f);
    c->positionAng = i%2==0 ? quat{0,sinf(a*0.5f),0,cosf(a*0.5f)} : quat{sinf(a*0.5f),0,0,cosf(a*0.5f)};
  }
  for(int i=0;i<900;i++){
    solver.step();
    float minZ=999; int under=0;
    for(Rigid* b=solver.bodies;b;b=b->next) if(b->shape.type==RIGID_SHAPE_CAPSULE){ if(b->positionLin.z<minZ) minZ=b->positionLin.z; if(b->positionLin.z<0.5f) under++; }
    if(i%90==0||under) printf("%d minZ=%.4f manifolds=%d hits=%d under=%d\n",i,minZ,solver.stats.manifoldCount,solver.stats.sphereHits,under);
    if(under) break;
  }
}
