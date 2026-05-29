#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"
int main(){
  Solver solver;
  new Rigid(&solver,{100,100,1},0,0.5f,{0,0,0});
  Rigid* s=Rigid::makeSphere(&solver,0.5f,1,0.5f,{0,0,1.1f},{8,0,0});
  for(int i=0;i<1800;i++){
    solver.step();
    if(i%120==0 || s->positionLin.z<0.5f) printf("%d x=%.2f z=%.4f vx=%.2f vz=%.2f manifolds=%d\n",i,s->positionLin.x,s->positionLin.z,s->velocityLin.x,s->velocityLin.z,solver.stats.manifoldCount);
    if(s->positionLin.z < -1.0f) break;
  }
}
