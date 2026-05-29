#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"
int main(){
  Solver solver; new Rigid(&solver,{100,100,1},0,0.5f,{0,0,0});
  Rigid* b=new Rigid(&solver,{1,1,1},1,0.5f,{0,0,1.1f},{8,0,0});
  for(int i=0;i<240;i++){
    solver.step();
    if(i%20==0 || b->positionLin.z<0.5f) printf("%d x=%.2f z=%.4f vx=%.2f vz=%.2f manifolds=%d\n",i,b->positionLin.x,b->positionLin.z,b->velocityLin.x,b->velocityLin.z,solver.stats.manifoldCount);
    if(b->positionLin.z < -1.0f) break;
  }
}
