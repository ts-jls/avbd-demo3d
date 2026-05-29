#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"
int main(){
  for(float start=-0.8f; start<=0.8f; start+=0.4f){
    Solver solver; new Rigid(&solver,{100,100,1},0,0.5f,{0,0,0}); Rigid* s=Rigid::makeSphere(&solver,0.5f,1,0.5f,{0,0,start});
    for(int i=0;i<60;i++) solver.step();
    printf("start %.2f -> z %.4f vz %.4f manifolds %d\n", start, s->positionLin.z, s->velocityLin.z, solver.stats.manifoldCount);
  }
}
