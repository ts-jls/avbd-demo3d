#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"
int main(){ Solver solver; new Rigid(&solver,{1000,1000,1},0,0.5f,{0,0,0}); Rigid* c=Rigid::makeCapsule(&solver,0.35f,0.9f,1,0.5f,{0,0,1.2f},{2,0,0}); float a=rad(90.0f); c->positionAng={0,sinf(a*0.5f),0,cosf(a*0.5f)}; for(int i=0;i<600;i++){ solver.step(); if(i%60==0||c->positionLin.z<0.5f) printf("%d x=%.2f z=%.4f vx=%.2f vz=%.2f manifolds=%d\n",i,c->positionLin.x,c->positionLin.z,c->velocityLin.x,c->velocityLin.z,solver.stats.manifoldCount); if(c->positionLin.z<0.0f) break; }}
