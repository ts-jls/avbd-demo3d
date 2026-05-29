#include <stdio.h>
#include "C:/code/avbd-demo3d/source/solver.h"

int main(){
  Solver solver;
  new Rigid(&solver,{100,100,1},0,0.5f,{0,0,0});
  Rigid* s=Rigid::makeSphere(&solver,0.5f,1,0.5f,{0,0,1.1f},{8,0,0});
  for(int i=0;i<130;i++){
    solver.step();
    if(i==20 || i==60 || i==85 || i==100 || i==120){
      printf("step %d pos=(%.3f %.3f %.4f) vel=(%.3f %.3f %.3f) ang=(%.3f %.3f %.3f)\n", i, s->positionLin.x,s->positionLin.y,s->positionLin.z,s->velocityLin.x,s->velocityLin.y,s->velocityLin.z,s->velocityAng.x,s->velocityAng.y,s->velocityAng.z);
      for(Force* f=solver.forces; f; f=f->next){
        if(Manifold* m=dynamic_cast<Manifold*>(f)){
          printf("  manifold bodies %d/%d contacts=%d fr=%.3f basis0=(%.3f %.3f %.3f)\n", m->bodyA->shape.type, m->bodyB->shape.type, m->numContacts, m->friction, m->basis[0].x,m->basis[0].y,m->basis[0].z);
          for(int c=0;c<m->numContacts;c++){
            float3 xA=transform(m->bodyA->positionLin,m->bodyA->positionAng,m->contacts[c].rA);
            float3 xB=transform(m->bodyB->positionLin,m->bodyB->positionAng,m->contacts[c].rB);
            float3 err=m->basis*(xA-xB)+float3{COLLISION_MARGIN,0,0};
            printf("    xA.z=%.4f xB.z=%.4f err=(%.4f %.4f %.4f) C0=(%.4f %.4f %.4f) pen=(%.1f %.1f %.1f) lam=(%.4f %.4f %.4f)\n", xA.z,xB.z,err.x,err.y,err.z,m->contacts[c].C0.x,m->contacts[c].C0.y,m->contacts[c].C0.z,m->contacts[c].penalty.x,m->contacts[c].penalty.y,m->contacts[c].penalty.z,m->contacts[c].lambda.x,m->contacts[c].lambda.y,m->contacts[c].lambda.z);
          }
        }
      }
    }
  }
}
