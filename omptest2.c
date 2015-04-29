//
// omptest.c
//
// simple embarassingly-parallel test program
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>//AVX: -mavx

typedef struct particlestruct{
    double x[3];
    double v[3];
    double a[3];
}PARTICLE;

void push_v(PARTICLE * p, double dt)
{
    int i;
    for (i=0;i<3;i++)	p->v[i] += p->a[i]*dt;
}
void push_x(PARTICLE * p, double dt)
{
    int i;
    for (i=0;i<3;i++)	p->x[i] += p->v[i]*dt;
}

void calc_a(PARTICLE * p)
{
    int i;
    double r2 = p->x[0]*p->x[0]+p->x[1]*p->x[1]+p->x[2]*p->x[2];
    double r2inv = 1.0/r2;
    double r3inv = r2inv*sqrt(r2inv);
    for (i=0;i<3;i++)	p->a[i] = -p->x[i]*r3inv;
}

    
void push_particle(PARTICLE * p, double dt)
{
    int i;
    calc_a(p);
    push_v(p,dt);
    push_x(p,dt);
}
void push_particle_initial(PARTICLE * p, double dt)
{
    int i;
    calc_a(p);
    push_v(p,dt*0.5);
    push_x(p,dt);
}
void push_particle_last(PARTICLE * p, double dt)
{
    int i;
    push_v(p,dt*0.5);
}

void initialize(PARTICLE * p, int n)
{
    int i;
    for(i=0;i<n;i++){
	p[i].x[0]=1;
	p[i].x[1]=0.1;
	p[i].x[2]=0.01;
	p[i].v[0]=0.01;
	p[i].v[1]=1.0;
	p[i].v[2]=0.1;
    }
}

void print_total(PARTICLE * p, int n)
{
    printf("x,v, xsum= %f %f %f %f %f %f\n",
	   p->x[0],p->x[1],p->x[2],
	   p->v[0],p->v[1],p->v[2]);
    double sum =0;
    int i;
    for (i=0;i<n;i++) sum += p[i].x[0];
    printf("x,v, xsum/n= %f %f %f %f %f %f %f\n",
	   p->x[0],p->x[1],p->x[2],
	   p->v[0],p->v[1],p->v[2],
	   sum/n);
    
}

void print_energy(PARTICLE * p,  char *s)
{
    printf("%s E=%f\n",s, 0.5*(p->v[0]*p->v[0]+p->v[1]*p->v[1]+p->v[2]*p->v[2])
	  - 1.0/sqrt(p->x[0]*p->x[0]+p->x[1]*p->x[1]+p->x[2]*p->x[2]));
}

int main(void){
  const int n = 102400;
  int nloop=10000;
  int istep,i;
  double dt=0.1;
  PARTICLE pp[n];
  initialize(pp, n);
  print_energy(pp, "Initial");
  for(i=0;i<n;i++) push_particle_initial(pp+i,  dt);
#pragma omp parallel
  {
      int i;
      int istart = 0;
      int iend = n;
#ifdef _OPENMP
      //      printf("hello world from %d of %d\n",
      //	     omp_get_thread_num(), omp_get_num_threads());
      int istep = (n-1)/omp_get_num_threads()+1;
      int myid = omp_get_thread_num();
      istart = istep*myid;
      iend = istart + istep;
      if (iend > n) iend = n;
      //      printf("thread %d does %d %d\n\n",myid, istart, iend);
#endif      
      for (istep=0;istep < nloop-1; istep++){
	  for(i=istart;i<iend;i++){
	      push_particle(pp+i, dt);
	  }
      }
  }
  for(i=0;i<n;i++) push_particle_last(pp+i,  dt);
  print_energy(pp, "Final");
  print_total(pp,n);
  return 0;

}
