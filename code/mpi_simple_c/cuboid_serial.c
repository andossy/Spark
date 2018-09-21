#include <math.h>
#include <assert.h>
#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "mpi.h"

#define MAX_LINE_LENGTH 80
//#define DIST_STATE
//#define __PAPI__

#ifdef   __PAPI__
#include <papi.h>
#endif
//#define RMAX 1000000
//#define RMAX_F (1.0/RMAX)

typedef int(*CONDCF)(int a, int b);

double timing();
inline double my_random();
double my_min(double* ar, int len);
double my_max(double* ar, int len);
void stern(double t, double* y0, double* y1, double Ca);
void stern_discrete(double dt, int* y0, int* y1, double Ca);

void laplace3D (int nx0, int ny0, int nz0, double* C0,
                int nx1, int ny1, int nz1, double* C1,
                double alpha);//, int num_threads)

void reaction3D (int nx0, int ny0, int nz0, double* Ca,
                 int nx1, int ny1, int nz1, double* buff,
                 double B_tot, double k_on, double k_off, double dt);//, int num_threads)

void serca3D (int nx0, int ny0, int nz0, double* Ca_i,
              int nx1, int ny1, int nz1, double* Ca_SR,
              double dt, double gamma, double fudge);//, int num_threads)

void update_ryr(int nx0, int ny0, int nz0, double* Ca_i, double* Ca_SR, double* Ca_CSQN, 
    double k_on_CSQN,double k_off_CSQN,double CSQN_tot,double gamma,double K,double dt,
    int ryr_len, int* i0_ryr, int* i1_ryr, int* i2_ryr,
    int csqn_len, int* i0_csqn, int* i1_csqn, int* i2_csqn,
    int* states0, int* states1);

void  store2Dmatrixfile_double_1D(char* outfile, double* ar, int rows, int cols);
void  store2Dmatrixfile_int_1D(char* outfile, int* ar, int rows, int cols);

//int   less(int a, int b);
//int   giant(int a, int b);
//int*  loadRyRindexfile_int(char* infile, CONDFN cf, int cond);
int*    loadRyRindexfile_int(char* infile, int* count);

int load_indices(
    int size_x, int size_y, int size_z,
    int nx, int ny, int nz, int h, 
    int** i0_ryr,  int** i1_ryr,  int** i2_ryr,  int* ryr_len,
    int** i0_csqn, int** i1_csqn, int** i2_csqn, int* csqn_len,
    MPI_Comm comm3d);
  
void readparam(int* iconf, double* conf);
void updateBound(double* C0, int C_flag, int nx0, int ny0, int nz0,
    int* neighbor, MPI_Comm comm, MPI_Comm comm3d, MPI_Datatype matrix_type_oyz, MPI_Datatype
    matrix_type_oxz, MPI_Datatype matrix_type_oxy);

int main(int argc, char **argv)
{ 

  int i,j,k;
#ifdef __PAPI__
  //  int Events[] = { PAPI_L1_DCA, PAPI_L1_DCM  };
//  int Events[] = {PAPI_L3_TCM, PAPI_L3_TCA, PAPI_L2_TCM,PAPI_L2_TCA};
  int Events[] = {PAPI_DP_OPS,PAPI_L3_TCM};
  int NUM_EVENTS = sizeof(Events)/sizeof(Events[0]);
  long long res_papi[NUM_EVENTS];
  char EventName[128];
  int num_hwcntrs = 0;
  int EventSet = PAPI_NULL;
  int retval;

  retval = PAPI_library_init( PAPI_VER_CURRENT );
  retval = PAPI_create_eventset( &EventSet );

  if (PAPI_add_events( EventSet, Events, NUM_EVENTS) != PAPI_OK){
    printf("PAPI_add_events failed\n");
  }

  for (i=0; i<NUM_EVENTS; i++){
    res_papi[i] = 0;
  }
#endif

  double time_main=0.0;
  double time_comm=0.0;
  double time_conc=0.0;
  double time_ryr=0.0; 
  double time_io=0.0; 
 
  int save_data=0;
  int idx;
  int h_scale=1;
  int h=30;
  int Lx=h*50;
  int Ly=h*100;
  int Lz=h*70;

  int TimeStep;
  int size_x, size_y, size_z, my_id, x_domains, y_domains, z_domains;
  int iconf[8];
  double conf[2];
   /* MPI variables */
  int sizes[3], subsizes1[3], subsizes2[3], subsizes3[3], starts[3];
  int nproc, ndims;
  MPI_Comm comm, comm3d;
  int dims[3];
  int periods[3];
  int reorganisation = 0;
  MPI_Datatype matrix_type_oyz, matrix_type_oxz, matrix_type_oxy;
  int ZN=0, ZP=1, YN=2, YP=3, XN=4, XP=5;
  int NeighBor[6];

  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm,&nproc);
  MPI_Comm_rank(comm,&my_id);
  
  if (my_id==0) {
    readparam(iconf, conf);
  }
  MPI_Bcast(iconf,8,MPI_INT,0,comm);
  MPI_Bcast(conf,2,MPI_DOUBLE,0,comm);

  h  = iconf[0];
  size_x = iconf[1];
  size_y = iconf[2];
  size_z = iconf[3];  
  x_domains = iconf[4];
  y_domains = iconf[5];
  z_domains = iconf[6];  
  TimeStep  = iconf[7];

  h_scale=30/h;
  if((my_id==0) && (nproc!=(x_domains*y_domains*z_domains))) {
    printf("Number of processes not equal to Number of subdomains\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if((my_id==0)&&(size_x%x_domains!=0)) {
    printf("Number of x_domains is not divisible in scale\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if((my_id==0)&&(size_y%y_domains!=0)) {
    printf("Number of y_domains is not divisible in scale\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if((my_id==0)&&(size_z%z_domains!=0)) {
    printf("Number of z_domains is not divisible in scale\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  
  /* Create 3D cartesian grid */

  periods[0] = 0;
  periods[1] = 0;
  periods[2] = 0;  

  ndims = 3;
  dims[0]=z_domains;
  dims[1]=y_domains;
  dims[2]=x_domains;  

  MPI_Cart_create(comm, ndims, dims, periods, reorganisation, &comm3d);

  /* Identify neighbors */

  NeighBor[0] = MPI_PROC_NULL;
  NeighBor[1] = MPI_PROC_NULL;
  NeighBor[2] = MPI_PROC_NULL;
  NeighBor[3] = MPI_PROC_NULL;
  NeighBor[4] = MPI_PROC_NULL;  
  NeighBor[5] = MPI_PROC_NULL;    

  /* Left/West and right/Est neigbors Z direction*/
  MPI_Cart_shift(comm3d,0,1,&NeighBor[ZN],&NeighBor[ZP]);

  /* Bottom/South and Upper/North neigbors Y direction*/
  MPI_Cart_shift(comm3d,1,1,&NeighBor[YN],&NeighBor[YP]);

  /* Zdown/South and Zup/North neigbors X direction*/
  MPI_Cart_shift(comm3d,2,1,&NeighBor[XN],&NeighBor[XP]);

//--------------------------------------------------------------------
  int nx=(size_x/x_domains);
  int ny=(size_y/y_domains);
  int nz=(size_z/z_domains);  
  int nx0,ny0,nz0;
  int nx1,ny1,nz1;
  nx0=nx+2; 
  ny0=ny+2; 
  nz0=nz+2; 
  nx1=nx+2; 
  ny1=ny+2; 
  nz1=nz+2; 
  int len;
  len=nx0*ny0*nz0;

  /* Create matrix data types to communicate */
  MPI_Type_vector(ny, nz, nz0, MPI_DOUBLE, &matrix_type_oyz);
  MPI_Type_commit(&matrix_type_oyz);

  /* Create matrix data type to communicate on vertical Oxz plan */

  MPI_Type_vector(nx, nz, ny0*nz0, MPI_DOUBLE, &matrix_type_oxz);
  MPI_Type_commit(&matrix_type_oxz);

  /* Create matrix data type to communicate on vertical Oxy plan */
  MPI_Datatype matrix_type_liney; 
  MPI_Type_vector(ny, 1, nz0, MPI_DOUBLE, &matrix_type_liney);
  MPI_Type_commit(&matrix_type_liney);

//  MPI_Type_vector(nx*ny, 1, nz0, MPI_DOUBLE, &matrix_type_oxy);
  MPI_Type_hvector(nx, 1, ny0*nz0*sizeof(double), matrix_type_liney, &matrix_type_oxy);
  MPI_Type_commit(&matrix_type_oxy);


  printf("Simulation Begin!\n");

  double Vfraction;
  //first set the numbers of RyR in a CaRU;
  int n_ryr;

  //All CaRU placed mid-sarcomere
  int mid_x=(size_x+1)/2;
  Vfraction=(30.0/h)*(30.0/h)*(30.0/h); // scaling of RyR when changing dx
  //Define where the RyRs are:
  int* i0_ryr;
  int* i1_ryr;
  int* i2_ryr;
  int* i0_csqn;
  int* i1_csqn;
  int* i2_csqn;
  int  ryr_len;
  int  csqn_len;

    h_scale=load_indices(size_x, size_y, size_z,nx,ny,nz,h,&i0_ryr,&i1_ryr,&i2_ryr,&ryr_len,&i0_csqn,&i1_csqn,&i2_csqn,&csqn_len, comm3d);
    n_ryr=ryr_len;
//  if(!my_id){
    printf("load indices over: h:%d, h_scale:%d, nx:%d, ny:%d, nz:%d, ryr_len:%d, csqn:%d \n",
	h,h_scale,nx,ny,nz,ryr_len,csqn_len);
//  }

// store2Dmatrixfile_int_1D("i0.txt",i0,n_ryr,1);
// store2Dmatrixfile_int_1D("i1.txt",i1,n_ryr,1);
// store2Dmatrixfile_int_1D("i2.txt",i2,n_ryr,1);

// Set constants and dt based on these:
double D_i=220e3;
double D_SR=73.3e3;
double D_ATP=140e3;
double D_CMDN=22e3;
double D_Fluo=42e3;

double dt=(1./6)*h*h/D_i;

double alpha_i = dt*D_i/(h*h);
double Ca0 = 140e-3;
double* Ca_i;
Ca_i=(double*)malloc(len*sizeof(double));
for ( i = 0; i < len; i += 1 ) {
  Ca_i[i]=Ca0;
} 

double alpha_SR = dt*D_SR/(h*h);
double* Ca_SR;
Ca_SR=(double*)malloc(len*sizeof(double));
for ( i = 0; i < len; i += 1 ) {
  Ca_SR[i]=1.3e3;
} 

double k_on_CMDN = 34e-3;
double k_off_CMDN = 238e-3;
double CMDN_tot = 24;
double alpha_CMDN = dt*D_CMDN/(h*h);

double k_on_ATP = 255e-3;
double k_off_ATP = 45;
double ATP_tot = 455;
double alpha_ATP = dt*D_ATP/(h*h);

double k_on_Fluo = 110e-3;
double k_off_Fluo = 110e-3;
double Fluo_tot = 25;
double alpha_Fluo = dt*D_Fluo/(h*h);

double k_on_TRPN = 32.7e-3;
double k_off_TRPN = 19.6e-3;
double TRPN_tot = 70;

double k_on_CSQN = 102e-3;
double k_off_CSQN = 65;
double CSQN_tot = 30e3;

double alpha[7];
double k_on[7];
double k_off[7];
double B_tot[7];

alpha[0]=alpha_i; alpha[1]=alpha_SR; alpha[2]=alpha_CMDN; alpha[3]=alpha_ATP; alpha[4]=alpha_Fluo; alpha[5]=0;          alpha[6]=0;
 k_on[0]=0      ;  k_on[1]=       0;  k_on[2]= k_on_CMDN;  k_on[3]=k_on_ATP ;  k_on[4]=k_on_Fluo ;  k_on[5]=k_on_TRPN;   k_on[6]=k_on_CSQN;
k_off[0]=0      ; k_off[1]=       0; k_off[2]=k_off_CMDN; k_off[3]=k_off_ATP; k_off[4]=k_off_Fluo; k_off[5]=k_off_TRPN; k_off[6]=k_off_CSQN;
B_tot[0]=0      ; B_tot[1]=       0; B_tot[2]=CMDN_tot  ; B_tot[3]=ATP_tot  ; B_tot[4]=Fluo_tot  ; B_tot[5]=TRPN_tot;   B_tot[6]=CSQN_tot;


// Calculate steady state IC for the buffers based on Ca_i ...
double Ca_CMDN0=B_tot[2]*Ca0/(Ca0+k_off[2]/k_on[2]);
double Ca_ATP0 =B_tot[3]*Ca0/(Ca0+k_off[3]/k_on[3]);
double Ca_Fluo0=B_tot[4]*Ca0/(Ca0+k_off[4]/k_on[4]);
double Ca_TRPN0=B_tot[5]*Ca0/(Ca0+k_off[5]/k_on[5]);

// and Ca_SR:
double Ca_CSQN0 = CSQN_tot*Ca_SR[0]/(Ca_SR[0] +  k_off_CSQN/k_on_CSQN);

printf("%f %f %f %f %f \n ", Ca_ATP0, Ca_CMDN0, Ca_Fluo0, Ca_TRPN0, Ca_CSQN0);

// Allocate the data structure for the solution
double *Ca_ATP  ;
double *Ca_CMDN ; 
double *Ca_Fluo ; 
double *Ca_TRPN ; 
double *Ca_CSQN ; 

Ca_ATP =(double*)malloc(len*sizeof(double));
Ca_CMDN=(double*)malloc(len*sizeof(double));
Ca_Fluo=(double*)malloc(len*sizeof(double));
Ca_TRPN=(double*)malloc(len*sizeof(double));
Ca_CSQN=(double*)malloc(len*sizeof(double));

for ( i = 0; i < len; i += 1 ) {
     Ca_ATP[i] = Ca_ATP0;
    Ca_CMDN[i] = Ca_CMDN0;
    Ca_Fluo[i] = Ca_Fluo0;
    Ca_TRPN[i] = Ca_TRPN0;
    Ca_CSQN[i] = Ca_CSQN0;
}

double* C0[7];
double* C1[7];
double* C_temp;

C0[0]=(double*)malloc(len*sizeof(double));
C1[0]=Ca_i;
memcpy(C0[0],C1[0],len*sizeof(double));

C0[1]=(double*)malloc(len*sizeof(double));
C1[1]=Ca_SR;
memcpy(C0[1],C1[1],len*sizeof(double));

C0[2]=(double*)malloc(len*sizeof(double));
C1[2]=Ca_CMDN;
memcpy(C0[2],C1[2],len*sizeof(double));

C0[3]=(double*)malloc(len*sizeof(double));
C1[3]=Ca_ATP;
memcpy(C0[3],C1[3],len*sizeof(double));

C0[4]=(double*)malloc(len*sizeof(double));
C1[4]=Ca_Fluo;
memcpy(C0[4],C1[4],len*sizeof(double));

C0[5]=Ca_TRPN;
C1[5]=Ca_TRPN;

C0[6]=Ca_CSQN;
C1[6]=Ca_CSQN;

//Ca = [[Ca_i.copy(),    Ca_i   ],
//      [Ca_SR.copy(),   Ca_SR  ],
//      [Ca_CMDN.copy(), Ca_CMDN],
//      [Ca_ATP.copy(),  Ca_ATP ],
//      [Ca_Fluo.copy(), Ca_Fluo],
//      [Ca_TRPN, Ca_TRPN],
//      [Ca_CSQN, Ca_CSQN]]

double gamma = 0.05; // SR volume fraction
int   cai=0;
int   sri=1;
int cmdni=2;
int  atpi=3;
int fluoi=4;
int trpni=5; 
int csqni=6;
double fraction[7]={1,1,1,1,1,1,1};
fraction[1]=gamma;
fraction[6]=gamma;

// Ryr conductance:
double k_s = (Vfraction)*150/2; // 1/ms, based on 0.5pA of Ca2+ into (30nm)^3.
double K = exp(-k_s*dt*(1+1/gamma)); // factor need in the integration below

//# Initial states of the RyRs
int* states0;
int* states1;
if(n_ryr>0){
  states0=(int*)malloc(n_ryr*sizeof(int));
  states1=(int*)malloc(n_ryr*sizeof(int));
  for ( i = 0; i < n_ryr; i += 1 ) {
    states0[i]=0;
    states1[i]=0;
  }
#ifdef DIST_STATE
  for ( i = 1; i < n_ryr; i =i+3 ) {
    if(my_random()>0.99)
      states0[i]=1;
  }
#endif
}
else
{
  states0=(int*)malloc(1*sizeof(int));
  states1=(int*)malloc(1*sizeof(int));
  states0[0]=0;
  states1[0]=0;
}

int *global_states0;
int  global_states_num;
int *states_rec_count;
int *states_rec_disp;
if(my_id==0){
  states_rec_count=malloc(nproc*sizeof(int));
   for (i = 0; i < nproc; i++) {
     states_rec_count[i]=0;
  }
}
printf("Thread%d: n_ryr=%d\n",my_id, n_ryr);
//MPI_Barrier(comm);
MPI_Gather(&n_ryr,1,MPI_INT,&states_rec_count[0],1,MPI_INT,0,comm);

if(my_id==0){
  for (i = 0; i < nproc; i++) {
    printf("states_rec_count[%d]: %d\n",i, states_rec_count[i]);
  }
  states_rec_disp =  malloc(nproc*sizeof(int));
  states_rec_disp[0] = 0;
  for (i = 1; i < nproc; i++) {
    states_rec_disp[i] = states_rec_disp[i-1] + states_rec_count[i-1];
  }
  global_states_num=states_rec_disp[nproc-1]+states_rec_count[nproc-1];
  printf("Global states Count %d\n", global_states_num);
  if(global_states_num>0)
    global_states0=malloc(global_states_num*sizeof(int));
  else
    global_states0=malloc(1*sizeof(int));
}


MPI_Bcast(&global_states_num,1,MPI_INT,0,comm);

//MPI_Gatherv(states0, n_ryr, MPI_INT, global_states0, states_rec_count, states_rec_disp, MPI_INT, 0, comm);

if(my_id==0){

  for ( i = 0; i < global_states_num; i++) 
    global_states0[i]=0;
  if(global_states_num>=23){
    for ( i = 1; i < 23; i =i+3 ) 
      global_states0[i]=1;
  }
  else
  {
    for ( i = 1; i < global_states_num ; i =i+3 ) 
      global_states0[i]=1; 
  }
}

//if(global_states_num>0)
  MPI_Scatterv(&global_states0[0], states_rec_count, states_rec_disp, MPI_INT, &states0[0], n_ryr, MPI_INT, 0, comm);


int T=10;
double t=0;
int counter=0;
int mean[7];
double DT=0.01; // plotting time step

time_main-=timing();

FILE *fpdata; 
char meanfile[200];
sprintf(meanfile,"outfile/mean%d.txt",my_id);
if(!my_id){
  if(save_data){
    if ((fpdata=fopen(meanfile, "w"))==NULL)
    {
      printf("fialed open output file ");
      printf("%s",meanfile);
      printf(" ! \n ");
      exit(0);
    }
  }
}


#ifdef __PAPI__
if ( PAPI_start( EventSet ) != PAPI_OK){
  printf("PAPI_read_counters failed\n");
}
#endif

for ( T = 0; T < TimeStep; T += 1 ) 
//while(t<T)
{
  t+=dt;

  time_comm-=timing();
  for ( i = 0; i < 5; i += 1 ) {
    updateBound(C0[i], i, nx0, ny0, nz0, NeighBor, comm, comm3d,  matrix_type_oyz, matrix_type_oxz,  matrix_type_oxy); 
  }
  time_comm+=timing();

  time_conc-=timing();
  for ( i = 0; i < 5; i += 1 ) {
    laplace3D(nx0,ny0,nz0,C0[i],nx1,ny1,nz1,C1[i],alpha[i]);
  }
  for ( i = 2; i < 6; i += 1 ) {
    reaction3D(nx1,ny1,nz1,C1[cai],nx1,ny1,nz1,C1[i],B_tot[i],k_on[i],k_off[i],dt);
  }
  serca3D(nx1,ny1,nz1, C1[cai],nx1,ny1,nz1, C1[sri], dt, gamma, 1.0);
  time_conc+=timing();

  // Update at RyRs, one at the time
  time_ryr-=timing();
  if(ryr_len>0)
  update_ryr(nx0, ny0, nz0, C1[cai], C1[sri], C1[csqni], k_on_CSQN, k_off_CSQN,CSQN_tot,gamma,K,dt,
      ryr_len, i0_ryr, i1_ryr,i2_ryr,csqn_len,i0_csqn, i1_csqn, i2_csqn, states0,states1);
  time_ryr+=timing();

  double sum_c_i_root[7];
  double sum_c_i[7];
  double cai_min;
  double cai_min_root=0.0;
  double cai_max;
  double cai_max_root=1.0;
  double sm;
  double ca[8];
  char caoutfile[100];
  //  if (fmod(t,DT)<dt){
  if (1){
    time_io-=timing();
    for(idx=0; idx<7; idx++){
      sum_c_i[idx]=0.0;
      for ( i = 1; i <= nx; i += 1 ) 
	for ( j = 1; j <= ny; j += 1 ) 
	  for ( k = 1; k <= nz; k += 1 ) 
	    sum_c_i[idx]+=C1[idx][i*ny0*nz0+j*nz0+k];
    }
    cai_min=my_min(C1[cai],len);
    cai_max=my_max(C1[cai],len);
    /* reduce operation comm*/
    MPI_Reduce(&sum_c_i[0],&sum_c_i_root[0],7,MPI_DOUBLE,MPI_SUM,0,comm);
    MPI_Reduce(&cai_min,&cai_min_root,1,MPI_DOUBLE,MPI_MIN,0,comm);
    MPI_Reduce(&cai_max,&cai_max_root,1,MPI_DOUBLE,MPI_MAX,0,comm);
    if(!my_id){
      sm = 0;
      ca[0] = t;
      if(save_data) fprintf(fpdata,"%f ", ca[0]);
      for(idx=0; idx<7; idx++){
	sm += fraction[idx]*sum_c_i_root[idx];
	ca[idx+1]  = sum_c_i_root[idx]/(nx*x_domains*ny*y_domains*nz*z_domains);
	printf("ca[%d]: %f , sum : %f, nx ny nz: %d %d %d \n",idx+1, ca[idx+1], 
	    sum_c_i_root[idx],nx*x_domains,ny*y_domains,nz*z_domains);
	if(save_data) fprintf(fpdata,"%f ", ca[idx+1]);
      }
      if(save_data) fprintf(fpdata,"\n ");
      printf("%d,%f,%f,%f,%f,%f,%f \n \n",
	  counter, t, ca[1], ca[2], cai_min_root,cai_max_root, sm);
    }
    if(save_data){
      for(i=0;i<7;i++){
	sprintf(caoutfile,"outfile/Ca%d_T%04d_rank%d.np",i,counter,my_id);
	//	Ca[i][1].dump(root+"Ca%d_T%04d.np"%(i,counter))
	store2Dmatrixfile_double_1D(caoutfile,C1[i],len,1);
      }
    }
    counter += 1;
  }
  //   # Update Ca
  for(i=0;i<7;i++){
    C_temp=C0[i];
    C0[i]=C1[i];
    C1[i]=C_temp;
  }

}
time_main+=timing();
if(!my_id){
  if(save_data) fclose(fpdata);
  printf("cubiod_c: h:%d size_x:%d size_y:%d size_z:%d\n", h, size_x, size_y, size_z); 
  printf("nx0:%d ny0:%d nz0:%d size/array:%7.3f MB total size:%7.3f MB\n", 
      nx0,ny0,nz0,len*8*1e-6,len*8*1e-6*12); 
#ifdef __PAPI__
  if ( PAPI_stop( EventSet, res_papi ) != PAPI_OK){
    printf("PAPI_accum_counters failed\n");
  }
  for (i = 0; i<NUM_EVENTS; i++){
    PAPI_event_code_to_name(Events[i], EventName);
    printf("PAPI Event name: %s, value: %lld\n", EventName, res_papi[i]);
  }
#endif

  printf("average computing time: %7.3f \n",  time_conc);
  printf("average ryr time: %7.3f \n",  time_conc);
  printf("main time: %7.3f \n",   time_main);

#ifdef __PAPI__
  printf("PAPI Performanc/core: %7.3f GFLOPS\n", res_papi[0]/1e9/time_conc);
#endif
}
for(i=0;i<5;i++){
  free(C0[i]);
  free(C1[i]);
}
free(C0[6]);
free(C0[5]);
if(ryr_len>0)
{
  free(i0_ryr);
  free(i1_ryr);
  free(i2_ryr);
}
if(csqn_len>0)
{
free(i0_csqn);
free(i1_csqn);
free(i2_csqn);
}
if(my_id==0){
free(global_states0);
free(states_rec_count);
free(states_rec_disp);
}
MPI_Barrier(comm);
MPI_Finalize();

return 0;
}

void laplace3D (int nx0, int ny0, int nz0, double* C0,
                int nx1, int ny1, int nz1, double* C1,
                double alpha)//, int num_threads)
{

  // Set num threads
 // omp_set_num_threads(num_threads);

  // Local variables
  int i, j, k;
  double C0_tmp;
  
  
  // Main kernel loop
 // #pragma omp parallel for private(i, j, k, C0_tmp) //collapse(3)
  for (i=1; i<nx0-1; i++)
  {
    for (j=1; j<ny0-1; j++)
    {
      for (k=1; k<nz0-1; k++)
      {
        // Main kernel
        C0_tmp = -6*C0[i*nz0*ny0+j*nz0+k] + 
           C0[(i-1)*nz0*ny0+j*nz0+k] + C0[(i+1)*nz0*ny0+j*nz0+k] + 
           C0[i*nz0*ny0+(j-1)*nz0+k] + C0[i*nz0*ny0+(j+1)*nz0+k] + 
           C0[i*nz0*ny0+j*nz0+k-1] + C0[i*nz0*ny0+j*nz0+k+1];
  
        // Put value back into return array with offset to indices 
        C1[i*nz1*ny1+j*nz1+k] = C0[i*nz1*ny1+j*nz1+k] + C0_tmp*alpha;
      }
    }  
  }    
}

void reaction3D (int nx0, int ny0, int nz0, double* Ca,
                 int nx1, int ny1, int nz1, double* buff,
                 double B_tot, double k_on, double k_off, double dt)//, int num_threads)
{

  // Set num threads
  //  omp_set_num_threads(num_threads);

  // Local variables
  int i, j, k;
  double J;

  // Use pointers reducing indexing into memory to once 
  double* Ca_ijk;
  double* buff_ijk;
  
  // Main kernel loop
  //  #pragma omp parallel for private(i, j, k, J, Ca_ijk, buff_ijk) //collapse(3)
  for (i=1; i<nx0-1; i++)
  {
    for (j=1; j<ny0-1; j++)
    {
      for (k=1; k<nz0-1; k++)
      {
        // Main kernel
        Ca_ijk = &Ca[i*nz0*ny0+j*nz0+k];
        buff_ijk = &buff[i*nz0*ny0+j*nz0+k];
        J = k_on*(B_tot - *buff_ijk)*(*Ca_ijk) - k_off*(*buff_ijk);
        *Ca_ijk -= dt*J;
        *buff_ijk += dt*J;
      }
    }  
  }    
}

void serca3D (int nx0, int ny0, int nz0, double* Ca_i,
              int nx1, int ny1, int nz1, double* Ca_SR,
              double dt, double gamma, double fudge)//, int num_threads)
{

  // Set num threads
//  omp_set_num_threads(num_threads);

  // Local variables
  int i, j, k;
  double J;

  // Use pointers reducing indexing into memory to once 
  double Ca_i2_ijk;
  double Ca_SR2_ijk;
  
  // Main kernel loop
//  #pragma omp parallel for private(i, j, k, J, Ca_i2_ijk, Ca_SR2_ijk) //collapse(3)
  for (i=1; i<nx0-1; i++)
  {
    for (j=1; j<ny0-1; j++)
    {
      for (k=1; k<nz0-1; k++)
      {
        // Main kernel
        Ca_i2_ijk = Ca_i[i*nz0*ny0+j*nz0+k];
        Ca_SR2_ijk = Ca_SR[i*nz0*ny0+j*nz0+k];
        Ca_i2_ijk *= Ca_i2_ijk;
        Ca_SR2_ijk *= Ca_SR2_ijk;
        J = fudge*(570997802.885875*Ca_i2_ijk - 0.0425239333622699*Ca_SR2_ijk)/(106720651.206402*Ca_i2_ijk + 182.498197548666*Ca_SR2_ijk + 5.35062954944879);
        Ca_i[i*nz0*ny0+j*nz0+k] -= dt*J;
        Ca_SR[i*nz0*ny0+j*nz0+k] += dt*J/gamma;
      }
    }  
  }    
}

void update_ryr(int nx0, int ny0, int nz0, double* Ca_i, double* Ca_SR, double* Ca_CSQN, 
    double k_on_CSQN,double k_off_CSQN,double CSQN_tot,double gamma,double K,double dt,
    int ryr_len, int* i0_ryr, int* i1_ryr, int* i2_ryr,
    int csqn_len, int* i0_csqn, int* i1_csqn, int* i2_csqn,
    int* states0, int* states1)
{
  int i;
  int x,y,z;
  int idx;
  double J;
  int open;
  double c0,c1;
  for(i=0;i<csqn_len;i+=1){
//    printf("line:%d\n",i);
    x=i0_csqn[i];
    y=i1_csqn[i];
    z=i2_csqn[i];
    idx=x*ny0*nz0+y*nz0+z;
    //CSQN step:
    J = k_on_CSQN*(CSQN_tot - Ca_CSQN[idx])*Ca_SR[idx] -  k_off_CSQN*Ca_CSQN[idx];
    Ca_SR[idx] -= dt*J;
    Ca_CSQN[idx] += dt*J;
  }

  for ( i = 0; i < ryr_len; i += 1 ) {
    x=i0_ryr[i];
    y=i1_ryr[i];
    z=i2_ryr[i];
    idx=x*ny0*nz0+y*nz0+z;
//	#Continous formulation
//	#states[:,i] += dt*stern(t, states[:,i], Ca_i[idx])
    stern_discrete(dt, &states0[i],&states1[i], Ca_i[idx]);
    open = states0[i]*(1-states1[i]);
//	#Exp Euler:
//	#J_RyR = k*open*(Ca_SR[idx]-Ca_i[idx])
//	#Ca_i[idx]  += dt*J_RyR
//	#Ca_SR[idx] -= dt*J_RyR/gamma;
//	#Analytical update:
    if (open){
      printf("open [%d] state0 %d  state1 %d \n", i, states0[i], states1[i]);
      c0 = (Ca_i[idx] + gamma*Ca_SR[idx])/(1+gamma);
      c1 = (Ca_i[idx] - Ca_SR[idx])/(1+1/gamma);
      Ca_i[idx] =  c0 + c1*K;
      Ca_SR[idx] = c0 - c1*K/gamma;
    }
  }
}

void stern(double t, double* y0, double* y1, double Ca){
   double m = *y0;
   double h = *y1;
   double kim = 0.005;
   double kom = 0.06;
   double K_i = 0.01*10;
   double K_o = 0.01*41.4;
   double ki = kim/K_i;
   double ko = kom/(K_o*K_o);
   double dm = ko*Ca*Ca*(1-m)-kom*m;
   double dh = ki*Ca*(1-h)-kim*h;
   *y0=dm;
   *y1=dh;
}

void stern_discrete(double dt, int* y0, int* y1, double Ca){
    double kim = 0*0.005;
    double kom = 0*0.06;
    double ki = Ca*0.5;
    double ko = 1e-2*Ca*Ca*35;
    double r;
    int m,h;

    m = *y0;
    if(m==1){
	r = my_random();
	m = 1 - (r<(dt*kom));
    }
    else
    {
        r=my_random();
	m = 1*(r<(dt*ko));
    }

    h = *y1;
    if(h==1){
	r = my_random();
	h = 1 - (r<(dt*kim));
    }
    else{
	r = my_random();
	h = 1*(r<(dt*ki));
    }
    *y0=m;
    *y1=h;
}

inline double my_random()
{
  double r;
  double x;
  r=(double)(rand()%1000000);
  x=(r*1e-6);
  return x;

}

void  store2Dmatrixfile_double_1D(char* outfile, double* ar, int rows, int cols){
    FILE *fpdata; 
    int i,j;       
    if ((fpdata=fopen(outfile, "w"))==NULL)
      {
	    printf("fialed open output file ");
	    printf("%s",outfile);
	    printf(" ! \n ");
	    exit(0);
      }
//    printf("----Generating list output to ");
//    printf("%s",outfile);
//    printf(" file----\n");
    for(i=0;i<rows;i++)
    {
      for(j=0;j<cols;j++)
	fprintf(fpdata,"%.7e ",ar[i*cols+j]);
      fprintf(fpdata,"\n");
    }
    fclose(fpdata);
    return;
}

void  store2Dmatrixfile_int_1D(char* outfile, int* ar, int rows, int cols){
    FILE *fpdata; 
    int i,j;       
    if ((fpdata=fopen(outfile, "w"))==NULL)
      {
	    printf("fialed open output file ");
	    printf("%s",outfile);
	    printf(" ! \n ");
	    exit(0);
      }
    printf("----Generating list output to ");
    printf("%s",outfile);
    printf(" file----\n");
    for(i=0;i<rows;i++)
    {
      for(j=0;j<cols;j++)
	fprintf(fpdata,"%d ",ar[i*cols+j]);
      fprintf(fpdata,"\n");
    }
    fclose(fpdata);
    return;
}

double my_min(double* ar, int len)
{
  double min=ar[0];
  int i;
  for ( i = 0; i < len; i += 1 ) {
    if(ar[i]<min) min=ar[i];
  }
  return min;
}

double my_max(double* ar, int len)
{
  double max=ar[0];
  int i;
  for ( i = 0; i < len; i += 1 ) {
    if(ar[i]>max) max=ar[i];
  }
  return max;
}

double timing(){
	double time;
	struct timeval timmer;
	gettimeofday(&timmer,NULL);
	time = 1000000*timmer.tv_sec + timmer.tv_usec;
	time /= 1000000;
	return time;
}


int load_indices(
    int size_x, int size_y, int size_z,
    int nx, int ny, int nz, int h, 
    int** i0_ryr,  int** i1_ryr,  int** i2_ryr,  int* ryr_len,
    int** i0_csqn, int** i1_csqn, int** i2_csqn, int* csqn_len,
    MPI_Comm comm3d)
{
  int  nx_s, nx_e;//nx start, nx end
  int  ny_s, ny_e;
  int  nz_s, nz_e;
  int  dims[3];
  int  periods[3];
  int  coords[3];
  MPI_Cart_get(comm3d, 3, dims, periods, coords);

  int i,j,k;
  int* i1;
  int* i2;
  int  i1_len;
  int  i2_len;
  int* i1_temp;
  int* i2_temp;
  int  i1_temp_len;
  int  i1_ryr_len;
  int  i1_csqn_len;
  int* i0_csqn_list;
  int* i1_csqn_list;
  int* i2_csqn_list;

  int  h_scale;
  int  mid_x;
  int  old_nz=nz;
  int  old_ny=ny;
  int  old_nx=nx;

  *ryr_len=0;
  *csqn_len=0;
  // Scale nx, xy, nz in terms of RyR
  if(30%h!=0){
    printf("30 must be divisible by h!");
    exit(1);
  }
  h_scale = 30/h;

  nz_s=nz*coords[0]/h_scale;
  ny_s=ny*coords[1]/h_scale;
  nx_s=nx*coords[2]/h_scale;

  nz_e=nz*(coords[0]+1)/h_scale;
  ny_e=ny*(coords[1]+1)/h_scale;
  nx_e=nx*(coords[2]+1)/h_scale;

  // All CaRU placed mid-sarcomere
  nz = nz/h_scale;
  ny = ny/h_scale;
  nx = nx/h_scale;

  mid_x = (size_x/h_scale+1)/2;

  if((mid_x>=nx_s)&&(mid_x<nx_e)){
    // load RyR indices from file
    i1=loadRyRindexfile_int("i_RyR_indices.dat",&i1_len);
    i2=loadRyRindexfile_int("j_RyR_indices.dat",&i2_len);

    //    # Only use the subset which are inside the geometry
    if(i1_len==i2_len)
      printf("num RyR before reduction: %d\n", i1_len);
    else
      printf("num RyR is wrong: i1_len!=i2_len\n");

    i1_temp_len=0;
    for ( i = 0; i < i1_len; i += 1 ) {
      i1[i]=i1[i]-ny_s;
      if((i1[i]>=0)&&(i1[i]<ny)) i1_temp_len++;
    }
    if(i1_temp_len>0){
      i1_temp=malloc(i1_temp_len*sizeof(int));
      i2_temp=malloc(i1_temp_len*sizeof(int));
    }
    else
    {
      i1_temp=malloc(1*sizeof(int));
      i2_temp=malloc(1*sizeof(int));
    }
    j=0;
    for ( i = 0; i < i1_len; i += 1 ) {
      if((i1[i]>=0)&&(i1[i]<ny)){
	i1_temp[j]=i1[i];
	i2_temp[j]=i2[i];
	j++;
      }
    }
    free(i1);
    free(i2);

    i1_ryr_len=0;
    for ( i = 0; i < i1_temp_len; i += 1 ) {
      i2_temp[i]=i2_temp[i]-nz_s;
      if((i2_temp[i]>=0)&&(i2_temp[i]<nz)) i1_ryr_len++;
    }
    if(i1_ryr_len>0){
      *i0_ryr=malloc(i1_ryr_len*sizeof(int));
      *i1_ryr=malloc(i1_ryr_len*sizeof(int));
      *i2_ryr=malloc(i1_ryr_len*sizeof(int));
    }
    else
    {
      *i0_ryr=malloc(1*sizeof(int));
      *i1_ryr=malloc(1*sizeof(int));
      *i2_ryr=malloc(1*sizeof(int));
    }
    j=0;
    for ( i = 0; i < i1_temp_len; i += 1 ) {
      if((i2_temp[i]>=0)&&(i2_temp[i]<nz)){ 
      (*i1_ryr)[j]=i1_temp[i];
      (*i2_ryr)[j]=i2_temp[i];
      j++;
    }
  }
  free(i1_temp);
  free(i2_temp);

// Scale indices and move to center of macro voxel
  for ( i = 0; i < i1_ryr_len; i += 1 ) {
    (*i0_ryr)[i] = (mid_x-nx_s)*h_scale+1;
    (*i1_ryr)[i] = (*i1_ryr)[i]*h_scale - floor((double)h_scale/2)+1;
    (*i2_ryr)[i] = (*i2_ryr)[i]*h_scale - floor((double)h_scale/2)+1;
  }
  *ryr_len=i1_ryr_len;
  
// load CSQN indices from file
  i1 = loadRyRindexfile_int("i_csqn_indices.dat", &i1_len); 
  i2 = loadRyRindexfile_int("j_csqn_indices.dat", &i2_len);
  if(i1_len==i2_len)
    printf("num CSQN before reduction: %d\n", i1_len);
  else
    printf("num CSQN is wrong: i1_len!=i2_len\n");

//# Only use the subset which are inside the geometry
//  i1_csqn = i1[i2<nz]*h_scale
//    i2_csqn = i2[i2<nz]*h_scale
//    i0_csqn = np.ones(len(i1_csqn), dtype=int)*mid_x*h_scale
  i1_temp_len=0;
  for ( i = 0; i < i1_len; i += 1 ) {
    i1[i]=i1[i]-ny_s;
    if((i1[i]>=0)&&(i1[i]<ny)) i1_temp_len++;
  }
  if(i1_temp_len>0){
    i1_temp=malloc(i1_temp_len*sizeof(int));
    i2_temp=malloc(i1_temp_len*sizeof(int));
  }
  else
  {
    i1_temp=malloc(1*sizeof(int));
    i2_temp=malloc(1*sizeof(int));
  }
  j=0;
  for ( i = 0; i < i1_len; i += 1 ) {
   if((i1[i]>=0)&&(i1[i]<ny)){
      i1_temp[j]=i1[i];
      i2_temp[j]=i2[i];
      j++;
    }
  }
  free(i1);
  free(i2);

  i1_csqn_len=0;
  for ( i = 0; i < i1_temp_len; i += 1 ) {
    i2_temp[i]=i2_temp[i]-nz_s;
    if((i2_temp[i]>=0)&&(i2_temp[i]<nz)) i1_csqn_len++;
  }
  if(i1_csqn_len>0){
    *i0_csqn=malloc(i1_csqn_len*sizeof(int));
    *i1_csqn=malloc(i1_csqn_len*sizeof(int));
    *i2_csqn=malloc(i1_csqn_len*sizeof(int));
  }
  else
  {
    *i0_csqn=malloc(1*sizeof(int));
    *i1_csqn=malloc(1*sizeof(int));
    *i2_csqn=malloc(1*sizeof(int));
  }
  j=0;
  for ( i = 0; i < i1_temp_len; i += 1 ) {
    if((i2_temp[i]>=0)&&(i2_temp[i]<nz)){
      (*i1_csqn)[j]=i1_temp[i];
      (*i2_csqn)[j]=i2_temp[i];
      j++;
    }
  }
  free(i1_temp);
  free(i2_temp);

// Scale indices and move to center of macro voxel
  for ( i = 0; i < i1_csqn_len; i += 1 ) {
    (*i0_csqn)[i] = (mid_x-nx_s)*h_scale+1;
    (*i1_csqn)[i] = (*i1_csqn)[i]*h_scale+1;
    (*i2_csqn)[i] = (*i2_csqn)[i]*h_scale+1; 
  }


//    # Add CSQN to all voxels covered by the original CSQN array
    if ((h_scale > 1)&&(i1_csqn_len>0)){
      i0_csqn_list=malloc(i1_csqn_len*h_scale*h_scale*sizeof(int));
      i1_csqn_list=malloc(i1_csqn_len*h_scale*h_scale*sizeof(int));
      i2_csqn_list=malloc(i1_csqn_len*h_scale*h_scale*sizeof(int));
//        # Add offsetted versions of the csqn
      for ( i = 0; i < h_scale; i += 1 ) {
	for ( j = 0; j < h_scale; j += 1 ) {
	  for ( k = 0; k < i1_csqn_len; k += 1 ) {
                i0_csqn_list[i*i1_csqn_len*h_scale+j*i1_csqn_len+k]=(*i0_csqn)[k];
                i1_csqn_list[i*i1_csqn_len*h_scale+j*i1_csqn_len+k]=(*i1_csqn)[k]+i;
                i2_csqn_list[i*i1_csqn_len*h_scale+j*i1_csqn_len+k]=(*i2_csqn)[k]+j;
	  }
	}
      }
      free(*i0_csqn);
      free(*i1_csqn);
      free(*i2_csqn);
      *i0_csqn=i0_csqn_list;
      *i1_csqn=i1_csqn_list;
      *i2_csqn=i2_csqn_list;
    }
      *csqn_len=i1_csqn_len*h_scale*h_scale;
      //check value
      for ( i = 0; i < *ryr_len; i += 1 ) {
	if(((*i0_ryr)[i]<0)||((*i0_ryr)[i]>(old_nx+1))) printf("i0_ryr overflow\n");
	if(((*i1_ryr)[i]<0)||((*i1_ryr)[i]>(old_ny+1))) printf("i1_ryr overflow\n");
	if(((*i2_ryr)[i]<0)||((*i2_ryr)[i]>(old_nz+1))) printf("i2_ryr overflow\n");
      }
      for ( i = 0; i < *csqn_len; i += 1 ) {
	if(((*i0_csqn)[i]<0)||((*i0_csqn)[i]>(old_nx+1))) printf("i0_csqn overflow\n");
	if(((*i1_csqn)[i]<0)||((*i1_csqn)[i]>(old_ny+1))) printf("i1_csqn overflow\n");
	if(((*i2_csqn)[i]<0)||((*i2_csqn)[i]>(old_nz+1))) printf("i2_csqn overflow\n");
      }
  }
  else
  {
    *i0_ryr=malloc(1*sizeof(int));
    *i1_ryr=malloc(1*sizeof(int));
    *i2_ryr=malloc(1*sizeof(int));
    *i0_csqn=malloc(1*sizeof(int));
    *i1_csqn=malloc(1*sizeof(int));
    *i2_csqn=malloc(1*sizeof(int));
    *ryr_len=0;
    *csqn_len=0;
  }
      return h_scale;
}

//int*  loadRyRindexfile_int(char* infile, CONDFN cf, int cond)
int*  loadRyRindexfile_int(char* infile,  int* count)
{
    FILE *fpdata; 
    int* arreturn; 
    int i;       
    int temp_d;
    *count=0;
    printf("Load file name: %s\n", infile);
    fpdata = fopen(infile, "r");    
    if(fpdata==NULL)
    {
        printf("\nFailure to open input file.\n");
        exit(0);
    }
    while(fscanf(fpdata, "%d", &temp_d)!=EOF){
//       if(cf(temp_d,cond)) count++;
         (*count)++;
//	 printf("%d,",temp_d);
    }
    printf("There are %d indices satisfy the condition\n",*count);
    arreturn = malloc((*count)*sizeof(int));
    if (arreturn == NULL)
    {
        printf("\nFailure trying to allocate room for array.\n");
        exit(0);
    }
    rewind(fpdata);
    i=0;
    while(fscanf(fpdata, "%d", &temp_d)!=EOF){
//      if(cf(temp_d,cond)) {
	arreturn[i]=temp_d;
	i++;
//      }
    }
    fclose(fpdata);
    if (*count != i)
    {
        printf("Wrong indices number\n");
        exit(0);
    }
    printf("load file %s over \n", infile);
    return arreturn;    
}


void readparam(int* iconf, double* conf)
{
  FILE* file2;
  char  Data[MAX_LINE_LENGTH];
  if((file2=fopen("param","r")) == NULL)
  { printf("Error opening param file\n");
    return;
  }

  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[0]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[1]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[2]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[3]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[4]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[5]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[6]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[7]);

  conf[0]=1.0;
  conf[1]=1.0;
//  fgets(Data,MAX_LINE_LENGTH,file2);
//  fscanf(file2,"%le\n",&conf[0]);
//  fgets(Data,MAX_LINE_LENGTH,file2);
//  fscanf(file2,"%le",&conf[1]);
  fclose(file2);
}   


void updateBound(double* C0, int C_flag, int nx0, int ny0, int nz0,
    int* neighbor,  MPI_Comm comm,MPI_Comm comm3d, MPI_Datatype matrix_type_oyz, MPI_Datatype
    matrix_type_oxz, MPI_Datatype matrix_type_oxy) 
{
  int i,j,k;
  MPI_Status status;
  int  dims[3];
  int  periods[3];
  int  coords[3];
  int  ZN=0, ZP=1, YN=2, YP=3, XN=4, XP=5;
  MPI_Cart_get(comm3d, 3, dims, periods, coords);
 // Ghost X end sheet
  if(coords[2]==0){
    i=0;
    for (j=1; j<ny0-1; j++)
      for (k=1; k<nz0-1; k++)
	C0[i*nz0*ny0+j*nz0+k] = C0[(i+1)*nz0*ny0+j*nz0+k];
  }
  if(coords[2]==(dims[2]-1))
  {
    i=nx0-1;
    for (j=1; j<ny0-1; j++)
      for (k=1; k<nz0-1; k++)
	C0[i*nz0*ny0+j*nz0+k] = C0[(i-1)*nz0*ny0+j*nz0+k];
  }
  MPI_Sendrecv(&C0[1*nz0*ny0+nz0+1],1,matrix_type_oyz,neighbor[XN],C_flag+1000,
      &C0[nz0+1],1,matrix_type_oyz,neighbor[XN],C_flag+1000,comm,&status);
  MPI_Sendrecv(&C0[(nx0-2)*nz0*ny0+nz0+1],1,matrix_type_oyz,neighbor[XP],C_flag+1000,
      &C0[(nx0-1)*nz0*ny0+nz0+1],1,matrix_type_oyz,neighbor[XP],C_flag+1000,comm,&status);

//  printf("exchange X end sheet ok! coords[%d,%d,%d]\n",coords[0],coords[1],coords[2]);
  // Ghost Y end sheet
  if(coords[1]==0){
    j=0;
    for (i=1; i<nx0-1; i++)
      for (k=1; k<nz0-1; k++)
	C0[i*nz0*ny0+j*nz0+k] = C0[i*nz0*ny0+(j+1)*nz0+k];
  }
  if(coords[1]==(dims[1]-1))
  {
    j=ny0-1;
    for (i=1; i<nx0-1; i++)
      for (k=1; k<nz0-1; k++)
	C0[i*nz0*ny0+j*nz0+k] = C0[i*nz0*ny0+(j-1)*nz0+k];
  }
  MPI_Sendrecv(&C0[1*nz0*ny0+nz0+1],1,matrix_type_oxz,neighbor[YN],C_flag+2000,
      &C0[1*nz0*ny0+1],1,matrix_type_oxz,neighbor[YN],C_flag+2000,comm,&status);
  MPI_Sendrecv(&C0[1*nz0*ny0+(ny0-2)*nz0+1],1,matrix_type_oxz,neighbor[YP],C_flag+2000,
      &C0[1*nz0*ny0+(ny0-1)*nz0+1],1,matrix_type_oxz,neighbor[YP],C_flag+2000,comm,&status);

//  printf("exchange Y end sheet ok! coords[%d,%d,%d]\n",coords[0],coords[1],coords[2]);
  // Ghost Z end sheet
  if(coords[0]==0){
    k=0;
    for (i=1; i<nx0-1; i++)
      for (j=1; j<ny0-1; j++)
	C0[i*nz0*ny0+j*nz0+k] = C0[i*nz0*ny0+j*nz0+k+1];
  }
  if(coords[0]==(dims[0]-1))
  {
    k=nz0-1;
    for (i=1; i<nx0-1; i++)
      for (j=1; j<ny0-1; j++)
	C0[i*nz0*ny0+j*nz0+k] = C0[i*nz0*ny0+j*nz0+k-1];
  }
  MPI_Sendrecv(&C0[1*nz0*ny0+nz0+1],1,matrix_type_oxy,neighbor[ZN],C_flag+3000,
      &C0[1*nz0*ny0+nz0],1,matrix_type_oxy,neighbor[ZN],C_flag+3000,comm,&status);
  MPI_Sendrecv(&C0[1*nz0*ny0+nz0+nz0-2],1,matrix_type_oxy,neighbor[ZP],C_flag+3000,
      &C0[1*nz0*ny0+nz0+nz0-1],1,matrix_type_oxy,neighbor[ZP],C_flag+3000,comm,&status);
//  printf("exchange Z end sheet ok! coords[%d,%d,%d]\n",coords[0],coords[1],coords[2]);
}


void *mpi_malloc (
   int id,     /* IN - Process rank */
   int bytes)  /* IN - Bytes to allocate */
{
   void *buffer;
   if ((buffer = malloc ((size_t) bytes)) == NULL) {
      printf ("Error: Malloc failed for process %d\n", id);
      fflush (stdout);
      MPI_Abort (MPI_COMM_WORLD, 4);
   }
   return buffer;
}
