/*
 * Accel.cpp
 *
 *  Created on: Jul 25, 2018
 *      Author: Edg@r j.
 */
#include "Accel.hpp"

#define NMAX      8192*2*16
#define NTHRE     512
#define ATYPE     8
#define ATYPE2    (ATYPE * ATYPE)
#define ThreadsPB 512
//////For NaCl Optimized if_kernel
#define NTHREOPT  512
#define NDIVBIT   4
#define NDIV      (1<<NDIVBIT)
#define NTHREOPT2 (NTHREOPT/NDIV)

typedef struct {
		  float pol;
		  float sigm;
		  float ipotro;
		  float pc;
		  float pd;
		  float zz;
} VG_MATRIX;

__device__ __constant__ float cpoly[3240];//(20*3*3*2)+(8*20*18) When Ditail = 20 is maximum
__device__ __constant__ float crtable[5];	//Radios Table
__device__ __constant__ float ccolor_table[40]; // Color Table

__device__ __constant__
VG_MATRIX c_matrix[4]={	{1.250000,2.340000,3.154574,0.072868,0.034699,1.000000},
											 	{1.000000,2.755000,3.154574,0.485784,0.602893,-1.000000},
											 	{1.000000,2.755000,3.154574,0.485784,0.602893,-1.000000},
											 	{0.750000,3.170000,3.154574,5.031334,10.106042,1.000000}};

__device__ __constant__
float d_color_table[5][4]={ {0.35	,0.19	,0.19	,1.0},
														{0.19	,0.275,0.19	,1.0},
														{1.0	,0.4	,1.0	,1.0},
														{0.0	,0.8	,1.0	,1.0},
														{1.0	,1.0	,1.0	,1.0} };

__device__ __constant__
float d_r_table[5]={2.443/2,3.487/2,3.156/2,0.7,0.7};

////////////////////OTher C Routines
void CircleTable_float (float **sint,float **cost,const int n){

    int i;
    /* Table size, the sign of n flips the circle direction */
    const int size = abs(n);
    /* Determine the angle between samples */
    const float angle = 2*M_PI/(float)( ( n == 0 ) ? 1 : n );
    /* Allocate memory for n samples, plus duplicate of first entry at the end */
    *sint = (float *) calloc(sizeof(float), size+1);
    *cost = (float *) calloc(sizeof(float), size+1);
    /* Bail out if memory allocation fails, fgError never returns */
    if (!(*sint) || !(*cost))
    {
        free(*sint);
        free(*cost);
        printf("Failed to allocate memory in fghCircleTable");
        exit(0);
    }
    /* Compute cos and sin around the circle */
    (*sint)[0] = 0.0;
    (*cost)[0] = 1.0;
    for (i=1; i<size; i++)
    {
        (*sint)[i] = sin(angle*i);
        (*cost)[i] = cos(angle*i);
    }
    /* Last sample is duplicate of the first */
    (*sint)[size] = (*sint)[0];
    (*cost)[size] = (*cost)[0];
}

void map_sphere (int ditail){

	int l,m,k,j,p;
	int	vert_body = (((ditail/2)-2)*ditail*3*3*2);
	int	vert_hats	= ditail*3*3*2;
	int stacks=ditail/2;
	float z0,z1,r0,r1;
	int polysize_vert = vert_body+vert_hats;

	float *f_pol;
	f_pol  = (float*)malloc(polysize_vert * sizeof(float));

	printf("Mapping coordinates for one Sphere - By CPU\n");

  /* Pre-computed circle */
  float *sint1,*cost1;
  float *sint2,*cost2;

  CircleTable_float(&sint1,&cost1,-ditail);
  CircleTable_float(&sint2,&cost2,stacks*2);

  ///////Compute one Circle/////////////////////////
  z0 = 1.0f;
  z1 = cost2[(stacks>0)?1:0];
  r0 = 0.0f;
  r1 = sint2[(stacks>0)?1:0];
  for(m=0;m<ditail;m++){
  	*(f_pol+(9*m))     =0;
  	*(f_pol+1+(9*m))   =0;
  	*(f_pol+2+(9*m))   =1;
  }
  l=3;
  for (j=ditail; j>0; j--){
  	*(f_pol+l)     = cost1[j]*r1;
  	*(f_pol+l+1)   = sint1[j]*r1;
  	*(f_pol+l+2)   = z1;

  	*(f_pol+l+3)   = cost1[j-1]*r1;
  	*(f_pol+l+4)   = sint1[j-1]*r1;
  	*(f_pol+l+5)   = z1;
  	l+=9;
  }
  l-=3;
  for( k=1; k<stacks-1; k++ ){
  	z0 = z1; z1 = cost2[k+1];
  	r0 = r1; r1 = sint2[k+1];
  	p=0;
  	for(j=0; j<ditail; j++){
  		*(f_pol+l+p)     = cost1[j]*r1;
  		*(f_pol+l+p+1)   = sint1[j]*r1;
  		*(f_pol+l+p+2)   =z1;

  		*(f_pol+l+p+3)   = cost1[j]*r0;
  		*(f_pol+l+p+4)   = sint1[j]*r0;
  		*(f_pol+l+p+5)   = z0;

  		*(f_pol+l+p+6)   = cost1[j+1]*r1;
  		*(f_pol+l+p+7)   = sint1[j+1]*r1;
  		*(f_pol+l+p+8)   = z1;
  		//////////////////First Triangle////////////////////////////////
  		*(f_pol+l+p+9)   = *(f_pol+l+p+6);
  		*(f_pol+l+p+10)  = *(f_pol+l+p+7);
  		*(f_pol+l+p+11)  = *(f_pol+l+p+8);

  		*(f_pol+l+p+12)   = *(f_pol+l+p+3);
  		*(f_pol+l+p+13)   = *(f_pol+l+p+4);
  		*(f_pol+l+p+14)   = *(f_pol+l+p+5);

  		*(f_pol+l+p+15)  = cost1[j+1]*r0;
  		*(f_pol+l+p+16)  = sint1[j+1]*r0;
  		*(f_pol+l+p+17)  = z0;
  		/////////////////////Second Triangle//////////////////////////////
  		p+=18;
  	}
  	l+=(ditail)*6*3;
  }
  	////////////////////
  z0 = z1;
  r0 = r1;
  for(m=0;m<ditail;m++){
  	*(f_pol+l+(9*m))   = 0;
  	*(f_pol+l+1+(9*m)) = 0;
  	*(f_pol+l+2+(9*m)) = -1;
  }
  p=3;
  for (j=0; j<ditail; j++){
  	*(f_pol+l+p)     = cost1[j]*r0;
  	*(f_pol+l+p+1)   = sint1[j]*r0;
  	*(f_pol+l+p+2)   = z0;

  	*(f_pol+l+p+3)   = cost1[j+1]*r0;
  	*(f_pol+l+p+4)   = sint1[j+1]*r0;
  	*(f_pol+l+p+5)   = z0;
  	p+=9;
  }
  	///////////////////Allocate Memory and Varibales in GPU//
  checkCudaErrors(cudaMemcpyToSymbol(cpoly,f_pol,(polysize_vert*sizeof(float))));
  // Free Memory
  free(f_pol);
}

////////FORCE CALCULATION WITH GPU/////////////////////////////////////
__global__
void update_coor_kernel(int n3, float *vl,float *xs,
                        float *fc,float *side, float *poss,
                        int *atype){
#ifdef KER
	int tid  = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < n3){
		vl[tid]   =  (vl[tid]*(1-(*xs))+fc[tid])/(1+(*xs));
    poss[tid]	+=	vl[tid];
    if (poss[tid] < 0 || poss[tid] > side[tid % 3]) vl[tid] *= -1;
}
#endif
}
//////////////////////////////////////////////////////////////////////////
__device__ __inline__
void inter_if(float xj[3], float xi[3], float fi[3], int t, float xmax,
		float xmax1) {
#ifdef KER

	int k;
	float dn2, r, inr, inr2, inr4, inr8, d3, dr[3];
	float pb = (float) (0.338e-19 / (14.39 * 1.60219e-19)), dphir;

	dn2 = 0.0f;
	for (k = 0; k < 3; k++) {
		dr[k] = xi[k] - xj[k];
		dr[k] -= rintf(dr[k] * xmax1) * xmax;
		dn2 += dr[k] * dr[k];
	}
	r = sqrtf(dn2);
#if 1
	inr = 1.0f / r;
#elif 0
	if(dn2 != 0.0f) inr = 1.0f / r;
	else inr = 0.0f;
#elif 0
	if(dn2 == 0.0f) inr = 0.0f;
	else inr = 1.0f / r;
#else
	inr = 1.0f / r;
	if(dn2 == 0.0f) inr = 0.0f;
#endif
	inr2 = inr * inr;
	inr4 = inr2 * inr2;
	inr8 = inr4 * inr4;
	d3 = pb * c_matrix[t].pol
			* expf((c_matrix[t].sigm - r) * c_matrix[t].ipotro);
	dphir =
			(d3 * c_matrix[t].ipotro * inr - 6.0f * c_matrix[t].pc * inr8
					- 8.0f * c_matrix[t].pd * inr8 * inr2
					+ inr2 * inr * c_matrix[t].zz);
#if 1
	if (dn2 == 0.0f)
		dphir = 0.0f;
#endif
	for (k = 0; k < 3; k++)
		fi[k] += dphir * dr[k];
#endif
}

__global__
void nacl_kernel_if2(int n, int nat, float xmax, float *fvec, float *poss, int *atype) {
#ifdef KER
	int tid 		= threadIdx.x;
	int jdiv 		= tid / NTHREOPT2;
	int i 			= blockIdx.x * NTHREOPT2 + (tid & (NTHREOPT2 - 1)); // Same + (tid %16)
	int j, k;
	float xmax1 = 1.0f / xmax;
	int atypei;
	float xi[3];

	__shared__ float		s_xjj[NTHREOPT][3];
	__shared__ int			s_xa[NTHREOPT];
	__shared__ float 		s_fi[NTHREOPT][3];

	for (k = 0; k < 3; k++)
		s_fi[tid][k] = 0.0f;
	for (k = 0; k < 3; k++){
		xi[k] = poss[i*3+k];
	}

	atypei = atype[i] * nat;
	int na;
	na = n 	/ NTHREOPT;
	na = na * NTHREOPT;
	for (j = 0; j < na; j += NTHREOPT) {
		__syncthreads();

		s_xjj	[tid][0] 	= poss	[j*3 + tid*3];
		s_xjj	[tid][1]	= poss	[j*3 + tid*3+1];
		s_xjj	[tid][2] 	= poss	[j*3 + tid*3+2];
		s_xa	[tid]			= atype	[j + tid];
		__syncthreads();
#pragma unroll 16
		for (int js = jdiv; js < NTHREOPT; js += NDIV)

			inter_if(s_xjj[js], xi, s_fi[tid], atypei + s_xa[js], xmax, xmax1);
	}
	for (j = na + jdiv; j < n; j += NDIV) {

		inter_if(poss+j*3, xi, s_fi[tid], atypei + atype[j], xmax, xmax1);
	}
#if NTHREOPT>=512 && NTHREOPT2<=256
	__syncthreads();
	if(tid<256) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+256][k];
#endif
#if NTHREOPT>=256 && NTHREOPT2<=128
	__syncthreads();
	if (tid < 128)
		for (k = 0; k < 3; k++)
			s_fi[tid][k] += s_fi[tid + 128][k];
#endif
#if NTHREOPT>=128 && NTHREOPT2<=64
	__syncthreads();
	if (tid < 64)
		for (k = 0; k < 3; k++)
			s_fi[tid][k] += s_fi[tid + 64][k];
#endif
#if NTHREOPT>=64 && NTHREOPT2<=32
	__syncthreads();
	if (tid < 32)
		for (k = 0; k < 3; k++)
			s_fi[tid][k] += s_fi[tid + 32][k];
#endif
#if NTHREOPT2<=16
	if (tid < 16)
		for (k = 0; k < 3; k++)
			s_fi[tid][k] += s_fi[tid + 16][k];
#endif
#if NTHREOPT2<=8
	if(tid<8) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+8][k];
#endif
#if NTHREOPT2<=4
	if(tid<4) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+4][k];
#endif
#if NTHREOPT2<=2
	if(tid<2) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+2][k];
#endif
#if NTHREOPT2<=1
	if(tid<1) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+1][k];
#endif
	if (jdiv == 0)
		for (k = 0; k < 3; k++)
			fvec[i * 3 + k] = s_fi[tid][k];
#endif
}

__global__
void velforce_kernel(int n3, float *fc, float *a_mass, float *vl,
                     int *atype_mat, float hsq,float *ekin1,
                     int *atype, float *poss, float *sideh){
#ifdef KER
	__shared__ float cache [ThreadsPB];
  int indx = threadIdx.x;
	int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	cache [indx] = 0;

	if (tid < n3 ){
		fc[tid]-= fc[tid]/(n3/3);
		fc[tid] *= hsq/a_mass[atype_mat[atype[tid/3]]];
		cache [indx] = vl[tid]*vl[tid]*a_mass[atype_mat[atype[tid/3]]];
	}
	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>0; s>>=1)
	{
		if (indx < s)
		{
			cache[indx] += cache[indx + s];
		}
		__syncthreads();
	}
	if (indx == 0) ekin1[blockIdx.x] = cache [0];

#endif
}

__global__
void reduction (float *ekin,float *mtemp,float *mpres,float *xs,float tscale,
                float nden, float vir,int s_num,int w_num,float rtemp,
                float lq,float hsq,float *ekin1, int limi){

#ifdef KER
	__shared__ float cache [NTHREOPT];

  int indx = threadIdx.x;

	cache [indx] = (indx < limi) ? ekin1[indx]:0.0f;

	__syncthreads();

	for (unsigned int s=NTHREOPT/2; s>0; s>>=1){
		if (indx < s)
		{
			cache[indx] += cache[indx + s];
		}
			__syncthreads();
	  }

	if (indx == 0){
		*ekin = cache [0];
		*ekin /= hsq;
		*mtemp = tscale * (*ekin);
		*mpres  = nden / 3.f * ((*ekin) - (vir)) / (s_num + w_num);
		*xs += (*mtemp - rtemp) /  lq * hsq *.5f;
	}

#endif
}
/////////////////////For INTEROPERABILITY
__global__
void mapVertex(int n3,float *poss, float *sideh, float *glVertex){

	int tid  = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < n3 ){
		glVertex[tid] = poss[tid] - sideh[tid % 3]; //sideh[tid % 3]; // for graphics VBO -- Position
	}

}

__global__
void colorn4(int n4,float *vl,int *atype_mat, float *colorvbo, int *atype){
#ifdef KER
	int 	tid  	= threadIdx.x + blockIdx.x * blockDim.x;
	int 	ipx		= (tid/4) * 3;
	float d0;
	float d0aux[4];

	d0 = (vl[ipx]*vl[ipx]+vl[ipx+1]*vl[ipx+1]+vl[ipx+2]*vl[ipx+2])*500;
	d0aux[0] 	= d0;
	d0aux[1] 	= d0/3;
	d0aux[2]	= d0/3;
	d0aux[3]	= 0;

	if (tid < n4){
		colorvbo[tid] = d_color_table[atype_mat[atype[tid/4]]][tid%4] + d0aux[tid%4];
	}
#endif
}

__global__
void rend_sphe_VerNor (	float *poly, float *polyn, float *poss, float *sideh,
				 	 	 	 					int n3, int *atype_mat, int *atype ,float radios,
				 	 	 	 					int ipx){
#ifdef KER
	int 	tid = threadIdx.x + blockIdx.x * blockDim.x;
	int 	n 	= n3/3;
	float radius;
	float cd_aux;

	if  (tid < ipx*n){
		cd_aux 			= poss[((tid/ipx)*3)+(tid%3)]-sideh[tid%3];
		radius 			= radios*d_r_table[atype_mat[atype[tid/ipx]]];
		poly[tid] 	= cd_aux+cpoly[tid%ipx]*radius;
		polyn[tid]	= cpoly[tid%ipx];
	}
#endif
}


__global__
void rend_sphe_Color(	float *polyc_a, float *poss, float *sideh,
				 	 	 	 				int n3, int *atype_mat, int *atype ,float radios,
				 	 	 	 				float *vl,int ipx){

#ifdef KER
	int 	tid = threadIdx.x + blockIdx.x * blockDim.x;
	int 	n 			= n3/3;
	int		ipx4		= (ipx/3)*4;		//Number of vertex per polygon
	int 	tidpx3	= (tid/ipx4)*3;
	float d0;
	float d0aux[4];

	d0 				= (vl[tidpx3]*vl[tidpx3]+vl[tidpx3+1]*vl[tidpx3+1]+vl[tidpx3+2]*vl[tidpx3+2])*500;
	d0aux[0] 	= d0;
	d0aux[1] 	= d0/3;
	d0aux[2]	= d0/3;
	d0aux[3]	= 0;

	if (tid < ipx4*n){
		polyc_a[tid] = d_color_table[atype_mat[atype[tid/ipx4]]][tid%4] + d0aux[tid%4];
	}

#endif
}

////////////////// for DYNAMIC KERNELL CALL
#ifdef DP
__global__
void md_loop_cuda (	int n3, float *vl,float *xs,float *fc,float *side,
										int n, int nat, float xmax,
										float *a_mass, int *atype_mat, float hsq,float *ekin1,
										float *ekin,float *mtemp,float *mpres,float tscale,
										float nden, float vir,int s_num,int w_num,float rtemp,
										float lq,int limi,
										int md_step, float *sideh, float *d_glColr, float *poss,
										int *atype, float *glCoord,
										int polysize_vert, float *glPolyVert, float *glPolyNor, float *glPolyColor,
										float fradios, InterOpCUDAgl igl, RenderParticleType rpt)
{

	int  blocksPGrid = (n3 + ThreadsPB - 1)/(ThreadsPB);
	dim3 THREADS(NTHRE);
	dim3 BLOCKS((n3 + ThreadsPB - 1)/(ThreadsPB));
	dim3 threads(NTHREOPT);
	dim3 grid((n * NDIV + NTHREOPT - 1) / NTHREOPT);
	dim3 colorgridn4(((n*4) + ThreadsPB - 1)/(ThreadsPB));

	dim3 TREE(NTHRE);
  dim3 BLOCKS_TREE  	 ((polysize_vert * n + NTHRE - 1)/(NTHRE));
  dim3 BLOCKS_TREE_CLR (((polysize_vert/3) * (n*4) + NTHRE - 1)/(NTHRE));

	for(int md_loop = 0; md_loop < md_step; md_loop++){
		update_coor_kernel<<<BLOCKS,THREADS>>>
				(n3,vl,xs,fc,side,poss,atype);
		nacl_kernel_if2<<<grid, threads>>>
				(n, nat, xmax,fc,poss,atype);
		velforce_kernel<<<BLOCKS,THREADS>>>
				(n3,fc,a_mass,vl,atype_mat,hsq,ekin1,atype,poss,sideh);
		reduction<<<1,NTHRE>>>
				(ekin,mtemp,mpres,xs,tscale,nden,vir,s_num,w_num,rtemp,lq,hsq,ekin1,blocksPGrid);
	}

#ifdef INTEROP
	if(igl == YES){
		if(rpt == POINTS){
			mapVertex<<<BLOCKS,THREADS>>>
			(n3,poss,sideh,glCoord);
			colorn4<<<colorgridn4,THREADS>>>
			(n*4,vl,atype_mat,d_glColr,atype);
		}
		if(rpt == SPHERE){
			rend_sphe_VerNor<<<BLOCKS_TREE,TREE>>>
			(glPolyVert,glPolyNor,poss,sideh,n3,atype_mat,atype,fradios,polysize_vert);
			rend_sphe_Color<<<BLOCKS_TREE_CLR,TREE>>>
			(glPolyColor,poss,sideh,n3,atype_mat,atype,fradios,vl,polysize_vert);
		}
	}
#endif
}
#endif
Accel::Accel() {
		// Initialize CUDA
		checkCudaErrors(cudaSetDevice(0));
		checkCudaErrors(cudaGetDevice(&m_cuDevice));
		checkCudaErrors(cudaGetDeviceProperties(&m_cuDevProp,m_cuDevice));
		//checkCudaErrors(cudaGLSetGLDevice(m_cuDevice));  //Drepecated function according to CUDA
		std::cout<<"Device "<<m_cuDevice<<": "<<m_cuDevProp.name<<" is used!"<<std::endl;

		// Setting up all pointers
		d_atypemat = d_atype = NULL;
		d_force = d_side = d_sideh = d_amass = d_vl = NULL;
		d_ekin1 = d_ekin = d_xs = d_mtemp = d_mpres = d_glColr = d_glCoord = d_poss = NULL;
		d_glPolyVert = d_glPolyNor = d_glPolyColor = NULL;
		d_glHPossVBO = d_glHColorVBO = NULL;
		d_glHPolyVert = d_glHPolyNor = d_glHPolyColor = NULL;

#ifdef INTEROP
		g_strucPossVBOCUDA		= NULL;
		g_strucColorVBOCUDA		= NULL;
		g_strucPolyVertVBO		= NULL;
		g_strucPolyNorVBO			= NULL;
		g_strucPolyColorVBO		= NULL;

		// CUDA related structs
		g_possVBO = g_colorVBO = g_polyColorVBO = g_polyVertVBO = g_polyNorVBO = 0;

		// VBO to Render with GPU using normal Kernels for POINTS
		m_fPossVBO 	= new float [ (NMAX+NTHREOPT2)*3 ];
		m_fColorVBO = new float [ (NMAX+NTHREOPT2)*4 ];

		// VBO to Render with GPU using normal Kernels for SPHERE
		m_fPolyVert = m_fPolyNor = m_fPolyColor = NULL;
#endif

		// Timer related
		m_fFlops = m_fStepsec = 0.0f;

		// Memory Flags related
		m_bChangeInterop = m_bChangeMalloc = true;
}

void Accel::gpuMDloop(int n3,int grape_flg,double phi [3],double *phir,double *iphi, double *vir,int s_num3,
			timeval time_v,double *md_time0,double *md_time,int *m_clock,int md_step,double *mtemp,
			double tscale,double *mpres,double nden,int s_num,int w_num,double rtemp,double lq,
			double x[], int n, int atype[], int nat,
			double pol[], double sigm[], double ipotro[],
		 	double pc[], double pd[],double zz[],
		 	int tblno, double xmax, int periodicflag,
		 	double force[],
			double hsq,double a_mass [], int atype_mat [], double *ekin,double *vl,
			double *xs,double side [],double sideh[],
			int rendermode,unsigned int ditail, double radios,Settings cnfg){

	//////////////VARIABLES FROM THE BEGINING/////////////////
	int i,j,p;
	float xmaxf;
  if((periodicflag & 1)==0) xmax*=2.0;
	xmaxf=xmax;
  int n4 = n*4;
  unsigned int vert_body 			= (((ditail/2)-2)*ditail*3*2); 	//ONLY VERTEX OF THE STRUCTURE
  unsigned int vert_hats			= ditail*3*2;									 	//ONLY VERTEX OF THE STRUCTURE
	unsigned int polysize_vert 	= 3*(vert_hats+vert_body);			//Times 3 for X,Y,Z per Vertex
	unsigned int polysize_color	=	4*(vert_hats+vert_body); 			//Times 4 for R,G,B,A

  /////////////////////////////////////////////////////////
	int  blocksPGrid 	= (n3 + ThreadsPB - 1)/(ThreadsPB);
	dim3 THREADS(NTHRE);
	dim3 BLOCKS((n3 + ThreadsPB - 1)/(ThreadsPB));
	dim3 threads(NTHREOPT);
	dim3 grid((n * NDIV + NTHREOPT - 1) / NTHREOPT);
	dim3 colorgridn4((n4 + ThreadsPB - 1)/(ThreadsPB));
  // For Sphere rendering
	dim3 TREE(NTHRE);
  dim3 BLOCKS_TREE  	 ((polysize_vert * n + NTHRE - 1)/(NTHRE));
  dim3 BLOCKS_TREE_CLR ((polysize_color * n + NTHRE - 1)/(NTHRE));

	float   fxs = *xs;
	float   fside[3],*ffc, fsideh[3];
	float   *vla;

	float   hsqf = hsq;
	float   *fvl,fa_mass[4];

	float 	ftscale = tscale,fnden = nden,frtemp = rtemp,flq = lq,fvir = 0;
	float 	fmtemp = *mtemp,fmpres = *mpres;

	float		fradios = (float) radios;

	float 	*fposs, *cord;
	int			*auxatype;

	vla					= (float*)	malloc(n3*sizeof(float));
	cord				= (float*)	malloc(n3*sizeof(float));

	if(m_bChangeMalloc || m_bChangeInterop){
		std::cout<<"CUDA malloc time...\n";
		// Allocating memory for float conversion.
		ffc 			= (float*)	malloc(n3*sizeof(float));
		fvl 			= (float*)	malloc(n3*sizeof(float));
		auxatype 	= (int*)		malloc((NMAX+NTHREOPT2)*sizeof(int));
		fposs			=	(float*)	malloc((NMAX+NTHREOPT2)*3*sizeof(float));
		// Conversion from Double to Float
		for (p=0;p<4;p++) fa_mass[p] = (float) a_mass[p];
		for (p=0;p<3;p++) fside[p] 	 = (float) side[p];
		for (p=0;p<3;p++) fsideh[p]  = (float) sideh[p];
		for (p=0;p<n3;p++){
			fvl     [p] =  (float) *(vl +p);
			ffc     [p] =  (float) *(force +p);
		}
		for (i = 0; i < (n + NTHREOPT2 - 1) / NTHREOPT2 * NTHREOPT2; i++) {
			if (i < n) {
				for (j = 0; j < 3; j++) {
					fposs[i * 3 + j] = x[i * 3 + j];
				}
				auxatype[i]		=	atype[i];
			}
			else {
				for (j = 0; j < 3; j++) {
					fposs[i * 3 + j] = 0.0f;
				}
				auxatype[i] = 0;
			}
		}

#ifdef INTEROP
		if(cnfg.igt == YES){
			if (cnfg.rpt == POINTS){//case for Only Points
				unsigned int size;
				// As for CUDA-OpenGL Inter-operability
				// Unregister and clean OpenGL and CUDA resources
				if (g_strucPossVBOCUDA  != NULL)
					checkCudaErrors(cudaGraphicsUnregisterResource(g_strucPossVBOCUDA));
				if (g_strucColorVBOCUDA != NULL)
					checkCudaErrors(cudaGraphicsUnregisterResource(g_strucColorVBOCUDA));

				glDeleteBuffers(1,&g_possVBO);
				glDeleteBuffers(1,&g_colorVBO);

				// Creation of share buffer between CUDA and OpenGL
				// For Position
				glGenBuffers(1, &g_possVBO);
				glBindBuffer(GL_ARRAY_BUFFER, g_possVBO);
				size = (NMAX+NTHREOPT2)*3*sizeof(float);
				glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				// For Color
				glGenBuffers(1, &g_colorVBO);
				glBindBuffer(GL_ARRAY_BUFFER, g_colorVBO);
				size = (NMAX+NTHREOPT2)*4*sizeof(float);
				glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				// Register CUDA and OpenGL Interop
				checkCudaErrors(cudaGraphicsGLRegisterBuffer(&g_strucPossVBOCUDA,
												g_possVBO,cudaGraphicsMapFlagsNone));
				checkCudaErrors(cudaGraphicsGLRegisterBuffer(&g_strucColorVBOCUDA,
												g_colorVBO,cudaGraphicsMapFlagsNone));
				// Position
				size_t vbosizepos;
				checkCudaErrors(cudaGraphicsMapResources(1,&g_strucPossVBOCUDA,0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glCoord,
																							&vbosizepos,g_strucPossVBOCUDA));
				// Color
				size_t vbosizecol;
				checkCudaErrors(cudaGraphicsMapResources(1,&g_strucColorVBOCUDA,0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glColr,
																								&vbosizecol,g_strucColorVBOCUDA));
			}

			else if (cnfg.rpt == SPHERE){//case for Spheres rendering
				// Map angles and vertex of only One Sphere and put in Constant Memory on GPU
				map_sphere(ditail);

				// As for CUDA-OpenGL Inter-operability
				// Unregister and clean OpenGL and CUDA resources
				if(g_strucPolyVertVBO	 != NULL)
					checkCudaErrors(cudaGraphicsUnregisterResource(g_strucPolyVertVBO));
				if(g_strucPolyNorVBO   != NULL)
					checkCudaErrors(cudaGraphicsUnregisterResource(g_strucPolyNorVBO));
				if(g_strucPolyColorVBO != NULL)
					checkCudaErrors(cudaGraphicsUnregisterResource(g_strucPolyColorVBO));

				glDeleteBuffers(1,&g_polyVertVBO);
				glDeleteBuffers(1,&g_polyNorVBO);
				glDeleteBuffers(1,&g_polyColorVBO);

				// Creation of share buffer between CUDA and OpenGL
				// For Vertex/Position
				glGenBuffers(1, &g_polyVertVBO);
				glBindBuffer(GL_ARRAY_BUFFER,g_polyVertVBO);
				//size = ((n*vert_hats)+(n*vert_body))*sizeof(float);
				glBufferData(GL_ARRAY_BUFFER, n*polysize_vert*sizeof(float), 0, GL_DYNAMIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				// For Normals
				glGenBuffers(1, &g_polyNorVBO);
				glBindBuffer(GL_ARRAY_BUFFER,g_polyNorVBO);
				//size = ((n*vert_hats)+(n*vert_body))*sizeof(float);
				glBufferData(GL_ARRAY_BUFFER, n*polysize_vert*sizeof(float), 0, GL_DYNAMIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				// For Color
				glGenBuffers(1, &g_polyColorVBO);
				glBindBuffer(GL_ARRAY_BUFFER,g_polyColorVBO);
				//size = ((n*vert_hats)+(n*vert_body))*sizeof(float);
				//size_color = (size/3)+size;
				glBufferData(GL_ARRAY_BUFFER, n*polysize_color*sizeof(float), 0, GL_DYNAMIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				// Register CUDA and OpenGL Inter-operability
				checkCudaErrors(cudaGraphicsGLRegisterBuffer(&g_strucPolyVertVBO,
																											g_polyVertVBO,cudaGraphicsMapFlagsNone));
				checkCudaErrors(cudaGraphicsGLRegisterBuffer(&g_strucPolyNorVBO,
																											g_polyNorVBO,cudaGraphicsMapFlagsNone));
				checkCudaErrors(cudaGraphicsGLRegisterBuffer(&g_strucPolyColorVBO,
																											g_polyColorVBO,cudaGraphicsMapFlagsNone));

				// Vertex
				size_t vbosizepos;
				checkCudaErrors(cudaGraphicsMapResources(1,&g_strucPolyVertVBO,0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glPolyVert,
																														&vbosizepos,g_strucPolyVertVBO));
				// Normal
				size_t vbosizenor;
				checkCudaErrors(cudaGraphicsMapResources(1,&g_strucPolyNorVBO,0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glPolyNor,
																														&vbosizenor,g_strucPolyNorVBO));
				// Color
				size_t vbosizecol;
				checkCudaErrors(cudaGraphicsMapResources(1,&g_strucPolyColorVBO,0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glPolyColor,
																														&vbosizecol,g_strucPolyColorVBO));
		}
	}

		// Half INTEROP and SPHERE
		if (cnfg.igt == HALF && cnfg.rpt == SPHERE){

			// Map angles and vertex of only One Sphere and put in Constant Memory on GPU
			map_sphere(ditail);

			// Free CUDA memory. In case we already allocate
			if(d_glHPolyVert	!= NULL)checkCudaErrors(cudaFree(d_glHPolyVert));
			if(d_glHPolyNor		!= NULL)checkCudaErrors(cudaFree(d_glHPolyNor));
			if(d_glHPolyColor	!= NULL)checkCudaErrors(cudaFree(d_glHPolyColor));
			// Allocate global memory to GPU
			checkCudaErrors(cudaMalloc((void**)&d_glHPolyVert,sizeof(float)*polysize_vert*n));
			checkCudaErrors(cudaMalloc((void**)&d_glHPolyNor,sizeof(float)*polysize_vert*n));
			checkCudaErrors(cudaMalloc((void**)&d_glHPolyColor,sizeof(float)*polysize_color*n));

			// Form memory on CPU
			if(m_fPolyVert 	!= NULL)delete [] m_fPolyVert;
			if(m_fPolyNor 	!= NULL)delete [] m_fPolyNor;
			if(m_fPolyColor != NULL)delete [] m_fPolyColor;

			m_fPolyVert 	= new float[polysize_vert*n];
			m_fPolyNor		= new float[polysize_vert*n];
			m_fPolyColor	= new float[polysize_color*n];
		}
#endif
		// Free CUDA memory. In case we already allocate
		if(d_poss 		!= NULL)checkCudaErrors(cudaFree(d_poss));
		if(d_force 		!= NULL)checkCudaErrors(cudaFree(d_force));
		if(d_side 		!= NULL)checkCudaErrors(cudaFree(d_side));
		if(d_sideh 		!= NULL)checkCudaErrors(cudaFree(d_sideh));
		if(d_amass 		!= NULL)checkCudaErrors(cudaFree(d_amass));
		if(d_vl 			!= NULL)checkCudaErrors(cudaFree(d_vl));
		if(d_atypemat != NULL)checkCudaErrors(cudaFree(d_atypemat));
		if(d_ekin 		!= NULL)checkCudaErrors(cudaFree(d_ekin));
		if(d_xs 			!= NULL)checkCudaErrors(cudaFree(d_xs));
		if(d_mtemp 		!= NULL)checkCudaErrors(cudaFree(d_mtemp));
		if(d_mpres 		!= NULL)checkCudaErrors(cudaFree(d_mpres));
		if(d_ekin1 		!= NULL)checkCudaErrors(cudaFree(d_ekin1));
		if(d_atype 		!= NULL)checkCudaErrors(cudaFree(d_atype));
		//if(d_glHPossVBO 	!= NULL)checkCudaErrors(cudaFree(d_glHPossVBO));
		//if(d_glHColorVBO	!= NULL)checkCudaErrors(cudaFree(d_glHColorVBO));


		// Allocate global memory to GPU
		checkCudaErrors(cudaMalloc((void**)&d_poss,sizeof(float)*(NMAX + NTHREOPT2)*3));
		checkCudaErrors(cudaMalloc((void**)&d_force,sizeof(float)*(NMAX + NTHREOPT2)*3));
		checkCudaErrors(cudaMalloc((void**)&d_side,3*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_sideh,3*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_amass,4*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_vl,n3*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_atypemat,20*sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_ekin,sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_xs,sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_mtemp,sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_mpres,sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_ekin1,blocksPGrid*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_atype,sizeof(int)*(NMAX + NTHREOPT2)));
		//checkCudaErrors(cudaMalloc((void**)&d_glHPossVBO,sizeof(float)*(NMAX + NTHREOPT2)*3));
		//checkCudaErrors(cudaMalloc((void**)&d_glHColorVBO,sizeof(float)*(NMAX + NTHREOPT2)*4));

		// Copy memory from CPU to GPU
		checkCudaErrors(cudaMemcpy(d_poss,fposs,
															sizeof(float)*3*((n+NTHREOPT2-1)/NTHREOPT2*NTHREOPT2),
															cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_side,fside,sizeof(float)*3,cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_sideh,fsideh,sizeof(float)*3,cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_mtemp,&fmtemp,sizeof(float),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_mpres,&fmpres,sizeof(float),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_xs,&fxs,sizeof(float),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vl,fvl,sizeof(float)*n3,cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_amass,fa_mass,sizeof(float)*4,cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_atypemat,atype_mat,sizeof(int)*20,cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_force,ffc,sizeof(float)*n*3,cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_atype,auxatype,
															sizeof(int)*((n+NTHREOPT2-1)/NTHREOPT2*NTHREOPT2),
															cudaMemcpyHostToDevice));
		// Free the memory used to convert from Double to Float
		free(ffc);
		free(fvl);
		free(fposs);
		free(auxatype);
	}

	else{
#ifdef INTEROP
		if(cnfg.igt == YES){
			if (cnfg.rpt == POINTS){ //For points
				// Position
				size_t vbosizepos;
				checkCudaErrors(cudaGraphicsMapResources(1,&g_strucPossVBOCUDA,0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glCoord,
																								&vbosizepos,g_strucPossVBOCUDA));
				// Color
				size_t vbosizecol;
				checkCudaErrors(cudaGraphicsMapResources(1,&g_strucColorVBOCUDA,0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glColr,
																								&vbosizecol,g_strucColorVBOCUDA));
			}
			else if (cnfg.rpt == SPHERE){ //For Spheres
				// Vertex
				size_t vbosizepos;
				checkCudaErrors(cudaGraphicsMapResources(1,&g_strucPolyVertVBO,0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glPolyVert,
																								&vbosizepos,g_strucPolyVertVBO));
				// Normal
				size_t vbosizenor;
				checkCudaErrors(cudaGraphicsMapResources(1,&g_strucPolyNorVBO,0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glPolyNor,
																								&vbosizenor,g_strucPolyNorVBO));
				// Color
				size_t vbosizecol;
				checkCudaErrors(cudaGraphicsMapResources(1,&g_strucPolyColorVBO,0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glPolyColor,
																								&vbosizecol,g_strucPolyColorVBO));
			}
		}
#endif
	}
///////Md_loop///////////////////////////////////////////////
if(cnfg.krt == DYNAMIC){
#ifdef DP
	float 			cudakernltimer = 0.0f;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start));

	md_loop_cuda<<<1,1>>>
	(n3,d_vl,d_xs,d_force,d_side,
	 n,nat,xmaxf,
	 d_amass,d_atypemat,hsqf,d_ekin1,
	 d_ekin,d_mtemp,d_mpres,ftscale,fnden,fvir,s_num,w_num,frtemp,flq,blocksPGrid,
	 md_step,d_sideh,d_glColr,d_poss,d_atype,d_glCoord,
	 polysize_vert,d_glPolyVert,d_glPolyNor,d_glPolyColor,fradios,cnfg.igt,cnfg.rpt);

	checkCudaErrors(cudaEventRecord(stop));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&cudakernltimer,start,stop));

	m_fStepsec 	= cudakernltimer/1000.0f;
	m_fFlops		= (double)n*(double)n*78/(cudakernltimer/1000.0f)*1e-9*md_step;

#endif
}
else if(cnfg.krt == NORMAL){

#ifndef DP
	float 			cudakernltimer = 0.0f;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start));

	for(int md_loop = 0; md_loop < md_step; md_loop++){
		update_coor_kernel<<<BLOCKS,THREADS>>>
		(n3,d_vl,d_xs,d_force,d_side,d_poss,d_atype);
		nacl_kernel_if2<<<grid, threads>>>
		(n, nat, xmaxf, d_force,d_poss,d_atype);
		velforce_kernel<<<BLOCKS,THREADS>>>
		(n3,d_force,d_amass,d_vl,d_atypemat,hsqf,d_ekin1,d_atype,d_poss,d_sideh);
		reduction<<<1,threads>>>
		(d_ekin,d_mtemp,d_mpres,d_xs,ftscale,fnden,fvir,s_num,w_num,frtemp,flq,hsqf,d_ekin1,blocksPGrid);
	}

	checkCudaErrors(cudaEventRecord(stop));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&cudakernltimer,start,stop));

	m_fStepsec 	= cudakernltimer/1000.0f;
	m_fFlops		= (double)n*(double)n*78/(cudakernltimer/1000.0f)*1e-9*md_step;
	//printf("CUDA measure time:%f\n",msec/1000);

	if(cnfg.igt == YES){
		if(cnfg.rpt == POINTS){
			mapVertex<<<BLOCKS,THREADS>>>
					(n3,d_poss,d_sideh,d_glCoord);
			colorn4<<<colorgridn4,THREADS>>>
					(n4,d_vl,d_atypemat,d_glColr,d_atype);
		}
		else if (cnfg.rpt == SPHERE){
			rend_sphe_VerNor<<<BLOCKS_TREE,TREE>>>
					(d_glPolyVert,d_glPolyNor,d_poss,d_sideh,n3,d_atypemat,d_atype,fradios,polysize_vert);
			rend_sphe_Color<<<BLOCKS_TREE_CLR,TREE>>>
					(d_glPolyColor,d_poss,d_sideh,n3,d_atypemat,d_atype,fradios,d_vl,polysize_vert);
		}
	}

#ifdef INTEROP
	if(cnfg.igt == HALF){
		if(cnfg.rpt == POINTS){
			mapVertex<<<BLOCKS,THREADS>>>
					(n3,d_poss,d_sideh,d_glHPossVBO);
			colorn4<<<colorgridn4,THREADS>>>
					(n4,d_vl,d_atypemat,d_glHColorVBO,d_atype);
		}
		else if (cnfg.rpt == SPHERE){
			rend_sphe_VerNor<<<BLOCKS_TREE,TREE>>>
					(d_glHPolyVert,d_glHPolyNor,d_poss,d_sideh,n3,d_atypemat,d_atype,fradios,polysize_vert);
			rend_sphe_Color<<<BLOCKS_TREE_CLR,TREE>>>
					(d_glHPolyColor,d_poss,d_sideh,n3,d_atypemat,d_atype,fradios,d_vl,polysize_vert);
		}
	}
#endif

#endif
}

#ifdef INTEROP
if(cnfg.igt == YES){
	if(cnfg.rpt == POINTS){
		checkCudaErrors(cudaGraphicsUnmapResources(1,&g_strucPossVBOCUDA,0));
		checkCudaErrors(cudaGraphicsUnmapResources(1,&g_strucColorVBOCUDA,0));
	}
	else if(cnfg.rpt == SPHERE){
		checkCudaErrors(cudaGraphicsUnmapResources(1,&g_strucPolyVertVBO,0));
		checkCudaErrors(cudaGraphicsUnmapResources(1,&g_strucPolyNorVBO,0));
		checkCudaErrors(cudaGraphicsUnmapResources(1,&g_strucPolyColorVBO,0));
	}
}


if(cnfg.igt == HALF){
	if(cnfg.rpt == POINTS){
		checkCudaErrors(cudaMemcpy(m_fPossVBO,d_glHPossVBO,n*3*sizeof(float),cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(m_fColorVBO,d_glHColorVBO,n*4*sizeof(float),cudaMemcpyDeviceToHost));
	}
	else if(cnfg.rpt == SPHERE){
		checkCudaErrors(cudaMemcpy(m_fPolyVert,d_glHPolyVert,n*polysize_vert*sizeof(float),cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(m_fPolyNor,d_glHPolyNor,n*polysize_vert*sizeof(float),cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(m_fPolyColor,d_glHPolyColor,n*polysize_color*sizeof(float),cudaMemcpyDeviceToHost));
	}
}
#endif

if(cnfg.igt == NO){
	checkCudaErrors(cudaMemcpy(vla,d_vl,n3*sizeof(float),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cord,d_poss,n3*sizeof(float),cudaMemcpyDeviceToHost));
	for(p=0;p<n3;p++) *(vl+p) = (double) vla[p];
	for(i=0;i<n;i++)for(j=0;j<3;j++) *(x+i*3+j) = (double)cord[j+i*3];
}

	free(vla);
	free(cord);
	m_bChangeMalloc		= false;
	m_bChangeInterop 	= false;

}

Accel::~Accel() {
	// Unregister if CUDA-InteropGL
#ifdef INTEROP
		std::cout<<"Unregistering CUDA-GL Resources...\n";
		if(g_strucPossVBOCUDA  != NULL)
			checkCudaErrors(cudaGraphicsUnregisterResource(g_strucPossVBOCUDA));
		if(g_strucColorVBOCUDA != NULL)
			checkCudaErrors(cudaGraphicsUnregisterResource(g_strucColorVBOCUDA));
		if(g_strucPolyVertVBO	 != NULL)
			checkCudaErrors(cudaGraphicsUnregisterResource(g_strucPolyVertVBO));
		if(g_strucPolyNorVBO   != NULL)
			checkCudaErrors(cudaGraphicsUnregisterResource(g_strucPolyNorVBO));
		if(g_strucPolyColorVBO != NULL)
			checkCudaErrors(cudaGraphicsUnregisterResource(g_strucPolyColorVBO));
#endif
	// Free memory for HALF interop
	delete [] m_fPossVBO;
	delete [] m_fColorVBO;
	delete [] m_fPolyVert;
	delete [] m_fPolyNor;
	delete [] m_fPolyColor;
}

