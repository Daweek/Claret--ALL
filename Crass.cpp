/*
 * Crass.cpp
 *
 *  Created on: Jul 10, 2018
 *      Author: Edg@r J.
 */

#include "cras36.h"  // Here is the whole initialization for the main engine
#include "Crass.h"
#include "Render.hpp"

void ForceCPU_omp(double x[], int n, int atype[], int nat,
                 double pol[], double sigm[], double ipotro[],
                 double pc[], double pd[], double zz[],
                 int tblno, double xmax, int periodicflag,
                 double force[])
{
  int i,j,k,t;
  double xmax1,dn2,r,inr,inr2,inr4,inr8,d3,dr[3],fi[3];
  double pb=0.338e-19/(14.39*1.60219e-19),dphir;
  if((periodicflag & 1)==0) xmax *= 2;
  xmax1 = 1.0 / xmax;
#pragma omp parallel for private(k,j,dn2,dr,r,inr,inr2,inr4,inr8,t,d3,dphir,fi)
  for(i=0; i<n; i++){
    for(k=0; k<3; k++) fi[k] = 0.0;
    for(j=0; j<n; j++){
      dn2 = 0.0;
      for(k=0; k<3; k++){
        dr[k] =  x[i*3+k] - x[j*3+k];
        dr[k] -= rint(dr[k] * xmax1) * xmax;
        dn2   += dr[k] * dr[k];
      }
      if(dn2 != 0.0){
        r     = sqrt(dn2);
        inr   = 1.0  / r;
        inr2  = inr  * inr;
        inr4  = inr2 * inr2;
        inr8  = inr4 * inr4;
        t     = atype[i] * nat + atype[j];
        d3    = pb * pol[t] * exp( (sigm[t] - r) * ipotro[t]);
        dphir = ( d3 * ipotro[t] * inr
                  - 6.0 * pc[t] * inr8
                  - 8.0 * pd[t] * inr8 * inr2
                  + inr2 * inr * zz[t] );
        for(k=0; k<3; k++) fi[k] += dphir * dr[k];
      }
    }
    for(k=0; k<3; k++) force[i*3+k] = fi[k];
  }
}
/////////////////////////////////////////////////////////////////////////////////

Crass::Crass(unsigned int npkey, unsigned int mdstep, double temperature,
						 unsigned int sphereDitail) {
	// TODO Auto-generated constructor stub

	np 					= npkey;  			// Number of particles from 1 - 12
  md_step			=	mdstep;				// Inner loop which calls to compute force
	temp 				= temperature;	// Initial Temperature
	ditail			= sphereDitail; // Set up variable for Number of vertex in Sphere

	init_MD();
	keep_mem(S_NUM_MAX,W_NUM_MAX*w_site);
	set_cd(1);

	// Getting pointers to object
	m_cd				= cd;
	m_atype			= atype;
	m_vl				= vl;

	m_iNp 			= &n1;
	m_iNpKey		= &np;
	m_npx				= &npx;
	m_npy				= &npy;
	m_npz				= &npz;

	m_iMdstep 	= &md_step;
	m_temp			= &temp;
	m_rtemp			= &rtemp;
	m_epsv			= &epsv;
	m_kb				= &kb;
	m_ditail		= &ditail;
	m_cflg			= &c_flg;
	m_cnum			= &c_num;
	m_mclock		= &m_clock;

	m_side			= &(side[0]);
	m_sideh			= &(sideh[0]);
	m_eyelen		= &eye_len;

	m_drow_flg 	= drow_flg;
	m_atype_mat	= atype_mat;

	m_pTrans		= NULL; // This one is initialized in Window.cpp

}

void Crass::md_run(Settings cnfg, Accel* gpu){
	int i;
	int md_loop;

	double zz2[2][2];  //,center[3];
	int ii,jj;
	static int n3_bak=0;

	if(n3!=n3_bak){
		if(n3_bak!=0)
		n3_bak=n3;
	}

	if (cnfg.hdt != GPU){
		// Measure time
		timespec time1, time2;
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

		for (md_loop = 0; md_loop < md_step; md_loop++){
			m_clock++;
			/////////////////////Update_coor_kernel///////////////////////////////
			for (i = 0; i < n3; i++) { /* update coordinations */
				if (atype[i / 3] <= 2 && atype[i / 3] != 8) {
					vl[i] = (vl[i] * (1 - xs) + fc[i]) / (1 + xs);
					cd[i] += vl[i];
				}
			}
			for (i = 0; i < n3; i += 3) {
				if (atype[i / 3] <= 2) {
					if (cd[i] < 0 || cd[i] > side[0])	vl[i] *= -1;
					if (cd[i + 1] < 0 || cd[i + 1] > side[1])	vl[i + 1] *= -1;
					if (cd[i + 2] < 0 || cd[i + 2] > side[2])	vl[i + 2] *= -1;
					} else {
					printf("atye[%d] is %d\n", i / 3, atype[i / 3]);
				}
			}

			for (i = 0; i < 2; i++)phi[i] = 0;
			phir = 0;		vir = 0;
			for (i = 0; i < n3; i++){fc[i] = 0;iphi[i] = 0;}
			for (i = 0; i < s_num3; i++)fc[i] 	= 0;
			for (i = 0; i < n1; i++) nig_num[i] = 0;

			if(cnfg.hdt == CPU_1){ //Original way for CPU, only 1 thread
				int i,j,i0,i1;
				double d0,d1,d2,d3,d4,d5,dphir;

				for (i = 0; i < n3; i += 3) {
					i0 = atype_mat[atype[i / 3]];
					for (j = i + 3; j < n3; j += 3) {
						d0 = cd[i] - cd[j];
						d1 = cd[i + 1] - cd[j + 1];
						d2 = cd[i + 2] - cd[j + 2];
						rd = d0 * d0 + d1 * d1 + d2 * d2;
						r = sqrt(rd);
						inr = 1. / r;
						i1 = atype_mat[atype[j / 3]];
						//d7 = phir;
						if (i0 < 2 && i1 < 2) {
								d3 = pb * pol[i0][i1]
										* exp((sigm[i0][i1] - r) * ipotro[i0][i1]);

								dphir = (d3 * ipotro[i0][i1] * inr
										- 6 * pc[i0][i1] * pow(inr, 8)
										- 8 * pd[i0][i1] * pow(inr, 10)
										+ inr * inr * inr * zz[i0][i1]);
						}
						vir -= rd * dphir;
						d3 = d0 * dphir;
						d4 = d1 * dphir;
						d5 = d2 * dphir;
						fc[i] += d3;
						fc[i + 1] += d4;
						fc[i + 2] += d5;
						fc[j] -= d3;
						fc[j + 1] -= d4;
						fc[j + 2] -= d5;
					}
				}

		}

		if(cnfg.hdt == CPU_OMP){
			for(ii=0;ii<2;ii++) for(jj=0;jj<2;jj++)	zz2[ii][jj]=zz[ii][jj];

			ForceCPU_omp(cd,n3/3,atype,2,(double *)pol,(double *)sigm,
						(double *)ipotro,(double *)pc,(double *)pd,
						(double*)zz2,8,side[0],0,fc);

		}
		//////////////////////////VelForce_Kernel & Reduction_Kernel///
		for (i = 0; i < n3; i++) {
			if (atype[i / 3] == 2)
				fc[i] *= hsq / (a_mass[2] + 2 * a_mass[3]);
			else if (atype[i / 3] == 0 || atype[i / 3] == 1)
				fc[i] *= hsq / a_mass[atype_mat[atype[i / 3]]];
		}
		for (i = 0; i < w_num3; i++)trq[i] *= hsq;
		ekin1 = 0;
		ekin2 = 0;
		for (i = 0; i < n3; i += 3) {
			ekin1 += (vl[i] * vl[i] + vl[i + 1] * vl[i + 1]
					+ vl[i + 2] * vl[i + 2]) * a_mass[atype_mat[atype[i / 3]]];
		}
		for (i = 0; i < w_num3; i += 3) {
			ekin2 += (moi[0] * agvph[i] * agvph[i]
					+ moi[1] * agvph[i + 1] * agvph[i + 1]
					+ moi[2] * agvph[i + 2] * agvph[i + 2]);
		}
		ekin1 	/= hsq;
		ekin2 	/= hsq;
		ekin 		= ekin1 + ekin2;
		mtemp 	= tscale * ekin;
		mpres 	= nden / 3. * (ekin - vir) / (s_num + w_num);
		xs 			+= (mtemp - rtemp) / lq * hsq * .5;
		}
	// Measure Time
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
	float totalsec = diff(time1,time2).tv_sec + (1.0e-9*(diff(time1,time2).tv_nsec));
	//std::cout<<diff(time1,time2).tv_sec<<":"<<diff(time1,time2).tv_nsec<<std::endl;
	//std::cout.precision(6);
	//std::cout<<totalsec<<std::endl;


	gpu->m_fStepsec = totalsec;
	gpu->m_fFlops		= (double)n1*(double)n1*78/(totalsec)*1e-9*md_step;

	}
	////////////////////////////////////End CPU MD
	else if(cnfg.hdt == GPU){
		for(ii=0;ii<2;ii++) for(jj=0;jj<2;jj++)	zz2[ii][jj]=zz[ii][jj];
			gpu->gpuMDloop(n3,grape_flg,phi,&phir,iphi,&vir,s_num3,time_v,&md_time0,&md_time,
					&m_clock,md_step,&mtemp,tscale,&mpres,nden,s_num,w_num,rtemp,lq,
					cd,n3/3,atype,2,(double *)pol,(double *)sigm,
					(double *)ipotro,(double *)pc,(double *)pd,
					(double*)zz2,8,side[0],0,fc,
					hsq,a_mass,atype_mat,&ekin,vl,
					&xs,side,sideh,
					rendermode,*m_ditail,radius,cnfg);

	}

}

timespec Crass::diff(timespec start, timespec end){
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
			temp.tv_sec = end.tv_sec-start.tv_sec-1;
			temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
			temp.tv_sec = end.tv_sec-start.tv_sec;
			temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

void Crass::handler_set_cd(int n){
	set_cd(n);
}

Crass::~Crass() {
	// TODO Auto-generated destructor stub
	// free memory from Cras36.h
	std::cout<<"Cleaning memory from crass.h ...\n";
	free(nli);
	free(nig_num);
	free(nig_data);
	free(atype);
	free(cd);
	free(vl);
	free(fc);
	free(fcc);
	free(iphi);
	free(ang);
	free(agv);
	free(agvp);
	free(angh);
	free(agvh);
	free(agvph);
	free(trq);
	free(w_index);
	free(w_rindex);
	free(w_info);
	free(erfct);
	for(int i = 0;i < VMAX;i++)free(vecn[i]);
}

