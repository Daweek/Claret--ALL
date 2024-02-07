/*
 * Accel.h
 *
 *  Created on: Jul 25, 2018
 *      Author: Edg@r j.
 */
#ifndef ACCEL_H_
#define ACCEL_H_
#pragma once

#include <iostream>
#include <assert.h>
#include <GL/glew.h>
// ***** CUDA includes
#include <cuda.h>
//#include <nvcuvid.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "cras36def.h"

const std::string cKernelType[] = {"NORMAL","DYNAMIC"};
const std::string cInterOpCUDAgl[] = {"YES","NO","HALF"};

class Accel {
	private:
		CUdevice						m_cuDevice;
		cudaDeviceProp			m_cuDevProp;

		int   	*d_atypemat;
		int			*d_atype;

		float 	*d_force;
		float   *d_side,*d_sideh;
		float   *d_amass,*d_vl;
		float 	*d_ekin1;
		float 	*d_ekin,*d_xs,*d_mtemp,*d_mpres;
		float		*d_poss;

		float		*d_glColr;
		float		*d_glCoord;
		float		*d_glPolyVert;
		float		*d_glPolyNor;
		float		*d_glPolyColor;

		float		*d_glHPossVBO;
		float		*d_glHColorVBO;
		float		*d_glHPolyVert;
		float		*d_glHPolyNor;
		float		*d_glHPolyColor;

#ifdef INTEROP
		// In case of using Points for render
		//GLuint g_possVBO, g_colorVBO;
		struct cudaGraphicsResource* g_strucPossVBOCUDA;
		struct cudaGraphicsResource* g_strucColorVBOCUDA;
		// In case we draw (polygon) using Sphere
		//GLuint g_polyVertVBO, g_polyNorVBO, g_polyColorVBO;
		struct cudaGraphicsResource* g_strucPolyVertVBO;
		struct cudaGraphicsResource* g_strucPolyNorVBO;
		struct cudaGraphicsResource* g_strucPolyColorVBO;
#endif

	public:
		float		m_fFlops;
		float		m_fStepsec;

		GLuint 	g_possVBO, g_colorVBO;
		GLuint 	g_polyVertVBO, g_polyNorVBO, g_polyColorVBO;

		float		*m_fPossVBO,*m_fColorVBO;
		float		*m_fPolyVert,*m_fPolyNor,*m_fPolyColor;

		bool		m_bChangeMalloc;
		bool		m_bChangeInterop;

		Accel();
		void gpuMDloop(int n3,int grape_flg,double phi [3],double *phir,double *iphi, double *vir,int s_num3,
				timeval time_v,double *md_time0,double *md_time,int *m_clock,int md_step,double *mtemp,
				double tscale,double *mpres,double nden,int s_num,int w_num,double rtemp,double lq,
				double x[], int n, int atype[], int nat,
				double pol[], double sigm[], double ipotro[],
			 	double pc[], double pd[],double zz[],
			 	int tblno, double xmax, int periodicflag,
			 	double force[],
				double hsq,double a_mass [], int atype_mat [], double *ekin,double *vl,
				double *xs,double side [], double sideh [],
				int rendermode,unsigned int ditail, double radios,Settings cnfg);
		~Accel();
};

#endif /* ACCEL_H_ */
