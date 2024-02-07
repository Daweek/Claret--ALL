/*
 * Crass.h
 *
 *  Created on: Jul 10, 2018
 *      Author: Edg@r J.
 */
#ifndef CRASS_H_
#define CRASS_H_
#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include <omp.h>
#include <math.h>
#include <time.h>

#include "cras36def.h"
#include "Accel.hpp"

const std::string cHardwareType[] = {"CPU_1","CPU_OMP","GPU"};

class Crass {
	private:


	public:
		// Variables to share on the global scope
		int* 					m_iNp;
		int* 					m_iNpKey;
		int* 					m_iMdstep;
		int*					m_drow_flg;
		int*					m_atype_mat;
		int*					m_atype;
		int*					m_cflg;
		int*					m_cnum;
		int*					m_mclock;
		int*					m_npx;
		int*					m_npy;
		int*					m_npz;

		unsigned int* m_ditail;

		double*				m_temp;
		double*				m_rtemp;
		double*				m_epsv;
		double*				m_kb;
		double*				m_side;
		double*				m_sideh;
		double*				m_eyelen;

		double*				m_cd;
		double*				m_vl;
		double*				m_pTrans;



		Crass(unsigned int npkey, unsigned int mdstep, double temperature,
					unsigned int sphereDitail);
		void md_run(Settings cnfg,Accel* gpu);
		timespec diff(timespec start, timespec end);

		void handler_set_cd(int n);


		~Crass();
};

#endif /* CRASS_H_ */
