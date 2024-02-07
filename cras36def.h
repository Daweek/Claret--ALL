#ifndef CRAS36DEF_H_
#define CRAS36DEF_H_
#pragma once
/*
		Definitions and Global variables.

    Edg@r J.
*/
// App related
#define VER 1.00
#define LAP_TIME
#define C_MASS
#define CROSS
#define INFO

#define KER
#define INTEROP //In case using rCUDA or other vGPUtools that does not support GL-Interop
//#define DP			//Need to find another solution

//////////////////Deprecated
//#define CUDATIMER
//#define TIME_MEMORY
//////////////////Deprecated

enum HardwareType{CPU_1,CPU_OMP,GPU};
enum RenderParticleType{POINTS,SPHERE,TEXTURE};
enum FrameBufferType{MAIN,FBO};
enum KernelType{NORMAL,DYNAMIC};
enum InterOpCUDAgl{YES,NO,HALF};

struct Settings{
		HardwareType 				hdt;
		RenderParticleType	rpt;
		FrameBufferType			fbt;
		KernelType					krt;
		InterOpCUDAgl				igt;
};

// Legacy code from cras36.c constants and definitions
#define GL_ON
#if defined(MDGRAPE3) || defined(VTGRAPE)
#define MDM 2      /* 0:host 2:m2 */
#else
#define MDM 0      /* 0:host 2:m2 */
#endif
#define SPC 0
#define ST2 0
#define TIP5P 1
#define SYS 0 /* 0:NaCl 1:water(fcc) 2:water(ice) 3:water(ice2) 4:NaCl-water */

// Memory related
#define S_NUM_MAX 10*10*10*8*10
#define W_NUM_MAX 10*10*10*8*10
#define ZERO_P 1
#define V_SCALE 0
#define T_CONST 1
#define P_CONST 0
#define KNUM 5                    /* number of particle type */
#define VMAX 462 /*1535*/        /* max value of wave nubmer vector */
#define EFT 12000
#define my_min(x,y) ((x)<(y) ? (x):(y))
#define my_max(x,y) ((x)>(y) ? (x):(y))

// Constant related
#define PI 		M_PI              /* pi */
#define PIT 	M_PI*2.0         /* 2 * pi */
#define PI2 	M_PI*M_PI        /* pi*pi */
#define IPI 	M_1_PI           /* 1/pi */
#define ISPI 	M_2_SQRTPI*0.5  /* 1 / sqrt(pi) */

#endif // Header
