/*
 * Main.cpp
 *
 *  Created on: Jul 3, 2018
 *      Author: Edg@r j.
 */
#include "cras36def.h"
#include "Crass.h"
#include "Accel.hpp"
#include "Render.hpp"
#include "WindowGL.hpp"
//////////////////////////////////////// Globals
// Window
unsigned int 				g_WinWidth  = 512*2;
unsigned int 				g_WinHeight = 512*2;
// Main Objects
Accel*							g_oGPU;
WindowGL*						g_oWindow;
Crass*							g_oCrass;
Render*							g_oRender;
// General options for Hardware & Render
Settings						g_sConfig;

int main(int argc, char **argv)
{
	// TODO catch main arguments
	std::cout<<"Init Claret++\n";

	// Main Variables for object options
	g_sConfig.fbt	= MAIN;			// MAIN,FBO

	g_sConfig.rpt = SPHERE;  	// POINTS,SPHERE
	g_sConfig.hdt	= GPU;			// CPU_1,CPU_OMP,GPU
	g_sConfig.igt = YES;   	// INTEROPERABILITY YES,NO,HALF

	g_sConfig.krt = NORMAL;		// NORMAL,DYNAMIC --> for DYNAMIC remember -rdc=true for compiling

	if (g_sConfig.hdt != GPU && g_sConfig.igt == YES){
		std::cerr<<"CUDA-OpenGL Inter-operability not a viable with CPU accelerator mode\n";
		g_sConfig.igt = NO;
		//assert(!"CUDA-OpenGL Inter-operability not a viable with CPU accelerator mode\n");
	}

	unsigned int 	npkey 			= 12; 	// Number of particles from 1 - 12 keyboard
	unsigned int 	mdstep			= 10; // Inner loop which calls to compute force
	unsigned int	ditail			= 20; // In case Rendering with Polygons (Spheres) Number of slides
	double				temperature = 300;// Obvious...

	// Main objects
	g_oGPU		= new Accel();
	g_oCrass	= new Crass(npkey,mdstep,temperature,ditail);
	g_oWindow = new	WindowGL(g_WinWidth,g_WinHeight,g_oRender,g_oCrass,g_oGPU);

	// Print info to console
	std::cout<<"Options: \n  npkey\t\t\t = "<<*g_oCrass->m_iNpKey<<std::endl<<
												"  np\t\t\t = "<<*g_oCrass->m_iNp<<std::endl<<
												"  mdstep\t\t = "<<mdstep<<std::endl<<
												"  hardware\t\t = "<<cHardwareType[g_sConfig.hdt]<<std::endl<<
												"  rendermode\t\t = "<<cRenderParticleType[g_sConfig.rpt]<<std::endl<<
												"  framebuffer\t\t = "<<cFrameBufferType[g_sConfig.fbt]<<std::endl;

	if(g_sConfig.hdt == GPU)std::cout<<
												"  CUDA-OpenGL interop\t = "<<cInterOpCUDAgl[g_sConfig.igt]<<std::endl<<
												"  kerneltype\t\t = "<<cKernelType[g_sConfig.krt]<<std::endl;

	// Prepare for FPS measure
	g_oWindow->m_dTimeOld = glfwGetTime();

	while (g_oWindow->continueRender()){

		if(g_oWindow->pauseSimulation() == false)
				g_oCrass->md_run(g_sConfig,g_oGPU);

		g_oWindow->renderScene(g_sConfig);

	}

	std::cout<<"\nCleaning objects...\n";
	delete g_oGPU;
	delete g_oCrass;
	delete g_oRender;
	delete g_oWindow;

	return 0;
}
