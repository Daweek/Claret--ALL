/*
 * WindowGL.cpp
 *
 *  Created on: Jul 3, 2018
 *      Author: Edg@r j.
 */
#include "WindowGL.hpp"

WindowGL::WindowGL(unsigned int w, unsigned int h,
										Render*& rnd,Crass*& crs, Accel*& gpu) {

	// glfw initialization
	glfwInit();
	glfwWindowHint(GLFW_SAMPLES,4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	m_pWinID = glfwCreateWindow(w, h, "Claret++ V 2.0",NULL,NULL);
	glfwMakeContextCurrent(m_pWinID);

	// Glew Initialization
	glewExperimental = true;
	if (glewInit() != GLEW_OK)
		assert(!"GLEW initialization\n");
	if(!glewIsSupported("GL_EXT_framebuffer_object"))
		assert(!"The GL_EXT_framebuffer_object extension is required.\n");

	// Callbacks for glfw and hints for window
	//glfwSetWindowPos(m_pWinID,12,12);
	glfwSetWindowPos(m_pWinID,100,100);
	glfwSetWindowUserPointer(m_pWinID, this);
	glfwSetKeyCallback(m_pWinID,keyboardCallback);
	glfwSetMouseButtonCallback(m_pWinID,mouseCallback);
	glfwSetCursorPosCallback(m_pWinID,motionCallback);

	//glfwSetInputMode(m_pWinID,GLFW_STICKY_KEYS,GLFW_STICKY_KEYS);
	glfwSetInputMode(m_pWinID,GLFW_CURSOR,GLFW_CURSOR_NORMAL);

	// Construct the render object
	rnd = new Render(w,h,crs->m_side,crs->m_sideh,crs->m_eyelen,crs->m_ditail,gpu);

	// Assign values to object memory
	m_dAngle[0] = m_dAngle[1] = m_dAngle[2] = 0.0;
	m_dTrans[0] = m_dTrans[1] = m_dTrans[2] = 0.0;
	m_bContinue		= true;
	m_bPauseSim		= false;
	m_uiWinWidth  = w;
	m_uiWinHeight =	h;
	m_iMouseButtonL = 0;
	m_iMouseButtonR = 0;
	m_iMouseButtonM = 0;

	//Values for measuring FPS
	m_dTimeCurrent 	= m_dTimeOld = 0.0;
	m_iFpsCount			= 0;

	// Get a handler for ...
	m_pCrs 		= crs;
	m_pRender = rnd;
	m_pAccel	= gpu;
	// Get a pointer for the light position from translation
	crs->m_pTrans = &m_dTrans[2];  // Position in Z

}

void WindowGL::renderScene(Settings cnfg){

	// Measure FPS performance
	m_dTimeCurrent 	= glfwGetTime();
	m_iFpsCount++;

	if(m_dTimeCurrent - m_dTimeOld >= 1.0){
		m_pRender->m_iFps = m_iFpsCount;
		m_iFpsCount = 0;
		m_dTimeOld = m_dTimeCurrent;
	}

	// Prepare camera variables
	m_pRender->camera(m_dAngle,m_dTrans);

	// Restart mouse controls. This is for keeping rotating effect
	m_dAngle[0] = 0;
	if(m_iMouseButtonL == 1 || m_iMouseButtonM == 1 || m_iMouseButtonR == 1){
		m_dAngle[1] = 0;
		m_dAngle[2] = 0;
	}

	// Draw the cube, the cross and the particles
	if(cnfg.fbt == MAIN)	m_pRender->renderToNormal(cnfg, m_pCrs);
	if(cnfg.fbt == FBO)		m_pRender->renderToFBO(cnfg, m_pCrs);

	// Swap the toilet...
	glfwSwapBuffers(m_pWinID);
	glfwPollEvents();
}

auto WindowGL::mouse(int button, int action, int mods, double x, double y) -> void {

  switch (button) {
  case GLFW_MOUSE_BUTTON_LEFT:
    if (action == GLFW_PRESS) {
    	m_iPos[0] = x;
    	m_iPos[1] = y;
    	m_iMouseButtonL = 1;
    }
    if (action == GLFW_RELEASE){
    	m_iMouseButtonL = 0;
    }
    break;
  case GLFW_MOUSE_BUTTON_MIDDLE:
    if (action == GLFW_PRESS) {
    	m_iPos[0] = x;
    	m_iPos[1] = y;
    	m_iMouseButtonM = 1;
    }
    if (action == GLFW_RELEASE) {
    	m_iMouseButtonM = 0;
    }
    break;
  case GLFW_MOUSE_BUTTON_RIGHT:
    if (action == GLFW_PRESS) {
    	m_iPos[0] = x;
    	m_iPos[1] = y;
    	m_iMouseButtonR = 1;
    }
    if (action == GLFW_RELEASE) {
    	m_iMouseButtonR = 0;
    }
    break;
  default:
    break;
  }
}

auto WindowGL::motion(double x, double y) -> void {

	double d0;
  double len = 10;

  len = (29.0641587/2.0)/tan(20.0 * PI / 180.0);

  // MOUSE MIDDLE OR (LEFT AND RIGHT)
  if(m_iMouseButtonM == 1 || (m_iMouseButtonL == 1 && m_iMouseButtonR == 1)){
  	m_dTrans[0] += (double)(x-m_iPos[0])*len*.001;
  	m_dTrans[1] -= (double)(y-m_iPos[1])*len*.001;
  }
  // MOUSE RIGHT
  else if(m_iMouseButtonR == 1){
  	m_dTrans[2] -= (double)(y-m_iPos[1])*len/150;
  	m_dAngle[2] =  (double)(x-m_iPos[0])*0.2;
  }
  // MOUSE LEFT
  else if(m_iMouseButtonL == 1){
    d0 = len/50;
    if(d0 > 1.0) d0 = 1.0;
    m_dAngle[0] = (double)(y-m_iPos[1])*d0;
    m_dAngle[1] = (double)(x-m_iPos[0])*d0;
  }

  if(m_iMouseButtonL == 1 || m_iMouseButtonM == 1 || m_iMouseButtonR == 1){
  	m_iPos[0] = x;
  	m_iPos[1] = y;
  }
}

auto WindowGL::keyboard(int key, int scancode, int action, int mods) -> void {
	// For only press one time....even we hold
	if(action == GLFW_PRESS){
		// Show Keyboard capabilities...
		if(key == GLFW_KEY_SLASH){
			if(mods == GLFW_MOD_SHIFT){
				std::cout<<"\nKeyboard Capabilities:"<<std::endl
								 <<" q,ESC\t  --> Exit"<<std::endl
								 <<" z\t  --> Pause simulation/Force Computation Pause  "<<std::endl
								 <<" !\t  --> Re-start position of particles and Camera "<<std::endl
								 <<" KEY UP\t  --> Increase # of Particle "<<std::endl
								 <<" KEY DOWN --> Decrease # of Particle "<<std::endl
								 <<" t\t  --> Increase (+) 100 degree the temperature  "<<std::endl
								 <<" g\t  --> Decrease (-) 100 degree the temperature  "<<std::endl
								 <<" r\t  --> Increase (+) 10 MD step  "<<std::endl
								 <<" f\t  --> Decrease (-) 10 MD step  "<<std::endl
								 <<" ?\t  --> Print this information "<<std::endl;

			}
		}
		// Quit
		if(key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE){
			m_bContinue = false;
		}
		// Pause simulation...
		if(key == GLFW_KEY_Z){
			m_bPauseSim = !m_bPauseSim;
		}
		// Re-start the camera and particles to initial state
		if(key == GLFW_KEY_1){
			if(mods == GLFW_MOD_SHIFT){
				m_pRender->m_m4SaveRotation = glm::mat4(1);
				m_dTrans[0] = 0;
				m_dTrans[1] = 0;
				m_dTrans[2] = 0;
				*m_pCrs->m_cflg			= 0;
				*m_pCrs->m_cnum			= 0;
				*m_pCrs->m_mclock		= 0;
				m_pCrs->handler_set_cd(0);
				m_pAccel->m_bChangeMalloc = true;
			}
		}
		// Increase the Number of Particles in the system
		if(key == GLFW_KEY_UP){
			m_pRender->m_m4SaveRotation = glm::mat4(1);
			m_dTrans[0] 				= 0;
			m_dTrans[1] 				= 0;
			m_dTrans[2] 				= 0;

			*m_pCrs->m_iNpKey = *m_pCrs->m_iNpKey + 1;
			*m_pCrs->m_npx = *m_pCrs->m_iNpKey;
			*m_pCrs->m_npy = *m_pCrs->m_iNpKey;
			*m_pCrs->m_npz = *m_pCrs->m_iNpKey;

			*m_pCrs->m_cflg			= 0;
			*m_pCrs->m_cnum			= 0;
			*m_pCrs->m_mclock		= 0;
			m_pCrs->handler_set_cd(0);
			m_pAccel->m_bChangeMalloc = true;
		}
		// Decrease the Number of Particles in the system
		if(key == GLFW_KEY_DOWN){
			if(*m_pCrs->m_iNpKey > 1){
				m_pRender->m_m4SaveRotation = glm::mat4(1);
				m_dTrans[0] 				= 0;
				m_dTrans[1] 				= 0;
				m_dTrans[2] 				= 0;

				*m_pCrs->m_iNpKey = *m_pCrs->m_iNpKey - 1;
				*m_pCrs->m_npx = *m_pCrs->m_iNpKey;
				*m_pCrs->m_npy = *m_pCrs->m_iNpKey;
				*m_pCrs->m_npz = *m_pCrs->m_iNpKey;

				*m_pCrs->m_cflg			= 0;
				*m_pCrs->m_cnum			= 0;
				*m_pCrs->m_mclock		= 0;
				m_pCrs->handler_set_cd(0);
				m_pAccel->m_bChangeMalloc = true;
			}
		}
		// Increase the temperature
		if(key == GLFW_KEY_T){
			*m_pCrs->m_temp		+= 100;
			*m_pCrs->m_rtemp 	= *(m_pCrs->m_temp) / *(m_pCrs->m_epsv) * *(m_pCrs->m_kb);
		}
		// Decrease the temperature
		if(key == GLFW_KEY_G){
			if(*m_pCrs->m_temp > 100){
				*m_pCrs->m_temp	-= 100;
				*m_pCrs->m_rtemp = *(m_pCrs->m_temp) / *(m_pCrs->m_epsv) * *(m_pCrs->m_kb);
			}
		}
		// Increase MD_STEP
		if(key == GLFW_KEY_R){
			//*m_pCrs->m_iMdstep+=10;
			*m_pCrs->m_iMdstep = 500;
		}
		// Decrease MD_STEP
		if(key == GLFW_KEY_F){
			if(*m_pCrs->m_iMdstep > 10){
				//*m_pCrs->m_iMdstep -= 10;
			}
			*m_pCrs->m_iMdstep = 100;
		}
		// Modify the triangles per sphere on SPHERE mode.
		if(key == GLFW_KEY_D){
			if(mods == GLFW_MOD_SHIFT){
				if(*m_pCrs->m_ditail < 20){
					 *m_pCrs->m_ditail += 1;
					 m_pAccel->m_bChangeInterop = true;
				}
			}
			else {
				if(*m_pCrs->m_ditail > 5){
					 *m_pCrs->m_ditail -= 1;
					 m_pAccel->m_bChangeInterop = true;
				}
			}
		}
	}

	// When we press and we hold...
	if(action == GLFW_REPEAT){
		// Increase the temperature on press
		if(key == GLFW_KEY_T){
			*(m_pCrs->m_temp)	+= 100;
			*(m_pCrs->m_rtemp) = *(m_pCrs->m_temp) / *(m_pCrs->m_epsv) * *(m_pCrs->m_kb);
		}
		// Decrease the temperature on press
		if(key == GLFW_KEY_G){
			if(*(m_pCrs->m_temp) > 100){
				*(m_pCrs->m_temp)	-= 100;
				*(m_pCrs->m_rtemp) = *(m_pCrs->m_temp) / *(m_pCrs->m_epsv) * *(m_pCrs->m_kb);
			}
		}
		// Increase MD_STEP
		if(key == GLFW_KEY_R){
			//*m_pCrs->m_iMdstep+=10;
		}
		// Decrease MD_STEP
		if(key == GLFW_KEY_F){
			if(*m_pCrs->m_iMdstep > 10){
				//*m_pCrs->m_iMdstep -= 10;
			}
		}

	}


}

WindowGL::~WindowGL() {

	std::cout<<"Closing glfw...\n";
	glfwWindowShouldClose(m_pWinID);
	glfwTerminate();

}

