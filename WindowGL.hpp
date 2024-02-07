/*
 * WindowGL.h
 *
 *  Created on: Jul 3, 2018
 *      Author: Edg@r j.
 */
#ifndef WINDOWGL_H_
#define WINDOWGL_H_
#pragma once

#include <iostream>
#include <assert.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "cras36def.h"
#include "Crass.h"
#include "Render.hpp"
#include "Accel.hpp"

class WindowGL {

	private:
		GLFWwindow* 			m_pWinID;
		Render*						m_pRender;
		Crass*						m_pCrs;
		Accel* 						m_pAccel;

		unsigned int			m_uiWinWidth;
		unsigned int			m_uiWinHeight;
		bool							m_bContinue;
		bool							m_bPauseSim;

		int								m_iPos[2];
		int								m_iMouseButtonL;
		int								m_iMouseButtonR;
		int								m_iMouseButtonM;

		double						m_dAngle[3];
		double						m_dTrans[3];

		inline static auto keyboardCallback(
			GLFWwindow *win,
			int key,
			int scancode,
			int action,
			int mods) -> void {
			WindowGL *window = static_cast<WindowGL*>(glfwGetWindowUserPointer(win));
			window->keyboard(key, scancode, action, mods);
		}

		inline static auto mouseCallback(
			GLFWwindow *win,
			int button,
			int action,
			int mods) -> void {
			WindowGL *window = static_cast<WindowGL*>(glfwGetWindowUserPointer(win));
			double x,y;
			glfwGetCursorPos(win,&x,&y);
			window->mouse(button, action, mods,x,y);
		}

		inline static auto motionCallback(
			GLFWwindow *win,
			double x,
			double y) -> void {
			WindowGL *window = static_cast<WindowGL*>(glfwGetWindowUserPointer(win));
			window->motion(x,y);
		}

	public:

		double						m_dTimeOld;
		double						m_dTimeCurrent;
		int								m_iFpsCount;

		WindowGL(unsigned int w, unsigned int h,Render*& rnd, Crass*& crs,
						 Accel*& gpu);
		void renderScene(Settings cnfg);
		auto mouse(int button, int action, int mods, double x, double y) -> void;
		auto motion(double x, double y) -> void;
		auto keyboard(int key, int scancode, int action, int mods) -> void;
		inline bool continueRender(){return m_bContinue;};
		inline bool pauseSimulation(){return m_bPauseSim;};
		~WindowGL();
};

#endif /* WINDOWGL_H_ */
