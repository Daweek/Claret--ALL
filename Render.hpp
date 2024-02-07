/*
 * Render.h
 *
 *  Created on: Jul 3, 2018
 *      Author: Edg@r j.
 */
#ifndef RENDER_H_
#define RENDER_H_
#pragma once

#include <iostream>
#include <assert.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

#include <GL/glew.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "cras36def.h"
#include "Crass.h"
#include "Accel.hpp"

#include "shader.hpp"
#include "texture.hpp"
#include "text2D.hpp"

const std::string cFrameBufferType[] = {"MAIN","FBO"};
const std::string cRenderParticleType[] = {"POINTS","SPHERE","TEXTURE"};

class Render{

	private:
		Accel*				m_pGpu;

		unsigned int	m_uiFboWidth;
		unsigned int	m_uiFboHeight;

		GLuint 				m_uiSimpleProgID;
		GLuint 				m_uiSimpleMVPID;

		GLuint				m_uiSphereProgID;
		GLuint				m_uiSphereMVP;
		GLuint				m_uiSphereViewMatrix;
		GLuint				m_uiSphereModelMatrix;
		GLuint				m_uiSphereLightPos;

		GLuint				m_uiTextureProgID;
		GLuint				m_uiTextureID;
		GLuint				m_uiQuadVertexBuffer;

		static const
		GLfloat  			m_fQuadVertexBufferData[18];

		GLuint				m_uiFboTexture;
		GLuint				m_uiFboDepth;
		GLuint				m_uiFboFramBuff;

		GLuint				m_uiVertArrayID;
		GLenum				m_eDrawBuffers[1];

		glm::vec3 		m_v3Eye;
		glm::vec3 		m_v3Dir;
		glm::vec3 		m_v3Up;
		glm::mat4 		m_m4Projection;
		glm::mat4 		m_m4View;
		glm::mat4 		m_m4Model;
		glm::mat4 		m_m4MVP;

		float					m_fColor_table[10][4];
		float					m_fR_table[5];

		double				m_dRadius;
		double*				m_dSide;
		double*				m_dSideh;
		double*				m_dEyelen;

	public:
		glm::mat4			m_m4SaveRotation;
		int						m_iFps;
		unsigned int*	m_uiSphereDitail;

		void camera(double [],double []);

		void renderToNormal(Settings cnfg, Crass* crs);
		void renderToFBO(		Settings cnfg, Crass* crs);
		void drawALL(				Settings cnfg,Crass* crs);
		void drawInfo(			Settings cnfg,Crass* crs);
		void drawHako (double d0, double d1);
		void drawCross(double d0, double d1);

		void drawParticlePoints(int n3,int* drow_flag,int* atype_mat,int* atype,
														double* cd,double* vl,double sideh,
														float color_table[10][4], int* drow_flg,Settings cnfg);

		void drawParticleSphere(int n3,int* drow_flag,int* atype_mat,int* atype,
														double* cd,double* vl,double msideh,
														float color_table[10][4], int* drow_flg, float r_table[5],
														double radius,int ditail,Settings cnfg);

		Render(unsigned int w, unsigned int h, double*& side, double*& sideh, double*& eye_len,
					 unsigned int*& sphereDitail, Accel*& gpu);
		~Render();
};

#endif /* RENDER_H_ */
