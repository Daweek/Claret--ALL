/*
 * Render.cpp
 *
 *  Created on: Jul 3, 2018
 *      Author: Edg@r j.
 */
#include "Render.hpp"
// Prepare static coordinates for camera
const GLfloat Render::m_fQuadVertexBufferData[18] = {
				-1.0f, -1.0f, 0.0f,
				 1.0f, -1.0f, 0.0f,
				-1.0f,  1.0f, 0.0f,
				-1.0f,  1.0f, 0.0f,
				 1.0f, -1.0f, 0.0f,
				 1.0f,  1.0f, 0.0f,
};

void CircleTable (float **sint,float **cost,const int n){
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

Render::Render(	unsigned int w, unsigned int h,double*& side, double*& sideh,double*& eye_len,
								unsigned int*& sphereDitail, Accel*& gpu) {

	std::cout<<"Loading shaders and preparing OpenGL buffers...\n";

	// Create Color Attachment for frame buffer
	glGenTextures(1, &m_uiFboTexture);
	glBindTexture(GL_TEXTURE_RECTANGLE, m_uiFboTexture);
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// Create Depth Attachment for frame buffer
	glGenRenderbuffers( 1, &m_uiFboDepth );
	glBindRenderbuffer( GL_RENDERBUFFER, m_uiFboDepth );
	glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
	glBindRenderbuffer( GL_RENDERBUFFER, 0 );

	// Create Frame buffer
	glGenFramebuffers(1, &m_uiFboFramBuff);
	glBindFramebuffer(GL_FRAMEBUFFER, m_uiFboFramBuff);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_uiFboTexture, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_uiFboDepth);

	// Check Frame buffer status
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		assert(!"Framebuffer is incomplete.\n");
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Load simple shaders
	m_uiSimpleProgID 	= LoadShaders("SimpleVert.glsl","SimpleFrag.glsl");
	m_uiSimpleMVPID		= glGetUniformLocation(m_uiSimpleProgID,"MVP");

	// Load Sphere shaders
	m_uiSphereProgID  		= LoadShaders("SphereVert.glsl", "SphereFrag.glsl");
	m_uiSphereMVP					= glGetUniformLocation(m_uiSphereProgID, "MVP");
	m_uiSphereViewMatrix	= glGetUniformLocation(m_uiSphereProgID, "V");
	m_uiSphereModelMatrix	= glGetUniformLocation(m_uiSphereProgID, "M");
	m_uiSphereLightPos		= glGetUniformLocation(m_uiSphereProgID, "LightPosition_worldspace");

	// Load shaders for text
	initText2D("Font.DDS");

	// Prepare shaders for FBO presentation on Texture
	glGenBuffers(1, &m_uiQuadVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_uiQuadVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_fQuadVertexBufferData), m_fQuadVertexBufferData, GL_STATIC_DRAW);

	m_uiTextureProgID = LoadShaders( "TextureFBOVert.glsl", "TextureFBOFrag.glsl" );
	m_uiTextureID 		= glGetUniformLocation(m_uiTextureProgID, "renderedTexture");

	// OpenGL options and hints
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glGenVertexArrays(1,&m_uiVertArrayID);
	glBindVertexArray(m_uiVertArrayID);

	m_eDrawBuffers[0] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, m_eDrawBuffers); // "1" is the size of DrawBuffers

	// Start variables for temporal rendering
	//m_dSide  = 29.0641587;
	//m_dSideh = m_dSide / 2;
	//m_dEyelen	= m_dSideh/tan(20.0 * PI / 180.0);
	m_dSide  	= side;
	m_dSideh 	= sideh;
	m_dEyelen	= eye_len;

	// Start camera variables
	m_v3Eye 					= glm::vec3(0.0,0.0,*m_dEyelen);
	m_v3Dir 					= glm::vec3(0.0,0.0,0.0);
	m_v3Up 						= glm::vec3(0.0,1.0,0.0);

	m_m4Projection		= glm::mat4(1);
	m_m4View 					= glm::mat4(1);
	m_m4Model 				= glm::mat4(1);
	m_m4MVP						=	glm::mat4(1);
	m_m4SaveRotation	= glm::mat4(1);

	// Colors and radio tables
	m_fColor_table[0][0] = 0.7 /2.0;
	m_fColor_table[0][1] = 0.38/2.0;
	m_fColor_table[0][2] = 0.38/2.0;
	m_fColor_table[0][3] = 1;
	m_fColor_table[1][0] = 0.38/2.0;
	m_fColor_table[1][1] = 0.55/2.0;
	m_fColor_table[1][2] = 0.38/2.0;
	m_fColor_table[1][3] = 1;
	m_fColor_table[2][0] = 1;
	m_fColor_table[2][1] = .4;
	m_fColor_table[2][2] = 1;
	m_fColor_table[2][3] = 1;
	m_fColor_table[3][0] = 0;
	m_fColor_table[3][1] = 0.8;
	m_fColor_table[3][2] = 1;
	m_fColor_table[3][3] = 1;
	m_fColor_table[4][0] = 1;
	m_fColor_table[4][1] = 1;
	m_fColor_table[4][2] = 1;
	m_fColor_table[4][3] = 1;

	m_fR_table[0] = 2.443/2;
	m_fR_table[1] = 3.487/2;
	m_fR_table[2] = 3.156/2;
	m_fR_table[3] = .7;
	m_fR_table[4] = .7;

	// Pass information for the main object
	m_uiFboWidth 	= w;
	m_uiFboHeight	= h;
	m_uiSphereDitail = sphereDitail;

	m_dRadius	= 0.45;

	// Holder for GPU object
	m_pGpu = gpu;

	// Timer
	m_iFps = 0;
}

void Render::camera(double ang[3], double trans[3]){

	m_v3Eye 				= glm::vec3(0.0,0.0,*m_dEyelen);
	m_v3Dir 				= glm::vec3(0.0,0.0,0.0);
	m_v3Up 					= glm::vec3(0.0,1.0,0.0);

	m_m4Projection	= glm::mat4(1);
	m_m4View 				= glm::mat4(1);
	m_m4Model 			= glm::mat4(1);
	m_m4MVP					= glm::mat4(1);

	// Projection and View
	m_m4Projection 	= glm::perspective(glm::radians(60.0f),(float)m_uiFboWidth/(float)m_uiFboHeight, 1.0f,10000.0f);
	m_m4View       	= glm::lookAt(m_v3Eye,m_v3Dir,m_v3Up);

	// Rotate
	glm::mat4 Rot(1.0);
	Rot	= glm::rotate(Rot,glm::radians((float)ang[0]),glm::vec3(1.0,0.0,0.0));
	Rot	= glm::rotate(Rot,glm::radians((float)ang[1]),glm::vec3(0.0,1.0,0.0));
	Rot	= glm::rotate(Rot,glm::radians((float)ang[2]),glm::vec3(0.0,0.0,1.0));

	// Rotate counting the previous state
	Rot							*= m_m4SaveRotation;
	// Save the current rotation
	m_m4SaveRotation =  Rot;

	// Translate the Model
	glm::mat4 Trans(1.0);
	Trans				= glm::translate(Trans,glm::vec3(trans[0],trans[1],trans[2]));

	// Compute the actual Final Model
	m_m4Model		= Trans * Rot;
	m_m4MVP 		= m_m4Projection * m_m4View  * m_m4Model;

}

void Render::drawInfo(Settings cnfg, Crass* crs){
	// Print some Text
	std::string buf;
	std::stringstream st1,st2,st3;
	int x,y,size;

	size	= 28;
	x 		= 10;
	y			= 570;

	st1<<std::fixed<<std::setprecision(0)<<*crs->m_temp;
	st2<<std::fixed<<std::setprecision(0)<<*crs->m_iNp;
	st3<<std::fixed<<std::setprecision(0)<<*crs->m_iNpKey;
	buf = "Temp=" + st1.str() + "\tNkey=" + st3.str() + "\tN="+st2.str();
	printText2D(buf.c_str(),x,y,size);

	y		-= size;
	buf = "Hardware=" + cHardwareType[cnfg.hdt];
	printText2D(buf.c_str(),x,y,size);

	if(cnfg.hdt == GPU){
		y		-= size;
		buf = "CUDA-GL Interop: "+ cInterOpCUDAgl[cnfg.igt];
		printText2D(buf.c_str(),x,y,size);

		y		-= size;
		buf = "CUDA Kernel Type: "+ cKernelType[cnfg.krt];
		printText2D(buf.c_str(),x,y,size);
	}

	y		-= size*2;
	buf = "md_step: "+ std::to_string(*crs->m_iMdstep);
	printText2D(buf.c_str(),x,y,size);

	if(cnfg.rpt == SPHERE){
		y		-= size;
		buf = "sphere_ditail: "+ std::to_string(*crs->m_ditail);
		printText2D(buf.c_str(),x,y,size);
	}

	y		-= size*12;
	buf = "FPS: "+ std::to_string(m_iFps);
	printText2D(buf.c_str(),x,y,size);

	y		-= size;
	buf = "Sec/Step: " + std::to_string(m_pGpu->m_fStepsec);
	printText2D(buf.c_str(),x,y,size);

	y		-= size;
	buf = "Gflops: " + std::to_string(m_pGpu->m_fFlops);
	printText2D(buf.c_str(),x,y,size);

}

void Render::drawALL(Settings cnfg, Crass* crs){
	// Put matrices to shader
	glUseProgram(m_uiSimpleProgID);
	glUniformMatrix4fv(m_uiSimpleMVPID, 1, GL_FALSE, &m_m4MVP[0][0]);

	// Draw the cube
	drawHako(*m_dSideh,0.5);

	// Draw the Cross
	drawCross(*m_dSideh/2.0,0.5);

	if(cnfg.rpt == POINTS){
			drawParticlePoints(*crs->m_iNp*3,crs->m_drow_flg,crs->m_atype_mat,crs->m_atype,
										 crs->m_cd,crs->m_vl,*m_dSideh,m_fColor_table,crs->m_drow_flg,cnfg);
	}

	else if (cnfg.rpt == SPHERE){
		glUseProgram(m_uiSphereProgID);
		glUniformMatrix4fv(m_uiSphereMVP				, 1, GL_FALSE, &m_m4MVP[0][0]);
		glUniformMatrix4fv(m_uiSphereViewMatrix	, 1, GL_FALSE, &m_m4View[0][0]);
		glUniformMatrix4fv(m_uiSphereModelMatrix, 1, GL_FALSE, &m_m4Model[0][0]);
		glm::vec3 lightPos = glm::vec3(0,0,*m_dEyelen+*(crs->m_pTrans)+225); //+200 is the intensity
		glUniform3f(m_uiSphereLightPos, lightPos.x, lightPos.y, lightPos.z);

		drawParticleSphere(*crs->m_iNp*3,crs->m_drow_flg,crs->m_atype_mat,crs->m_atype,
											 crs->m_cd,crs->m_vl,*m_dSideh,m_fColor_table,crs->m_drow_flg,
											 m_fR_table,m_dRadius,*m_uiSphereDitail,cnfg);
	}
	// Render Info
	drawInfo(cnfg,crs);
}

void Render::renderToNormal(Settings cnfg,Crass* crs){

	// Prepare first window clearing color and enabling OpenGL states/behavior
	glViewport(0, 0, m_uiFboWidth, m_uiFboHeight);
	glClearColor(0.0,0.0,0.0,0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	drawALL(cnfg,crs);
}
void Render::renderToFBO(Settings cnfg,Crass* crs){

	glBindFramebuffer(GL_FRAMEBUFFER, m_uiFboFramBuff);
		// Prepare first window clearing color and enabling OpenGL states/behavior
		glViewport(0, 0, m_uiFboWidth, m_uiFboHeight);
		glClearColor(0.2,0.2,0.2,0.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		drawALL(cnfg,crs);

	glBindFramebuffer(GL_FRAMEBUFFER,0);
////////////////////////////////
	// Draw to a Texture rectangle
	glClearColor(0.0,0.0,0.0,0.0);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0,0,m_uiFboWidth,m_uiFboHeight);

	// Use our shader
	glUseProgram(m_uiTextureProgID);

	// Bind our texture in Texture Unit 0
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE, m_uiFboTexture);
	// Set our "renderedTexture" sampler to user Texture Unit 0
	glUniform1i(m_uiTextureID, 0);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, m_uiQuadVertexBuffer);
	glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);

	// Draw the triangles !
	glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles
}

void Render::drawParticlePoints(int n3,int* drow_flag,int* atype_mat,int* atype,
																double* cd,double* vl,double sideh,float color_table[10][4],
																int* drow_flg, Settings cnfg){
	if(cnfg.igt == NO){
		GLuint buf[2];
		glGenBuffers(2, buf);

		int 		n = n3/3;
		int 		q = 0;
		double 	d0;

		float *f_pointA,*f_clr;
		unsigned int size 			= (n*3)*sizeof(float);
		unsigned int size_color = (n*4)*sizeof(float);

		f_pointA    = (float*)malloc(n*3*sizeof(float));
		f_clr	  		= (float*)malloc(n*4*sizeof(float));

		for(int i=0; i<n3;i+=3){
			if(drow_flg[atype_mat[atype[i/3]]] == 1){
				// Compute coordinates
				f_pointA[i] 	= cd[i]-sideh;
				f_pointA[i+1]	= cd[i+1]-sideh;
				f_pointA[i+2]	= cd[i+2]-sideh;
				// Compute Color
				d0 = (vl[i]*vl[i]+vl[i+1]*vl[i+1]+vl[i+2]*vl[i+2])*500;
				*(f_clr+0+q) = color_table[atype_mat[atype[i/3]]][0]+d0;
				*(f_clr+1+q) = color_table[atype_mat[atype[i/3]]][1]+d0/3;
				*(f_clr+2+q) = color_table[atype_mat[atype[i/3]]][2]+d0/3;
				*(f_clr+3+q) = color_table[atype_mat[atype[i/3]]][3];
				q+=4;
				}
		}

		glBindBuffer(GL_ARRAY_BUFFER, buf[0]);
		glBufferData(GL_ARRAY_BUFFER,size,f_pointA, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, buf[1]);
		glBufferData(GL_ARRAY_BUFFER,size_color,f_clr, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,0,(void*)0);

		glPointSize(4.0);
		glDrawArrays(GL_POINTS,0,n);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDeleteBuffers(2, buf);

		free(f_pointA);
		free(f_clr);
	}

#ifdef INTEROP
	if(cnfg.igt == HALF){
		GLuint buf[2];
		glGenBuffers(2, buf);

		int 		n = n3/3;
		unsigned int size 			= (n*3)*sizeof(float);
		unsigned int size_color = (n*4)*sizeof(float);

		glBindBuffer(GL_ARRAY_BUFFER, buf[0]);
		glBufferData(GL_ARRAY_BUFFER,size, m_pGpu->m_fPossVBO, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, buf[1]);
		glBufferData(GL_ARRAY_BUFFER,size_color,m_pGpu->m_fColorVBO, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,0,(void*)0);

		glDrawArrays(GL_POINTS,0,n);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDeleteBuffers(2, buf);
	}

	if(cnfg.igt == YES){
		int n = n3/3;
		///////////////////////////
		glBindBuffer(GL_ARRAY_BUFFER, m_pGpu->g_possVBO);
		glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, m_pGpu->g_colorVBO);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,0,(void*)0);

		glDrawArrays(GL_POINTS,0,n);

		///////////////////////////////
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}
#endif
}

void Render::drawParticleSphere(int n3,int* drow_flag,int* atype_mat,int* atype,
																double* cd,double* vl,double msideh,float color_table[10][4],
																int* drow_flg,float r_table[5],double radius,int ditail,
																Settings cnfg){

	int	n = n3/3;
	unsigned int vert_body 			= (((ditail/2)-2)*ditail*3*2); 	//ONLY VERTEX OF THE STRUCTURE
	unsigned int vert_hats			= ditail*3*2;									 	//ONLY VERTEX OF THE STRUCTURE
	unsigned int poly_size			= vert_hats+vert_body;					//Number of Vertex in one sphere
	unsigned int polysize_vert 	= 3*(vert_hats+vert_body);			//Times 3 for X,Y,Z per Vertex
	unsigned int polysize_color	=	4*(vert_hats+vert_body); 			//Times 4 for R,G,B,A per Vertex

	if(cnfg.igt == NO){
		double 	sideh[3] 	= {msideh,msideh,msideh};
		GLuint buf[3];
		glGenBuffers(3, buf);

		int n = n3/3;
		int q,m,l=3,k,p,t=0,r=0,slices = ditail,stacks = ditail/2,j;
		float z0,z1,r0,r1,radios;
		float d0;

		float *f_pol,*f_pol_n,*f_clr_a;
		float *sint1,*cost1;
		float *sint2,*cost2;

		unsigned int size 			= n*polysize_vert*sizeof(float);
		unsigned int size_color = n*polysize_color*sizeof(float);

		f_pol    = (float*)malloc(n*polysize_vert*sizeof(float));
		f_pol_n  = (float*)malloc(n*polysize_vert*sizeof(float));
		f_clr_a	 = (float*)malloc(n*polysize_color*sizeof(float));

		/* Pre-computed circle */
		CircleTable(&sint1,&cost1,-slices);
		CircleTable(&sint2,&cost2,stacks*2);
	//////Mapping The circle////////////////////////////////////////////////////
		for(int i=0; i<n3;i+=3){
			if(drow_flg[atype_mat[atype[i/3]]] == 1){
			 /////////////////////Compute Color
				for(q=0;q<poly_size;q++){
					d0 = (vl[i]*vl[i]+vl[i+1]*vl[i+1]+vl[i+2]*vl[i+2])*500;
					*(f_clr_a+0+r+q*4) = color_table[atype_mat[atype[i/3]]][0]+d0;
					*(f_clr_a+1+r+q*4) = color_table[atype_mat[atype[i/3]]][1]+d0/3;
					*(f_clr_a+2+r+q*4) = color_table[atype_mat[atype[i/3]]][2]+d0/3;
					*(f_clr_a+3+r+q*4) = color_table[atype_mat[atype[i/3]]][3];
				}

				radios=radius*r_table[atype_mat[atype[i/3]]];
				///////Compute one Circle/////////////////////////
				z0 = 1.0f;
				z1 = cost2[(stacks>0)?1:0];
				r0 = 0.0f;
				r1 = sint2[(stacks>0)?1:0];

				for(m=0;m<slices;m++){
					*(f_pol+t+(9*m))     =cd[i]-sideh[0];
					*(f_pol+t+1+(9*m))   =cd[i+1]-sideh[1];
					*(f_pol+t+2+(9*m))   =cd[i+2]-sideh[2]+radios;
					*(f_pol_n+t+(9*m))    =0;
					*(f_pol_n+t+1+(9*m))  =0;
					*(f_pol_n+t+2+(9*m))  =1;
				}
				l=3;

				for (j=slices; j>0; j--){
					*(f_pol+t+l)     =(cd[i]-sideh[0])+(cost1[j]*r1*radios);
					*(f_pol+t+l+1)   =(cd[i+1]-sideh[1])+(sint1[j]*r1*radios);
					*(f_pol+t+l+2)   =(cd[i+2]-sideh[2])+ (z1*radios);
					*(f_pol_n+t+l)   =cost1[j]*r1;
					*(f_pol_n+t+l+1) =sint1[j]*r1;
					*(f_pol_n+t+l+2) =z1;

					*(f_pol+t+l+3)   =(cd[i]-sideh[0])  + (cost1[j-1]*r1*radios);
					*(f_pol+t+l+4)   =(cd[i+1]-sideh[1])+ (sint1[j-1]*r1*radios);
					*(f_pol+t+l+5)   =(cd[i+2]-sideh[2])+ (z1*radios);
					*(f_pol_n+t+l+3) =cost1[j-1]*r1;
					*(f_pol_n+t+l+4) =sint1[j-1]*r1;
					*(f_pol_n+t+l+5) =z1;
					l+=9;
				}

				l-=3;
				for( k=1; k<stacks-1; k++ ){
					z0 = z1; z1 = cost2[k+1];
					r0 = r1; r1 = sint2[k+1];
					p=0;
					for(j=0; j<slices; j++){
						//////////////////First Triangle////////////////////////////////
						*(f_pol+t+l+p)     = (cd[i]-sideh[0] ) +(cost1[j]*r1*radios);
						*(f_pol+t+l+p+1)   = (cd[i+1]-sideh[1])+(sint1[j]*r1*radios);
						*(f_pol+t+l+p+2)   = (cd[i+2]-sideh[2])+(z1*radios);
						*(f_pol_n+t+l+p)   = cost1[j]*r1;
						*(f_pol_n+t+l+p+1) = sint1[j]*r1;
						*(f_pol_n+t+l+p+2) = z1;

						*(f_pol+t+l+p+3)   = (cd[i]-sideh[0])  +(cost1[j]*r0*radios);
						*(f_pol+t+l+p+4)   = (cd[i+1]-sideh[1])+(sint1[j]*r0*radios);
						*(f_pol+t+l+p+5)   = (cd[i+2]-sideh[2])+(z0*radios);
						*(f_pol_n+t+l+p+3) = cost1[j]*r0;
						*(f_pol_n+t+l+p+4) = sint1[j]*r0;
						*(f_pol_n+t+l+p+5) = z0;

						*(f_pol+t+l+p+6)   = (cd[i]-sideh[0])  +(cost1[j+1]*r1*radios);
						*(f_pol+t+l+p+7)   = (cd[i+1]-sideh[1])+(sint1[j+1]*r1*radios);
						*(f_pol+t+l+p+8)   = (cd[i+2]-sideh[2])+(z1*radios);
						*(f_pol_n+t+l+p+6) = cost1[j+1]*r1;
						*(f_pol_n+t+l+p+7) = sint1[j+1]*r1;
						*(f_pol_n+t+l+p+8) = z1;
						/////////////////////Second Triangle//////////////////////////////
						*(f_pol+t+l+p+9)   = *(f_pol+t+l+p+6);
						*(f_pol+t+l+p+10)  = *(f_pol+t+l+p+7);
						*(f_pol+t+l+p+11)  = *(f_pol+t+l+p+8);
						*(f_pol_n+t+l+p+9) = *(f_pol_n+t+l+p+6);
						*(f_pol_n+t+l+p+10)= *(f_pol_n+t+l+p+7);
						*(f_pol_n+t+l+p+11)= *(f_pol_n+t+l+p+8);

						*(f_pol+t+l+p+12)   = *(f_pol+t+l+p+3);
						*(f_pol+t+l+p+13)   = *(f_pol+t+l+p+4);
						*(f_pol+t+l+p+14)   = *(f_pol+t+l+p+5);
						*(f_pol_n+t+l+p+12) = *(f_pol_n+t+l+p+3);
						*(f_pol_n+t+l+p+13) = *(f_pol_n+t+l+p+4);
						*(f_pol_n+t+l+p+14) = *(f_pol_n+t+l+p+5);

						*(f_pol+t+l+p+15)  =(cd[i]-sideh[0] ) +(cost1[j+1]*r0*radios);
						*(f_pol+t+l+p+16)  =(cd[i+1]-sideh[1])+(sint1[j+1]*r0*radios);
						*(f_pol+t+l+p+17)  =(cd[i+2]-sideh[2])+(z0*radios);
						*(f_pol_n+t+l+p+15)=cost1[j+1]*r0;
						*(f_pol_n+t+l+p+16)=sint1[j+1]*r0;
						*(f_pol_n+t+l+p+17)=z0;

						p+=18;
					}
					l+=(slices)*6*3;
				}
				z0 = z1;
				r0 = r1;
				for(m=0;m<slices;m++){
					*(f_pol+t+l+(9*m))     = cd[i]-sideh[0];
					*(f_pol+t+l+1+(9*m))   = cd[i+1]-sideh[1];
					*(f_pol+t+l+2+(9*m))   = cd[i+2]-sideh[2]-radios;
					*(f_pol_n+t+l+(9*m))   = 0;
					*(f_pol_n+t+l+1+(9*m)) = 0;
					*(f_pol_n+t+l+2+(9*m)) = -1;
				}
				p=3;
				for (j=0; j<slices; j++){
					*(f_pol+t+l+p)     = (cd[i]-sideh[0])  + (cost1[j]*r0*radios);
					*(f_pol+t+l+p+1)   = (cd[i+1]-sideh[1])+ (sint1[j]*r0*radios);
					*(f_pol+t+l+p+2)   = (cd[i+2]-sideh[2])+ (z0*radios);
					*(f_pol_n+t+l+p)   = cost1[j]*r0;
					*(f_pol_n+t+l+p+1) = sint1[j]*r0;
					*(f_pol_n+t+l+p+2) = z0;

					*(f_pol+t+l+p+3)   = (cd[i]-sideh[0] ) +(cost1[j+1]*r0*radios);
					*(f_pol+t+l+p+4)   = (cd[i+1]-sideh[1])+(sint1[j+1]*r0*radios);
					*(f_pol+t+l+p+5)   = (cd[i+2]-sideh[2])+(z0*radios);
					*(f_pol_n+t+l+p+3) = cost1[j+1]*r0;
					*(f_pol_n+t+l+p+4) = sint1[j+1]*r0;
					*(f_pol_n+t+l+p+5) = z0;
					p+=9;
				}
			}
			t+=polysize_vert;
			r+=polysize_color;
		}

		glBindBuffer(GL_ARRAY_BUFFER, buf[0]);
		glBufferData(GL_ARRAY_BUFFER,size,f_pol, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, buf[1]);
		glBufferData(GL_ARRAY_BUFFER,size,f_pol_n, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,(void*)0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, buf[2]);
		glBufferData(GL_ARRAY_BUFFER,size_color,f_clr_a, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE,0,(void*)0);
		glEnableVertexAttribArray(2);

		glDrawArrays(GL_TRIANGLES,0,poly_size*n);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
		glDeleteBuffers(3, buf);

		free(f_clr_a);
		free(f_pol_n);
		free(f_pol);
		free(sint1);
		free(cost1);
		free(sint2);
		free(cost2);
	}

	if (cnfg.igt == HALF){
		GLuint buf[3];
		glGenBuffers(3, buf);

		unsigned int size_vert	= n*polysize_vert*sizeof(float);
		unsigned int size_color = n*polysize_color*sizeof(float);

		glBindBuffer(GL_ARRAY_BUFFER, buf[0]);
		glBufferData(GL_ARRAY_BUFFER,size_vert, m_pGpu->m_fPolyVert, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, buf[1]);
		glBufferData(GL_ARRAY_BUFFER,size_vert,m_pGpu->m_fPolyNor, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,(void*)0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, buf[2]);
		glBufferData(GL_ARRAY_BUFFER,size_color,m_pGpu->m_fPolyColor, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE,0,(void*)0);
		glEnableVertexAttribArray(2);

		glDrawArrays(GL_TRIANGLES,0,n*poly_size);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
		glDeleteBuffers(3, buf);
	}

	if(cnfg.igt == YES){

		glBindBuffer(GL_ARRAY_BUFFER, m_pGpu->g_polyVertVBO);
		glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, m_pGpu->g_polyNorVBO);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,(void*)0);

		glBindBuffer(GL_ARRAY_BUFFER, m_pGpu->g_polyColorVBO);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE,0,(void*)0);

		//////////draw //////////////////////////////
		glDrawArrays(GL_TRIANGLES,0,n*poly_size);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
	}
}

void Render::drawCross(double d0, double d1){

	float fd0 = (float) d0;
	float fd1 = (float) d1;

	GLfloat cross_vertex[] = {
					0,	  0,	-fd0,
					0,	  0,	 fd0,
					0,	-fd0,		0,
					0,	 fd0,		0,
				-fd0,		0,		0,
				 fd0,		0,		0
	};

	GLfloat cross_color[] = {
				fd1,	  fd1,		fd1,
				fd1,	  fd1,		fd1,
				fd1,		fd1,		fd1,
				fd1,	 	fd1,		fd1,
				fd1,		fd1,		fd1,
				fd1,		fd1,		fd1
	};

	/////////////Draw the cross
	GLuint buf2[2];
	glGenBuffers(2, buf2);

	glBindBuffer(GL_ARRAY_BUFFER, buf2[0]);
	glBufferData(GL_ARRAY_BUFFER,sizeof(cross_vertex),cross_vertex, GL_STATIC_DRAW);
	glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, buf2[1]);
	glBufferData(GL_ARRAY_BUFFER,sizeof(cross_color),cross_color, GL_STATIC_DRAW);
	glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,(void*)0);
	glEnableVertexAttribArray(1);

	glLineWidth(2.0);
	glDrawArrays(GL_LINES,0,6);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDeleteBuffers(2, buf2);

}

void Render::drawHako(double d0, double d1){

	float fd0 = (float) d0;
	float fd1 = (float) d1;

	GLfloat hako_vertex[] = {
				-fd0,fd0,-fd0,
				 fd0,fd0,-fd0,
				 fd0,fd0, fd0,
				-fd0,fd0, fd0,

				-fd0,-fd0,-fd0,
				 fd0,-fd0,-fd0,
				 fd0,-fd0, fd0,
				-fd0,-fd0, fd0,

	};

	GLfloat hako_color[]={
				fd1,	  fd1,		fd1,
				fd1,	  fd1,		fd1,
				fd1,		fd1,		fd1,
				fd1,	 	fd1,		fd1,
				fd1,		fd1,		fd1,
				fd1,		fd1,		fd1,
				fd1,		fd1,		fd1,
				fd1,		fd1,		fd1
	};

	GLshort hako_index[] = {
				0,1,
				1,2,
				2,3,
				3,0,

				4,5,
				5,6,
				6,7,
				7,4,

				1,5,
				2,6,
				3,7,
				0,4
	};

	GLuint buf3[3];
	glGenBuffers(3,buf3);

	glBindBuffer(GL_ARRAY_BUFFER, buf3[0]);
	glBufferData(GL_ARRAY_BUFFER,sizeof(hako_vertex),hako_vertex, GL_STATIC_DRAW);
	glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, buf3[1]);
	glBufferData(GL_ARRAY_BUFFER,sizeof(hako_color),hako_color, GL_STATIC_DRAW);
	glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,(void*)0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,buf3[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(hako_index),hako_index,GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buf3[2]);

	glLineWidth(2.0);
	glDrawElements(GL_LINES,24,GL_UNSIGNED_SHORT,(void*)0);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	glDeleteBuffers(3, buf3);

}

Render::~Render() {
	std::cout<<"Deleting OpenGL Buffers...\n";
	glDeleteTextures(1, &m_uiFboTexture);
	glDeleteTextures(1, &m_uiFboDepth);
	glDeleteFramebuffers(1, &m_uiFboFramBuff);
	glDeleteVertexArrays(0,&m_uiVertArrayID);

	std::cout<<"Cleaning Text2D...\n";
	cleanupText2D();
}

