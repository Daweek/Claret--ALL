################################################################################
#Makefile to Generate Clare++      													 	Edg@r J 2018
################################################################################
#Compilers
GCC				= gcc
CXX 			= g++

CUDA 			= /usr/local/cuda
CUDA_SDK	= $(CUDA)/samples
NVCC     	= $(CUDA)/bin/nvcc

#Include Paths
CUDAINC   = -I$(CUDA)/include -I$(CUDA_SDK)/common/inc
INC				= -I.
#Library Paths
CUDALIB		= -L/usr/lib/x86_64-linux-gnu -L$(CUDA)/lib64 \
						-lcuda -lcudart -lcudadevrt
GLLIB  		= -lGL -lGLU -lGLEW -lglfw
LIB 			= $(CUDALIB) $(GLLIB) -lm

################ Choosing architecture code for GPU ############################
NVCC_ARCH			=
HOSTNAME		 	= $(shell uname -n)

ifeq ("$(HOSTNAME)","edgar-msi")
	NVCC_ARCH		= -gencode arch=compute_61,code=sm_61
endif

ifeq ("$(HOSTNAME)","narumiken-LAP")
	NVCC_ARCH		= -gencode arch=compute_52,code=sm_52
endif

ifeq ("$(HOSTNAME)","edgar-PC")
	NVCC_ARCH		= -gencode arch=compute_61,code=sm_61
endif

ifeq ("$(HOSTNAME)","edgar-PC2")
	NVCC_ARCH		= -gencode arch=compute_61,code=sm_61
endif

###############	Debug, 0 -> False,  1-> True
DEBUGON						:= 0
ifeq (1,$(DEBUGON))
	CXXDEBUG 				:= -g -Wall
	CXXOPT					:= -std=c++11 -O0
	NVCCDEBUG				:= -g -G
#	NVCCDEBUG				:= 
	NVCCOPT					:= -std=c++11 -O0
	NVCCFLAGSXCOMP 	:= -Xcompiler -g,-Wall,-O0,-fopenmp	
else
	CXXDEBUG 				:= 
	CXXOPT					:= -std=c++11 -O3 -ffast-math -funroll-loops
	NVCCDEBUG				:= 
	NVCCOPT					:= -std=c++11 -O3 --cudart=shared -use_fast_math
	NVCCFLAGSXCOMP 	:= -Xcompiler -O3,-ffast-math,-funroll-loops,-fopenmp
endif
###############################################################################
#NVCC_DP					= -rdc=true
CXXFLAGS				= $(CXXDEBUG) $(CXXOPT) -fopenmp
NVCCFLAGS 			= $(NVCCDEBUG) $(NVCC_DP) --compile $(NVCCOPT) $(NVCC_ARCH)
NVCCFLAGSLINK		= $(NVCCDEBUG) $(NVCC_DP) $(NVCCOPT) $(NVCC_ARCH)
###############################################################################

TARGET = claret++

all: $(TARGET)

OBJLIST = Main.o Crass.o WindowGL.o Render.o Accel.o shader.o text2D.o texture.o
				
claret++ : $(OBJLIST)
	$(NVCC) $(NVCCFLAGSLINK) $(NVCCFLAGSXCOMP) $(CUDAINC) -o $@ $(OBJLIST) $(LIB) 

Main.o: Main.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(CUDAINC) $(CUDAINC) -c $< -o $@ 
	
Crass.o: Crass.cpp 
	$(CXX) $(CXXFLAGS) $(INC) $(CUDAINC) -c $< -o $@		
	
WindowGL.o: WindowGL.cpp 
	$(CXX) $(CXXFLAGS) $(INC) $(CUDAINC) -c $< -o $@	

Render.o: Render.cpp 
	$(CXX) $(CXXFLAGS) $(INC) $(CUDAINC) -c $< -o $@
	
Accel.o: Accel.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGSXCOMP) $(INC) $(CUDAINC) $< -o $@ 
	
text2D.o: text2D.cpp 
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@
	
shader.o: shader.cpp 
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@	
	
texture.o: texture.cpp 
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@	

clean:
	-rm -f *.o 
	-rm -f $(TARGET)
