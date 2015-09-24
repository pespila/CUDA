primaldual_lagrange: primaldual_lagrange.cu aux.cu aux.h Makefile
	nvcc -o primaldual_lagrange primaldual_lagrange.cu aux.cu -lcublas --ptxas-options=-v --use_fast_math -arch sm_35 --compiler-options -Wall -lopencv_highgui -lopencv_core
