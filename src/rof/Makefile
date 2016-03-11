primaldual: primaldual.cu aux.cu aux.h Makefile
	nvcc -o primaldual primaldual.cu aux.cu -lcublas --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core
