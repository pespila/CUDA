primaldual: primaldual.cu aux.cu aux.h Makefile
	nvcc -o primaldual primaldual.cu aux.cu -lcublas --ptxas-options=-v --use_fast_math -arch sm_35 --compiler-options -Wall -lopencv_highgui -lopencv_core


VERSION = 1.0
CC = clang++
CPPFLAGS = -O3 -ftree-vectorize -funroll-loops -fomit-frame-pointer -ffast-math -Wall

DEPS =	Image.h \
		Projection.h \
		LinearOperator.h \
		Algebra.h \
		Vector.h

OBJ =	main.o \
		Image.o \
		Projection.o \
		LinearOperator.o \
		Algebra.o \
		Vector.o

LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lm
LDFLAGS = -L/opt/local/lib
IDFLAGS = -I/opt/local/include

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CPPFLAGS) $(IDFLAGS)

Projection: $(OBJ)
	$(CC) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

# clean:
# 	rm *.o