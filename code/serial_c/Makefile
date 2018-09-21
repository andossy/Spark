APP = cuboid_serial
CC = gcc
#CC = /usr/local/mpi/bin/mpicc
#CC = mpicc
#CFLAGS = -fopenmp -O3 
#CFLAGS = -msse4.2 -O3 
#CFLAGS = -xAVX -fopenmp -O3 -lpapi 
#CFLAGS = -xAVX -O3 
#CFLAGS = -g 
#CFLAGS = -funroll-loops -O2 
CFLAGS = -g -O3 -Wall #-vec-report1
#-lpapi
#CFLAGS = -O3 -openmp -Wall -vec-report1 -mmic 
#-opt-prefetch-distance=32,16
#INC=   -I/home/nfs/u107/tools/papi-install/include
#LIBS = -L/home/nfs/u107/tools/papi-install/lib -lpapi
#INC=-I/usr/local/include -I/usr/include -I/opt/intel/composer_xe2013.0.079/compiler/include
#LIBS=-L/lib64 -L/usr/local/lib -L/usr/local/lib64 -L/usr/lib64 -L/opt/intel/composer_xe2013.0.079/bin/sourcechecker/lib/intel64/pinruntime/glibc -L/opt/intel/composer_xe2013.0.079/compiler/lib/intel64

#LIBS = -L/home/nanwu/local/lib 
#INC  = -I/home/nanwu/local/include 

#LIBS = -L/usit/abel/u1/nanwu/lib
#INC =  -I/usit/abel/u1/nanwu/include

LIBS=-L/usr/local/lib -lm
INC=-I/usr/local/include 

OBJS = cuboid_serial.o

$(APP): $(OBJS)
	$(CC) $(CFLAGS) -o $(APP) $(OBJS) $(LIBS)

cuboid_serial.o: cuboid_serial.c Makefile 
	$(CC) $(CFLAGS) $(INC) -c cuboid_serial.c

.PHONY: clean
clean:
	rm -rf cuboid_serial $(OBJS)
