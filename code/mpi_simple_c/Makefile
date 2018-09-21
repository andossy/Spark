APP = cuboid_mpi
#CC = gcc
#CC = /usr/local/mpi/bin/mpicc
CC = mpicc
#CFLAGS = -fopenmp -O3 
#CFLAGS = -msse4.2 -O3 
#CFLAGS = -xAVX -fopenmp -O3 -lpapi 
#CFLAGS = -xAVX -O3 
#CFLAGS = -g 
#CFLAGS = -funroll-loops -O2 

####### ABEL #####################
CFLAGS = -O3 -Wall -vec-report1
LIBS = -L/cluster/software/VERSIONS/hdf5-1.8.12_intel/lib -lhdf5

####### HAKE LOCAL ###############
#CFLAGS = -O3 -Wall -I/home/hake/local/include
#LIBS = -lhdf5 -L/home/hake/local/lib
VECTORIZE = -ftree-vectorize -ftree-vectorizer-verbose=1 -msse -msse2 -msse3 
CFLAGS = -O3 -Wall -g -march=native -mtune=native -fPIC -ffast-math $(VECTORIZE)
LIBS = -lhdf5 -lm

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


OBJS = cuboid_mpi.o

$(APP): $(OBJS)
	$(CC) $(CFLAGS) -o $(APP)$(APP_SUFFIX) $(OBJS) $(LIBS)

cuboid_mpi.o: cuboid_mpi.c Makefile 
	$(CC) $(CFLAGS) $(INC) -c cuboid_mpi.c

.PHONY: clean
clean:
	rm -rf $(APP) $(OBJS)
