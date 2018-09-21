APP = merge
CC = gcc
CFLAGS = -std=c99 -O2 -Wall -g #-vec-report1
LIBS=-L/usr/local/lib
INC=-I/usr/local/include 
OBJS = merge.o
SOURCE = merge.c

$(APP): $(OBJS)
	$(CC) $(CFLAGS) -o $(APP) $(OBJS) $(LIBS)

cuboid_mpi.o: $(SOURCE) Makefile 
	$(CC) $(CFLAGS) $(INC) -c $(SOURCE) 

.PHONY: clean
clean:
	rm -rf $(APP) $(OBJS)
