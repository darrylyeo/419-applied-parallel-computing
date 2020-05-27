CC = gcc
CFLAGS = -Wall -ansi -g -openmp

k-nearest-neighbors: $(OBJS)
	$(CC) $(CFLAGS) -o k-nearest-neighbors k-nearest-neighbors.c

k-nearest-neighbors-openmp: $(OBJS)
	$(CC) $(CFLAGS) -o k-nearest-neighbors-openmp k-nearest-neighbors-openmp.c