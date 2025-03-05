CC=gcc

CFLAGS=-g3 -Wall

OFLAGS=-march=native -Ofast -fopenmp -fopenmp-simd

LFLAGS=-lm

all: memcpybench

memcpybench: main.c
		$(CC) $(CFLAGS) $(OFLAGS) utils.c memcpy.c $< -o $@ $(LFLAGS)

clean:
	rm -Rf memcpybench
