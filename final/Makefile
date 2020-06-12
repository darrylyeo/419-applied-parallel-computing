clean:
	rm integrate histogram-cuda histogram-openmp histogram-pthreads


integrate: integrate.cu
	/usr/local/cuda-9.0/bin/nvcc -run -o integrate integrate.cu

nvprof-integrate: integrate
	/usr/local/cuda-9.0/bin/nvprof ./integrate


histogram-cuda: histogram-cuda.cu
	/usr/local/cuda-9.0/bin/nvcc -run -o histogram-cuda histogram-cuda.cu

nvprof-histogram-cuda: histogram-cuda
	/usr/local/cuda-9.0/bin/nvprof ./integrate


histogram-openmp: histogram-openmp.c
	gcc -Wall -ansi -g -openmp -o histogram-openmp histogram-openmp.c


histogram-pthreads: histogram-pthreads.c
	gcc -Wall -ansi -g -pthread -o histogram-pthreads histogram-pthreads.c
