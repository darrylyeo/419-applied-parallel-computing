#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define N 1024
#define THREADS_PER_BLOCK 32

__global__ void SingleBlockLoop(){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N)
		printf("%d\n", i);
}

int main(void){
	cudaError_t err = cudaSuccess;

	SingleBlockLoop<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>();
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	
	return 0;
}