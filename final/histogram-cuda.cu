#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define N 20
#define THREADS_PER_BLOCK 3

__global__ void histogram(char *buffer, int *frequencies){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N)
		frequencies[buffer[i]]++;
}

int main(void){
	cudaError_t err = cudaSuccess;

	size_t size = N * sizeof(int);
	char *buffer;
	cudaMallocManaged(&buffer, size); // Use `buffer` on the CPU and/or on any GPU in the accelerated system.

	int *frequencies;
	cudaMallocManaged(&frequencies, 127);

	for(int i = 0; i < N; i++)
		buffer[i] = 32 + rand() % 95;

	histogram<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(buffer, frequencies);

	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();

	for(int i = 32; i < 127; i++)
		printf("%c: %d\n", i, frequencies[i]);

	cudaFree(buffer);

	return 0;
}