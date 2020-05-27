#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define N 1000000

__global__ void doubleElements(int *a){
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
		a[i] *= 2;
}

int main(void){
	cudaError_t err = cudaSuccess;

	size_t size = N * sizeof(int);
	int *a;
	cudaMallocManaged(&a, size); // Use `a` on the CPU and/or on any GPU in the accelerated system.

	for(int i = 0; i < N; i++)
		a[i] = i;

	size_t threads_per_block = 256;
	size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
	doubleElements<<<number_of_blocks, threads_per_block>>>(a);
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();

	for(int i = 0; i < N; i++)
		if(a[i] != i * 2){
			printf("Failed\n");
			break;
		}
	printf("Done\n");
	
	cudaFree(a);
	
	return 0;
}