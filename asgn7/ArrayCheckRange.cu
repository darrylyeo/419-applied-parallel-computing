#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

int N = 1000;
size_t threads_per_block = 256;
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

__global__ void initializeElementsTo(int *a, int *value){
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < N)
		a[value] = *value;
}

int main(void){
	cudaError_t err = cudaSuccess;

	size_t size = N * sizeof(int);
	int *a;
	cudaMallocManaged(&a, size);

	initializeElementsTo<<<number_of_blocks, threads_per_block>>>(a, 123);
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();

	for(int i = 0; i < N; i++)
		if(a[i] != 123){
			printf("Failed\n");
			break;
		}
	printf("Done\n");
	
	cudaFree(a);
	
	return 0;
}