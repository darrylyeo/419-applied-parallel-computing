#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define N 1000000

__global__ void addVectors(int *a, int *b, int *c){
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
		c[i] = a[i] + b[i];
}

int main(void){
	cudaError_t err = cudaSuccess;

	size_t size = N * sizeof(int);

	int *a, *b, *c;
	cudaMallocManaged(&a, size);
	cudaMallocManaged(&b, size);
	cudaMallocManaged(&c, size);

	for(int i = 0; i < N; i++){
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	size_t threads_per_block = 256;
	size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
	addVectors<<<number_of_blocks, threads_per_block>>>(a, b, c);
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();

	for(int i = 0; i < N; i++)
		printf("%d ", c[i];
	printf("\n");

	printf("Done\n");
	
	cudaFree(a);
	
	return 0;
}