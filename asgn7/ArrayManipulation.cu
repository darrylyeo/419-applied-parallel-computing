#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define N 20

__global__ void doubleElements(int *a){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N)
		a[i] *= 2;
}

int main(void){
	cudaError_t err = cudaSuccess;

	size_t size = N * sizeof(int);
	int *a;
	cudaMallocManaged(&a, size); // Use `a` on the CPU and/or on any GPU in the accelerated system.

	for(int i = 0; i < N; i++)
		a[i] = i;

	doubleElements<<<2,10>>>(a);
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();

	for(int i = 0; i < N; i++)
		printf("%d ", a[i]);
	printf("\n");
	
	cudaFree(a);
	
	return 0;
}