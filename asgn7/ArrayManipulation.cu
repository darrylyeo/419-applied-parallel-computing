#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define N 20

__global__ void doubleArray(int *a){
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

	doubleArray<<<2,10>>>(a);

	for(int i = 0; i < N; i++)
		printf("%d ", a[i]);
	printf("\n");
	
	cudaFree(a);
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	
	return 0;
}