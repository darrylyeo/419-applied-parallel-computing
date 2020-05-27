#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

__global__ void HelloWorld(){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("i = %d, thread %d, block %d", i, threadIdx.x, blockIdx.x);
	printf("Hello World\n");
}

int main(void){
	cudaError_t err = cudaSuccess;

	printf("HelloWorld<<<1,5>>>();");
	HelloWorld<<<1,5>>>();

	printf("HelloWorld<<<5,5>>>();");
	HelloWorld<<<2,5>>>();

	printf("HelloWorld<<<5,5>>>();");
	HelloWorld<<<2,5>>>();
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	
	return 0;
}