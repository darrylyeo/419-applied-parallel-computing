#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

__global__ void HelloWorld(){
	printf("Hello World");
}

int main(void){
	cudaError_t err = cudaSuccess;

	HelloWorld<<<5,1>>>();
	HelloWorld<<<5,5>>>();
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	
	return 0;
}