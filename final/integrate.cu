#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define THREADS_PER_BLOCK 3

__global__ void calculate(char *buffer, double start, double step, int N, double (*f) (double)){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if((double) i < N){
		double x = start + i * step;
		buffer[i] = f(x);
	}
}

double integrate(char *buffer, double start, double end, int div, double (*f) (double)){
	unsigned long N = (unsigned long) div;
	double step = (end - start) / div;

	cudaMallocManaged(buffer, sizeof(int) * N);
	calculate<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(buffer, start, step, N, f);

	double result = (f(start) + f(end)) / 2;
	for(int i = 0; i < N; i++)
		result += buffer[i]; // f(start + i * step);
	return result;
}

double f(double x){
	return x*x;
}

int main(void){
	cudaError_t err = cudaSuccess;

	char *buffer = NULL;

	double result = integrate(buffer, 0, 10, 0.1, f);
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();

	printf("Result: %d\n", result);
	
	cudaFree(buffer);
	
	return 0;
}