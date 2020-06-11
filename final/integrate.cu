#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define THREADS_PER_BLOCK 3

__global__ void calculate(double start, double step, int div, double (*f) (double), char *buffer){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if((double) i < div){
		double x = start + i * step;
		buffer[i] = f(x);
	}
}

double integrate(double start, double end, int div, double (*f) (double)){
	int N = div;
	double step = (end - start) / div;

	char *buffer;
	cudaMallocManaged(&buffer, sizeof(int) * N);
	calculate<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(start, step, N, f, buffer);

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

	double result = integrate(0, 10, 0.1, f);
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();

	printf("Result: %d\n", result);
	
	cudaFree(buffer);
	
	return 0;
}