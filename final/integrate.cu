#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define THREADS_PER_BLOCK 3

double f(double x){
	return x*x;
}

__global__ void calculate(double *buffer, double start, double step, int N, double (*f) (double)){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N){
		double x = start + i * step;
		buffer[i] = f(x);
	}
}

double integrate(double *buffer, double start, double end, int div, double (*f) (double)){
	int N = div;
	double step = (end - start) / div;

	cudaMallocManaged(&buffer, sizeof(double) * N);
	calculate<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(buffer, start, step, N, f);

	cudaDeviceSynchronize();

	for(int i = 0; i < N; i++)
		printf("%d ", buffer[i]);
	printf("\n");

	double result = (f(start) + f(end)) / 2;
	for(int i = 0; i < N; i++)
		result += buffer[i]; // f(start + i * step);
	return result;
}

int main(void){
	cudaError_t err = cudaSuccess;

	double *buffer = NULL;

	double result = integrate(buffer, 0, 10, 100, f);

	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Result: %d\n", result);

	cudaFree(buffer);

	return 0;
}