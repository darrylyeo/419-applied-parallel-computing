#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define THREADS_PER_BLOCK 3

__global__ void calculate(char *buffer, double (*f) (double)){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < div){
		double x = start + i * step;
		buffer[i] = f(x);
	}
}

double integrate(double start, double end, int div, double (*f) (double)){
	double step = (end - start) / div;
	double result = (f(start) + f(end)) / 2;

	char *buffer;
	cudaMallocManaged(&buffer, sizeof(int) * div);
	calculate<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(buffer, f);

	for(int i = 0; i < div; i++)
		result += buffer[i]; // f(start + i * step);
}

double f(double x){
	return x;
}

int main(void){
	cudaError_t err = cudaSuccess;

	integrate(st, en, div, f);
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();

	for(int i = 0; i < N; i++)
		printf("%d ", buffer[i]);
	printf("\n");
	
	cudaFree(buffer);
	
	return 0;
}