#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define THREADS_PER_BLOCK 32

__global__ void multiplyMatrices(int *a, int *b, int *c, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m){
		int sum = 0;
		for(int i = 0; i < n; i++)
			sum += a[row * n + i] * b[i * k + col];
		c[row * k + col] = sum;
	}
} 

int main(void){
	cudaError_t err = cudaSuccess;

	// a: m x n
	// b: n x k
	// c: m x k
	int *a, *b, *c;
	int m = 1000, n = 1000, k = 1000;

	cudaMallocManaged(&a, sizeof(int) * m * n);
	cudaMallocManaged(&b, sizeof(int) * n * k);
	cudaMallocManaged(&c, sizeof(int) * m * k);

	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			a[i * n + j] = rand() % 10;

	for(int i = 0; i < n; i++)
		for(int j = 0; j < k; j++)
			b[i * k + j] = rand() % 10;
	
	multiplyMatrices<<<
		(
			(m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
			(k + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
		),
		(
			THREADS_PER_BLOCK,
			THREADS_PER_BLOCK
		)
	>>>(
		a, b, c,
		m, n, k
	);
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();

	for(int i = 0; i < m; i++){
		for(int j = 0; j < k; j++)
			printf("%d ", c[i * k + j]);
		printf("\n");
	}

	printf("Done\n");
	
	cudaFree(a);
	
	return 0;
}