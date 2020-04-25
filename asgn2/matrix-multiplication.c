#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>


#define N 10
#define NUM_THREADS 4

int matrixA[N][N];
int matrixB[N][N];
int matrixC[N][N];


int step = 0;

void *multiply(void *arg){
	/* Have each thread compute 1/Nth of the operations */
	for(int i = step * N / NUM_THREADS; i < (step + 1) * N / NUM_THREADS; i++)
		for(int j = 0; j < N; j++)
			for(int k = 0; k < N; k++)
				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
	
	step++;
	
	return NULL;
}


void printMatrix(char *name, int matrix[N][N]){
	printf("%s\n", name);
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++)
			printf("%d ", matrix[i][j]);
		printf("\n");
	}
	printf("\n");
}


int main(){
	int i, j;

	/* Generate random values for matrixA and matrixB */
	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			matrixA[i][j] = rand() % 10;
			matrixB[i][j] = rand() % 10;
		}
	}

	printMatrix("A", matrixA);
	printMatrix("B", matrixB);

	/* Create threads */
	pthread_t threads[NUM_THREADS];
	for(i = 0; i < NUM_THREADS; i++)
		pthread_create(&threads[i], NULL, multiply, NULL);

	/* Wait for all threads to complete */
	for(i = 0; i < NUM_THREADS; i++)
		pthread_join(threads[i], NULL);

	printMatrix("C", matrixC);

	return 0;
}
