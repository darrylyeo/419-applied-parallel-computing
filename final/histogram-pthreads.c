#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>


#define N 1000
#define NUM_THREADS 4


char buffer[N];
int frequencies[127];

int threadNum = 0;

void *histogram(void *arg){
	int i;
	for(i = threadNum * N / NUM_THREADS; i < (threadNum + 1) * N / NUM_THREADS; i++)
		frequencies[(int) buffer[i]]++;

	threadNum++;

	return NULL;
}

int main(void){
	int i;

	for(i = 0; i < N; i++)
		buffer[i] = 32 + rand() % 95;

	pthread_t threads[NUM_THREADS];
	for(i = 0; i < NUM_THREADS; i++)
		pthread_create(&threads[i], NULL, histogram, NULL);

	for(i = 0; i < NUM_THREADS; i++)
		pthread_join(threads[i], NULL);

	for(i = 32; i < 127; i++)
		printf("%c: %d\n", i, frequencies[i]);

	return 0;
}