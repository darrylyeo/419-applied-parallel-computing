#include <stdio.h>

#define N 1000

int main(void){
	char buffer[N];
	for(int i = 0; i < N; i++)
		buffer[i] = 32 + rand() % 95;

	int frequencies[127];
	#pragma omp for reduce(frequencies) local(i)
	for(int i = 0; i < N; i++)
		frequencies[buffer[i]]++;

	for(int i = 32; i < 127; i++)
		printf("%c: %d\n", i, frequencies[i]);

	return 0;
}