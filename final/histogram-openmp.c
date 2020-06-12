#include <stdio.h>

#define N 1000

int main(void){
	int i;

	char buffer[N];
	for(i = 0; i < N; i++)
		buffer[i] = 32 + rand() % 95;

	int frequencies[127];
	#pragma omp for reduce(frequencies) local(i)
	for(i = 0; i < N; i++)
		frequencies[(int) buffer[i]]++;

	for(i = 32; i < 127; i++)
		printf("%c: %d\n", i, frequencies[i]);

	return 0;
}