#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>


int writers;
int writing;
int reading;

pthread_cond_t turn;
pthread_mutex_t mutex;

int fd;
#define BUFFER_SIZE 2000
char buffer[BUFFER_SIZE];


void *reader(char *filename){
	pthread_mutex_lock(&mutex);
	while(writers > 0)
		pthread_cond_wait(&turn, &mutex);
	reading++;
	pthread_mutex_unlock(&mutex);

	/* Read stuff */
	if(fd != -1){
		fd = open(filename, O_RDONLY, 0640);
		read(fd, &buffer, BUFFER_SIZE);
	}else{
		perror("open");
		exit(EXIT_FAILURE);
	}

	pthread_mutex_lock(&mutex);
	reading--;
	pthread_cond_broadcast(&turn);
	pthread_mutex_unlock(&mutex);

	return NULL;
}


void *writer(){
	pthread_mutex_lock(&mutex);
	writers++;
	while(reading || writing)
		pthread_cond_wait(&turn, &mutex);
	writing++;
	pthread_mutex_unlock(&mutex);

	/* Write stuff */
	write(1, &buffer, BUFFER_SIZE);
	printf("\n");

	pthread_mutex_lock(&mutex);
	writing--;
	writers--;
	pthread_cond_broadcast(&turn);
	pthread_mutex_unlock(&mutex);

	return NULL;
}


int main(){
	pthread_t threads[4];

	pthread_create(&threads[0], NULL, &reader, "reader-writer.c");
	pthread_create(&threads[1], NULL, &writer, NULL);
	pthread_create(&threads[2], NULL, &reader, "reader-writer.c");
	pthread_create(&threads[3], NULL, &writer, NULL);

	pthread_join(threads[0], NULL);
	pthread_join(threads[1], NULL);
	pthread_join(threads[2], NULL);
	pthread_join(threads[3], NULL);

	return 0;
}
