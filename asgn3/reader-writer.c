#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#define INPUT_FILE "reader-writer.c"

#define NUM_THREADS 4
#define CHUNK_SIZE 100

char *searchStrings[] = {"thread", "reader", "writer", "search"};

int fd;
char *buffer;
int bufferSize = 0;


pthread_cond_t turn;
pthread_mutex_t mutex;

int writers;
int writing;
int reading;

void *reader(){
	int size;

	pthread_mutex_lock(&mutex);
	while(writers > 0)
		pthread_cond_wait(&turn, &mutex);
	reading++;
	pthread_mutex_unlock(&mutex);

	/* Read */
	lseek(fd, 0, 0);
	while((size = read(fd, &buffer[bufferSize], CHUNK_SIZE)))
		bufferSize += size;
		/*printf("%d\n", *id);*/

	pthread_mutex_lock(&mutex);
	reading--;
	pthread_cond_broadcast(&turn);
	pthread_mutex_unlock(&mutex);

	return NULL;
}

void *writer(void *arg){
	char *searchString = (char *) arg;
	/*char substring[CHUNK_SIZE];*/
	int searchLength = strlen(searchString);
	/*int searchCursor = 0;*/
	char *searchPointer;
	char *searchResult;

	pthread_mutex_lock(&mutex);
	writers++;
	while(reading > 0 || writing > 0)
		pthread_cond_wait(&turn, &mutex);
	writing++;
	pthread_mutex_unlock(&mutex);

	/* Write */
	for(
		searchPointer = buffer;
		searchPointer < buffer + bufferSize && (searchResult = strstr(searchPointer, searchString)) != NULL;
		searchPointer = searchResult + searchLength
	){
		printf("\"%s\" found at index %d\n", searchString, (int) (searchResult - buffer));
	}
	/*while(searchCursor < bufferSize){
		printf("%s %d %d\n", searchString, searchCursor, bufferSize);
		int size = searchCursor + CHUNK_SIZE < bufferSize
			? CHUNK_SIZE
			: bufferSize - CHUNK_SIZE;
		memcpy(substring, &buffer[searchCursor], size);
		printf("\nSearching: %s\n%s\n", searchString, substring);

		if((searchPointer = strstr(searchString, substring)) != NULL)
			printf("\"%s\" found at index %d\n", searchString, (int) searchPointer - searchCursor);

		searchCursor += size - searchLength;
	}*/

	/*while(writeCursor < bufferSize)
		write(1, &chunks[writeCursor++], CHUNK_SIZE);*/

	pthread_mutex_lock(&mutex);
	writing--;
	writers--;
	pthread_cond_broadcast(&turn);
	pthread_mutex_unlock(&mutex);

	return NULL;
}


int main(){
	int i;
	pthread_t readerThreads[NUM_THREADS];
	pthread_t writerThreads[NUM_THREADS];

	buffer = calloc(50000, sizeof(char));

	if(fd != -1){
		fd = open(INPUT_FILE, O_RDONLY, 0640);
	}else{
		perror("open");
		return EXIT_FAILURE;
	}

	for(i = 0; i < NUM_THREADS; i++)
		pthread_create(&readerThreads[i], NULL, &reader, &i);
	for(i = 0; i < NUM_THREADS; i++)
		pthread_create(&writerThreads[i], NULL, &writer, searchStrings[i]);
	
	for(i = 0; i < NUM_THREADS; i++)
		pthread_join(readerThreads[i], NULL);
	for(i = 0; i < NUM_THREADS; i++)
		pthread_join(writerThreads[i], NULL);

	return EXIT_SUCCESS;
}
