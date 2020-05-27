CC = /usr/local/cuda-9.0/bin/nvcc
CFLAGS = -run

clean:
	rm HelloWorld HelloWorldBlockThread

HelloWorld: $(OBJS)
	$(CC) $(CFLAGS) -o HelloWorld HelloWorld.cu

HelloWorldBlockThread: $(OBJS)
	$(CC) $(CFLAGS) -o HelloWorldBlockThread HelloWorldBlockThread.cu
