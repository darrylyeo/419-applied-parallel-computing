CC = /usr/local/cuda-9.0/bin/nvcc
CFLAGS = -run

clean:
	rm HelloWorld HelloWorldBlockThread SingleBlockLoop ArrayManipulation ArrayCheckRange

HelloWorld: $(OBJS)
	$(CC) $(CFLAGS) -o HelloWorld HelloWorld.cu

HelloWorldBlockThread: $(OBJS)
	$(CC) $(CFLAGS) -o HelloWorldBlockThread HelloWorldBlockThread.cu

SingleBlockLoop: $(OBJS)
	$(CC) $(CFLAGS) -o SingleBlockLoop SingleBlockLoop.cu

ArrayManipulation: $(OBJS)
	$(CC) $(CFLAGS) -o ArrayManipulation ArrayManipulation.cu

ArrayCheckRange: $(OBJS)
	$(CC) $(CFLAGS) -o ArrayCheckRange ArrayCheckRange.cu
