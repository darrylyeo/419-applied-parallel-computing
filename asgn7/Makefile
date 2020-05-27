CC = /usr/local/cuda-9.0/bin/nvcc
CFLAGS = -run

clean:
	rm HelloWorld HelloWorldBlockThread SingleBlockLoop ArrayManipulation ArrayCheckRange ArrayManipulationGridStride 01-vector-add 01-matrix-multiply


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

ArrayManipulationGridStride: $(OBJS)
	$(CC) $(CFLAGS) -o ArrayManipulationGridStride ArrayManipulationGridStride.cu


01-vector-add: $(OBJS)
	$(CC) $(CFLAGS) -o 01-vector-add 01-vector-add.cu

01-matrix-multiply: $(OBJS)
	$(CC) $(CFLAGS) -o 01-matrix-multiply 01-matrix-multiply.cu
