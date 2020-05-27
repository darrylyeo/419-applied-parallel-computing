# Introduction to CUDA Writing Application Code for the GPU

CUDA provides extensions for many common programming languages, in the case of this lab, C/C++. These language extensions easily allow developers to run functions in their source code on a GPU.

Below is a .cu file (.cu is the file extension for CUDA-accelerated programs). It contains two functions, the first which will run on the CPU, the second which will run on the GPU. Spend a little time identifying the differences between the functions, both in terms of how they are defined, and how they are invoked.

```c
void CPUFunction(){
	printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction(){
	printf("This function is defined to run on the GPU.\n");
}

int main(){
	CPUFunction();

	GPUFunction<<<1, 1>>>();
	cudaDeviceSynchronize();
}
```

Here are some important lines of code to highlight, as well as some other common terms used in accelerated computing:

```c
__global__ void GPUFunction()
```

* The `__global__` keyword indicates that the following function will run on the GPU, and can be invoked **globally**, which in this context means either by the CPU, or, by the GPU.
* Often, code executed on the CPU is referred to as **host** code, and code running on the GPU is referred to as device code.
* Notice the return type void. It is required that functions defined with the `__global__` keyword return type void.

```c
GPUFunction<<<1, 1>>>();
```

* Typically, when calling a function to run on the GPU, we call this function a kernel, which is launched.
* When launching a kernel, we must provide an execution configuration, which is done by using the <<< ... >>> syntax just prior to passing the kernel any expected arguments.
* At a high level, execution configuration allows programmers to specify the thread hierarchy for a kernel launch, which defines the number of thread groupings (called blocks), as well as how many threads to execute in each block. Execution configuration will be explored at great length later in the lab, but for the time being, notice the kernel is launching with 1 block of threads (the first execution configuration argument) which contains 1 thread (the second configuration argument).

```c
cudaDeviceSynchronize();
```

* Unlike much C/C++ code, launching kernels is asynchronous: the CPU code will continue to execute without waiting for the kernel launch to complete.
* A call to cudaDeviceSynchronize, a function provided by the CUDA runtime, will cause the host (CPU) code to wait until the device (GPU) code completes, and only then resume execution on the CPU.


## Exercise: Hello GPU Kernel

	nvcc -o hello-gpu 01-hello/01-hello-gpu.cu -run

nvcc is the command line command for using the nvcc compiler.


```c
#include <stdio.h>
#include <cuda_runtime.h> // => may not be necesary to include
#include <helper_cuda.h>

__global__ void HelloWorld()
{
	printf(“Hello World”);
}

int main(void){
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Launch the Vector Add CUDA Kernel
	HelloWorld<<<1,1>>>();
	err = cudaGetLastError();
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();
	printf("Done\n");
	return 0;
}
```


## Launching Parallel Kernels

The execution configuration allows programmers to specify details about launching the kernel to run in parallel on multiple GPU threads. More precisely, the execution configuration allows programmers to specifiy how many groups of threads - called thread blocks, or just blocks - and how many threads they would like each thread block to contain. The syntax for this is:

	<<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>

A kernel is executed once for every thread in every thread block configured when the kernel is launched.
Thus, under the assumption that a kernel called someKernel has been defined, the following are true:

* someKernel<<<1, 1>>() is configured to run in a single thread block which has a single thread and will therefore run only once.
* someKernel<<<1, 10>>() is configured to run in a single thread block which has 10 threads and will therefore run 10 times.
* someKernel<<<10, 1>>() is configured to run in 10 thread blocks which each have a single thread and will therefore run 10 times.
* someKernel<<<10, 10>>() is configured to run in 10 thread blocks which each have 10 threads and will therefore run 100 times.


## Exercise: Launch Parallel Kernels

Follow the steps below to refactor HelloWorld to run on the GPU to:
* Refactor the HelloWorld kernel to execute in parallel on 5 threads, all executing in a single thread block. You should see the output message printed 5 times after compiling and running the code.
* Refactor the HelloWorld kernel again, this time to execute in parallel inside 5 thread blocks, each containing 5 threads. You should see the output message printed 25 times now after compiling and running.

	nvcc -o basic-parallel HelloWorld.cu -run


```c
#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

__global__ void HelloWorld(){
	printf("Hello World\n");
}

int main(void){
	cudaError_t err = cudaSuccess;

	printf("HelloWorld<<<1,5>>>();");
	HelloWorld<<<1,5>>>();

	printf("HelloWorld<<<5,5>>>();");
	HelloWorld<<<5,5>>>();
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	
	return 0;
}
```


## Thread and Block Indices
Each thread is given an index within its thread block, starting at 0. Additionally, each block is given an index, starting at 0. Just as threads are grouped into thread blocks, blocks are grouped into a grid, which is the highest entity in the CUDA thread hierarchy. In summary, CUDA kernels are executed in a grid of 1 or more blocks, with each block containing the same number of 1 or more threads.
CUDA kernels have access to special variables identifying both the index of the thread (within the block) that is executing the kernel, and, the index of the block (within the grid) that the thread is within. These variables are threadIdx.x and blockIdx.x respectively.


## Exercise: Launch Parallel Kernel:
* Refactor the HelloWorld kernel print the threadID and BlockID, execute in parallel on 5 threads, all executing in a single thread block.
* Refactor the HelloWorld kernel print the threadID and BlockID, execute in parallel on 5 threads, all executing in a two thread blocks.
* Refactor the HelloWorld kernel so each thread prints helloWorld and one number from 0 to 9, execute in parallel on 5 threads, all executing in two thread blocks.

Write here the GPU kernel function

```c
#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

__global__ void HelloWorld(){
	printf("Hello World\n");

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("i = %d, thread %d, block %d\n", i, threadIdx.x, blockIdx.x);
}

int main(void){
	cudaError_t err = cudaSuccess;

	printf("HelloWorld<<<1,5>>>();");
	HelloWorld<<<1,5>>>();

	printf("HelloWorld<<<5,5>>>();");
	HelloWorld<<<2,5>>>();

	printf("HelloWorld<<<5,5>>>();");
	HelloWorld<<<2,5>>>();
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	
	return 0;
}
```

	nvcc  -o basic-parallel HelloWorld.cu -run


## Accelerating For Loops
For loops in CPU-only applications are ripe for acceleration: rather than run each iteration of the loop serially, each iteration of the loop can be run in parallel in its own thread. Consider the following for loop, and notice, though it is obvious, that it controls how many times the loop will execute, as well as defining what will happen for each iteration of the loop:

```c
int N = 2<<20;
for (int i = 0; i < N; ++i){
	printf("%d\n", i);
}
```

In order to parallelize this loop, 2 steps must be taken:
* A kernel must be written to do the work of a single iteration of the loop.
* Because the kernel will be agnostic of other running kernels, the execution configuration must be such that the kernel executes the correct number of times, for example, the number of times the loop would have iterated.


## Exercise: Accelerating a For Loop with a Single Block of Threads
Refactor the loop function to be a CUDA kernel which will launch to execute N iterations in parallel. After successfully refactoring, the numbers 0 through 9 should still be printed.

	nvcc -o single-block-loop 04-loops/01-single-block-loop.cu -run

Write here the GPU kernel function

```c
#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define N 1024
#define THREADS_PER_BLOCK 32

__global__ void SingleBlockLoop(){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N)
		printf("%d\n", i);
}

int main(void){
	cudaError_t err = cudaSuccess;

	SingleBlockLoop<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>();
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	
	return 0;
}
```


## Using Block Dimensions for More Parallelization
There is a limit to the number of threads that can exist in a thread block: 1024 to be precise. In order to increase the amount of parallelism in accelerated applications, we must be able to coordinate among multiple thread blocks.
CUDA Kernels have access to a special variable that gives the number of threads in a block: blockDim.x. Using this variable, in conjunction with blockIdx.x and threadIdx.x, increased parallelization can be accomplished by organizing parallel execution accross multiple blocks of multiple threads with the idiomatic expression threadIdx.x + blockIdx.x * blockDim.x. Here is a detailed example.
The execution configuration <<<10, 10>>> would launch a grid with a total of 100 threads, contained in 10 blocks of 10 threads. We would therefore hope for each thread to have the ability to calculate some index unique to itself between 0 and 99.
* If block blockIdx.x equals 0, then blockIdx.x * blockDim.x is 0. Adding to 0 the possible threadIdx.x values 0 through 9, then we can generate the indices 0 through 9 within the 100 thread grid.
* If block blockIdx.x equals 1, then blockIdx.x * blockDim.x is 10. Adding to 10 the possible threadIdx.x values 0 through 9, then we can generate the indices 10 through 19 within the 100 thread grid.
* If block blockIdx.x equals 5, then blockIdx.x * blockDim.x is 50. Adding to 50 the possible threadIdx.x values 0 through 9, then we can generate the indices 50 through 59 within the 100 thread grid.
* If block blockIdx.x equals 9, then blockIdx.x * blockDim.x is 90. Adding to 90 the possible threadIdx.x values 0 through 9, then we can generate the indices 90 through 99 within the 100 thread grid.


## Exercise: Accelerating a For Loop with Multiple Blocks of Threads

Currently, the loop function inside runs a for loop that will serially print the numbers 0 through 9. Refactor the loop function to be a CUDA kernel which will launch to execute N iterations in parallel. After successfully refactoring, the numbers 0 through 9 should still be printed. For this exercise, as an additional constraint, use an execution configuration that launches at least 2 blocks of threads.

Write here the GPU kernel function

```c
#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define N 10
#define THREADS_PER_BLOCK 3

__global__ void SingleBlockLoop(){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N)
		printf("%d\n", i);
}

int main(void){
	cudaError_t err = cudaSuccess;

	SingleBlockLoop<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>();
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	
	return 0;
}
```


## Allocating Memory to be accessed on the GPU and the CPU

More recent versions of CUDA (version 6 and later) have made it easy to allocate memory that is available to both the CPU host and any number of GPU devices, and while there are many intermediate and advanced techniques for memory management that will support the most optimal performance in accelerated applications, the most basic CUDA memory management technique we will now cover supports fantastic performance gains over CPU-only applications with almost no developer overhead.
To allocate and free memory, and obtain a pointer that can be referenced in both host and device code, replace calls to malloc and free with cudaMallocManaged and cudaFree as in the following example:

```c
// CPU-only
int N = 2<<20;
size_t size = N * sizeof(int);
int *a;
a = (int *)malloc(size); // Use `a` in CPU-only program.
...
free(a);

// Accelerated
int N = 2<<20;
size_t size = N * sizeof(int);
int *a;
cudaMallocManaged(&a, size); // Use `a` on the CPU and/or on any GPU in the accelerated system.
cudaFree(a);
```


## Exercise: Array Manipulation on both the Host and Device
Program allocates an array, initializes it with integer values on the host, attempts to double each of these values in parallel on the GPU, and then confirms whether or not the doubling operations were successful, on the host.
Write here the GPU kernel function and the Memory Allocation Part of Host code

```c
#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#define N 20

__global__ void double(int *a){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N)
		a[i] *= 2;
}

int main(void){
	cudaError_t err = cudaSuccess;

	size_t size = N * sizeof(int);
	int *a;
	cudaMallocManaged(&a, size); // Use `a` on the CPU and/or on any GPU in the accelerated system.

	for(int i = 0; i < N; i++)
		a[i] = i;

	double<<<2,10>>>(&a);

	for(int i = 0; i < N; i++)
		print("%d ", a[i]);
	print("\n");
	
	cudaFree(a);
	
	if ((err = cudaGetLastError()) != cudaSuccess){
		fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	
	return 0;
}
```


## Handling Block Configuration Mismatches to Number of Needed Threads
It may be the case that an execution configuration cannot be expressed that will create the exact number of threads needed for parallelizing a loop.
A common example has to do with the desire to choose optimal block sizes. For example, due to GPU hardware traits, blocks that contain a number of threads that are a multiple of 32 are often desirable for performance benefits. Assuming that we wanted to launch blocks each containing 256 threads (a multiple of 32), and needed to run 1000 parallel tasks (a trivially small number for ease of explanation), then there is no number of blocks that would produce an exact total of 1000 threads in the grid, since there is no integer value 32 can be multiplied by to equal exactly 1000.
This scenario can be easily addressed in the following way:
* Write an execution configuration that creates more threads than necessary to perform the allotted work.
* Pass a value as an argument into the kernel that represents the N number of times the kernel ought to run.
* After calculating the thread's index within the grid (using tid+bid*bdim), check that this index does not exceed N, and only perform the pertinent work of the kernel if it does not.
Here is an example of an idiomatic way to write an execution configuration when both N and the number of threads in a block are known, and an exact match between the number of threads in the grid and N cannot be guaranteed. It ensures that there are always at least as many threads as needed for N, and only 1 additional block's worth of threads extra, at most:

```c
// Assume `N` is known
int N = 100000;

// Assume we have a desire to set `threads_per_block` exactly to `256`
size_t threads_per_block = 256;

// Ensure there are at least `N` threads in the grid, but only 1 block's worth extra
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

some_kernel<<<number_of_blocks, threads_per_block>>>(N);
```


Because the execution configuration above results in more threads in the grid than N, care will need to be taken inside of the some_kernel definition so that some_kernel does not attempt to access out of range data elements, when being executed by one of the "extra" threads:

```c
__global__ some_kernel(int N){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N) // Check to make sure `idx` maps to some value within `N`
	{
		// Only do work if it does
	}
}
```


## Exercise: Accelerating a For Loop with a Mismatched Execution Configuration
* Assign a value to number_of_blocks that will make sure there are at least as many threads as there are elements in a to work on.

Write here the GPU kernel function

```c

```


Update the initializeElementsTo kernel to make sure that it does not attempt to work on data elements that are out of range.


```c
```


## Data Sets Larger than the Grid
Either by choice, often to create the most performant execution configuration, or out of necessity, the number of threads in a grid may be smaller than the size of a data set. Consider an array with 1000 elements, and a grid with 250 threads (using trivial sizes here for ease of explanation). Here, each thread in the grid will need to be used 4 times. One common method to do this is to use a grid-stride loop within the kernel.
In a grid-stride loop, each thread will calculate its unique index within the grid using tid+bid*bdim, perform its operation on the element at that index within the array, and then, add to its index the number of threads in the grid and repeat, until it is out of range of the array. For example, for a 500 element array and a 250 thread grid, the thread with index 20 in the grid would:
* Perform its operation on element 20 of the 500 element array
* Increment its index by 250, the size of the grid, resulting in 270
* Perform its operation on element 270 of the 500 element array
* Increment its index by 250, the size of the grid, resulting in 520
* Because 520 is now out of range for the array, the thread will stop its work
CUDA provides a special variable giving the number of blocks in a grid, gridDim.x. Calculating the total number of threads in a grid then is simply the number of blocks in a grid multiplied by the number of threads in each block, gridDim.x * blockDim.x. With this in mind, here is a verbose example of a grid-stride loop within a kernel:

```c

__global void kernel(int *a, int N)
{
	int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
	int gridStride = gridDim.x * blockDim.x;

	for (int i = indexWithinTheGrid; i < N; i += gridStride)
	{
		// do work on a[i];
	}
}

```


## Exercise: Use a Grid-Stride Loop to Manipulate an Array Larger than the Grid

Refactor the doubleElements kernel, in order that the grid, which is smaller than N, can reuse threads to cover every element in the array. The program will print whether or not every element in the array has been doubled, currently the program accurately prints FALSE.

Write here the GPU kernel function and the Memory Allocation Part of Host code

```c
```

## Error Handling
As in any application, error handling in accelerated CUDA code is essential.

```c
cudaError_t err;
err = cudaMallocManaged(&a, N)                      // Assume the existence of `a` and `N`.

if (err != cudaSuccess)                             // `cudaSuccess` is provided by CUDA.
{
	printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString` is provided by CUDA.
}
```

Launching kernels, which are defined to return void, do not return a value of type cudaError_t. To check for errors occuring at the time of a kernel launch, for example if the launch configuration is erroneous, CUDA provides the cudaGetLastError function, which does return a value of type cudaError_t.

```c
/* * This launch should cause an error, but the kernel itself cannot return it. */
someKernel<<<1, -1>>>();  // -1 is not a valid number of threads.
cudaError_t err;
err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
if (err != cudaSuccess)
{
	printf("Error: %s\n", cudaGetErrorString(err));
}
```


Finally, in order to catch errors that occur asynchronously, for example during the execution of an asynchronous kernel, it is essential to check the error returned by cudaDeviceSynchronize, which will return an error if one of the kernel executions it is synchronizing on should fail.


## CUDA Error Handling Macro

It can be helpful to create a macro that wraps CUDA function calls for checking errors. Here is an example, feel free to use it in the remaining exercises:

```c
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

int main()
{
	/* The macro can be wrapped around any function returning
	* a value of type `cudaError_t`. */
	checkCuda( cudaDeviceSynchronize() )
}
```