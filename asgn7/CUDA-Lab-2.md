# Using nvprof and Unified Memory

The _[CUDA Best Practices Guide](http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)_, a highly recommended followup to this and other CUDA fundamentals labs


## Objectives

By the time you complete this lab, you will be able to:

- Use the **NVIDIA Command Line Profiler** ( **nprof** ) to profile accelerated application performance.
- Leverage an understanding of **Streaming Multiprocessors** to optimize execution configurations.
- Understand the behavior of **Unified Memory** with regard to page faulting and data migrations.
- Use **asynchronous memory prefetching** to reduce page faults and data migrations for increased performance.
- Employ an iterative development cycle to rapidly accelerate and deploy applications.


## Iterative Optimizations with the NVIDIA Command Line Profiler

The only way to be assured that attempts at optimizing accelerated code bases are actually successful is to profile the application for quantitative information about the application&#39;s performance. `nvprof` is the NVIDIA command line profiler. It ships with the CUDA toolkit, and is a powerful tool for profiling accelerated applications. Its most basic usage is to simply pass it the path to an executable compiled with `nvcc`. `nvprof` will proceed to execute the application, after which it will print a summary output of the application&#39;s GPU activities, CUDA API calls, as well as information about **Unified Memory** activity, a topic which will be covered extensively later in this lab.

When accelerating applications, or optimizing already-accelerated applications, take a scientific and iterative approach. Profile your application after making changes, take note, and record the implications of any refactoring on performance. Make these observations early and often: frequently, enough performance boost can be gained with little effort such that you can ship your accelerated application. Additionally, frequent profiling will teach you how specific changes to your CUDA codebases impact its actual performance: knowledge that is hard to acquire when only profiling after many kinds of changes in your codebase.


### Exercise: Profile an Application with nvprof

Use your VectorAddition Code.

After profiling the application, answer the following questions using information displayed in the profiling output:

- What was the name of the only CUDA kernel called in this application?
- How many times did this kernel run?
- How long did it take this kernel to run? Record this time somewhere: you will be optimizing this application and will want to know how much faster you can make it.

```
nvcc -o single-thread-vector-add 01-vector-add/01-vector-add.cu -run
nvprof ./single-thread-vector-add
```


### Exercise: Optimize Iteratively

- Find out if you code needs optimization. Start by listing 3 to 5 different ways you will update the execution configuration, being sure to cover a range of different grid and block size combinations.
- Edit the [vector-add.cu](http://ec2-18-188-127-217.us-east-2.compute.amazonaws.com/XJ6244SS/nbconvert/edit/02_AC_UM_NVPROF/01-vector-add/01-vector-add.cu) program in one of the ways you listed.
- Compile and profile your updated code with the two code execution cells below.
- Record the runtime of the kernel execution, as given in the profiling output.
- Repeat the edit/profile/record cycle for each possible optimzation you listed above

Which of the execution configurations you attempted proved to be the fastest?



## Streaming Multiprocessors and Querying the Device


### Streaming Multiprocessors and Warps

The GPUs that CUDA applications run on have processing units called **streaming multiprocessors** , or **SMs**. During kernel execution, blocks of threads are given to SMs to execute. SMs execute identical instructions on parallel threads within a block in groupings of 32 threads called **warps**.

Because of this feature of the hardware, block sizes containing a number of threads that are multiples of 32 often times yields performance gains, since it supports the SMs executing parallel instructions on as many threads at once as possible.

Additionally, in order to support the GPUs ability to perform as many parallel operations as possible, performance gains can often be had by choosing a grid size that has a number of blocks that is a multiple of the number of SMs on a given GPU.


### Programmatically Querying GPU Device Properties

In order to support portability, since the number of SMs on a GPU can differ depending on the specific GPU being used, the number of SMs should not be hard-coded into a codebase. Rather, this information should be acquired programatically. The following shows how, in CUDA C/C++, to obtain a C struct which contains many properties about the currently active GPU device, including its number of SMs:

```c
int deviceId;
cudaGetDevice(&deviceId);                  // `deviceId` now points to the id of the currently active GPU.

cudaDeviceProp props;
cudaGetDeviceProperties(&props, deviceId); // `props` now has many useful properties about the active GPU device.
```


### Exercise: Query the Device in the MPAC Lab




### Exercise: Optimize Vector Add with Grids Sized to Number of Sms and profile the difference

Utilize your ability to query the device for its number of SMs to refactor the `addVectorsInto` kernel you have been working on inside [01-vector-add.cu](http://ec2-18-188-127-217.us-east-2.compute.amazonaws.com/XJ6244SS/nbconvert/edit/02_AC_UM_NVPROF/01-vector-add/01-vector-add.cu) so that it launches with a grid containing a number of blocks that is a multiple of the number of SMs on the device.

Depending on other specific details in the code you have written, this refactor may or may not improve, or significantly change, the performance of your kernel. Therefore, as always, be sure to use `nvprof` so that you can quantitatively evaulate performance changes. Record the results with the rest of your findings thus far, based on the profiling output.

```
Add here a screen capture of the profiled code
```

## Unified Memory Details

You have been allocting memory intended for use either by host or device code with `cudaMallocManaged` and up until now have enjoyed the benefits of this method - automatic memory migration, ease of programming - without diving into the details of how the **Unified Memory** ( **UM** ) allocated by `cudaMallocManaged` actual works. `nvprof` provides details about UM management in accelerated applications, and using this information, in conjunction with a more-detailed understanding of how UM works, provides additional opportunities to optimize accelerated applications.


### Unified Memory Migration

When UM is allocated, the memory is not resident yet on either the host or the device. When either the host or device attempts to access the memory, a [page fault](https://en.wikipedia.org/wiki/Page_fault) will occur, at which point the host or device will migrate the needed data in batches. Similarly, at any point when the CPU, or any GPU in the accelerated system, attempts to access memory not yet resident on it, page faults will occur and trigger its migration.

The ability to page fault and migrate memory on demand is tremendously helpful for ease of development in your accelerated applications. Additionally, when working with data that exhibits sparse access patterns, for example when it is impossible to know which data will be required to be worked on until the application actually runs, and for scenarios when data might be accessed by multiple GPU devices in an accelerated system with multiple GPUs, on-demand memory migration is remarkably beneficial.

There are times - for example when data needs are known prior to runtime, and large contiguous blocks of memory are required - when the overhead of page faulting and migrating data on demand incurs an overhead cost that would be better avoided.

Much of the remainder of this lab will be dedicated to understanding on-demand migration, and how to identify it in the profiler&#39;s output. With this knowledge you will be able to reduce the overhead of it in scenarios when it would be beneficial.


### Exercise: Explore UM Page Faulting

`nvprof` provides output describing UM behavior for the profiled application. In this exercise, you will make several modifications to a simple application, and make use of `nvprof`&#39;s Unified Memory output section after each change, to explore how UM data migration behaves.

`Write a function that` contains a `hostFunction` and a `gpuKernel`, both which could be used to initialize the elements of a `2&lt;&lt;24` element vector with the number `1`.

- What happens when unified memory is accessed only by the CPU?
- What happens when unified memory is accessed only by the GPU?
- What happens when unified memory is accessed first by the CPU then the GPU?
- What happens when unified memory is accessed first by the GPU then the CPU?

```

```

### Exercise: Revisit UM Behavior for Vector Add Program

Returning to the vectoradd.cu program you have been working on throughout this lab, review the codebase in its current state, and hypothesize about what kinds of page faults you expect to occur. Look at the profiling output for your last refactor (either by scrolling up to find the output or by executing the code execution cell just below), observing the Unified Memory section of the profiler output. Can you explain the page faulting descriptions based on the contents of the code base?



### Write yur conclusions here



### Exercise: Initialize Vector in Kernel

When `nvprof` gives the amount of time that a kernel takes to execute, the host-to-device page faults and data migrations that occur during this kernel&#39;s execution are included in the displayed execution time.

With this in mind, refactor the `initWith` host function in your vector-add.cu program to instead be a CUDA kernel, initializing the allocated vector in parallel on the GPU. After successfully compiling and running the refactored application, but before profiling it, hypothesize about the following:

- How do you expect the refactor to affect UM page-fault behavior?
- How do you expect the refactor to affect the reported run time of `addVectorsInto`?



Write here the Memory Allocation Part of Host code



## Asynchronous Memory Prefetching

A powerful technique to reduce the overhead of page faulting and on-demand memory migrations, both in host-to-device and device-to-host memory transfers, is called **asynchronous memory prefetching**. Using this technique allows programmers to asynchronously migrate unified memory (UM) to any CPU or GPU device in the system, in the background, prior to its use by application code. By doing this, GPU kernels and CPU function performance can be increased on account of reduced page fault and on-demand data migration overhead.

Prefetching also tends to migrate data in larger chunks, and therefore fewer trips, than on-demand migration. This makes it an excellent fit when data access needs are known before runtime, and when data access patterns are not sparse.

CUDA Makes asynchronously prefetching managed memory to either a GPU device or the CPU easy with its `cudaMemPrefetchAsync` function. Here is an example of using it to both prefetch data to the currently active GPU device, and then, to the CPU:

```
int deviceId;
cudaGetDevice(&deviceId);                                         // The ID of the currently active GPU device.

cudaMemPrefetchAsync(pointerToSomeUMData, size, deviceId);        // Prefetch to GPU device.
cudaMemPrefetchAsync(pointerToSomeUMData, size, cudaCpuDeviceId); // Prefetch to host. `cudaCpuDeviceId` is a built-in CUDA variable.
```


### Exercise: Prefetch Memory

Conduct 3 experiments using `cudaMemPrefetchAsync` inside of your vector-add.cu application to understand its impact on page-faulting and memory migration.

- What happens when you prefetch one of the initialized vectors to the host?
- What happens when you prefetch two of the initialized vectors to the host?
- What happens when you prefetch all three of the initialized vectors to the host?



### Exercise: Prefetch Memory Back to the CPU

Add additional prefetching back to the CPU for the function that verifies the correctness of the `addVectorInto` kernel. Again, hypothesize about the impact on UM before profiling in `nvprof` to confirm.

Write here the Memory Allocation Part of Host code

