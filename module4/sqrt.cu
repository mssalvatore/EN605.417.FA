// This program fills two arrays with numbers from 1 to N. One array is allocated in pageable
// memory and the other is allocated in pinned memory. The GPU is used to calculate the square root
// of each element in the array. A timer is used to measure the total execution time (including
// memory copy) of the paged vs pinned memory.

#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <math.h>
#include <chrono>

__global__ 
void cudaSqrt(float * data)
{
    const unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    data[index] = sqrt(data[index]);
}

uint64_t runKernel(float * data, uint64_t arraySize, uint32_t N)
{
    // Allocate global memory on device
    float *gpu_block;
    cudaMalloc((void **)&gpu_block, arraySize);

    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(gpu_block, data, arraySize, cudaMemcpyHostToDevice);
    cudaSqrt<<<(N+255)/256, 256>>>(gpu_block);
    cudaMemcpy(data, gpu_block, arraySize, cudaMemcpyDeviceToHost );
    auto stop = std::chrono::high_resolution_clock::now();

    cudaFree(gpu_block);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
}

int main()
{
    // Calculate the size of the data set
    uint64_t N = 1024*1024*1024;
    uint64_t arraySize = N * sizeof(float);

    // Allocate paged and pinned memory
    float * pagedData = (float *)malloc(arraySize);
    float * pinnedData;
    cudaMallocHost((void**)&pinnedData, arraySize);

    // Populate arrays with data
    for (int i = 0; i < N; i++) {
        pagedData[i] = i;
        pinnedData[i] = i;
    }

    // Run kernel with pageable host memory
    uint64_t totalPageableTimeNs = runKernel(pagedData, arraySize, N);
    std::cout << "Total time for pageable memcpy and execution: " <<  totalPageableTimeNs << "ns" << std::endl;

    // Run kernel with pinned host memory
    uint64_t totalPinnedTimeNs = runKernel(pinnedData, arraySize, N);
    std::cout << "Total time for pinned memcpy and execution:   " <<  totalPinnedTimeNs << "ns" << std::endl;

    std::cout << std::endl << "The total execution time using pageable memory is " << (double)totalPageableTimeNs / totalPinnedTimeNs << "x the total execution time using pinned memory." << std::endl;

    /* Free all memory on host and GPU */
    cudaFreeHost(pinnedData);
    free(pagedData);

    return EXIT_SUCCESS;
}
