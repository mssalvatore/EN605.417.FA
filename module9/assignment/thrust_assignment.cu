#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <chrono>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/execution_policy.h>

const uint32_t MAX_DATA_SET_SIZE_PER_BLOCK = 1024;
const uint32_t DEFAULT_DATA_SIZE = 8192;

typedef uint32_t (*operation_function)(uint32_t operand1, uint32_t operand2);

// Reduction adapted from http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
__device__ void reduce(uint32_t *dataSet, uint32_t* sharedData, uint32_t *result, operation_function op)
{
    // Calculate indecies
    unsigned int threadId = threadIdx.x;
    unsigned int globalThreadId = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    
    // Perform the first reduction to get memory from global to shared
    sharedData[threadId] = op(dataSet[globalThreadId], dataSet[globalThreadId + blockDim.x]);
    __syncthreads();

    // Perform reduction using the provided operation_function
    for (unsigned int i=blockDim.x/2; i>0; i >>= 1)
    {
        if (threadId < i)
        {
            sharedData[threadId] = op(sharedData[threadId], sharedData[threadId + i]);
        }
        __syncthreads();
    }

    // Copy result back to global memory
    if (threadId == 0)
    {
        result[blockIdx.x] = sharedData[0];
    }
}

// Return the max of 2 elements without blocking (using a conditional)
__device__ uint32_t maximum(uint32_t element1, uint32_t element2)
{
    uint32_t findMax[2];
    findMax[0] = element1;
    findMax[1] = element2;

    return findMax[element1 < element2];
}

// Get the max of all elements in dataSet
__global__ void getMax(uint32_t* dataSet, uint32_t *result)
{
    extern __shared__ uint32_t sharedData[];
    reduce(dataSet, sharedData, result, &maximum);
}

uint32_t mod7GetMax(uint32_t dataSize, uint32_t numBlocks, uint32_t * data)
{
    // Allocate CPU memory
    int32_t numThreads = dataSize / (numBlocks * 2);
    uint32_t *cpuMaxResult = (uint32_t *)malloc(numBlocks * sizeof(uint32_t));

    // Allocate GPU memory
    uint32_t *gpuMaxResult;
    cudaMalloc((void **)&gpuMaxResult, numBlocks * sizeof(uint32_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    getMax<<<numBlocks, numThreads, numThreads * sizeof(uint32_t)>>>(data, gpuMaxResult);
    cudaMemcpy(cpuMaxResult, gpuMaxResult, numBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);

    uint32_t max = 0;
    for (size_t i = 0; i < numBlocks; i++)
    {
        if (cpuMaxResult[i] > max) {
            max = cpuMaxResult[i];
        }
    }

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free GPU memory
    cudaFree(gpuMaxResult);

    // Free CPU memory
    free(cpuMaxResult);

    return max;
}

uint32_t thrustGetMax(uint32_t dataSize, uint32_t * data)
{
    return thrust::reduce(thrust::cuda::par, data, data + dataSize, 0, thrust::maximum<uint32_t>());
}

uint32_t * generateDataSet(uint32_t dataSize)
{
    // Allocate pinned memory for data set
    uint32_t * data;
    cudaMallocHost((void**)&data, dataSize * sizeof(uint32_t));

    uint32_t range = dataSize * 4;
    for(size_t i = 0; i < dataSize; i++){ 
        data[i] = rand() % range + 1;
    }

    return data;
}

int main(int argc, char* argv[])
{
    srand((unsigned)time(NULL)); 

    uint32_t dataSize = DEFAULT_DATA_SIZE;
    uint32_t numBlocks = 8;

    // Check command line for number of blocks argument
    if (argc > 1) {
        numBlocks = atoi(argv[1]);

        if ((numBlocks % 2) != 0) {
            printf("Must enter a multiple of 2\n");
            return 1;
        }
        dataSize = numBlocks * MAX_DATA_SET_SIZE_PER_BLOCK ;
    }

    uint32_t * data = generateDataSet(dataSize);

    // Allocate device memory for data set
    uint32_t *gpuData;
    cudaMalloc((void **)&gpuData, dataSize * sizeof(uint32_t));
    cudaMemcpy(gpuData, data, dataSize * sizeof(uint32_t), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    uint32_t max = mod7GetMax(dataSize, numBlocks, gpuData);
    auto stop = std::chrono::high_resolution_clock::now();

    int32_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("Using module 7 reduction, a max of %d was found in %dus\n", max, runTime);

    start = std::chrono::high_resolution_clock::now();
    max = thrustGetMax(dataSize, gpuData);
    stop = std::chrono::high_resolution_clock::now();

    runTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("Using thurst reduction, a max of %d was found in %dus\n", max, runTime);

    cudaFreeHost(data);
    cudaFree(gpuData);
}
