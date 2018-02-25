#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <chrono>

const uint32_t MAX_DATA_SET_SIZE_PER_BLOCK = 1024;
const uint32_t DEFAULT_DATA_SIZE = 8192;

typedef uint32_t (*operation_function)(uint32_t operand1, uint32_t operand2);

// Reduction adapted from http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
__device__ void reduce(uint32_t *dataSet, uint32_t* sharedData, uint32_t *result, operation_function op)
{
    unsigned int threadId = threadIdx.x;
    unsigned int globalThreadId = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sharedData[threadId] = op(dataSet[globalThreadId], dataSet[globalThreadId + blockDim.x]);
    __syncthreads();

    for (unsigned int i=blockDim.x/2; i>0; i >>= 1)
    {
        if (threadId < i)
        {
            sharedData[threadId] = op(sharedData[threadId], sharedData[threadId + i]);
        }
        __syncthreads();
    }

    if (threadId == 0)
    {
        result[blockIdx.x] = sharedData[0];
    }
}

__device__ uint32_t add(uint32_t operand1, uint32_t operand2)
{
    return operand1 + operand2;
}

__device__ uint32_t maximum(uint32_t operand1, uint32_t operand2)
{
    uint32_t findMax[2];
    findMax[0] = operand1;
    findMax[1] = operand2;

    return findMax[operand1 < operand2];
}

__device__ uint32_t minimum(uint32_t operand1, uint32_t operand2)
{
    uint32_t findMax[2];
    findMax[0] = operand1;
    findMax[1] = operand2;

    return findMax[operand1 > operand2];
}

__global__ void sum(uint32_t* dataSet, uint32_t *result)
{
    extern __shared__ uint32_t sharedData[];
    reduce(dataSet, sharedData, result, &add);
}

__global__ void getMax(uint32_t* dataSet, uint32_t *result)
{
    extern __shared__ uint32_t sharedData[];
    reduce(dataSet, sharedData, result, &maximum);
}

__global__ void getMin(uint32_t* dataSet, uint32_t *result)
{
    extern __shared__ uint32_t sharedData[];
    reduce(dataSet, sharedData, result, &minimum);
}

void getMaxMinAvg(uint32_t dataSize, uint32_t numBlocks, uint32_t * data, uint32_t *max, uint32_t *min, double *avg)
{
    int32_t numThreads = dataSize / (numBlocks * 2);
    uint32_t *cpuMaxResult = (uint32_t *)malloc(numBlocks * sizeof(uint32_t));
    uint32_t *cpuMinResult = (uint32_t *)malloc(numBlocks * sizeof(uint32_t));
    uint32_t *cpuSumResult = (uint32_t *)malloc(numBlocks * sizeof(uint32_t));

    uint32_t *gpuMaxResult;
    cudaMalloc((void **)&gpuMaxResult, numBlocks * sizeof(uint32_t));
    uint32_t *gpuMinResult;
    cudaMalloc((void **)&gpuMinResult, numBlocks * sizeof(uint32_t));
    uint32_t *gpuSumResult;
    cudaMalloc((void **)&gpuSumResult, numBlocks * sizeof(uint32_t));

    cudaEvent_t gotMax, gotMin, gotSum;
    cudaEventCreate(&gotMax);
    cudaEventCreate(&gotMin);
    cudaEventCreate(&gotSum);

    cudaStream_t maxStream, minStream, sumStream;
    cudaStreamCreate(&maxStream);
    cudaStreamCreate(&minStream);
    cudaStreamCreate(&sumStream);

    auto start = std::chrono::high_resolution_clock::now();
    getMax<<<numBlocks, numThreads, numThreads * sizeof(uint32_t), maxStream>>>(data, gpuMaxResult);
    cudaMemcpyAsync(cpuMaxResult, gpuMaxResult, numBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost, maxStream);
    cudaEventRecord(gotMax, maxStream);

    getMin<<<numBlocks, numThreads, numThreads * sizeof(uint32_t), minStream>>>(data, gpuMinResult);
    cudaMemcpyAsync(cpuMinResult, gpuMinResult, numBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost, minStream);
    cudaEventRecord(gotMin, minStream);

    sum<<<numBlocks, numThreads, numThreads * sizeof(uint32_t), sumStream>>>(data, gpuSumResult);
    cudaMemcpyAsync(cpuSumResult, gpuSumResult, numBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost, sumStream);
    cudaEventRecord(gotSum, sumStream);

    bool maxFinished = false;
    bool minFinished = false;
    bool sumFinished = false;

    while (!maxFinished && !minFinished && !sumFinished)
    {
        if (!maxFinished && (cudaEventQuery(gotMax) == cudaSuccess))
        {
            auto now = std::chrono::high_resolution_clock::now();
            printf("Get Maximum finished after %dus\n", std::chrono::duration_cast<std::chrono::microseconds>(now - start).count());
            maxFinished = true;
        }
        if (!minFinished && (cudaEventQuery(gotMin) == cudaSuccess))
        {
            auto now = std::chrono::high_resolution_clock::now();
            printf("Get Minimum finished after %dus\n", std::chrono::duration_cast<std::chrono::microseconds>(now - start).count());
            minFinished = true;
        }
        if (!sumFinished && (cudaEventQuery(gotSum) == cudaSuccess))
        {
            auto now = std::chrono::high_resolution_clock::now();
            printf("Get Sum finished after %dus\n", std::chrono::duration_cast<std::chrono::microseconds>(now - start).count());
            sumFinished = true;
        }
    }
    cudaDeviceSynchronize();

    *max = 0;
    *min = UINT_MAX;
    double sum = 0;

    for (size_t i = 0; i < numBlocks; i++)
    {
        sum += cpuSumResult[i];

        if (cpuMaxResult[i] > *max) {
            *max = cpuMaxResult[i];
        }
        if (cpuMinResult[i] < *min) {
            *min = cpuMinResult[i];
        }
    }

    *avg = sum / dataSize;

    cudaEventDestroy(gotMax);
    cudaEventDestroy(gotMin);
    cudaEventDestroy(gotSum);

    cudaStreamDestroy(maxStream);
    cudaStreamDestroy(minStream);
    cudaStreamDestroy(sumStream);

    cudaFree(gpuMaxResult);
    cudaFree(gpuMinResult);
    cudaFree(gpuSumResult);
}

int main(int argc, char* argv[])
{
    uint32_t dataSize = DEFAULT_DATA_SIZE;
    uint32_t numBlocks = 8;
    uint32_t numRuns = 2;
    if (argc > 1) {
        numBlocks = atoi(argv[1]);

        if ((numBlocks % 2) != 0) {
            printf("Must enter a multiple of 2\n");
            return 1;
        }
        dataSize = numBlocks * MAX_DATA_SET_SIZE_PER_BLOCK ;
    }

    if (argc > 2) {
        numRuns = atoi(argv[2]);
    }

    uint32_t * data;
    cudaMallocHost((void**)&data, dataSize * sizeof(uint32_t));

    uint32_t *gpuData;
    cudaMalloc((void **)&gpuData, dataSize * sizeof(uint32_t));

    srand((unsigned)time(NULL)); 
    for (uint32_t x = 0; x < numRuns; x++)
    {
        printf("Run %d\n--------------------------------------------------\n", x);

        uint32_t range = dataSize * 4;
        for(size_t i = 0; i < dataSize; i++){ 
            data[i] = rand() % range + 1;
        }

        cudaMemcpy(gpuData, data, dataSize * sizeof(uint32_t), cudaMemcpyHostToDevice);

        double avg;
        uint32_t max;
        uint32_t min;
        getMaxMinAvg(dataSize, numBlocks, gpuData, &max, &min, &avg);

        printf("\n\n");
        printf("Average is %f\n", avg);
        printf ("Max is %d\n", max);
        printf ("Min is %d\n", min);
        printf("\n\n\n");
    }

    cudaFreeHost(data);
    cudaFree(gpuData);
}
