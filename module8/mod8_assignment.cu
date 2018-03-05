#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cublas.h>

#include <chrono>
#include <cmath>

#define BLOCK_SIZE 256
#define MAX 1000000
#define CUBLAS_RUNS 20
#define CURAND_RUNS 16

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states)
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
            tid, /* the sequence number should be different for each core (unless you want all
                    cores to get the same sequence of numbers for some reason - use thread id! */
            0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
            &states[tid]);
}

// Generate numbers randomly
__global__ void genRandoms(curandState_t* states, float * numbers)
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    numbers[tid] = curand_uniform(&states[tid]) * MAX;
}

// Generate numbers in sequence
__global__ void sequence(float * numbers)
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    numbers[tid] = tid;
}

// Get the absolute value sum of dataArray using cuBLAS
int32_t getSum(int dataSetSize, float *dataArray, float * sum)
{

    // Calculate the sum of the absolute values of the data set elements
    cublasStatus status;
    auto start = std::chrono::high_resolution_clock::now();
    *sum = cublasSasum(dataSetSize, dataArray, 1);
    auto stop = std::chrono::high_resolution_clock::now();

    // If there was an error, print alert the user and return
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! kernel execution error.\n");
      return 0;
    }

    // Tell the user how long took to run the cuBLAS routine
    int32_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    // Free memory
    cudaFree(dataArray);

    return runTime;
}

// runCublasExample(int dataSetSize) generates a set of data of size dataSetSize. It populates
// the data set with sequential numbers on the interval [0, dataSetSize). It then uses the cuBLAS
// function cublasSasum() to get the sum of all values in the data set.
void runCublasExample(int dataSetSize)
{
    // calculate the number of blocks
    int numBlocks = dataSetSize / BLOCK_SIZE;

    // Allocate memory on the device
    float * dataArray;
    cudaMalloc((void**) &dataArray, dataSetSize * sizeof(float));

    // populate data set with values on the interval [0, dataSetSize)
    sequence<<<numBlocks, BLOCK_SIZE>>>(dataArray);

    float sum;
    int32_t runTime = getSum(dataSetSize, dataArray, &sum);
    printf("The random data set of size %d has an absolute value sum of %f. It took %dus to calculate\n", dataSetSize, sum, runTime);
}

// Generates random numbers and fills randomArray with the values
void genRandomNumbers(size_t numBlocks, int dataSetSize, float *randomArray)
{
  curandState_t* states;
  cudaMalloc((void**) &states, dataSetSize * sizeof(curandState_t));

  init<<<numBlocks, BLOCK_SIZE>>>(time(0), states);
  genRandoms<<<numBlocks, BLOCK_SIZE>>>(states, randomArray);

  cudaFree(states);
}

// Uses cuRAND to generate a random data set and finds the average using cuBLAS
void runCurandExample(int dataSetSize, int blockSize)
{
    int numBlocks = dataSetSize / blockSize;

    float sum;
    float * randomArray;
    cudaMalloc((void**) &randomArray, dataSetSize * sizeof(float));

    // Get the average of a set of random numbers
    auto start = std::chrono::high_resolution_clock::now();
    genRandomNumbers(numBlocks, dataSetSize, randomArray);
    getSum(dataSetSize, randomArray, &sum);
    float avg = sum / dataSetSize;
    auto stop = std::chrono::high_resolution_clock::now();

    auto runTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("It took %d us generate the random data set of size %d. A block size of %d was used. The avg of random values was %f\n", runTime, dataSetSize, blockSize, avg);
    cudaFree(randomArray);
}

// Main function
int main(int argc, char* argv[])
{
    // Use cuBLAS to find the sum of sequential numbers
    printf("Run cuBLAS kernel\n");
    runCublasExample(256);
    for (int x = 0; x < CUBLAS_RUNS; x++)
    {
        runCublasExample(256 << x);
    }

    printf("\n\n");
    printf("Random numbers are between 0 and %d\n\n", MAX);

    // Use cuRAND and cuBLAS to find the average of a random set of numbers. Use block size of BLOCK_SIZE for random number generation.
    int blockSize = BLOCK_SIZE;
    printf("Run cuRAND kernel with block size %d\n", blockSize);
    runCurandExample(256, blockSize);
    for (int x = 2; x < CURAND_RUNS; x++)
    {
        runCurandExample(256 << x, blockSize);
    }

    printf("\n\n");

    // Use cuRAND and cuBLAS to find the average of a random set of numbers. Use block size of 2 * BLOCK_SIZE for random number generation.
    blockSize = BLOCK_SIZE * 2;
    printf("Run cuRAND kernel with block size %d\n", blockSize);
    for (int x = 2; x < CURAND_RUNS; x++)
    {
        runCurandExample(256 << x, blockSize);
    }

    printf("\n\n");

    // Use cuRAND and cuBLAS to find the average of a random set of numbers. Use block size of 4 * BLOCK_SIZE for random number generation.
    blockSize = BLOCK_SIZE * 4;
    printf("Run cuRAND kernel with block size %d\n", blockSize);
    for (int x = 2; x < CURAND_RUNS; x++)
    {
        runCurandExample(256 << x, blockSize);
    }
    return 0;
}
