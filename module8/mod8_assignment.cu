#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cublas.h>

#include <chrono>

#define BLOCK_SIZE 256
#define MAX 1000000

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

__global__ void randoms(curandState_t* states, float * numbers)
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    numbers[tid] = curand_uniform(&states[tid]) * MAX;
}

void genRandomNumbers(size_t numBlocks, int dataSetSize, float *randomArray)
{
  curandState_t* states;
  cudaMalloc((void**) &states, dataSetSize * sizeof(curandState_t));

  init<<<numBlocks, BLOCK_SIZE>>>(time(0), states);
  randoms<<<numBlocks, BLOCK_SIZE>>>(states, randomArray);

  cudaFree(states);
}

void runCublasExample(int dataSetSize)
{
    int numBlocks = dataSetSize / BLOCK_SIZE;

    float * randomArray;
    cudaMalloc((void**) &randomArray, dataSetSize * sizeof(float));
    genRandomNumbers(numBlocks, dataSetSize, randomArray);

    cublasStatus status;

    auto start = std::chrono::high_resolution_clock::now();
    float sum = cublasSasum(dataSetSize, randomArray, 1);
    auto stop = std::chrono::high_resolution_clock::now();

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! kernel execution error.\n");
      return;
    }

    auto runTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("The random data set of size %d has an absolute value sum of %f. It took %dus to calculate\n", dataSetSize, sum, runTime);
    cudaFree(randomArray);
}

// Main function
int main(int argc, char* argv[])
{
 
  runCublasExample(32768);
  runCublasExample(32768);
  runCublasExample(65536);
  runCublasExample(131072);
  return 0;
}
