#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>

#include "options.cpp"

#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE MAX_THREADS_PER_BLOCK
#define WARP_SIZE 32

__constant__  static const float PHI = 1.618033;

/* this GPU kernel function is used to initialize the random states */
__global__ void initRandom(unsigned int seed, curandState_t* states)
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
__device__ void genRandoms(curandState_t* states, float* numbers, int count)
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = tid * count;
    for (int i = 0; i<count; i++)
    {
        numbers[row + i] = curand_uniform(&states[tid]);
    }
}

__device__ int integerPow(int num, int exponent)
{
    int result = 1;

    for (int i = 0; i < exponent; i++)
    {
        result *= num;
    }

    return result;
}

__global__ void martingale(float winProbability, curandState_t* states, float* spinData, int spinsPerRun, int bettingFactor = 2)
{
    genRandoms(states, spinData, spinsPerRun);

    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = tid * spinsPerRun;
    int purse = 0;
    int betSize = 1;
    int lossCount = 0;
    int winLossFactor[] = {1, -1};
    int totalLosses = 0;

    printf("Win probability: %f\n\n", winProbability);
    for (int i = 0; i < spinsPerRun; i++)
    {
        printf("!!Begin!! TID: %d -- Run: %d -- Purse: %d -- Bet: %d -- Losses: %d -- Spin: %f\n", tid, i, purse, betSize, lossCount, spinData[row+i]);
        int lostSpin = (spinData[row + i] >= winProbability);
        purse += winLossFactor[lostSpin] * betSize;

        lossCount = lossCount * lostSpin + lostSpin;
        totalLosses += (lossCount > 0);
        betSize = integerPow(bettingFactor, lossCount);
        printf("!!END  !! TID: %d -- Run: %d -- Purse: %d -- Bet: %d -- Losses: %d -- Spin: %f\n\n", tid, i, purse, betSize, lossCount, spinData[row+i]);
    }

    printf("Purse: %d\n", purse);
    printf("TotalLosses %d\n", totalLosses);
}

__global__ void dalembert(float winProbability, curandState_t* states, float* spinData, int spinsPerRun, int bettingFactor = 1)
{
    genRandoms(states, spinData, spinsPerRun);

    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = tid * spinsPerRun;
    int purse = 0;
    int betSize = bettingFactor;
    int lossCount = 0;
    int winLossFactor[] = {1, -1};

    printf("Win probability: %f\n\n", winProbability);
    for (int i = 0; i < spinsPerRun; i++)
    {
        printf("!!Begin!! TID: %d -- Run: %d -- Purse: %d -- Bet: %d -- Losses: %d -- Spin: %f\n", tid, i, purse, betSize, lossCount, spinData[row+i]);
        int lostSpin = (spinData[row + i] >= winProbability);
        int wonSpin = !(spinData[row + i] >= winProbability);
        purse += winLossFactor[lostSpin] * betSize;

        //lossCount = lossCount * lostSpin + lostSpin;
        lossCount = (lossCount + winLossFactor[wonSpin]);
        lossCount *= (lossCount > 0);
        betSize = bettingFactor + (bettingFactor * lossCount);
        printf("!!END  !! TID: %d -- Run: %d -- Purse: %d -- Bet: %d -- Losses: %d -- Spin: %f\n\n", tid, i, purse, betSize, lossCount, spinData[row+i]);
    }

    printf("Purse: %d\n", purse);
}

// http://www.maths.surrey.ac.uk/hosted-sites/R.Knott/Fibonacci/fibFormula.html
__device__ int calculateFibonacciNumber(int n)
{
    return round((pow(PHI, n) - pow(-PHI, -n)) / sqrtf(5));
}

__global__ void fibonacci(float winProbability, curandState_t* states, float* spinData, int spinsPerRun, int bettingFactor = 2)
{
    genRandoms(states, spinData, spinsPerRun);

    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = tid * spinsPerRun;
    int purse = 0;
    int betSize = 1;
    int lossCount = 0;
    int winLossFactor[] = {1, -1};
    int totalLosses = 0;

    printf("Win probability: %f\n\n", winProbability);
    for (int i = 0; i < spinsPerRun; i++)
    {
        printf("!!Begin!! TID: %d -- Run: %d -- Purse: %d -- Bet: %d -- Losses: %d -- Spin: %f\n", tid, i, purse, betSize, lossCount, spinData[row+i]);
        int lostSpin = (spinData[row + i] >= winProbability);
        purse += winLossFactor[lostSpin] * betSize;

        lossCount = lossCount * lostSpin + lostSpin;
        totalLosses += (lossCount > 0);
        betSize = bettingFactor * calculateFibonacciNumber(lossCount + 1);
        printf("!!END  !! TID: %d -- Run: %d -- Purse: %d -- Bet: %d -- Losses: %d -- Spin: %f\n\n", tid, i, purse, betSize, lossCount, spinData[row+i]);
    }

    printf("Purse: %d\n", purse);
    printf("TotalLosses %d\n", totalLosses);
}

curandState_t* initializeRandom(int numRuns)
{
  curandState_t* states;
  cudaMalloc((void**) &states, numRuns * sizeof(curandState_t));

  //init<<<numRuns / BLOCK_SIZE, BLOCK_SIZE>>>(time(0), states);
  initRandom<<<1, numRuns>>>(time(0), states);

  return states;
}

void playRoulette(int numRuns, int spinsPerRun, float winProbability, BettingStrategy strategy, int bettingFactor = 2)
{
    // Get the average of a set of random numbers
    auto start = std::chrono::high_resolution_clock::now();
    curandState_t* states = initializeRandom(numRuns);

    float * spinData;
    cudaMalloc((void**) &spinData, numRuns * spinsPerRun * sizeof(float));
    if (strategy == MARTINGALE)
    {
        martingale<<<1, numRuns>>>(winProbability, states, spinData, spinsPerRun, bettingFactor);
    }
    else if (strategy == DALEMBERT)
    {
        dalembert<<<1, numRuns>>>(winProbability, states, spinData, spinsPerRun, bettingFactor);
    }
    else if (strategy == FIBONACCI)
    {
        fibonacci<<<1, numRuns>>>(winProbability, states, spinData, spinsPerRun, bettingFactor);
    }
    cudaDeviceSynchronize();

    cudaFree(states);
    auto stop = std::chrono::high_resolution_clock::now();

    auto runTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("It took %d us \n", runTime);
}

// Main function
int main(int argc, char* argv[])
{
    ProgramOptions options = parseOptions(argc, argv);
    playRoulette(options.numRuns, options.spinsPerRun, options.winProbability, options.bettingStrategy, options.bettingFactor);
}
