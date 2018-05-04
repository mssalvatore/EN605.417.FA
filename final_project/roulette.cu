#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>

#include "options.h"

#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE MAX_THREADS_PER_BLOCK
#define WARP_SIZE 32

__constant__  static const float PHI = 1.618033;

typedef int (*betsize_calculator_function)(int bettingFactor, int lossCount);
typedef int (*loss_calculator_function)(int currentLossCount, int spinResult, int winLossFactor[]);

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

__device__ void executeBettingStrategy(loss_calculator_function calcLossCount, betsize_calculator_function calcBetSize, float winProbability, curandState_t* states, float* spinData, int spinsPerRun, int bettingFactor = 2, int startingBet = 1)
{
    genRandoms(states, spinData, spinsPerRun);

    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = tid * spinsPerRun;
    int purse = 0;
    int betSize = startingBet;
    int lossCount = 0;
    int winLossFactor[] = {1, -1};
    int totalLosses = 0;

    printf("Win probability: %f\n\n", winProbability);
    printf("bettingFactor : %d\n\n", bettingFactor);
    for (int i = 0; i < spinsPerRun; i++)
    {
        printf("!!Begin!! TID: %d -- Run: %d -- Purse: %d -- Bet: %d -- Losses: %d -- Spin: %f\n", tid, i, purse, betSize, lossCount, spinData[row+i]);
        int lostSpin = (spinData[row + i] >= winProbability);
        purse += winLossFactor[lostSpin] * betSize;

        lossCount = calcLossCount(lossCount, lostSpin, winLossFactor);
        totalLosses += lostSpin;
        betSize = calcBetSize(bettingFactor, lossCount);
        printf("!!END  !! TID: %d -- Run: %d -- Purse: %d -- Bet: %d -- Losses: %d -- Spin: %f\n\n", tid, i, purse, betSize, lossCount, spinData[row+i]);
    }

    printf("Purse: %d\n", purse);
    printf("TotalLosses %d\n", totalLosses);
}

__device__ int lossCountResetOnWin(int currentLossCount, int spinResult, int * /*[] winLossFactor */)
{
    return currentLossCount * spinResult + spinResult;
}

__device__ int calculateDalembertLossCount(int currentLossCount, int spinResult, int winLossFactor[])
{
        int lossCount = (currentLossCount + winLossFactor[!spinResult]);
        return lossCount * (lossCount > 0);
}

__device__ int calculateDalembertBetSize(int bettingFactor, int lossCount)
{
        return bettingFactor + (bettingFactor * lossCount);
}

// http://www.maths.surrey.ac.uk/hosted-sites/R.Knott/Fibonacci/fibFormula.html
__device__ int calculateFibonacciBetSize(int bettingFactor, int lossCount)
{
    int n = lossCount + 1;
    return bettingFactor * (round((pow(PHI, n) - pow(-PHI, -n)) / sqrtf(5))) ;
}

__device__ int calculateMartingaleBetSize(int bettingFactor, int lossCount)
{
    return integerPow(bettingFactor, lossCount);
}

__global__ void dalembert(float winProbability, curandState_t* states, float* spinData, int spinsPerRun, int bettingFactor = 1)
{
    executeBettingStrategy(&calculateDalembertLossCount, &calculateDalembertBetSize, winProbability, states, spinData, spinsPerRun, bettingFactor);
}

__global__ void fibonacci(float winProbability, curandState_t* states, float* spinData, int spinsPerRun, int bettingFactor = 1)
{
    executeBettingStrategy(&lossCountResetOnWin, &calculateFibonacciBetSize, winProbability, states, spinData, spinsPerRun, bettingFactor);
}

__global__ void martingale(float winProbability, curandState_t* states, float* spinData, int spinsPerRun, int bettingFactor = 2)
{
    executeBettingStrategy(&lossCountResetOnWin, &calculateMartingaleBetSize, winProbability, states, spinData, spinsPerRun, bettingFactor);
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
