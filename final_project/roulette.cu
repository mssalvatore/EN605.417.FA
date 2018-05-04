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
#include "util.cu"
#include "commonBetting.cu"
#include "dalembert.cu"
#include "fibonacci.cu"
#include "martingale.cu"

#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE MAX_THREADS_PER_BLOCK
#define WARP_SIZE 32

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
