#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
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
#include "analytics.h"

#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE MAX_THREADS_PER_BLOCK
#define WARP_SIZE 32
#define NUM_ROULETTE_SLOTS 37

curandState_t* initializeRandom(int numRuns)
{
  curandState_t* states;
  cudaMalloc((void**) &states, numRuns * sizeof(curandState_t));

  //init<<<numRuns / BLOCK_SIZE, BLOCK_SIZE>>>(time(0), states);
  initRandom<<<1, numRuns>>>(time(0), states);

  return states;
}

template <class T>
void printArray(T * data, size_t size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout<<data[i] << " ";
    }
    std::cout<<std::endl;
}

void playRoulette(float * spinData, int numBlocks, int numThreads, int spinsPerRun, float winProbability, BettingStrategy strategy, int bettingFactor = 2)
{
    int * gpuPurse;
    int * gpuMaxPurse;
    int * gpuMinPurse;
    int64_t * gpuIntegral;
    cudaMalloc((void**)&gpuPurse, numBlocks * numThreads * sizeof(int));
    cudaMalloc((void**)&gpuMaxPurse, numBlocks * numThreads * sizeof(int));
    cudaMalloc((void**)&gpuMinPurse, numBlocks * numThreads * sizeof(int));
    cudaMalloc((void**)&gpuIntegral, numBlocks * numThreads * sizeof(int64_t));

    auto start = std::chrono::high_resolution_clock::now();
    if (strategy == MARTINGALE)
    {
        martingale<<<numBlocks, numThreads>>>(gpuPurse, gpuMaxPurse, gpuMinPurse, gpuIntegral, winProbability, spinData, spinsPerRun, bettingFactor);
    }
    else if (strategy == DALEMBERT)
    {
        dalembert<<<numBlocks, numThreads>>>(gpuPurse, gpuMaxPurse, gpuMinPurse, gpuIntegral, winProbability, spinData, spinsPerRun, bettingFactor);
    }
    else if (strategy == FIBONACCI)
    {
        fibonacci<<<numBlocks, numThreads>>>(gpuPurse, gpuMaxPurse, gpuMinPurse, gpuIntegral, winProbability, spinData, spinsPerRun, bettingFactor);
    }
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();

    auto runTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("It took %d us \n", runTime);

    int * purse = (int*) malloc(numBlocks * numThreads * sizeof(int));
    int * maxPurse = (int*) malloc(numBlocks * numThreads * sizeof(int));
    int * minPurse = (int*) malloc(numBlocks * numThreads * sizeof(int));
    int64_t * integral = (int64_t*) malloc(numBlocks * numThreads * sizeof(int64_t));

    cudaMemcpy(purse, gpuPurse, numBlocks * numThreads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(maxPurse, gpuMaxPurse, numBlocks * numThreads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(minPurse, gpuMinPurse, numBlocks * numThreads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(integral, gpuIntegral, numBlocks * numThreads * sizeof(int64_t), cudaMemcpyDeviceToHost);

    cudaFree(gpuPurse);
    cudaFree(gpuMaxPurse);
    cudaFree(gpuMinPurse);
    cudaFree(gpuIntegral);

    runAnalytics(purse, maxPurse, minPurse, integral, numBlocks * numThreads);
}

float * prepareRandomData(int numBlocks, int numThreads, int spinsPerRun)
{
    curandState_t* states = initializeRandom(numBlocks * numThreads);

    float * spinData;
    cudaMalloc((void**) &spinData, numBlocks * numThreads * spinsPerRun * sizeof(float));

    genRandoms<<<numBlocks, numThreads>>>(states, spinData, spinsPerRun);

    cudaFree(states);
    return spinData;
}

float * prepareRealData(char * fileName)
{
    std::ifstream file(fileName);
    if (!file) {
        std::cout << "Error opening input file " << fileName << std::endl;
        exit(1);
    }

    float *gpuRealSpins;
    float *realSpins;
    int numBlocks;
    int numThreads;
    std::vector<int> realSpinsFromFile = readLineFromFile(&file);
    size_t numRealSpins = realSpinsFromFile.size();

    getDimensions(&numBlocks, &numThreads, MAX_THREADS_PER_BLOCK, numRealSpins);

    cudaMallocHost((void**)&realSpins, realSpinsFromFile.size() * sizeof(float));
    std::copy(realSpinsFromFile.begin(), realSpinsFromFile.end(), realSpins);

    cudaMemcpyToSymbol(cudaColorTranslation, &hostColorTranslation, NUM_ROULETTE_SLOTS * sizeof(float));
    cudaMalloc((void **)&gpuRealSpins, numRealSpins * sizeof(float));
    cudaMemcpy(gpuRealSpins, realSpins, numRealSpins * sizeof(float), cudaMemcpyHostToDevice);

    translateSpinsToColors<<<numBlocks,numThreads>>>(gpuRealSpins);

    return gpuRealSpins;
}

// Main function
int main(int argc, char* argv[])
{
    ProgramOptions options = parseOptions(argc, argv);
    /*
    std::cout<<"numBlocks " << options.numBlocks <<std::endl;
    std::cout<<"numThreads " << options.numThreads <<std::endl;
    std::cout<<"spinsPerRun " << options.spinsPerRun <<std::endl;
    std::cout<<"winProbability " << options.winProbability <<std::endl;
    std::cout<<"bettingFactor " << options.bettingFactor <<std::endl;
    std::cout<<"bettingStrategy " << options.bettingStrategy <<std::endl;
    std::cout<<"fileName " << options.fileName <<std::endl;
    */
    float * spinData;
    if (options.fileName)
    {
        spinData = prepareRealData(options.fileName);
    }
    else {
        spinData = prepareRandomData(options.numBlocks, options.numThreads, options.spinsPerRun);
    }
    playRoulette(spinData, options.numBlocks, options.numThreads, options.spinsPerRun, options.winProbability, options.bettingStrategy, options.bettingFactor);

    cudaFree(spinData);
    exit(0);
}
