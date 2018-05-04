#include <curand.h>
#include <curand_kernel.h>

__device__ int calculateMartingaleBetSize(int bettingFactor, int lossCount)
{
    return integerPow(bettingFactor, lossCount);
}

__global__ void martingale(float winProbability, curandState_t* states, float* spinData, int spinsPerRun, int bettingFactor = 2)
{
    executeBettingStrategy(&lossCountResetOnWin, &calculateMartingaleBetSize, winProbability, states, spinData, spinsPerRun, bettingFactor);
}

