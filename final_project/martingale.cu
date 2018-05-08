#include <curand.h>
#include <curand_kernel.h>

__device__ int calculateMartingaleBetSize(int bettingFactor, int lossCount)
{
    return integerPow(bettingFactor, lossCount);
}

__global__ void martingale(int * outPurse, int * outMaxPurse, int * outMinPurse, int64_t * outIntegral, float winProbability, float* spinData, int spinsPerRun, int bettingFactor = 2)
{
    executeBettingStrategy(outPurse, outMaxPurse, outMinPurse, outIntegral, &lossCountResetOnWin, &calculateMartingaleBetSize, winProbability, spinData, spinsPerRun, bettingFactor);
}

