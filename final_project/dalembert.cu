#include <curand.h>
#include <curand_kernel.h>

__device__ int calculateDalembertLossCount(int currentLossCount, int spinResult, int winLossFactor[])
{
        int lossCount = (currentLossCount + winLossFactor[!spinResult]);
        return lossCount * (lossCount > 0);
}

__device__ int calculateDalembertBetSize(int bettingFactor, int lossCount)
{
        return bettingFactor + (bettingFactor * lossCount);
}

__global__ void dalembert(int * outPurse, int * outMaxPurse, int * outMinPurse, int64_t * outIntegral, float winProbability, float* spinData, int spinsPerRun, int bettingFactor = 1)
{
    executeBettingStrategy(outPurse, outMaxPurse, outMinPurse, outIntegral, &calculateDalembertLossCount, &calculateDalembertBetSize, winProbability, spinData, spinsPerRun, bettingFactor);
}

