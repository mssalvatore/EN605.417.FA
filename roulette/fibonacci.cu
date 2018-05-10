#include <curand.h>
#include <curand_kernel.h>

__constant__  static const float PHI = 1.618033;

// http://www.maths.surrey.ac.uk/hosted-sites/R.Knott/Fibonacci/fibFormula.html
__device__ int calculateFibonacciBetSize(int bettingFactor, int lossCount)
{
    int n = lossCount + 1;
    return bettingFactor * (round((pow(PHI, n) - pow(-PHI, -n)) / sqrtf(5))) ;
}

__global__ void fibonacci(int * outPurse, int * outMaxPurse, int * outMinPurse, int64_t * outIntegral, float winProbability, float* spinData, int spinsPerRun, int bettingFactor = 1)
{
    executeBettingStrategy(outPurse, outMaxPurse, outMinPurse, outIntegral, &lossCountResetOnWin, &calculateFibonacciBetSize, winProbability, spinData, spinsPerRun, bettingFactor);
}

