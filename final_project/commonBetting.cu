typedef int (*betsize_calculator_function)(int bettingFactor, int lossCount);
typedef int (*loss_calculator_function)(int currentLossCount, int spinResult, int winLossFactor[]);

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
    int64_t integral = 0;

    for (int i = 0; i < spinsPerRun; i++)
    {
        //printf("!!Begin!! TID: %d -- Run: %d -- Purse: %d -- Bet: %d -- Losses: %d -- Spin: %f\n", tid, i, purse, betSize, lossCount, spinData[row+i]);
        int lostSpin = (spinData[row + i] >= winProbability);
        purse += winLossFactor[lostSpin] * betSize;
        integral += purse;

        lossCount = calcLossCount(lossCount, lostSpin, winLossFactor);
        totalLosses += lostSpin;
        betSize = calcBetSize(bettingFactor, lossCount);
        //printf("!!END  !! TID: %d -- Run: %d -- Purse: %d -- Bet: %d -- Losses: %d -- Spin: %f\n\n", tid, i, purse, betSize, lossCount, spinData[row+i]);
    }

    printf("The area under the curve is %d\n", integral);
}

__device__ int lossCountResetOnWin(int currentLossCount, int spinResult, int * /*[] winLossFactor */)
{
    return currentLossCount * spinResult + spinResult;
}

