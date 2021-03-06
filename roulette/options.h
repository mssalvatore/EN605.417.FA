enum BettingStrategy { MARTINGALE, DALEMBERT, FIBONACCI };

struct ProgramOptions
{
    public:
        int numBlocks;
        int numThreads;
        int spinsPerRun;
        float winProbability;
        int bettingFactor;
        BettingStrategy bettingStrategy;
        char* fileName;

        ProgramOptions(int inNumBlocks = 1, int inNumRuns = 10, int inSpinsPerRun = 100, float inWinProbability = .486486, int inBettingFactor = 2, BettingStrategy inBettingStrategy = MARTINGALE);
};

int parseIntArgument(char* argument);
float parseFloatArgument(char* argument);
ProgramOptions parseOptions(int argc, char* argv[]);
void showUsage();
