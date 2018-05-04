enum BettingStrategy { MARTINGALE, DALEMBERT, FIBONACCI };

struct ProgramOptions
{
    public:
        int numRuns;
        int spinsPerRun;
        float winProbability;
        int bettingFactor;
        BettingStrategy bettingStrategy;

        ProgramOptions(int inNumRuns = 1, int inSpinsPerRun = 100, float inWinProbability = .4737, int inBettingFactor = 2, BettingStrategy inBettingStrategy = MARTINGALE);
};

int parseIntArgument(char* argument);
float parseFloatArgument(char* argument);
ProgramOptions parseOptions(int argc, char* argv[]);
