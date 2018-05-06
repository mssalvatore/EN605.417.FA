#include "options.h"
#include <sstream>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include "InvalidArgumentException.h"

int parseIntArgument(char* argument)
{
    std::istringstream ss(argument);
    int x;

    if (!(ss >> x))
    {
        throw InvalidArgumentException(argument, "integer");
    }

    return x;
}

float parseFloatArgument(char* argument)
{
    float x = atof(argument);

    if (x < 0.0 || x > 1.0)
    {
        throw InvalidArgumentException(argument, "float between 0.0 and 1.0 inclusive");
    }

    return x;
}

ProgramOptions::ProgramOptions(int inNumBlocks, int inNumThreads, int inSpinsPerRun, float inWinProbability, int inBettingFactor, BettingStrategy inBettingStrategy): numBlocks(inNumBlocks), numThreads(inNumThreads), spinsPerRun(inSpinsPerRun), winProbability(inWinProbability), bettingFactor(inBettingFactor), bettingStrategy(inBettingStrategy) {}

ProgramOptions parseOptions(int argc, char* argv[])
{
    ProgramOptions options;
    if (argc >= 2)
    {
        options.numBlocks = parseIntArgument(argv[1]);
    }
    if (argc >= 3)
    {
        options.numThreads = parseIntArgument(argv[2]);
    }
    if (argc >= 4)
    {
        options.spinsPerRun = parseIntArgument(argv[3]);
    }
    if (argc >= 5)
    {
        options.winProbability = parseFloatArgument(argv[4]);
    }
    if (argc >= 6)
    {
        options.bettingFactor = parseIntArgument(argv[5]);
    }
    if (argc >= 7)
    {
        {
            if (strcmp(argv[6], "martingale") == 0)
            {
                options.bettingStrategy = MARTINGALE;
            }
            else if (strcmp(argv[6], "dalembert") == 0)
            {
                options.bettingStrategy = DALEMBERT;
            }
            else if (strcmp(argv[6], "fibonacci") == 0)
            {
                options.bettingStrategy = FIBONACCI;
            }
            else {
                throw InvalidArgumentException(argv[5], "betting strategy");
            }
        }
    }

    return options;
}
