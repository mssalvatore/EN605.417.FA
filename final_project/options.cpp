#include "options.h"
#include <sstream>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include "InvalidArgumentException.h"
#include <getopt.h>

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
    int opt;

    ProgramOptions options;
    while ((opt = getopt(argc, argv, "f:b:t:n:p:m:s")) != -1)
    {
        switch (opt) {
            case 'f':
                options.fileName = optarg;
                break;
            case 'b':
                options.numBlocks = parseIntArgument(optarg);
                break;
            case 't':
        options.numThreads = parseIntArgument(optarg);
                break;
            case 'n':
        options.spinsPerRun = parseIntArgument(optarg);
                break;
            case 'p':
        options.winProbability = parseFloatArgument(optarg);
                break;
            case 'm':
        options.bettingFactor = parseIntArgument(optarg);
                break;
            case 's':
                if (strcmp(optarg, "martingale") == 0)
                {
                    options.bettingStrategy = MARTINGALE;
                }
                else if (strcmp(optarg, "dalembert") == 0)
                {
                    options.bettingStrategy = DALEMBERT;
                }
                else if (strcmp(optarg, "fibonacci") == 0)
                {
                    options.bettingStrategy = FIBONACCI;
                }
                else {
                    throw InvalidArgumentException(optarg, "betting strategy");
                }
                break;
        }
    }
    return options;
}
