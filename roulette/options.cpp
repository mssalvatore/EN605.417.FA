#include "options.h"
#include <sstream>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <iostream>
#include "InvalidArgumentException.h"
#include "IncompatibleArgumentException.h"
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

ProgramOptions::ProgramOptions(int inNumBlocks, int inNumThreads, int inSpinsPerRun, float inWinProbability, int inBettingFactor, BettingStrategy inBettingStrategy): numBlocks(inNumBlocks), numThreads(inNumThreads), spinsPerRun(inSpinsPerRun), winProbability(inWinProbability), bettingFactor(inBettingFactor), bettingStrategy(inBettingStrategy), fileName(0) {}

void showUsage()
{
    std::cout<<"Run a roulette simulation using the Martingale, D'Alembert, or Fibonacci betting strategies."<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Usage:"<<std::endl;
    std::cout<<"\troulette"<<std::endl;
    std::cout<<"\troulette -h"<<std::endl;
    std::cout<<"\troulette -f <inputFile> -b <numBlocks> -t <numThreads> -n <numSpins> -p <probabilityOfWinning> -m <bettingFactor> -s <strategy>"<<std::endl;
    std::cout<<"\troulette -b <numBlocks> -t <numThreads> -n <numSpins> -p <probabilityOfWinning> -m <bettingFactor> -s <strategy>"<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Usage:"<<std::endl;
    std::cout<<"\t-h\tShow help."<<std::endl;
    std::cout<<"\t-f\tAn input file path containing spin data. This option is not compatible with the '-p' option."<<std::endl;
    std::cout<<"\t-b\tThe number of GPU blocks."<<std::endl;
    std::cout<<"\t-t\tThe number of GPU threads per block (max 1024)."<<std::endl;
    std::cout<<"\t-n\tThe number of spins per each simulation."<<std::endl;
    std::cout<<"\t-p\tThe probability of winning (a float between 0.0 and 1.0 inclusive). This option is not compatible with the '-f' option."<<std::endl;
    std::cout<<"\t-m\tThe betting factor. This is a multiplier that affects the size of each bet."<<std::endl;
    std::cout<<"\t-s\tThe betting strategy to use. Options are \"martingale\", \"dalembert\", or \"fibonacci\""<<std::endl;
    std::cout<<std::endl;

    exit(0);
}
ProgramOptions parseOptions(int argc, char* argv[])
{
    ProgramOptions options;
    bool fileFlag, probabilityFlag = false;

    try
    {
        int opt;
        while ((opt = getopt(argc, argv, "hf:b:t:n:p:m:s:")) != -1)
        {
            switch (opt)
            {
                case 'h':
                    showUsage();
                    break;
                case 'f':
                    fileFlag = true;
                    if (probabilityFlag)
                    {
                        throw IncompatibleArgumentException("The -p and -f flags may not be used together.");
                    }
                    options.fileName = optarg;
                    break;
                case 'b':
                    options.numBlocks = parseIntArgument(optarg);
                    break;
                case 't':
                    options.numThreads = parseIntArgument(optarg);
                    if (options.numThreads <= 0 || options.numThreads > 1024)
                    {
                        throw InvalidArgumentException(optarg, "integer between 1 and 1024 inclusive");
                    }
                    break;
                case 'n':
                    options.spinsPerRun = parseIntArgument(optarg);
                    break;
                case 'p':
                    probabilityFlag = true;
                    if (fileFlag)
                    {
                        throw IncompatibleArgumentException("The -p and -f flags may not be used together.");
                    }
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
                    else
                    {
                        throw InvalidArgumentException(optarg, "betting strategy");
                    }
                    break;
            }
        }
    }
    catch(InvalidArgumentException ex)
    {
        std::cerr << "ERROR: " << ex.what() <<std::endl<<std::endl;
        showUsage();
    }
    catch(IncompatibleArgumentException ex)
    {
        std::cerr << "ERROR: " << ex.what() <<std::endl<<std::endl;
        showUsage();
    }

    return options;
}
