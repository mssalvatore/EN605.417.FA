#include "analytics.h"
#include <iostream>

template <class T>
void printArray(T * data, size_t size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout<<data[i] << " ";
    }
    std::cout<<std::endl;
}

template <class T>
double calculateMean(T * data, size_t size)
{
    double total = 0;
    for (int i = 0; i < size; i++)
    {
        total += (((double)data[i]) / size);
    }

    return total;
}

template <class T>
T calculateMax(T * data, size_t size)
{
    T max = INT_MIN;
    for (int i = 0; i < size; i++)
    {
        if (data[i] > max)
        {
            max = data[i];
        }
    }

    return max;
}

    template <class T>
T calculateMin(T * data, size_t size)
{
    T min = INT_MAX;
    for (int i = 0; i < size; i++)
    {
        if (data[i] < min)
        {
            min = data[i];
        }
    }

    return min;
}
 
void runAnalytics(int * purse, int * maxPurse, int * minPurse, int64_t * integral, size_t size)
{
    //printArray(purse, size);
    //printArray(maxPurse, size);
    //printArray(minPurse, size);
    //printArray(integral, size);

    std::cout<<"Avg Purse: " << calculateMean(purse, size) << std::endl;
    std::cout<<"Max Purse: " << calculateMax(purse, size) << std::endl;
    std::cout<<"Min Purse: " << calculateMin(purse, size) << std::endl;

    std::cout<<"Max Max Purse: " << calculateMax(maxPurse, size) <<std::endl;
    std::cout<<"Min Min Purse: " << calculateMin(minPurse, size) <<std::endl;

    std::cout<<"Avg Integral: " << calculateMean(integral, size) << std::endl;
    std::cout<<"Max Integral: " << calculateMax(integral, size) << std::endl;
    std::cout<<"Min Integral: " << calculateMin(integral, size) << std::endl;
}
