#include <limits.h>
#include <cstddef>
#include <stdint.h>
template <class T>
double calculateMean(T * data, size_t size);

template <class T>
T calculateMax(T * data, size_t size);

template <class T>
T calculateMin(T * data, size_t size);

void runAnalytics(int * purse, int * maxPurse, int * minPurse, int64_t * integral, size_t size);
