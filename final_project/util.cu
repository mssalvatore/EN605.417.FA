#include <curand.h>
#include <curand_kernel.h>

/* this GPU kernel function is used to initialize the random states */
__global__ void initRandom(unsigned int seed, curandState_t* states)
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
            tid, /* the sequence number should be different for each core (unless you want all
                    cores to get the same sequence of numbers for some reason - use thread id! */
            0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
            &states[tid]);
}

__global__ void genRandoms(curandState_t* states, float* numbers, int count)
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = tid * count;
    for (int i = 0; i<count; i++)
    {
        numbers[row + i] = curand_uniform(&states[tid]);
    }
}

__device__ int integerPow(int num, int exponent)
{
    int result = 1;

    for (int i = 0; i < exponent; i++)
    {
        result *= num;
    }

    return result;
}

