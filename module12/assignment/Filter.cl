__kernel void filter (
	__global  uint * filter,
    const int filterSize,
    __global  float * const output,
    const int outputOffset)
{
    printf("Begin processing thread %d\n", outputOffset);
    float total = 0;
    for (int i = 0; i < filterSize; i++) {
        total += filter[i];
    }

    float average = total / filterSize;

    for (int i = 0; i < filterSize; i++) {
         output[(outputOffset * filterSize) + i] = average;
    }

    printf("Finished processing thread %d\n", outputOffset);
}
