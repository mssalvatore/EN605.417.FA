//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// Convolution.cl
//
//    This is a simple kernel performing convolution.

__kernel void filter (
	__global  uint * filter,
    const int filterSize,
    __global  float * const output,
    const int outputOffset)
{
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
