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
	const __global  uint * const input,
	const __global  uint * const filter,
    __global  float * const output,
    const int inputWidth)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    //printf("(%d, %d) -- %d\n", row, col, input[x][y]);
    printf("(%d, %d) -- %d\n", row, col, filter[(row * 2)  + col]);
}
