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

__kernel void convolve(
	const __global  uint * const input,
    __constant uint * const mask,
    __global  float * const output,
    const int inputWidth,
    const int maskWidth)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float maskCenter = maskWidth / 2.0;

    float sum = 0;
    for (int r = 0; r < maskWidth; r++)
    {
        const int idxIntmp = (y + r) * inputWidth + x;

        for (int c = 0; c < maskWidth; c++)
        {
            float xDist = pow((float) (c - maskCenter), 2);
            float yDist = pow((float) (r - maskCenter), 2);
            float distance = sqrt(xDist + yDist);

            float gradient = 0;
            if (distance <= 1) {
                gradient = 1;
            }
            else if (distance <= 2) {
                gradient = .75;
            } else if (distance <= 3) {
                gradient = .50;
            } else if (distance <= 4) {
                gradient = .25;
            }
			sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c] * gradient;
        }
    } 
    
	output[y * get_global_size(0) + x] = sum;
}
