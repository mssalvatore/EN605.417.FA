// Filter.cpp
//
//    This is a simple example that demonstrates OpenCL buffer/sub-buffer use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

///
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

void populateBuffer(cl_uint * buffer, size_t size)
{
    for (int i = 0; i < size; i++) {
        //buffer[i] = (i - (i%2)) + 1;
        buffer[i] = i;
    }
}

///
//	main() for filter example
//
int main(int argc, char** argv)
{
    const int BUFFER_SIZE = 16;
    const int FILTER_SIZE = 4;
    const int NUM_KERNELS = BUFFER_SIZE / FILTER_SIZE;
    cl_uint input[BUFFER_SIZE];
    cl_float output[BUFFER_SIZE];

    populateBuffer(input, BUFFER_SIZE);

    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_program program;
	cl_mem inputBuffer;
	cl_mem outputBuffer;
    cl_mem filters[NUM_KERNELS];
    cl_kernel kernels[NUM_KERNELS];
    cl_command_queue queues[NUM_KERNELS];

    // First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 
 
	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("Filter.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Filter.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context, 
		1, 
		&src, 
		&length, 
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
			buildLog, 
			NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

	// Now allocate buffers
	inputBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * BUFFER_SIZE,
		static_cast<void *>(input),
		&errNum);
	checkErr(errNum, "clCreateBuffer(input)");

	outputBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * BUFFER_SIZE,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

    //int x = 0;
    for (int x = 0; x < NUM_KERNELS; x++) {
        cl_buffer_region region = 
            {
                FILTER_SIZE * x * sizeof(int), 
                FILTER_SIZE * sizeof(int)
            };
        cl_mem filterBuffer = clCreateSubBuffer(
            inputBuffer,
            CL_MEM_READ_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");
        filters[x] = filterBuffer;
    }

    for (int x = 0; x< NUM_KERNELS; x++) {
	// Pick the first device and create command queue.
        cl_command_queue queue = clCreateCommandQueue(
                context,
                deviceIDs[0],
                0,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues[x] = queue;

        // Create kernel object
        cl_kernel kernel = clCreateKernel(
                program,
                "filter",
                &errNum);
        checkErr(errNum, "clCreateKernel");

        errNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &filters[x]);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &FILTER_SIZE);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuffer);
        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &x);
        checkErr(errNum, "clSetKernelArg");

        kernels[x] = kernel;
    }






	//errNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &filters[1]);
	//errNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &filters);
    //errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
	//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &FILTER_SIZE);
	//checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[1] = {1};
    const size_t localWorkSize[1]  = { 1 };

    // Queue the kernel up for execution across the array
    auto start = std::chrono::high_resolution_clock::now();
    for (int x = 0; x < NUM_KERNELS; x++) {
    errNum = clEnqueueNDRangeKernel(
            queues[x], 
            kernels[x], 
            1, 
            NULL,
            globalWorkSize, 
            localWorkSize,
            0, 
            NULL, 
            NULL);
        checkErr(errNum, "clEnqueueNDRangeKernel");
    }
    
	errNum = clEnqueueReadBuffer(
		queues[0], 
		outputBuffer, 
		CL_TRUE,
        0, 
		sizeof(cl_float) * BUFFER_SIZE, 
		output,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");
    auto stop = std::chrono::high_resolution_clock::now();
    int32_t runTime = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    std::cout << "Run time for the filter was " << runTime << " nanoseconds" << std::endl << std::endl;



    std::cout << std::endl << "Executed program succesfully." << std::endl;

    printf("INPUT: \n");
    for (int x = 0; x < BUFFER_SIZE; x++) {
        printf("%d ", input[x]);
    }
    printf("\n\nOUTPUT: \n");
    for (int x = 0; x < BUFFER_SIZE; x++) {
        printf("%f ", output[x]);
    }

    printf("\n\n");

	return 0;
}
