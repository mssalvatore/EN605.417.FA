//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <stdio.h>
#include <string.h>
#include <cstring>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

size_t readLineFromFile(int * outValues, std::ifstream * file)
{
    std::string line;
    if (! std::getline(*file, line)) {
        return 0;
    }

    std::vector<int> nums;
    char * dup = strdup(line.c_str());
    char delim[] = " ";
    char * token = std::strtok(dup, delim);
    while (token != NULL) {
        nums.push_back(atoi(token));
        token = std::strtok(NULL, delim);
    }

    outValues = new int[nums.size()];
    std::copy(nums.begin(), nums.end(), outValues);

    return nums.size();
    
}

void enqueueCommand(int * inputOutput, size_t inputSize, cl_uint numDevices, cl_context context, cl_command_queue queue, cl_kernel kernel, cl_event* event)
{
    cl_int errNum;

    // create a single buffer to cover all the input data
    cl_mem buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(int) * inputSize * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer);
    checkErr(errNum, "clSetKernelArg(square)");

    // Write input data
    errNum = clEnqueueWriteBuffer(
      queue,
      buffer,
      CL_TRUE,
      0,
      sizeof(int) * inputSize * numDevices,
      (void*)inputOutput,
      0,
      NULL,
      NULL);

    errNum = clEnqueueNDRangeKernel(
      queue, 
      kernel, 
      1, 
      NULL,
      (const size_t*)&inputSize, 
      (const size_t*)NULL, 
      0, 
      0, 
      event);

 	errNum = clEnqueueWaitForEvents(queue, 1, event);

   	clEnqueueReadBuffer(
            queue,
            buffer,
            CL_TRUE,
            0,
            sizeof(int) * inputSize * numDevices,
            (void*)inputOutput,
            0,
            NULL,
            event);
}

void printInts(int * input, size_t inputSize) {
    // Display output in rows
    for (unsigned elems = 0; elems < inputSize; elems++)
    {
     std::cout << " " << input[elems];
    }
    std::cout << std::endl;
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context0;
    cl_program program0;
    int * inputOutput0;
    cl_context context1;
    cl_program program1;
    int * inputOutput1;

    int platform = DEFAULT_PLATFORM; 

    std::cout << "Simple events/queues Example" << std::endl;

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("square.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading square.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[platform], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[platform], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    context0 = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    // Create program from source
    program0 = clCreateProgramWithSource(
        context0, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");
 
    // Build program
    errNum = clBuildProgram(
        program0,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
 
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog0[16384];
        clGetProgramBuildInfo(
            program0, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog0), 
            buildLog0, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog0;
            checkErr(errNum, "clBuildProgram");
    }

    // create buffers and sub-buffers
    inputOutput0 = new int[NUM_BUFFER_ELEMENTS * numDevices];
    inputOutput1 = new int[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput0[i] = i;
        inputOutput1[i] = i;
    }
 
    // Create command queues
    InfoDevice<cl_device_type>::display(
     	deviceIDs[0], 
     	CL_DEVICE_TYPE, 
     	"CL_DEVICE_TYPE");

    cl_command_queue queue0 = 
     	clCreateCommandQueue(
     	context0,
     	deviceIDs[0],
     	0,
     	&errNum);
    checkErr(errNum, "clCreateCommandQueue");
 
    cl_kernel kernel0 = clCreateKernel(
     program0,
     "square",
     &errNum);
    checkErr(errNum, "clCreateKernel(square)");

    std::ifstream file("input.txt");
        if (!file) {
            std::cout <<"Error opening input file " << "input.txt" << std::endl;
            exit(EXIT_FAILURE);
        }
 
    //std::queue<cl_event*> events;
    //while (true) {
        size_t inputSize = readLineFromFile(inputOutput0, &file);
        printInts(inputOutput0, inputSize);
        cl_event *event = new cl_event;
        enqueueCommand(inputOutput0, inputSize, numDevices, context0, queue0, kernel0, event);

        //errNum = clEnqueueWaitForEvents(queue0, 1, event);

    printInts(inputOutput0, inputSize);
    //printInts(inputOutput1, NUM_BUFFER_ELEMENTS);
 
        inputSize = readLineFromFile(inputOutput0, &file);
        printInts(inputOutput0, inputSize);
        event = new cl_event;
        enqueueCommand(inputOutput0, inputSize, numDevices, context0, queue0, kernel0, event);
    printInts(inputOutput0, inputSize);
 
    std::cout << "Program completed successfully" << std::endl;

    return 0;
}
