// DESCRIPTION HERE
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <chrono>

#define NUM_THREADS 448
#define NUM_CHARS 26
#define SHARED_MEM_SIZE NUM_THREADS * NUM_CHARS

// Calculate how much padding is needed to make the file evenly divided into 448 chunks
size_t calculatePadding(size_t fileSize)
{
    size_t paddingBytes = NUM_THREADS - (fileSize % NUM_THREADS);
    return paddingBytes;
}

// Pad the data so it is evenly divided into 448 chunks
void padData(uint8_t * buf, size_t bytesRead, size_t numPadBytes)
{
    for (size_t i = 0; i < numPadBytes; i++)
    {
        buf[bytesRead + i] = 'a';
    }
}

// Read a file into a byte array
uint8_t * readFile(const char * filename, size_t * outBytesRead, size_t * paddingBytes)
{
    FILE *handle = fopen(filename, "rb");
    fseek(handle, 0, SEEK_END);
    *outBytesRead = ftell(handle);
    *paddingBytes = calculatePadding(*outBytesRead);
    rewind(handle);

    uint8_t * buf = (uint8_t *) malloc((*outBytesRead + *paddingBytes)*sizeof(uint8_t));
    fread(buf, *outBytesRead, 1, handle);
    fclose(handle);

    padData(buf, *outBytesRead, *paddingBytes);

    return buf;
}

// Shift all ascii letters so that 'a' is index 0, 'b' is index 1, etc.
__device__ __constant__ int shiftAmount;
__global__ void shiftLetters(uint8_t *data)
{
    uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    data[threadId] = data[threadId] - shiftAmount;
}

// Zero out the letter counts
__device__ void zeroLetterCounts(uint32_t * letterCounts)
{
    for (size_t i = 0; i < NUM_CHARS; i++) {
        letterCounts[(threadIdx.x * NUM_CHARS) + i] = 0;
    }
}

__device__ void tallyLetters(uint8_t *data, uint32_t *threadLetterCounts, size_t chunkSize)
{
    for (size_t i = 0; i < chunkSize; i++)
    {
        threadLetterCounts[data[i]]++;
    }
}

__device__ void sumThreadResults(uint32_t *globalLetterCounts, uint32_t *threadLetterCounts)
{
    // Total local thread tallys
    if (threadIdx.x < NUM_CHARS)
    {
        for (size_t i = 1; i < NUM_THREADS; i++)
        {
            threadLetterCounts[threadIdx.x] += threadLetterCounts[threadIdx.x + (i * NUM_CHARS)];
        }
        globalLetterCounts[threadIdx.x] = threadLetterCounts[threadIdx.x];
    }
}

// Count the occurence of each letter in *data
__device__ void countLetters(uint8_t *data, uint32_t *letterCounts, uint32_t *threadLetterCounts, size_t chunkSize)
{

    memset(&threadLetterCounts[(threadIdx.x * NUM_CHARS)], 0, NUM_CHARS * sizeof(uint32_t));
    //zeroLetterCounts(threadLetterCounts);
    __syncthreads();

    // Tally letters for each thread
    tallyLetters(&data[(threadIdx.x * chunkSize)], &threadLetterCounts[(threadIdx.x * NUM_CHARS)], chunkSize);

    __syncthreads();

    sumThreadResults(letterCounts, threadLetterCounts);
}

// Count the occurence of each letter in *data using shared memory
__global__ void countLettersShared(uint8_t *data, uint32_t *letterCounts, size_t chunkSize)
{
    __shared__ uint32_t sharedLetterCounts[SHARED_MEM_SIZE];
    countLetters(data, letterCounts, sharedLetterCounts, chunkSize);
}

// Count the occurence of each letter in *data using global memory
__global__ void countLettersGlobal(uint8_t *data, uint32_t *letterCounts, uint32_t * threadLetterCounts, size_t chunkSize)
{
    countLetters(data, letterCounts, threadLetterCounts, chunkSize);
}

// Count the occurence of each letter in *data using shared memory
__global__ void countLettersRegisters(uint8_t *data, uint32_t *letterCounts, size_t chunkSize)
{
    __shared__ uint32_t sharedLetterCounts[SHARED_MEM_SIZE];
    uint32_t threadLetterCounts[NUM_CHARS];
    memset(threadLetterCounts, 0, NUM_CHARS * sizeof(uint32_t));

    // Tally letters for each thread
    tallyLetters(&data[(threadIdx.x * chunkSize)], threadLetterCounts, chunkSize);

    // Copy registers to shared memory so threads can be summed up.
    memcpy(&sharedLetterCounts[(threadIdx.x * NUM_CHARS)], threadLetterCounts, NUM_CHARS * sizeof(uint32_t));
    __syncthreads();

    sumThreadResults(letterCounts, sharedLetterCounts);
}


// Remove any padding so that letter counts are accurage
void unpadResult(uint32_t * letterCounts, size_t paddingBytes)
{
    letterCounts[0] -= paddingBytes;
}

// Count the occurence of each letter in *data using shared memory
uint64_t countWithGPUShared(uint8_t * data, size_t dataSize, uint32_t * letterCounts, size_t textChunkSize)
{
    // Declare cuda memory
    uint8_t *gpuData;
    uint32_t *gpuLetterCounts;
    cudaMalloc((void **)&gpuData, dataSize);
    cudaMemcpy(gpuData, data, dataSize, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&gpuLetterCounts, NUM_CHARS * sizeof(uint32_t));
    cudaMemcpy(gpuLetterCounts, letterCounts, NUM_CHARS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Run Kernel
    shiftLetters<<<textChunkSize, NUM_THREADS>>>(gpuData);
    auto start = std::chrono::high_resolution_clock::now();
    countLettersShared<<<1, NUM_THREADS>>>(gpuData, gpuLetterCounts, textChunkSize);
    auto stop = std::chrono::high_resolution_clock::now();

    cudaMemcpy(letterCounts, gpuLetterCounts, NUM_CHARS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free the arrays on the GPU as now we're done with them
    cudaFree(gpuData);
    cudaFree(gpuLetterCounts);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
}

// Count the occurence of each letter in *data using register memory
uint64_t countWithGPURegisters(uint8_t * data, size_t dataSize, uint32_t * letterCounts, size_t textChunkSize)
{
    // Declare cuda memory
    uint8_t *gpuData;
    uint32_t *gpuLetterCounts;
    cudaMalloc((void **)&gpuData, dataSize);
    cudaMemcpy(gpuData, data, dataSize, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&gpuLetterCounts, NUM_CHARS * sizeof(uint32_t));
    cudaMemcpy(gpuLetterCounts, letterCounts, NUM_CHARS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Run Kernel
    shiftLetters<<<textChunkSize, NUM_THREADS>>>(gpuData);
    auto start = std::chrono::high_resolution_clock::now();
    countLettersRegisters<<<1, NUM_THREADS>>>(gpuData, gpuLetterCounts, textChunkSize);
    auto stop = std::chrono::high_resolution_clock::now();

    cudaMemcpy(letterCounts, gpuLetterCounts, NUM_CHARS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free the arrays on the GPU as now we're done with them
    cudaFree(gpuData);
    cudaFree(gpuLetterCounts);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
}

// Count the occurence of each letter in *data using global memory
uint64_t countWithGPUGlobal(uint8_t * data, size_t dataSize, uint32_t * letterCounts, size_t textChunkSize)
{
    // Declare cuda memory
    uint8_t *gpuData;
    uint32_t *gpuLetterCounts;
    uint32_t *threadLetterCounts;
    cudaMalloc((void **)&gpuData, dataSize);
    cudaMemcpy(gpuData, data, dataSize, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&gpuLetterCounts, NUM_CHARS * sizeof(uint32_t));
    cudaMemcpy(gpuLetterCounts, letterCounts, NUM_CHARS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&threadLetterCounts, SHARED_MEM_SIZE);

    // Run Kernel
    shiftLetters<<<textChunkSize, NUM_THREADS>>>(gpuData);
    auto start = std::chrono::high_resolution_clock::now();
    countLettersGlobal<<<1, NUM_THREADS>>>(gpuData, gpuLetterCounts, threadLetterCounts, textChunkSize);
    auto stop = std::chrono::high_resolution_clock::now();

    cudaMemcpy(letterCounts, gpuLetterCounts, NUM_CHARS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    /* Free the arrays on the GPU as now we're done with them */
    cudaFree(gpuData);
    cudaFree(gpuLetterCounts);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
}

// Display letter counts
void displayResults(uint32_t * letterCounts)
{
    printf("\n\n");
    for (size_t i = 0; i < NUM_CHARS; i++)
    {
        printf("Found %d %c's\n", letterCounts[i], i + 0x61);
    }

    printf("\n\n");
}

// Display and analyze the run times (shared vs. global vs. register)
void displayTimingResults(uint64_t gpuSharedDuration, uint64_t gpuGlobalDuration, uint64_t gpuRegisterDuration)
{
    printf("Took %dns to run processing on GPU with shared memory\n", gpuSharedDuration);
    printf("Took %dns to run processing on GPU with global memory\n", gpuGlobalDuration);
    printf("Took %dns to run processing on GPU with register memory\n", gpuRegisterDuration);

    printf("\n");
    printf("Shared Memory runs %fx faster than global memory\n", ((double)gpuGlobalDuration) / gpuSharedDuration);
    printf("Register Memory runs %fx faster than shared memory\n", ((double)gpuSharedDuration) / gpuRegisterDuration);
    printf("Register Memory runs %fx faster than global memory\n", ((double)gpuGlobalDuration) / gpuRegisterDuration);
    printf("\n");
}

int main(int argc, char* argv[])
{
    // Read command line args
    std::string fileName = "all_letter.shakespeare.txt";
    if (argc > 1) {
        fileName = argv[1];
    }

    // Copy from host to constant memory
    const int ascii_a = 0x61;
    cudaMemcpyToSymbol(shiftAmount, &ascii_a, sizeof(uint8_t));

    // Declare some variables
    uint32_t letterCounts[NUM_CHARS];
    size_t bytesRead;
    size_t paddingBytes;

    // Read file
    uint8_t * data = readFile(fileName.c_str(), &bytesRead, &paddingBytes);

    // Calculate run-time parameters
    size_t dataSize = bytesRead + paddingBytes;
    size_t textChunkSize = dataSize / NUM_THREADS;
    printf("Bytes read: %d\n", bytesRead);
    printf("Padding bytes: %d\n", paddingBytes);

    uint8_t *pinnedData;
    cudaMallocHost((void**)&pinnedData, dataSize);
    memcpy(pinnedData, data, dataSize);

    // Run letter counter on the GPU with global memory
    memset(letterCounts, 0, NUM_CHARS * sizeof(uint32_t));
    uint64_t gpuGlobalDuration = countWithGPUGlobal(pinnedData, dataSize, letterCounts, textChunkSize);

    // Run letter counter on the GPU with shared memory
    memset(letterCounts, 0, NUM_CHARS * sizeof(uint32_t));
    uint64_t gpuSharedDuration = countWithGPUShared(pinnedData, dataSize, letterCounts, textChunkSize);

    // Run letter counter on the GPU with registers
    memset(letterCounts, 0, NUM_CHARS * sizeof(uint32_t));
    uint64_t gpuRegisterDuration = countWithGPURegisters(pinnedData, dataSize, letterCounts, textChunkSize);

    unpadResult(letterCounts, paddingBytes);

    // Display letter counts and timing
    displayResults(letterCounts);
    displayTimingResults(gpuSharedDuration, gpuGlobalDuration, gpuRegisterDuration);

    uint16_t numRuns = 10;
    double ratioSum = 0;
    for (uint16_t i = 0; i < numRuns; i++)
    {
        // Run letter counter on the GPU with shared memory
        memset(letterCounts, 0, NUM_CHARS * sizeof(uint32_t));
        gpuSharedDuration = countWithGPUShared(pinnedData, dataSize, letterCounts, textChunkSize);

        // Run letter counter on the GPU with registers
        memset(letterCounts, 0, NUM_CHARS * sizeof(uint32_t));
        gpuRegisterDuration = countWithGPURegisters(pinnedData, dataSize, letterCounts, textChunkSize);

        ratioSum += ((double)gpuRegisterDuration) / gpuSharedDuration;
    }

    printf("On average over %d runs, using registers performed %fx faster than shared memory\n", numRuns, ratioSum / numRuns);



    return EXIT_SUCCESS;
}
