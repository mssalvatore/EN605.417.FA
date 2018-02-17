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

__constant__  static const unsigned int const_ascii_a = 0x61;
__global__ void shiftLetters(uint8_t *data)
{
    uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    data[threadId] = data[threadId] - const_ascii_a;
}

__device__ void zeroLetterCounts(uint32_t * letterCounts)
{
    for (size_t i = 0; i < NUM_CHARS; i++) {
        letterCounts[(threadIdx.x * NUM_CHARS) + i] = 0;
    }
}

__global__ void countLetters(uint8_t *data, uint32_t *letterCounts, size_t chunkSize)
{
    __shared__ uint32_t sharedLetterCounts[SHARED_MEM_SIZE];

    zeroLetterCounts(sharedLetterCounts);
    __syncthreads();

    for (size_t i = 0; i < chunkSize; i++)
    {
        sharedLetterCounts[(threadIdx.x * NUM_CHARS) + data[(threadIdx.x * chunkSize) + i]]++;
    }

    __syncthreads();

    if (threadIdx.x < NUM_CHARS)
    {
        for (size_t i = 0; i < NUM_THREADS; i++)
        {
            letterCounts[threadIdx.x] += sharedLetterCounts[threadIdx.x + (i * NUM_CHARS)]; 
        }
    }
    __syncthreads();
}

void unpadResult(uint32_t * letterCounts, size_t paddingBytes)
{
    letterCounts[0] -= paddingBytes;
}

void countWithGPU(uint8_t * data, size_t dataSize, uint32_t * letterCounts, uint64_t * outProcessDuration, uint64_t * outTotalDuration)
{

    size_t textChunkSize = dataSize / NUM_THREADS;
	printf("Text Chunk Size (Num Chars): %d\n", textChunkSize);


    auto start = std::chrono::high_resolution_clock::now();
	// Allocate memory on GPU
	uint8_t *gpuData;
	cudaMalloc((void **)&gpuData, dataSize);
	cudaMemcpy(gpuData, data, dataSize, cudaMemcpyHostToDevice);

    uint32_t *gpuLetterCounts;
	cudaMalloc((void **)&gpuLetterCounts, NUM_CHARS * sizeof(uint32_t));
	cudaMemcpy(gpuLetterCounts, letterCounts, NUM_CHARS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    auto processStart = std::chrono::high_resolution_clock::now();
    shiftLetters<<<textChunkSize, NUM_THREADS>>>(gpuData);
    countLetters<<<1, NUM_THREADS>>>(gpuData, gpuLetterCounts, textChunkSize);
    auto processStop = std::chrono::high_resolution_clock::now();

	cudaMemcpy(letterCounts, gpuLetterCounts, NUM_CHARS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	/* Free the arrays on the GPU as now we're done with them */
	cudaFree(gpuData);
	cudaFree(gpuLetterCounts);
    auto stop = std::chrono::high_resolution_clock::now();

    *outProcessDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(processStop - processStart).count();
    *outTotalDuration =  std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
}

uint64_t countWithCPU(uint8_t * data, size_t dataSize, uint32_t * letterCounts)
{
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < dataSize; i++)
    {
        letterCounts[data[i] - 0x61]++;
    }
    auto stop = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
}

void displayResults(uint32_t * letterCounts, uint64_t gpuProcessingDuration, uint64_t gpuTotalDuration, uint64_t cpuDuration)
{
    printf("\n\n");
    for (size_t i = 0; i < NUM_CHARS; i++)
    {
        printf("Found %d %c's\n", letterCounts[i], i + 0x61);
    }
    printf("\n\n");

    printf("Took %dns to run processing on GPU\n", gpuProcessingDuration);
    printf("Took %dns total run time (including device <=> host memory copy) on GPU\n", gpuTotalDuration);
    printf("Took %dns to run on CPU\n", cpuDuration);
    printf("\n");

    if (gpuProcessingDuration < cpuDuration)
    {
        printf("GPU processing is %fx faster than CPU\n", ((double)cpuDuration) / gpuProcessingDuration);
    }
    else
    {
        printf("CPU processing is %fx faster than GPU\n", ((double)gpuProcessingDuration) / cpuDuration);
    }

    printf("\n");

    if (gpuTotalDuration < cpuDuration)
    {
        printf("GPU total duration is %fx faster than CPU\n", ((double)cpuDuration) / gpuTotalDuration);
    }
    else
    {
        printf("CPU processing is %fx faster than GPU total duration\n", ((double)gpuTotalDuration) / cpuDuration);
    }

    printf("\n");
}

int main(int argc, char* argv[])
{
	std::string fileName = "all_letter.shakespeare.txt";
	if (argc > 1) {
		fileName = argv[1];
	}

	size_t bytesRead;
	size_t paddingBytes;

	uint8_t * data = readFile(fileName.c_str(), &bytesRead, &paddingBytes);
	size_t dataSize = bytesRead + paddingBytes;
	printf("Bytes read: %d\n", bytesRead);
	printf("Padding bytes: %d\n", paddingBytes);

    uint32_t letterCounts[NUM_CHARS];
    memset(letterCounts, 0, NUM_CHARS * sizeof(uint32_t));

    uint64_t cpuDuration = countWithCPU(data, dataSize, letterCounts);
    memset(letterCounts, 0, NUM_CHARS * sizeof(uint32_t));

    uint64_t gpuProcessingDuration;
    uint64_t gpuTotalDuration;
    countWithGPU(data, dataSize, letterCounts, &gpuProcessingDuration, &gpuTotalDuration);
    unpadResult(letterCounts, paddingBytes);
    displayResults(letterCounts, gpuProcessingDuration, gpuTotalDuration, cpuDuration);




	return EXIT_SUCCESS;
}
