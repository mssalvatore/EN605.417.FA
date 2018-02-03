// The jumbler is a cryptographically insecure way to "encrypt" a file.
// It jumbles a file by taking 8 byte chunks and using bit rotation based on a "key".
// Next, in takes the now-jumbled file and further jumbles 4, 2, and 1 byte chunks.
#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <math.h>

struct JumbleThreadAllocation {
    uint32_t BlockSize8Byte;
    uint32_t Threads8Byte;
    uint32_t BlockSize4Byte;
    uint32_t Threads4Byte;
    uint32_t BlockSize2Byte;
    uint32_t Threads2Byte;
    uint32_t BlockSize1Byte;
    uint32_t Threads1Byte;
};

template <typename T>
__global__ 
void jumble(T * data, uint8_t rotateNumBits)
{
	size_t numBits = sizeof(T) * 8;
    rotateNumBits = rotateNumBits % numBits;
	const unsigned int dataIndex = ((blockIdx.x * blockDim.x) + threadIdx.x);

	data[dataIndex] = (data[dataIndex] >> rotateNumBits) | (data[dataIndex] << (numBits - rotateNumBits));
}

template <typename T>
__global__ 
void unjumble(T * data, uint8_t rotateNumBits)
{
	size_t numBits = sizeof(T) * 8;
    rotateNumBits = rotateNumBits % numBits;
	const unsigned int dataIndex = ((blockIdx.x * blockDim.x) + threadIdx.x);

	data[dataIndex] = (data[dataIndex] << rotateNumBits) | (data[dataIndex] >> (numBits - rotateNumBits));
}

size_t calculatePadding(size_t fileSize)
{
    size_t paddingBytes = 8 - (fileSize % 8);
    return paddingBytes;
}

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

    *outBytesRead += *paddingBytes;
    return buf;
}

void writeFile(const char * filename, uint8_t * dataToWrite, size_t bytesToWrite)
{
    FILE *handle = fopen(filename, "w");

    fwrite(dataToWrite, 1, bytesToWrite, handle);
    fclose(handle);
}

JumbleThreadAllocation calculateThreadAllocation(size_t numBytes)
{
    return 
        (JumbleThreadAllocation)
            { .BlockSize8Byte = 512,
              .Threads8Byte = numBytes/8,
              .BlockSize4Byte = 256,
              .Threads4Byte = numBytes/4,
              .BlockSize2Byte = 128,
              .Threads2Byte = numBytes/2,
              .BlockSize1Byte = 64,
              .Threads1Byte = numBytes
            };
}

void jumble(JumbleThreadAllocation jta, uint8_t *gpu_block, size_t numBytes) {
	jumble<<<ceil(((double)jta.Threads8Byte) / jta.BlockSize8Byte), jta.BlockSize8Byte>>>((uint64_t *)gpu_block, (uint8_t)'A');
	jumble<<<ceil(((double)jta.Threads4Byte) / jta.BlockSize4Byte), jta.BlockSize4Byte>>>((uint32_t *)gpu_block, (uint8_t)'B');
	jumble<<<ceil(((double)jta.Threads2Byte) / jta.BlockSize2Byte), jta.BlockSize2Byte>>>((uint16_t *)gpu_block, (uint8_t)'C');
	jumble<<<ceil(((double)jta.Threads1Byte) / jta.BlockSize1Byte), jta.BlockSize1Byte>>>((uint8_t *)gpu_block, (uint8_t)'D');
}

void unjumble(JumbleThreadAllocation jta, uint8_t *gpu_block, size_t numBytes) {
	unjumble<<<ceil(((double)jta.Threads1Byte) / jta.BlockSize1Byte), jta.BlockSize1Byte>>>((uint8_t *)gpu_block, (uint8_t)'D');
	unjumble<<<ceil(((double)jta.Threads2Byte) / jta.BlockSize2Byte), jta.BlockSize2Byte>>>((uint16_t *)gpu_block, (uint8_t)'C');
	unjumble<<<ceil(((double)jta.Threads4Byte) / jta.BlockSize4Byte), jta.BlockSize4Byte>>>((uint32_t *)gpu_block, (uint8_t)'B');
	unjumble<<<ceil(((double)jta.Threads8Byte) / jta.BlockSize8Byte), jta.BlockSize8Byte>>>((uint64_t *)gpu_block, (uint8_t)'A');
}

void main_sub()
{

    size_t bytesRead;
    size_t paddingBytes;
    uint8_t *data = readFile("t8.shakespeare.txt", &bytesRead, &paddingBytes);
    printf("Bytes read %d", bytesRead);
    printf("\n\n");

	/* Declare pointers for GPU based params */
    JumbleThreadAllocation jta = calculateThreadAllocation(bytesRead);
	uint8_t *gpu_block;

	cudaMalloc((void **)&gpu_block, bytesRead);
	cudaMemcpy(gpu_block, data, bytesRead, cudaMemcpyHostToDevice);

    jumble(jta, gpu_block, bytesRead);
	cudaMemcpy(data, gpu_block, bytesRead, cudaMemcpyDeviceToHost );
	writeFile("t8.shakespeare.jumbled.txt", data, bytesRead);

    unjumble(jta, gpu_block, bytesRead);
	cudaMemcpy(data, gpu_block, bytesRead, cudaMemcpyDeviceToHost );
	writeFile("t8.shakespeare.unjumbled.txt", data, (bytesRead - paddingBytes));

	/* Execute our kernel */

	/* Free the arrays on the GPU as now we're done with them */
	//cudaMemcpy( data, gpu_block, bytesRead, cudaMemcpyDeviceToHost );
	cudaFree(gpu_block);
}


int main()
{
	main_sub();

	return EXIT_SUCCESS;
}
