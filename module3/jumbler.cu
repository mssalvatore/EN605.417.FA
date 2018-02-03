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


__global__ 
void jumble(uint8_t * data, uint8_t dataBlockSize, size_t numBytes, uint8_t rotateNumBits)
{
    rotateNumBits = rotateNumBits % (dataBlockSize * 8);
    
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t dataToManipulate = (uint64_t) (data[thread_idx * dataBlockSize]);
    uint64_t manipulatedData = (dataToManipulate >> rotateNumBits) | 
						(dataToManipulate << ((dataBlockSize * 8) - rotateNumBits));
	data[thread_idx * dataBlockSize] = manipulatedData;
}

__global__ 
void unjumble(uint8_t * data, uint8_t dataBlockSize, size_t numBytes, uint8_t rotateNumBits)
{
    rotateNumBits = rotateNumBits % (dataBlockSize * 8);
    
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t dataToManipulate = (uint64_t) (data[thread_idx * dataBlockSize]);
    uint64_t manipulatedData = (dataToManipulate << rotateNumBits) | 
						(dataToManipulate >> ((dataBlockSize * 8) - rotateNumBits));
	data[thread_idx * dataBlockSize] = manipulatedData;
}

size_t calculatePadding(size_t fileSize)
{
    size_t paddingBytes = 8 - (fileSize % 8);
    return paddingBytes;
}

uint8_t * readFile(const char * filename, size_t * outBytesRead)
{
    FILE *handle = fopen(filename, "rb");
    fseek(handle, 0, SEEK_END);
    *outBytesRead = ftell(handle);
    size_t paddingBytes = calculatePadding(*outBytesRead);
    rewind(handle);

    uint8_t * buf = (uint8_t *) malloc((*outBytesRead + paddingBytes)*sizeof(uint8_t));
    fread(buf, *outBytesRead, 1, handle);
    fclose(handle);

    *outBytesRead += paddingBytes;
    return buf;
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
	jumble<<<ceil(((double)jta.Threads8Byte) / jta.BlockSize8Byte), jta.BlockSize8Byte>>>(gpu_block, 8, numBytes, (uint8_t)'A');
	//jumble<<<ceil(((double)jta.Threads4Byte) / jta.BlockSize4Byte), jta.BlockSize4Byte>>>(gpu_block, 4, numBytes, (uint8_t)'B');
	//jumble<<<ceil(((double)jta.Threads2Byte) / jta.BlockSize2Byte), jta.BlockSize2Byte>>>(gpu_block, 2, numBytes, (uint8_t)'C');
	//jumble<<<ceil(((double)jta.Threads1Byte) / jta.BlockSize1Byte), jta.BlockSize1Byte>>>(gpu_block, 1, numBytes, (uint8_t)'D');
}

void unjumble(JumbleThreadAllocation jta, uint8_t *gpu_block, size_t numBytes) {
	//unjumble<<<ceil(((double)jta.Threads1Byte) / jta.BlockSize1Byte), jta.BlockSize1Byte>>>(gpu_block, 1, numBytes, (uint8_t)'D');
	//unjumble<<<ceil(((double)jta.Threads2Byte) / jta.BlockSize2Byte), jta.BlockSize2Byte>>>(gpu_block, 2, numBytes, (uint8_t)'C');
	//unjumble<<<ceil(((double)jta.Threads4Byte) / jta.BlockSize4Byte), jta.BlockSize4Byte>>>(gpu_block, 4, numBytes, (uint8_t)'B');
	unjumble<<<ceil(((double)jta.Threads8Byte) / jta.BlockSize8Byte), jta.BlockSize8Byte>>>(gpu_block, 8, numBytes, (uint8_t)'A');
}

void main_sub()
{

    size_t bytesRead;
    uint8_t *data = readFile("t8.shakespeare.txt", &bytesRead);
    printf("Bytes read %d", bytesRead);
    printf("\n\n");

	/* Declare pointers for GPU based params */
    JumbleThreadAllocation jta = calculateThreadAllocation(bytesRead);
	uint8_t *gpu_block;

	cudaMalloc((void **)&gpu_block, bytesRead);
	cudaMemcpy(gpu_block, data, bytesRead, cudaMemcpyHostToDevice);
	printf("%c\n", data[0]);
    jumble(jta, gpu_block, bytesRead);
	cudaMemcpy(data, gpu_block, bytesRead, cudaMemcpyDeviceToHost );
	printf("%c\n", data[0]);
    unjumble(jta, gpu_block, bytesRead);
	cudaMemcpy( data, gpu_block, bytesRead, cudaMemcpyDeviceToHost );
	printf("%c\n", data[0]);

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
