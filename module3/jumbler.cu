// The jumbler is a cryptographically insecure way to "encrypt" a file.
// It jumbles a file by taking 8 byte chunks and using bit rotation based on a "key".
// Next, in takes the now-jumbled file and further jumbles 4, 2, and 1 byte chunks.
#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <math.h>

// Used to hold the different thread/block sizes for the different rounds of "encryption"
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

// CUDA kernel for jumbling up the data
template <typename T>
__global__ void jumble(T * data, uint8_t rotateNumBits)
{
	size_t numBits = sizeof(T) * 8;
	rotateNumBits = rotateNumBits % numBits;
	const unsigned int dataIndex = ((blockIdx.x * blockDim.x) + threadIdx.x);

	data[dataIndex] = (data[dataIndex] >> rotateNumBits) | (data[dataIndex] << (numBits - rotateNumBits));
}

// CUDA kernel for unjumbling the data
template <typename T>
__global__ void unjumble(T * data, uint8_t rotateNumBits)
{
	size_t numBits = sizeof(T) * 8;
	rotateNumBits = rotateNumBits % numBits;
	const unsigned int dataIndex = ((blockIdx.x * blockDim.x) + threadIdx.x);

	data[dataIndex] = (data[dataIndex] << rotateNumBits) | (data[dataIndex] >> (numBits - rotateNumBits));
}

// Calculate how much padding is needed to make the file evenly divided into 8-byte chunks
size_t calculatePadding(size_t fileSize)
{
	size_t paddingBytes = 8 - (fileSize % 8);
	return paddingBytes;
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

	return buf;
}

// Write a file out to disk
void writeFile(std::string filename, uint8_t * dataToWrite, size_t bytesToWrite)
{
	FILE *handle = fopen(filename.c_str(), "w");

	fwrite(dataToWrite, 1, bytesToWrite, handle);
	fclose(handle);
}

// Populate a JumbleThreadAllocation struct based on the number of bytes in the data set
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

// Call the jumble kernel with the key "ABCD" to jumble up the data
void jumble(JumbleThreadAllocation jta, uint8_t *gpu_block, size_t numBytes) {
	jumble<<<ceil(((double)jta.Threads8Byte) / jta.BlockSize8Byte), jta.BlockSize8Byte>>>((uint64_t *)gpu_block, (uint8_t)'A');
	jumble<<<ceil(((double)jta.Threads4Byte) / jta.BlockSize4Byte), jta.BlockSize4Byte>>>((uint32_t *)gpu_block, (uint8_t)'B');
	jumble<<<ceil(((double)jta.Threads2Byte) / jta.BlockSize2Byte), jta.BlockSize2Byte>>>((uint16_t *)gpu_block, (uint8_t)'C');
	jumble<<<ceil(((double)jta.Threads1Byte) / jta.BlockSize1Byte), jta.BlockSize1Byte>>>((uint8_t *)gpu_block, (uint8_t)'D');
}

// Call the unjumble kernel with the key "ABCD" to decode the data
void unjumble(JumbleThreadAllocation jta, uint8_t *gpu_block, size_t numBytes) {
	unjumble<<<ceil(((double)jta.Threads1Byte) / jta.BlockSize1Byte), jta.BlockSize1Byte>>>((uint8_t *)gpu_block, (uint8_t)'D');
	unjumble<<<ceil(((double)jta.Threads2Byte) / jta.BlockSize2Byte), jta.BlockSize2Byte>>>((uint16_t *)gpu_block, (uint8_t)'C');
	unjumble<<<ceil(((double)jta.Threads4Byte) / jta.BlockSize4Byte), jta.BlockSize4Byte>>>((uint32_t *)gpu_block, (uint8_t)'B');
	unjumble<<<ceil(((double)jta.Threads8Byte) / jta.BlockSize8Byte), jta.BlockSize8Byte>>>((uint64_t *)gpu_block, (uint8_t)'A');
}

void printJumbleThreadAllocation(JumbleThreadAllocation jta)
{
	printf("Rotate 8 bytes chunks using %g blocks of size %d and %d total threads.\n", ceil((double)jta.Threads8Byte / jta.BlockSize8Byte), jta.BlockSize8Byte, jta.Threads8Byte);
	printf("Rotate 4 bytes chunks using %g blocks of size %d and %d total threads.\n", ceil((double)jta.Threads4Byte / jta.BlockSize4Byte), jta.BlockSize4Byte, jta.Threads4Byte);
	printf("Rotate 2 bytes chunks using %g blocks of size %d and %d total threads.\n", ceil((double)jta.Threads2Byte / jta.BlockSize2Byte), jta.BlockSize2Byte, jta.Threads2Byte);
	printf("Rotate 1 bytes chunks using %g blocks of size %d and %d total threads.\n", ceil((double)jta.Threads1Byte / jta.BlockSize1Byte), jta.BlockSize1Byte, jta.Threads1Byte);
}

int main(int argc, char* argv[])
{
	std::string fileName = "t8.shakespeare.txt";
	if (argc > 1) {
		fileName = argv[1];
	}
	size_t bytesRead;
	size_t paddingBytes;
	size_t dataSize;
	uint8_t *data = readFile(fileName.c_str(), &bytesRead, &paddingBytes);
	dataSize = bytesRead + paddingBytes;
	printf("Bytes read %d\n", bytesRead);
	printf("Padding bytes %d\n", paddingBytes);

	JumbleThreadAllocation jta = calculateThreadAllocation(dataSize);
	printJumbleThreadAllocation(jta);

	// Allocate memory on GPU
	uint8_t *gpu_block;
	cudaMalloc((void **)&gpu_block, dataSize);
	cudaMemcpy(gpu_block, data, dataSize, cudaMemcpyHostToDevice);

	// Jumble up the data and dump it to disk
	jumble(jta, gpu_block, dataSize);
	cudaMemcpy(data, gpu_block, dataSize, cudaMemcpyDeviceToHost);
	writeFile(fileName + ".jumbled", data, dataSize);

	// Unjumble the data and dump it to disk
	unjumble(jta, gpu_block, dataSize);
	cudaMemcpy(data, gpu_block, dataSize, cudaMemcpyDeviceToHost);
	writeFile(fileName + ".unjumbled", data, bytesRead);

	/* Free the arrays on the GPU as now we're done with them */
	cudaFree(gpu_block);

	return EXIT_SUCCESS;
}
