// The jumbler is a cryptographically insecure way to "encrypt" a file.
// It jumbles a file by taking 8 byte chunks and using bit rotation based on a "key".
// Next, in takes the now-jumbled file and further jumbles 4, 2, and 1 byte chunks.
#include <stdio.h>
#include <iostream>
#include <stdint.h>

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

void main_sub()
{

    size_t bytesRead;
    uint8_t *data = readFile("t8.shakespeare.txt", &bytesRead);
    printf("Bytes read %d", bytesRead);
    printf("\n\n");

	/* Declare pointers for GPU based params */
	uint8_t *gpu_block;

	cudaMalloc((void **)&gpu_block, bytesRead);

	cudaMemcpy(data, gpu_block, bytesRead, cudaMemcpyHostToDevice);
	printf("%d\n", data[0]);

	cudaMemcpy( data, gpu_block, bytesRead, cudaMemcpyDeviceToHost );
	printf("%d\n", data[0]);

	cudaFree(gpu_block);
}


int main()
{
	main_sub();

	return EXIT_SUCCESS;
}
