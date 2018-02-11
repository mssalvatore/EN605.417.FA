CC=nvcc
OPTIONS=-L /usr/local/cuda/lib -lcudart -std=c++11 
OUTFILE=sqrt

all: 
	$(CC) -o $(OUTFILE) sqrt.cu $(OPTIONS)

clean:
	@rm sqrt
