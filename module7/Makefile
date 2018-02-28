CC=nvcc
OPTIONS=-L /usr/local/cuda/lib -lcudart -std=c++11 
OUTFILE=streams

all: 
	$(CC) -o $(OUTFILE) streams.cu $(OPTIONS)

stretch:
	$(CC) -o stretch module7_stretch_problem.cu $(OPTIONS)

clean:
	@-rm streams
	@-rm stretch
