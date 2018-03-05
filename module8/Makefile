CC=nvcc
OPTIONS=-L /usr/local/cuda/lib -lcudart -lcuda -lcurand -lcublas -std=c++11 
OUTFILE=mod8
INFILE=mod8_assignment.cu

all: 
	$(CC) -o $(OUTFILE) $(INFILE) $(OPTIONS)

clean:
	@-rm $(OUTFILE)
