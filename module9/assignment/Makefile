CC=nvcc
T_OPTIONS=-L /usr/local/cuda/lib -lcudart -std=c++11 
T_OUTFILE=thrust
T_INFILE=thrust_assignment.cu

all: 
	$(CC) -o $(T_OUTFILE) $(T_INFILE) $(T_OPTIONS)

clean:
	@-rm $(T_OUTFILE)
