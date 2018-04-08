CC=nvcc
T_OPTIONS=-L /usr/local/cuda/lib -lcudart -std=c++11 
T_OUTFILE=roulette
T_INFILE=roulette.cu

all: roulette

roulette:
	$(CC) -o $(T_OUTFILE) $(T_INFILE) $(T_OPTIONS)


clean:
	@-rm $(T_OUTFILE)
