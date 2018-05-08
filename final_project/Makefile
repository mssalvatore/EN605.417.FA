CC=nvcc
T_OPTIONS=-L /usr/local/cuda/lib -lcudart -std=c++11 
T_OUTFILE=roulette
T_OBJECT=roulette.o
T_INFILE=roulette.cu

OBJECTS = options.o InvalidArgumentException.o IncompatibleArgumentException.o

all: roulette

roulette: $(OBJECTS)
	$(CC) -o $(T_OUTFILE)  $(T_INFILE) $(T_OPTIONS) $(OBJECTS)

clean:
	@-rm $(T_OUTFILE)
	@-rm *.o
