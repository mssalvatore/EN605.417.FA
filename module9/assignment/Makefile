CC=nvcc
HC=-ccbin g++
ALL_CCFLAGS=-m64
GENCODE_FLAGS=-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60
INCLUDES  := -I../../common/inc
LIBRARIES=-lnvgraph
T_OPTIONS=-L /usr/local/cuda/lib -lcudart -std=c++11 
T_OUTFILE=thrust
T_INFILE=thrust_assignment.cu
NV_OUTFILE=sssp
NV_INFILE=sssp.cu

all: thrust sssp

thrust:
	$(CC) -o $(T_OUTFILE) $(T_INFILE) $(T_OPTIONS)

sssp.o:$(NV_INFILE)
	$(CC) $(HC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $(NV_OUTFILE).o -c $<

sssp: $(NV_OUTFILE).o
	$(CC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	@-rm $(NV_OUTFILE).o

clean:
	@-rm $(T_OUTFILE)
	@-rm $(NV_OUTFILE)
