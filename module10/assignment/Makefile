UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	OPTIONS=-lOpenCL
endif
ifeq ($(UNAME), Darwin)
	OPTIONS=-framework OpenCL
endif


CC=g++
OPTIONS += -std=c++11
OUTFILE=math_funcs
INFILE=MathFuncs.cpp

all: 
	$(CC) -o $(OUTFILE) $(INFILE) $(OPTIONS)

clean:
	@-rm $(OUTFILE)
