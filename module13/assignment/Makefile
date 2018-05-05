UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	OPTIONS=-lOpenCL
endif
ifeq ($(UNAME), Darwin)
	OPTIONS=-framework OpenCL
endif


CC=g++
OPTIONS += -std=c++11
OUTFILE=square
INFILE=square.cpp

all: 
	$(CC) -o $(OUTFILE) $(INFILE) $(OPTIONS)

clean:
	@-rm $(OUTFILE)
