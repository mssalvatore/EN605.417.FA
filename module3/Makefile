CC=nvcc
OPTIONS=-L /usr/local/cuda/lib -lcudart
OUTFILE=jumbler
DATA_FILE=t8.shakespeare.txt

all: get_shakespeare
	$(CC) -o $(OUTFILE) jumbler.cu $(OPTIONS)
get_shakespeare:
	@if [ ! -f $(DATA_FILE) ];then \
		wget https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt; \
	fi

clean:
	@-rm t8.shakespeare.*txt*
	@rm jumbler
