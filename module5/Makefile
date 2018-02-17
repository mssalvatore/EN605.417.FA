CC=nvcc
OPTIONS=-L /usr/local/cuda/lib -lcudart
SOURCE_FILE=letter_counter.cu
OUT_FILE=letter_counter
RAW_DATA_FILE=t8.shakespeare.txt
LOWER_DATA_FILE=lower.shakespeare.txt
DATA_FILE=all_letter.shakespeare.txt

all: get_shakespeare
	$(CC) -o $(OUT_FILE) $(SOURCE_FILE) $(OPTIONS)
get_shakespeare:
	@if [ ! -f $(DATA_FILE) ];then \
		wget https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt; \
		tr '[:upper:]' '[:lower:]' < $(RAW_DATA_FILE) > $(LOWER_DATA_FILE); \
		perl -pe 's/[^a-z]+//g' <<< echo $(LOWER_DATA_FILE) > $(DATA_FILE); \
	fi

clean:
	@-rm *.shakespeare.*txt*
	@rm $(OUT_FILE)
