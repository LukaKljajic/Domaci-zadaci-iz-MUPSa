# C compiler
CC = /usr/local/cuda/bin/nvcc
CC_FLAGS = -lm 
OUTPUT = needle
SRC = dz4z2.cu
ADDITIONAL_FILES = 
ADDITIONAL_FILES += result_sequential
ADDITIONAL_FILES += result_parallel

$(OUTPUT): $(SRC)
	$(CC) $(CC_FLAGS) $< -o $@

clean:
	rm -f $(OUTPUT) $(ADDITIONAL_FILES)
