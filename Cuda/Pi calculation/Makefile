# C compiler
CC = /usr/local/cuda/bin/nvcc
CC_FLAGS = -lm
OUTPUT = piCalc
SRC = dz4z1.cu

$(OUTPUT): $(SRC)
	$(CC) $(CC_FLAGS) $< -o $@

clean:
	rm -f $(OUTPUT)
