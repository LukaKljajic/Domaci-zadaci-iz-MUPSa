# C compiler
CC = /usr/local/cuda/bin/nvcc
CC_FLAGS = -lm
OUTPUT = nbody
SRC = dz4z3.cu

$(OUTPUT): $(SRC)
	$(CC) $(CC_FLAGS) $< -o $@

clean:
	rm -f $(OUTPUT)
