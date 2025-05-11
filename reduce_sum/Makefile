NVCC        = /usr/local/cuda/bin/nvcc
NVCC_FLAGS  = -O0 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
BUILD_DIR   = build
EXE         = $(BUILD_DIR)/reduce_sum
OBJ         = $(BUILD_DIR)/reduce_sum.o

default: $(EXE)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(OBJ): reduce_sum.cu | $(BUILD_DIR)
	$(NVCC) -c -o $@ reduce_sum.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf $(BUILD_DIR)
