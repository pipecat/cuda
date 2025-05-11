NVCC        = /usr/local/cuda/bin/nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
BUILD_DIR   = build
EXE         = $(BUILD_DIR)/vectoradd
OBJ         = $(BUILD_DIR)/vectoradd.o

default: $(EXE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OBJ): vectoradd.cu | $(BUILD_DIR)
	$(NVCC) -c -o $@ vectoradd.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf $(BUILD_DIR)

