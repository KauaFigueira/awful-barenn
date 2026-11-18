CXX       ?= g++
ARCH      ?= native
CXXFLAGS  ?= -std=c++17 -O3 -march=$(ARCH) -ffp-contract=off -fopenmp
BIN       := barenn

# GPU (Cluster Franky: RTX A1000 = sm_86). --fmad=false mantém a mesma
# aritmética da CPU (sem contração em FMA).
NVCC      ?= nvcc
GPU_ARCH  ?= sm_86
NVCCFLAGS ?= -std=c++17 -O3 -arch=$(GPU_ARCH) --fmad=false \
             -Xcompiler "-O3 -march=$(ARCH) -ffp-contract=off -fopenmp"
BIN_CUDA  := barenn_cuda

all: $(BIN)

$(BIN): barenn.cpp barenn.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

cuda: $(BIN_CUDA)

$(BIN_CUDA): barenn_cuda.cu barenn.hpp
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(BIN)
	./$(BIN)

validate: $(BIN)
	./scripts/validate.sh ./$(BIN)

bench: $(BIN)
	./scripts/bench.sh ./$(BIN)

clean:
	rm -f $(BIN) $(BIN_CUDA)

.PHONY: all cuda run validate bench clean
