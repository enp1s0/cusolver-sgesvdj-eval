CXXFLAGS+=-ltmglib -llapacke -llapack -lblas -lgfortran

NVCC=nvcc
NVCCFLAGS=-std=c++17
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=$(CXXFLAGS) -Xcompiler=-fopenmp
NVCCFLAGS+=-lcusolver -I./src/cutf/include
NVCCFLAGS+=-I./src/mateval/include -L./src/mateval/build -lmateval_cuda

TARGET=cusolver-svdj.test

$(TARGET):src/main.cu libmateval_cuda
	$(NVCC) $< -o $@ $(NVCCFLAGS)

libmateval_cuda:
	[ -z ./src/mateval/build ] || mkdir -p ./src/mateval/build
	cd ./src/mateval/build && cmake .. && make -j
  
clean:
	rm -f $(TARGET)
