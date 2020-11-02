#CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
#NVCC_OPTS=-O3 -arch=sm_37 -Xcompiler -Wall -Xcompiler -Wextra -m64

reduce: main.cu Makefile
	nvcc main.cu -o ./build/reduce

clean:
	rm -f *.o reduce
