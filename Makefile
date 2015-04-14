
all: seq cuda


H_FILES = kde.h config.h

# ----
#  SEQ

OPTFLAGS  = -g
INCFLAGS = -I.
CFLAGS = $(OPTFLAGS) $(INCFLAGS)
LDFLAGS = $(OPTFLAGS)
LIBS =

CPP = g++

.c.o:
	$(CC) $(CFLAGS) -c $<

SEQ_SRC = kde_seq.cc main.cc file_io.cc wtime.cc

SEQ_OBJ = $(SEQ_SRC:%.cc=%.o)

seq: seq_main $(H_FILES)
seq_main: $(SEQ_OBJ)
	$(CPP) $(LDFLAGS) $^ -o $@ $(LIBS)

# -----
#  CUDA

NVCC = nvcc
NVCCFLAGS = $(CFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

CUDA_H_FILES = $(H_FILES) cuda_util.h
CUDA_C_SRC = main.cc file_io.cc wtime.cc
CUDA_CU_SRC = kde_cuda.cu
CUDA_C_OBJ = $(CUDA_C_SRC:%.cc=%.o)
CUDA_CU_OBJ = $(CUDA_CU_SRC:%.cu=%.o)

cuda_main: $(CUDA_C_OBJ) $(CUDA_CU_OBJ)
	$(NVCC) $(LDFLAGS) $^ -o $@ $(LIBS)
cuda: cuda_main $(CUDA_H_FILES)

clean:
	rm -rf *.o seq_main cuda_main
