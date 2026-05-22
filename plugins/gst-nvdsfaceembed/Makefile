CXX        := g++
NVCC       := nvcc
CUDA_ARCH  ?= 87
DS_INC     := /opt/nvidia/deepstream/deepstream/sources/includes
DS_LIB     := /opt/nvidia/deepstream/deepstream-7.1/lib
CUDA_INC   := /usr/local/cuda/include
CUDA_LIB   := /usr/local/cuda/lib64
TRT_INC    := /usr/include/aarch64-linux-gnu
TRT_LIB    := /lib/aarch64-linux-gnu

PKGS       := gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
GST_CFLAGS := $(shell pkg-config --cflags $(PKGS) 2>/dev/null)
GST_LIBS   := $(shell pkg-config --libs $(PKGS) 2>/dev/null)

CXXFLAGS := -fPIC -O2 -std=c++14 -Wall \
            -I. \
            -I../common \
            -I../../preprocess \
            -I../../parsers \
            -I$(DS_INC) \
            -I$(CUDA_INC) \
            -I$(TRT_INC) \
            $(GST_CFLAGS)

LDFLAGS := -shared -Wl,-no-undefined \
           -L$(CUDA_LIB) -L$(DS_LIB) -L$(TRT_LIB) \
           -lcudart -lcuda \
           -lnvbufsurface -lnvbufsurftransform \
           -lnvds_meta -lnvdsgst_meta \
           -lnvinfer \
           -Wl,-rpath,$(DS_LIB) \
           $(GST_LIBS)

NVCCFLAGS := -O2 -std=c++14 -Xcompiler -fPIC \
             -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
             -I. \
             -I$(CUDA_INC)

OBJS   := gstnvdsfaceembed.o face_align_solver.o face_align_cuda.o
TARGET := libgstnvdsfaceembed.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp nvds_face_embed_meta.h ../common/nvds_face_landmarks_meta.h
	$(CXX) -c -o $@ $(CXXFLAGS) $<

face_align_solver.o: ../../preprocess/face_align_solver.cpp ../../preprocess/face_align_solver.h
	$(CXX) -c -o $@ $(CXXFLAGS) $<

face_align_cuda.o: face_align_cuda.cu
	$(NVCC) -c -o $@ $(NVCCFLAGS) $<

clean:
	rm -f $(OBJS) $(TARGET)
