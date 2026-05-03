# Build libnvds_face_align.so — custom nvdspreprocess library.
# Phase 6.3 = NPP path active (NV12→RGB, warp affine, normalize).

CXX        := g++
DS_INC     := /opt/nvidia/deepstream/deepstream/sources/includes
DS_LIB     := /opt/nvidia/deepstream/deepstream-7.1/lib
DS_PRE_INC := /opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdspreprocess/include
DS_PRE_LIB := /opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdspreprocess/nvdspreprocess_lib
CUDA_INC   := /usr/local/cuda/include
CUDA_LIB   := /usr/local/cuda/lib64
GST_INC    := $(shell pkg-config --cflags gstreamer-1.0 2>/dev/null)
GST_LIBS   := $(shell pkg-config --libs gstreamer-1.0 2>/dev/null)

CXXFLAGS := -shared -fPIC -O2 -std=c++14 -Wall \
            -I. \
            -I$(DS_INC) \
            -I$(DS_PRE_INC) \
            -I$(DS_PRE_LIB) \
            -I$(CUDA_INC) \
            $(GST_INC)

LDFLAGS  := -L$(CUDA_LIB) -L$(DS_LIB) \
            -lcudart -lnppig -lnppidei -lnppicc -lnppial -lnppc \
            -lnvbufsurftransform -lnvbufsurface -lcuda \
            $(GST_LIBS)

SRCS    := nvds_face_align.cpp face_align_solver.cpp
TARGET  := libnvds_face_align.so

all: $(TARGET)

$(TARGET): $(SRCS) face_align_solver.h
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)
