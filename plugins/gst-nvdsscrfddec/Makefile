CXX        := g++
DS_INC     := /opt/nvidia/deepstream/deepstream/sources/includes
DS_LIB     := /opt/nvidia/deepstream/deepstream-7.1/lib

PKGS       := gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
GST_CFLAGS := $(shell pkg-config --cflags $(PKGS) 2>/dev/null)
GST_LIBS   := $(shell pkg-config --libs $(PKGS) 2>/dev/null)

CXXFLAGS := -fPIC -O2 -std=c++14 -Wall \
            -I. \
            -I../common \
            -I../../parsers \
            -I$(DS_INC) \
            $(GST_CFLAGS)

LDFLAGS := -shared -Wl,-no-undefined \
           -L$(DS_LIB) \
           -lnvds_meta -lnvdsgst_meta -lnvbufsurface \
           -Wl,-rpath,$(DS_LIB) \
           $(GST_LIBS)

OBJS   := gstnvdsscrfddec.o scrfd_decode.o
TARGET := libgstnvdsscrfddec.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp ../common/nvds_face_landmarks_meta.h
	$(CXX) -c -o $@ $(CXXFLAGS) $<

scrfd_decode.o: ../../parsers/scrfd_decode.cpp ../../parsers/scrfd_decode.h
	$(CXX) -c -o $@ $(CXXFLAGS) $<

clean:
	rm -f $(OBJS) $(TARGET)
