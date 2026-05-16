CXX     := g++
DS_INC  := /opt/nvidia/deepstream/deepstream/sources/includes

CXXFLAGS := -fPIC -O2 -std=c++14 -Wall \
            -I. -I$(DS_INC) -I/usr/local/cuda/include

OBJS   := nvds_scrfd_parser.o scrfd_decode.o
TARGET := libnvds_scrfd_parser.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -shared -o $@ $^

nvds_scrfd_parser.o: nvds_scrfd_parser.cpp scrfd_decode.h
	$(CXX) -c $(CXXFLAGS) $< -o $@

scrfd_decode.o: scrfd_decode.cpp scrfd_decode.h
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
