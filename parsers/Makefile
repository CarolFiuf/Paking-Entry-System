CXX     := g++
DS_INC  := /opt/nvidia/deepstream/deepstream/sources/includes

CXXFLAGS := -shared -fPIC -O2 -std=c++14 -Wall \
            -I$(DS_INC) -I/usr/local/cuda/include

all: libnvds_scrfd_parser.so

libnvds_scrfd_parser.so: nvds_scrfd_parser.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f libnvds_scrfd_parser.so
