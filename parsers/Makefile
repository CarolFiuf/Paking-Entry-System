CXX     := g++
DS_INC  := /opt/nvidia/deepstream/deepstream/sources/includes

CXXFLAGS := -fPIC -O2 -std=c++14 -Wall \
            -I. -I$(DS_INC) -I/usr/local/cuda/include

# ── SCRFD parser ──────────────────────────────────────────────
SCRFD_OBJS   := nvds_scrfd_parser.o scrfd_decode.o
SCRFD_TARGET := libnvds_scrfd_parser.so

# ── YOLOv8 plate detector parser ──────────────────────────────
YOLO_OBJS    := nvds_yolov8_parser.o
YOLO_TARGET  := libnvds_yolov8_parser.so

# ── YOLOv8 plate OCR (36 class) parser ────────────────────────
OCR_OBJS     := nvds_plate_ocr_parser.o
OCR_TARGET   := libnvds_plate_ocr_parser.so

all: $(SCRFD_TARGET) $(YOLO_TARGET) $(OCR_TARGET)

# SCRFD
$(SCRFD_TARGET): $(SCRFD_OBJS)
	$(CXX) -shared -o $@ $^

nvds_scrfd_parser.o: nvds_scrfd_parser.cpp scrfd_decode.h
	$(CXX) -c $(CXXFLAGS) $< -o $@

scrfd_decode.o: scrfd_decode.cpp scrfd_decode.h
	$(CXX) -c $(CXXFLAGS) $< -o $@

# YOLOv8 plate det
$(YOLO_TARGET): $(YOLO_OBJS)
	$(CXX) -shared -o $@ $^

nvds_yolov8_parser.o: nvds_yolov8_parser.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

# YOLOv8 plate OCR
$(OCR_TARGET): $(OCR_OBJS)
	$(CXX) -shared -o $@ $^

nvds_plate_ocr_parser.o: nvds_plate_ocr_parser.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(SCRFD_OBJS) $(SCRFD_TARGET) \
	      $(YOLO_OBJS) $(YOLO_TARGET) \
	      $(OCR_OBJS) $(OCR_TARGET)

ocr: $(OCR_TARGET)
