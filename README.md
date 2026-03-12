# Smart Parking System

Jetson Orin Nano | DeepStream + GStreamer | PostgreSQL + pgvector | Web Dashboard

## Kiến trúc

```
iPhone RTMP ──→ nginx ──→ DeepStream / GStreamer
                          ├── nvinfer (plate YOLOv8)     ← GPU zero-copy
                          └── probe → Python
                              ├── PaddleOCR (biển số)
                              ├── InsightFace (mặt)
                              ├── PostgreSQL + pgvector
                              └── FastAPI WebSocket → Dashboard
```

## Setup

```bash
pip install -r requirements.txt --break-system-packages
python setup.py --init-db     # PostgreSQL + pgvector
python setup.py               # Verify GPU + models
```

## Chạy

```bash
python main.py --entry              # Entry mode + web dashboard
python main.py --exit               # Exit mode
python main.py --entry --no-show    # Headless (chỉ web)
python main.py --entry --no-web     # Chỉ CV2 window
python main.py --benchmark test.mp4 # Đo tốc độ
```

Dashboard: http://jetson-ip:8080

## Files

```
├── main.py              # Orchestrator
├── pipeline.py          # DeepStream (fallback GStreamer)
├── engine.py            # OCR + face embedding
├── database.py          # PostgreSQL + pgvector
├── web.py               # FastAPI dashboard
├── templates/
│   └── dashboard.html   # Dashboard UI
├── configs/
│   └── plate_det_config.txt  # nvinfer config
├── config.yaml
└── models/
```

## DeepStream

Tự động dùng DeepStream nếu `pyds` available (JetPack SDK).
Fallback sang GStreamer + Python inference nếu không có.

Cài DeepStream Python bindings:
```bash
# Có sẵn trong JetPack 6.x, hoặc:
sudo apt install deepstream-7.0
pip install pyds-ext
```

## Phím tắt (CV2 window)

- **Q**: Thoát
- **M**: Chuyển Entry ↔ Exit
- **S**: In stats
