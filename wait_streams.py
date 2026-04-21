#!/usr/bin/env python3
"""
Continuously probe both RTSP/RTMP streams until they come online.
Usage: python wait_streams.py
"""
import cv2
import time
import threading

STREAMS = {
    "plate": "rtmp://192.168.118.112/live/plate",
    "face":  "rtmp://192.168.118.112/live/face",
}
RETRY_INTERVAL = 1.0  # seconds between retries


def wait_for_stream(name, url):
    print(f"[{name}] Waiting for stream: {url}")
    while True:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            cap.release()
            print(f"[{name}] CONNECTED ✓")
            return
        cap.release()
        time.sleep(RETRY_INTERVAL)


threads = [
    threading.Thread(target=wait_for_stream, args=(name, url), daemon=True)
    for name, url in STREAMS.items()
]

for t in threads:
    t.start()
for t in threads:
    t.join()

print("Both streams are live. You can start the system now.")
