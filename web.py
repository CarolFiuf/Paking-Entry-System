"""
web.py — FastAPI Web Dashboard
Real-time monitoring qua WebSocket.

Endpoints:
  GET  /              → Dashboard HTML
  GET  /api/stats     → Current stats
  GET  /api/active    → Active vehicles
  GET  /api/history   → Recent events
  WS   /ws            → Live events stream
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("web")

app = FastAPI(title="Smart Parking Dashboard")

# Global references (set by main.py)
_db = None
_state = {
    "mode": "entry",
    "fps": 0,
    "last_event": None,
    "plate_cam_ok": False,
    "face_cam_ok": False,
    "deepstream": False,
}


# Connected WebSocket clients
_clients: list[WebSocket] = []

# Frame buffers cho MJPEG stream (latest frame only)
_frames: dict[str, bytes] = {}  # {"plate": jpeg_bytes, "face": jpeg_bytes}
_frame_lock = __import__("threading").Lock()


def update_frame(name: str, frame):
    """Gọi từ pipeline thread — encode + lưu JPEG."""
    if frame is None:
        return
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    with _frame_lock:
        _frames[name] = jpg.tobytes()


def init(db, state: dict = None):
    """Gọi từ main.py để inject database reference."""
    global _db, _state
    _db = db
    if state:
        _state.update(state)


async def broadcast(event: dict):
    """Push event tới tất cả WebSocket clients."""
    msg = json.dumps(event, default=str)
    dead = []
    for ws in _clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.remove(ws)


def notify_sync(event: dict):
    """Gọi từ sync code (pipeline thread) → async broadcast."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(broadcast(event))
    except RuntimeError:
        pass  # No event loop, skip



# ── REST endpoints ──
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/stats")
async def api_stats():
    if not _db:
        return {"error": "DB not ready"}
    stats = _db.stats()
    stats.update({
        "mode": _state["mode"],
        "fps": _state["fps"],
        "plate_cam": _state["plate_cam_ok"],
        "face_cam": _state["face_cam_ok"],
        "deepstream": _state["deepstream"],
    })
    return stats


@app.get("/api/active")
async def api_active():
    if not _db:
        return []
    return _db.active_vehicles()


@app.get("/api/history")
async def api_history():
    if not _db:
        return []
    return _db.recent_events()

from fastapi.responses import StreamingResponse
import asyncio


# async def _mjpeg_gen(name: str):
#     """Generator — yield JPEG frames liên tục."""
#     while True:
#         with _frame_lock:
#             jpg = _frames.get(name)
#         if jpg:
#             yield (b"--frame\r\n"
#                    b"Content-Type: image/jpeg\r\n\r\n"
#                    + jpg + b"\r\n")
#         await asyncio.sleep(0.05)


# @app.get("/stream/{name}")
# async def stream(name: str):
#     """MJPEG stream: /stream/plate hoặc /stream/face"""
#     if name not in ("plate", "face"):
#         return {"error": "invalid stream name"}
#     return StreamingResponse(
#         _mjpeg_gen(name),
#         media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/{name}")
async def stream(name: str):
    if name not in ("plate", "face"):
        return {"error": "invalid"}

    async def generate():
        try:
            while True:
                with _frame_lock:
                    jpg = _frames.get(name)
                if jpg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            # Client disconnect → cleanup
            return

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        })


# ── WebSocket ──
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    log.info(f"WS client connected ({len(_clients)} total)")
    try:
        # Gửi stats ban đầu
        if _db:
            await ws.send_text(json.dumps({
                "type": "stats", "data": _db.stats()
            }, default=str))

        # Keep alive — chờ client disconnect
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Ping
                await ws.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _clients:
            _clients.remove(ws)
        log.info(f"WS client disconnected ({len(_clients)} total)")
