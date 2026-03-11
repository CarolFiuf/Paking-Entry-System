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
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
import asyncio

log = logging.getLogger("web")

app = FastAPI(title="Smart Parking Dashboard")

# Global references (set by main.py)
_db = None
# _state = {
#     "mode": "entry",
#     "fps": 0,
#     "last_event": None,
#     "plate_cam_ok": False,
#     "face_cam_ok": False,
#     "deepstream": False,
# }

_state = {}

# Connected WebSocket clients
_clients: list[WebSocket] = []


# Frame buffers cho MJPEG stream (latest frame only)
_frames: dict[str, bytes] = {}  # {"plate": jpeg_bytes, "face": jpeg_bytes}
# _frame_lock = __import__("threading").Lock()
_loop = None # lưu reference tới uvicorn event loop


def update_frame(name: str, frame):
    """Gọi từ pipeline thread — encode + lưu JPEG."""
    if frame is None:
        return
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    _frames[name] = jpg.tobytes()  # atomic reference assignment in CPython


def init(db, state: dict = None):
    """Gọi từ main.py để inject database reference."""
    global _db, _state
    _db = db
    if state:
        # _state.update(state)
        _state = state
        
@app.on_event("startup")
async def _save_loop():
    global _loop
    _loop = asyncio.get_event_loop()


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


# def notify_sync(event: dict):
#     """Gọi từ sync code (pipeline thread) → async broadcast."""
#     try:
#         loop = asyncio.get_event_loop()
#         if loop.is_running():
#             asyncio.ensure_future(broadcast(event))
#     except RuntimeError:
#         pass  # No event loop, skip


def notify_sync(event: dict):
    if _loop and _loop.is_running():
        _loop.call_soon_threadsafe(
            asyncio.ensure_future, broadcast(event))



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
        "mode": _state.get("mode", "?"),
        "fps": _state.get("fps", 0),
        "plate_cam": _state.get("plate_cam_ok", False),
        "face_cam": _state.get("face_cam_ok", False),
        "deepstream": _state.get("deepstream", False),
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


@app.get("/stream/{name}")
async def snapshot(name: str):
    if name not in ("plate", "face"):
        return Response(status_code=404)

    jpg = _frames.get(name)
    if not jpg:
        # 1x1 transparent pixel để tránh broken image
        return Response(
            content=b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01'
                    b'\x00\x00\x01\x00\x01\x00\x00\xff\xd9',
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache"}
        )

    return Response(
        content=jpg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        }
    )


# ── WebSocket ──
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    log.info(f"WS client connected ({len(_clients)} total)")
    try:
        # Gửi stats ban đầu
        if _db:
            stats = _db.stats()
            stats.update({
                "mode": _state.get("mode", "?"),
                "fps": _state.get("fps", 0),
                "plate_cam": _state.get("plate_cam_ok", False),
                "face_cam": _state.get("face_cam_ok", False),
                "deepstream": _state.get("deepstream", False),
            })
            await ws.send_text(json.dumps({
                "type": "stats", "data": stats
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
