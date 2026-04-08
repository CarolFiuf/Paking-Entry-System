"""
database.py — PostgreSQL + pgvector
Vector search trong DB luôn, không cần numpy FaceIndex riêng.

Fixes vs v2:
  - PostgreSQL thay SQLite (ACID, multi-client, scale)
  - pgvector cosine search thay numpy matmul
  - Connection pool thay open/close mỗi query (FIX #2)
  - Cached stats thay query mỗi frame (FIX #1)

Setup PostgreSQL:
  sudo apt install postgresql postgresql-contrib
  sudo -u postgres createuser parking --pwprompt
  sudo -u postgres createdb parking --owner=parking
  
  # Cài pgvector extension
  sudo apt install postgresql-16-pgvector   # hoặc build from source
  sudo -u postgres psql -d parking -c 'CREATE EXTENSION vector;'
"""

import numpy as np
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional
import threading

import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector

log = logging.getLogger("db")

# Embedding dimension (ArcFace MobileFaceNet)
DIM = 512


class ParkingDB:
    """
    PostgreSQL + pgvector.
    Connection pool, vector search, cached stats.
    """

    def __init__(self, host: str = "localhost", port: int = 5432,
                 dbname: str = "parking", user: str = "parking",
                 password: str = "parking123", max_cap: int = 500):
        self.max_cap = max_cap

        # Connection pool: min 1, max 5 connections
        # Tránh tạo connection mới mỗi query (FIX #2)
        self._pool = pool.ThreadedConnectionPool(
            minconn=1, maxconn=5,
            host=host, port=port, dbname=dbname,
            user=user, password=password
        )
        
        self._registered_conns = set()

        self._init_schema()
        self._stats_lock = threading.Lock()
        # Cache stats — chỉ cập nhật khi entry/exit (FIX #1)
        self._stats_cache = self._query_stats()
        log.info(f"DB ready: {self._stats_cache['current']} vehicles loaded")

    # @contextmanager
    # def _conn(self):
    #     """Lấy connection từ pool, tự trả lại khi xong."""
    #     conn = self._pool.getconn()
    #     try:
    #         register_vector(conn)
    #         yield conn
    #         conn.commit()
    #     except Exception:
    #         conn.rollback()
    #         raise
    #     finally:
    #         self._pool.putconn(conn)
    
    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            # Chỉ register 1 lần per connection
            conn_id = id(conn)
            if conn_id not in self._registered_conns:
                register_vector(conn)
                self._registered_conns.add(conn_id)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _init_schema(self):
        """Tạo tables + pgvector extension."""
        with self._conn() as conn:
            cur = conn.cursor()

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS active (
                    id SERIAL PRIMARY KEY,
                    plate TEXT NOT NULL,
                    embedding vector({DIM}) NOT NULL,
                    entry_time TIMESTAMP DEFAULT now(),
                    conf_plate REAL DEFAULT 0,
                    conf_face REAL DEFAULT 0
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS parking_log (
                    id SERIAL PRIMARY KEY,
                    plate TEXT NOT NULL,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP DEFAULT now(),
                    duration_min INTEGER,
                    match_conf REAL
                )
            """)

            # Index cho plate lookup (duplicate check)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_active_plate
                ON active(plate)
            """)

            # IVFFlat index cho vector search
            # lists = sqrt(N) ~ 22 cho 500 records
            # Nếu < 100 records, sequential scan nhanh hơn → Postgres tự chọn
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_active_embedding
                ON active USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 22)
            """)

    # ── ENTRY ──
    def entry(self, plate: str, embedding: np.ndarray,
              conf_plate: float = 0, conf_face: float = 0) -> int:
        """
        Đăng ký xe vào.
        Returns: record_id > 0 | -1 (full) | -2 (duplicate plate)
        """
        if self._stats_cache["current"] >= self.max_cap:
            return -1

        emb_list = embedding.astype(np.float32).tolist()

        with self._conn() as conn:
            cur = conn.cursor()

            # Check trùng biển số
            cur.execute("SELECT 1 FROM active WHERE plate = %s", (plate,))
            if cur.fetchone():
                return -2

            cur.execute(
                "INSERT INTO active (plate, embedding, conf_plate, conf_face) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (plate, emb_list, conf_plate, conf_face)
            )
            rid = cur.fetchone()[0]

        # Cập nhật cache
        with self._stats_lock:
            self._stats_cache["current"] += 1
            self._stats_cache["pct"] = round(
            100 * self._stats_cache["current"] / max(self.max_cap, 1), 1)

        log.info(f"ENTRY: {plate} (id={rid}, "
                 f"total={self._stats_cache['current']})")
        return rid

    # ── EXIT: plate lookup + face verify ──
    def find_by_plate(self, plate: str) -> Optional[dict]:
        """Tìm xe trong bảng active theo biển số, trả về id + embedding."""
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, plate, embedding FROM active WHERE plate = %s",
                (plate,))
            row = cur.fetchone()

        if not row:
            return None

        rid, plate, emb_raw = row
        embedding = np.array(emb_raw, dtype=np.float32)
        return {"id": rid, "plate": plate, "embedding": embedding}

    def match_exit(self, embedding: np.ndarray,
                   threshold: float = 0.45) -> Optional[dict]:
        """
        Cosine similarity search bằng pgvector.
        1 - (embedding <=> query) = cosine similarity
        <=> là cosine distance operator của pgvector
        """
        emb_list = embedding.astype(np.float32).tolist()

        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, plate,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM active
                ORDER BY embedding <=> %s::vector
                LIMIT 3
            """, (emb_list, emb_list))

            rows = cur.fetchall()

        if not rows:
            return None

        # Filter by threshold
        matches = [(rid, plate, float(sim))
                    for rid, plate, sim in rows if sim >= threshold]

        if not matches:
            return None

        rid, plate, sim = matches[0]
        return {"id": rid, "plate": plate, "sim": sim, "all": matches}

    def exit(self, record_id: int, match_conf: float = 0) -> bool:
        """Đăng ký xe ra, chuyển vào parking_log."""
        with self._conn() as conn:
            cur = conn.cursor()

            cur.execute(
                "SELECT plate, entry_time FROM active WHERE id = %s",
                (record_id,))
            row = cur.fetchone()
            if not row:
                return False

            plate, entry_time = row
            dur = 0
            if entry_time:
                dur = int((datetime.now() - entry_time).total_seconds() / 60)

            cur.execute(
                "INSERT INTO parking_log "
                "(plate, entry_time, duration_min, match_conf) "
                "VALUES (%s, %s, %s, %s)",
                (plate, entry_time, dur, match_conf))

            cur.execute("DELETE FROM active WHERE id = %s", (record_id,))

        # Cập nhật cache
        with self._stats_lock:
            self._stats_cache["current"] = max(0,
                                                self._stats_cache["current"] - 1)
            self._stats_cache["pct"] = round(
                100 * self._stats_cache["current"] / max(self.max_cap, 1), 1)

        log.info(f"EXIT: {plate} ({dur}min, "
                 f"remain={self._stats_cache['current']})")
        return True

    # ── STATS ──
    def stats(self) -> dict:
        """Trả về cached stats. Không query DB mỗi lần gọi."""
        with self._stats_lock:
            return self._stats_cache.copy()

    def _query_stats(self) -> dict:
        """Query thật từ DB — chỉ gọi khi init."""
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM active")
            current = cur.fetchone()[0]
        return {
            "current": current,
            "capacity": self.max_cap,
            "pct": round(100 * current / max(self.max_cap, 1), 1)
        }

    def close(self):
        """Đóng connection pool."""
        self._pool.closeall()

    # ── Dashboard queries ──
    def active_vehicles(self, limit: int = 50) -> list:
        """Danh sách xe đang trong bãi."""
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, plate, entry_time, conf_plate, conf_face "
                "FROM active ORDER BY entry_time DESC LIMIT %s",
                (limit,))
            rows = cur.fetchall()
        return [{"id": r[0], "plate": r[1],
                 "entry_time": r[2].isoformat() if r[2] else "",
                 "conf_plate": r[3], "conf_face": r[4]}
                for r in rows]

    def recent_events(self, limit: int = 20) -> list:
        """Lịch sử vào/ra gần nhất."""
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT plate, entry_time, exit_time, "
                "duration_min, match_conf "
                "FROM parking_log ORDER BY exit_time DESC LIMIT %s",
                (limit,))
            rows = cur.fetchall()
        return [{"plate": r[0],
                 "entry": r[1].isoformat() if r[1] else "",
                 "exit": r[2].isoformat() if r[2] else "",
                 "duration": r[3], "conf": r[4]}
                for r in rows]
