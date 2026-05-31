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
        # Tránh tạo connection mới mỗi query
        self._pool = pool.ThreadedConnectionPool(
            minconn=1, maxconn=5,
            host=host, port=port, dbname=dbname,
            user=user, password=password
        )
        
        self._registered_conns = set()

        self._init_schema()
        self._stats_lock = threading.Lock()
        # Cache stats — chỉ cập nhật khi entry/exit
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
        """Tạo tables + pgvector extension. Idempotent + migrate schema cũ."""
        with self._conn() as conn:
            cur = conn.cursor()

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS active (
                    id SERIAL PRIMARY KEY,
                    plate TEXT NOT NULL,
                    entry_time TIMESTAMP DEFAULT now(),
                    conf_plate REAL DEFAULT 0,
                    conf_face REAL DEFAULT 0
                )
            """)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS active_faces (
                    id SERIAL PRIMARY KEY,
                    active_id INTEGER NOT NULL
                        REFERENCES active(id) ON DELETE CASCADE,
                    embedding vector({DIM}) NOT NULL,
                    conf REAL DEFAULT 0,
                    quality REAL DEFAULT 0
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

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_active_plate
                ON active(plate)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_active_faces_active_id
                ON active_faces(active_id)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_active_faces_embedding
                ON active_faces USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 22)
            """)

            # Migrate schema cũ: nếu active.embedding còn tồn tại thì chuyển
            # dữ liệu sang active_faces rồi drop column.
            cur.execute("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'active' AND column_name = 'embedding'
            """)
            if cur.fetchone():
                cur.execute("""
                    INSERT INTO active_faces (active_id, embedding, conf, quality)
                    SELECT id, embedding, conf_face, 1.0 FROM active
                    WHERE NOT EXISTS (
                        SELECT 1 FROM active_faces
                        WHERE active_faces.active_id = active.id
                    )
                """)
                cur.execute("ALTER TABLE active DROP COLUMN embedding")
                log.info("Migrated active.embedding → active_faces")

    @staticmethod
    def _parse_embedding(raw) -> np.ndarray:
        """
        pgvector thường trả về ndarray sau register_vector(), nhưng một số
        connection vẫn có thể trả về chuỗi dạng "[0.1,...]". Chuẩn hóa tại đây
        để logic exit không phụ thuộc vào typecaster của từng connection.
        """
        if isinstance(raw, np.ndarray):
            emb = raw.astype(np.float32, copy=False)
        elif isinstance(raw, str):
            text = raw.strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            sep = "," if "," in text else " "
            emb = np.fromstring(text, sep=sep, dtype=np.float32)
        else:
            emb = np.asarray(raw, dtype=np.float32)

        emb = emb.reshape(-1)
        if emb.size != DIM:
            raise ValueError(f"Invalid embedding dimension: {emb.size} != {DIM}")
        return emb

    # ── ENTRY ──
    def entry(self, plate: str, slots: list,
              conf_plate: float = 0) -> int:
        """
        Đăng ký xe vào với N identity slot (1 ≤ N ≤ MAX_SLOTS).
        slots: list[dict] với keys {embedding, conf, quality}.
        Returns: record_id > 0 | -1 (full) | -2 (duplicate plate) | -3 (no slot)
        """
        if not slots:
            return -3
        if self._stats_cache["current"] >= self.max_cap:
            return -1

        max_face_conf = max(float(s.get("conf", 0)) for s in slots)

        with self._conn() as conn:
            cur = conn.cursor()

            cur.execute("SELECT 1 FROM active WHERE plate = %s", (plate,))
            if cur.fetchone():
                return -2

            cur.execute(
                "INSERT INTO active (plate, conf_plate, conf_face) "
                "VALUES (%s, %s, %s) RETURNING id",
                (plate, conf_plate, max_face_conf)
            )
            rid = cur.fetchone()[0]

            for s in slots:
                emb_list = np.asarray(s["embedding"],
                                      dtype=np.float32).tolist()
                cur.execute(
                    "INSERT INTO active_faces "
                    "(active_id, embedding, conf, quality) "
                    "VALUES (%s, %s, %s, %s)",
                    (rid, emb_list,
                     float(s.get("conf", 0)),
                     float(s.get("quality", 0)))
                )

        # Cập nhật cache
        with self._stats_lock:
            self._stats_cache["current"] += 1
            self._stats_cache["pct"] = round(
            100 * self._stats_cache["current"] / max(self.max_cap, 1), 1)

        log.info(f"ENTRY: {plate} (id={rid}, slots={len(slots)}, "
                 f"total={self._stats_cache['current']})")
        return rid

    # ── EXIT: plate lookup + face verify ──
    def find_by_plate(self, plate: str) -> Optional[dict]:
        """Tìm xe trong bảng active theo biển số, trả về id + list embedding."""
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT a.id, af.embedding
                FROM active a
                JOIN active_faces af ON af.active_id = a.id
                WHERE a.plate = %s
            """, (plate,))
            rows = cur.fetchall()

        if not rows:
            return None

        rid = rows[0][0]
        embeddings = [self._parse_embedding(r[1]) for r in rows]
        return {"id": rid, "plate": plate, "embeddings": embeddings}

    def match_exit(self, embedding: np.ndarray,
                   threshold: float = 0.45) -> Optional[dict]:
        """
        Cosine similarity search bằng pgvector qua bảng active_faces.
        Mỗi xe có thể có nhiều embedding → group theo active_id, lấy max.
        """
        emb_list = embedding.astype(np.float32).tolist()

        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT a.id, a.plate,
                       MAX(1 - (af.embedding <=> %s::vector)) AS similarity
                FROM active a
                JOIN active_faces af ON af.active_id = a.id
                GROUP BY a.id, a.plate
                ORDER BY similarity DESC
                LIMIT 3
            """, (emb_list,))

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
