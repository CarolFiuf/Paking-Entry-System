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
        """
        Tạo tables + pgvector extension.

        Phase 7 migration (idempotent, transactional):
          - active_face_embeddings table (N embeddings/plate, FK CASCADE)
          - UNIQUE constraint on active.plate
          - active.embedding NULLable (kept for rollback path)
          - Backfill afe rows từ active.embedding hiện có
        """
        with self._conn() as conn:
            cur = conn.cursor()

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS active (
                    id SERIAL PRIMARY KEY,
                    plate TEXT NOT NULL,
                    embedding vector({DIM}),
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

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS active_face_embeddings (
                    id          SERIAL PRIMARY KEY,
                    active_id   INT NOT NULL
                                REFERENCES active(id) ON DELETE CASCADE,
                    embedding   vector({DIM}) NOT NULL,
                    quality     REAL,
                    track_id    BIGINT,
                    source_id   INT,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Existing schemas may have NOT NULL on active.embedding.
            # Drop it so Phase 7+ entries can store embeddings only in afe.
            cur.execute("""
                ALTER TABLE active ALTER COLUMN embedding DROP NOT NULL
            """)

            # UNIQUE on plate. Older schemas had a non-unique idx_active_plate;
            # add a unique index alongside (CREATE UNIQUE INDEX IF NOT EXISTS).
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_active_plate_unique
                ON active(plate)
            """)
            # Old non-unique index becomes redundant; drop if present.
            cur.execute("DROP INDEX IF EXISTS idx_active_plate")

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_afe_active
                ON active_face_embeddings(active_id)
            """)

            # IVFFlat index cho vector search trên active.embedding (legacy
            # path / rollback). Plate-scoped match dùng JOIN trên afe nên
            # không cần ivfflat trên afe.
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_active_embedding
                ON active USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 22)
            """)

            # Backfill: cho mỗi active row có embedding nhưng chưa có afe row,
            # copy 1 row sang afe. Idempotent qua NOT EXISTS.
            cur.execute("""
                INSERT INTO active_face_embeddings
                    (active_id, embedding, quality)
                SELECT a.id, a.embedding, a.conf_face
                FROM active a
                WHERE a.embedding IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM active_face_embeddings afe
                      WHERE afe.active_id = a.id
                  )
            """)

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
    def entry(self, plate: str, embeddings,
              conf_plate: float = 0, conf_face: float = 0,
              *,
              qualities: Optional[list] = None,
              track_ids: Optional[list] = None,
              source_ids: Optional[list] = None) -> int:
        """
        Đăng ký xe vào. Hỗ trợ N embeddings/xe (Phase 7).

        Args:
            embeddings: np.ndarray (1 emb, back-compat) hoặc list[np.ndarray].
            qualities, track_ids, source_ids: list song song với embeddings.
                Nếu None → fill bằng (conf_face, None, None).

        Side effects:
            - INSERT 1 row vào `active`, embedding[0] giữ nguyên cho rollback.
            - Bulk INSERT N rows vào `active_face_embeddings`.

        Returns: record_id > 0 | -1 (full) | -2 (duplicate plate)
        """
        if self._stats_cache["current"] >= self.max_cap:
            return -1

        # Normalize: ndarray → [ndarray]
        if isinstance(embeddings, np.ndarray):
            emb_list_np = [embeddings]
        else:
            emb_list_np = list(embeddings)
        if not emb_list_np:
            raise ValueError("entry() requires at least 1 embedding")

        n = len(emb_list_np)
        emb_lists = [e.astype(np.float32).reshape(-1).tolist()
                     for e in emb_list_np]
        for el in emb_lists:
            if len(el) != DIM:
                raise ValueError(
                    f"Invalid embedding dim {len(el)} != {DIM}")

        qual_list = qualities if qualities is not None \
            else [conf_face] * n
        tid_list = track_ids if track_ids is not None else [None] * n
        sid_list = source_ids if source_ids is not None else [None] * n
        if not (len(qual_list) == len(tid_list) == len(sid_list) == n):
            raise ValueError(
                "qualities/track_ids/source_ids length mismatch")

        with self._conn() as conn:
            cur = conn.cursor()

            # Atomic duplicate-plate check: ON CONFLICT trên UNIQUE(plate)
            # tránh race giữa SELECT-then-INSERT khi 2 entry() đồng thời.
            cur.execute(
                "INSERT INTO active (plate, embedding, conf_plate, conf_face) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (plate) DO NOTHING "
                "RETURNING id",
                (plate, emb_lists[0], conf_plate, conf_face)
            )
            row = cur.fetchone()
            if row is None:
                return -2
            rid = row[0]

            # Bulk insert vào afe.
            afe_rows = [
                (rid, emb_lists[i], qual_list[i], tid_list[i], sid_list[i])
                for i in range(n)
            ]
            cur.executemany(
                "INSERT INTO active_face_embeddings "
                "(active_id, embedding, quality, track_id, source_id) "
                "VALUES (%s, %s, %s, %s, %s)",
                afe_rows
            )

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
        """
        Tìm xe trong bảng active theo biển số.

        Trả về:
            {id, plate, embeddings: list[np.ndarray], embedding: np.ndarray}
        `embedding` (singular, = embeddings[0]) giữ cho back-compat với
        main.py tới khi Phase 8 chuyển sang multi-emb.
        """
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT a.id, a.plate, afe.embedding
                FROM active a
                LEFT JOIN active_face_embeddings afe
                    ON afe.active_id = a.id
                WHERE a.plate = %s
                ORDER BY afe.id
            """, (plate,))
            rows = cur.fetchall()

        if not rows:
            return None

        rid = rows[0][0]
        plate_out = rows[0][1]
        embeddings = [self._parse_embedding(r[2]) for r in rows
                      if r[2] is not None]
        if not embeddings:
            return None
        return {
            "id": rid,
            "plate": plate_out,
            "embeddings": embeddings,
            "embedding": embeddings[0],
        }

    def match_exit_by_plate(self, plate: str, candidate_emb: np.ndarray,
                            threshold: float = 0.45) -> Optional[dict]:
        """
        Plate-scoped face verify (Phase 7).

        Pass nếu BẤT KỲ embedding nào của plate đó có cosine sim ≥ threshold
        với candidate (= "any of N matches"). 1 query, JOIN + MIN distance.

        Returns: {active_id, plate, sim, n_embeddings} | None
        """
        emb_list = candidate_emb.astype(np.float32).reshape(-1).tolist()
        if len(emb_list) != DIM:
            raise ValueError(f"Invalid embedding dim {len(emb_list)} != {DIM}")
        max_dist = 1.0 - float(threshold)

        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT a.id, a.plate,
                       MIN(afe.embedding <=> %s::vector) AS dist,
                       COUNT(afe.id) AS n
                FROM active a
                JOIN active_face_embeddings afe ON afe.active_id = a.id
                WHERE a.plate = %s
                GROUP BY a.id, a.plate
                HAVING MIN(afe.embedding <=> %s::vector) <= %s
                LIMIT 1
            """, (emb_list, plate, emb_list, max_dist))
            row = cur.fetchone()

        if not row:
            return None
        active_id, plate_out, dist, n = row
        return {
            "active_id": int(active_id),
            "plate": plate_out,
            "sim": float(1.0 - dist),
            "n_embeddings": int(n),
        }

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
