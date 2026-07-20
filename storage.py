from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except Exception:  # pragma: no cover - optional in local dev
    psycopg = None  # type: ignore
    dict_row = None  # type: ignore
    Jsonb = None  # type: ignore


CAMERAS = "cctv_ai_cameras"
STREAM_SESSIONS = "cctv_ai_stream_sessions"
TARGETS = "cctv_ai_targets"
TELEMETRY = "cctv_ai_telemetry"
VISION_EVENTS = "cctv_ai_vision_events"
AUDIT_LOG = "cctv_ai_audit_log"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, default=str)


class CCTVStore:
    def __init__(self, database_url: str | None, local_database_path: Path) -> None:
        self.database_url = database_url
        self.local_database_path = local_database_path
        self.use_postgres = bool(database_url and database_url.startswith(("postgres://", "postgresql://")))
        if self.use_postgres and psycopg is None:
            raise RuntimeError("DATABASE_URL is PostgreSQL but psycopg is not installed.")
        if not self.use_postgres:
            self.local_database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def _connect(self):
        if self.use_postgres:
            assert psycopg is not None
            conn = psycopg.connect(self.database_url, row_factory=dict_row)  # type: ignore[arg-type]
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()
            return
        conn = sqlite3.connect(self.local_database_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init(self) -> None:
        if self.use_postgres:
            statements = [
                f"""
                CREATE TABLE IF NOT EXISTS {CAMERAS} (
                    camera_id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    zone TEXT,
                    stream_url TEXT,
                    status TEXT NOT NULL DEFAULT 'registered',
                    topology JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """,
                f"""
                CREATE TABLE IF NOT EXISTS {STREAM_SESSIONS} (
                    session_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    org_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    stopped_at TIMESTAMPTZ,
                    details JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """,
                f"""
                CREATE TABLE IF NOT EXISTS {TARGETS} (
                    target_id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_camera_id TEXT,
                    last_zone TEXT,
                    embedding JSONB NOT NULL DEFAULT '[]'::jsonb,
                    attributes JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    status TEXT NOT NULL DEFAULT 'active'
                )
                """,
                f"""
                CREATE TABLE IF NOT EXISTS {TELEMETRY} (
                    id BIGSERIAL PRIMARY KEY,
                    request_id TEXT,
                    org_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    zone TEXT,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    bbox JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    movement_vector JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    reid_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                    event_type TEXT NOT NULL,
                    snapshot_ref TEXT,
                    predicted_destination JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    qwen_status TEXT NOT NULL DEFAULT 'not_requested',
                    raw JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """,
                f"""
                CREATE TABLE IF NOT EXISTS {VISION_EVENTS} (
                    id BIGSERIAL PRIMARY KEY,
                    request_id TEXT,
                    org_id TEXT NOT NULL,
                    target_id TEXT,
                    camera_id TEXT,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                    analysis JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """,
                f"""
                CREATE TABLE IF NOT EXISTS {AUDIT_LOG} (
                    id BIGSERIAL PRIMARY KEY,
                    org_id TEXT,
                    actor TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    details JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """,
                f"CREATE INDEX IF NOT EXISTS idx_{CAMERAS}_org ON {CAMERAS}(org_id)",
                f"CREATE INDEX IF NOT EXISTS idx_{TARGETS}_org_last_seen ON {TARGETS}(org_id, last_seen_at DESC)",
                f"CREATE INDEX IF NOT EXISTS idx_{TELEMETRY}_target_time ON {TELEMETRY}(target_id, timestamp DESC)",
                f"CREATE INDEX IF NOT EXISTS idx_{TELEMETRY}_org_camera_time ON {TELEMETRY}(org_id, camera_id, timestamp DESC)",
                f"CREATE INDEX IF NOT EXISTS idx_{VISION_EVENTS}_org_target ON {VISION_EVENTS}(org_id, target_id, created_at DESC)",
                f"CREATE INDEX IF NOT EXISTS idx_{AUDIT_LOG}_org_time ON {AUDIT_LOG}(org_id, created_at DESC)",
            ]
            with self._connect() as conn:
                with conn.cursor() as cur:
                    for statement in statements:
                        cur.execute(statement)
            return

        with self._connect() as conn:
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS {CAMERAS} (
                    camera_id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    zone TEXT,
                    stream_url TEXT,
                    status TEXT NOT NULL DEFAULT 'registered',
                    topology TEXT NOT NULL DEFAULT '{{}}',
                    metadata TEXT NOT NULL DEFAULT '{{}}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS {STREAM_SESSIONS} (
                    session_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    org_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    stopped_at TEXT,
                    details TEXT NOT NULL DEFAULT '{{}}'
                );

                CREATE TABLE IF NOT EXISTS {TARGETS} (
                    target_id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    last_camera_id TEXT,
                    last_zone TEXT,
                    embedding TEXT NOT NULL DEFAULT '[]',
                    attributes TEXT NOT NULL DEFAULT '{{}}',
                    status TEXT NOT NULL DEFAULT 'active'
                );

                CREATE TABLE IF NOT EXISTS {TELEMETRY} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    org_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    zone TEXT,
                    timestamp TEXT NOT NULL,
                    bbox TEXT NOT NULL DEFAULT '{{}}',
                    movement_vector TEXT NOT NULL DEFAULT '{{}}',
                    reid_confidence REAL NOT NULL DEFAULT 0,
                    event_type TEXT NOT NULL,
                    snapshot_ref TEXT,
                    predicted_destination TEXT NOT NULL DEFAULT '{{}}',
                    qwen_status TEXT NOT NULL DEFAULT 'not_requested',
                    raw TEXT NOT NULL DEFAULT '{{}}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS {VISION_EVENTS} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    org_id TEXT NOT NULL,
                    target_id TEXT,
                    camera_id TEXT,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    analysis TEXT NOT NULL DEFAULT '{{}}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS {AUDIT_LOG} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id TEXT,
                    actor TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    details TEXT NOT NULL DEFAULT '{{}}',
                    created_at TEXT NOT NULL
                );
                """
            )

    def health(self) -> dict[str, Any]:
        return {
            "status": "success",
            "database": "supabase_postgres" if self.use_postgres else "sqlite_dev",
            "path": None if self.use_postgres else str(self.local_database_path),
        }

    @staticmethod
    def _coerce_json(value: Any, fallback: Any) -> Any:
        if value is None:
            return fallback
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return fallback

    @classmethod
    def _row(cls, row: Any | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        for key, fallback in {
            "topology": {},
            "metadata": {},
            "details": {},
            "embedding": [],
            "attributes": {},
            "bbox": {},
            "movement_vector": {},
            "predicted_destination": {},
            "raw": {},
            "analysis": {},
            "details_json": {},
        }.items():
            if key in item:
                item[key] = cls._coerce_json(item[key], fallback)
        for key in ("created_at", "updated_at", "started_at", "stopped_at", "first_seen_at", "last_seen_at", "timestamp"):
            if isinstance(item.get(key), datetime):
                item[key] = item[key].isoformat()
        return item

    @property
    def _ph(self) -> str:
        return "%s" if self.use_postgres else "?"

    def _json_param(self, value: Any) -> Any:
        if self.use_postgres:
            assert Jsonb is not None
            return Jsonb(value)
        return _json(value)

    def register_camera(self, camera: dict[str, Any]) -> dict[str, Any]:
        timestamp = now_iso()
        existing = self.get_camera(camera["camera_id"])
        created_at = existing.get("created_at") if existing else timestamp
        sql = f"""
            INSERT INTO {CAMERAS} (camera_id, org_id, name, zone, stream_url, status, topology, metadata, created_at, updated_at)
            VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph})
            ON CONFLICT(camera_id) DO UPDATE SET
                org_id=excluded.org_id,
                name=excluded.name,
                zone=excluded.zone,
                stream_url=excluded.stream_url,
                status=excluded.status,
                topology=excluded.topology,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
        """
        params = (
            camera["camera_id"],
            camera["org_id"],
            camera["name"],
            camera.get("zone"),
            camera.get("stream_url"),
            camera.get("status", "registered"),
            self._json_param(camera.get("topology") or {}),
            self._json_param(camera.get("metadata") or {}),
            created_at,
            timestamp,
        )
        with self._connect() as conn:
            with conn.cursor() if self.use_postgres else conn:
                conn.execute(sql, params)
        self.audit(camera["org_id"], "camera_registered", "camera", camera["camera_id"], {"zone": camera.get("zone")})
        return self.get_camera(camera["camera_id"]) or camera

    def get_camera(self, camera_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT * FROM {CAMERAS} WHERE camera_id = %s", (camera_id,))
                    return self._row(cur.fetchone())
            return self._row(conn.execute(f"SELECT * FROM {CAMERAS} WHERE camera_id = ?", (camera_id,)).fetchone())

    def list_cameras(self, org_id: str | None = None) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM {CAMERAS}"
        params: tuple[Any, ...] = ()
        if org_id:
            sql += f" WHERE org_id = {self._ph}"
            params = (org_id,)
        sql += " ORDER BY name"
        with self._connect() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return [self._row(row) or {} for row in cur.fetchall()]
            return [self._row(row) or {} for row in conn.execute(sql, params).fetchall()]

    def set_camera_status(self, camera_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(f"UPDATE {CAMERAS} SET status = {self._ph}, updated_at = {self._ph} WHERE camera_id = {self._ph}", (status, now_iso(), camera_id))

    def create_stream_session(self, session: dict[str, Any]) -> dict[str, Any]:
        sql = f"""
            INSERT INTO {STREAM_SESSIONS} (session_id, camera_id, org_id, status, started_at, details)
            VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph})
        """
        with self._connect() as conn:
            conn.execute(sql, (session["session_id"], session["camera_id"], session["org_id"], session["status"], session["started_at"], self._json_param(session.get("details") or {})))
        self.audit(session["org_id"], "stream_started", "camera", session["camera_id"], session)
        return session

    def stop_stream_session(self, session_id: str) -> dict[str, Any] | None:
        found = None
        timestamp = now_iso()
        with self._connect() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT * FROM {STREAM_SESSIONS} WHERE session_id = %s", (session_id,))
                    found = self._row(cur.fetchone())
                    if not found:
                        return None
                    cur.execute(f"UPDATE {STREAM_SESSIONS} SET status = 'stopped', stopped_at = %s WHERE session_id = %s", (timestamp, session_id))
                    cur.execute(f"SELECT * FROM {STREAM_SESSIONS} WHERE session_id = %s", (session_id,))
                    result = self._row(cur.fetchone())
            else:
                row = conn.execute(f"SELECT * FROM {STREAM_SESSIONS} WHERE session_id = ?", (session_id,)).fetchone()
                found = self._row(row)
                if not found:
                    return None
                conn.execute(f"UPDATE {STREAM_SESSIONS} SET status = 'stopped', stopped_at = ? WHERE session_id = ?", (timestamp, session_id))
                result = self._row(conn.execute(f"SELECT * FROM {STREAM_SESSIONS} WHERE session_id = ?", (session_id,)).fetchone())
        if result:
            self.audit(result["org_id"], "stream_stopped", "camera", result["camera_id"], {"session_id": session_id})
        return result

    def upsert_target(self, target: dict[str, Any]) -> dict[str, Any]:
        timestamp = now_iso()
        existing = self.get_target(target["target_id"])
        sql = f"""
            INSERT INTO {TARGETS} (target_id, org_id, first_seen_at, last_seen_at, last_camera_id, last_zone, embedding, attributes, status)
            VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph})
            ON CONFLICT(target_id) DO UPDATE SET
                last_seen_at=excluded.last_seen_at,
                last_camera_id=excluded.last_camera_id,
                last_zone=excluded.last_zone,
                embedding=excluded.embedding,
                attributes=excluded.attributes,
                status=excluded.status
        """
        params = (
            target["target_id"],
            target["org_id"],
            existing["first_seen_at"] if existing else target.get("first_seen_at", timestamp),
            target.get("last_seen_at", timestamp),
            target.get("last_camera_id"),
            target.get("last_zone"),
            self._json_param(target.get("embedding") or []),
            self._json_param(target.get("attributes") or {}),
            target.get("status", "active"),
        )
        with self._connect() as conn:
            conn.execute(sql, params)
        return self.get_target(target["target_id"]) or target

    def get_target(self, target_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT * FROM {TARGETS} WHERE target_id = %s", (target_id,))
                    return self._row(cur.fetchone())
            return self._row(conn.execute(f"SELECT * FROM {TARGETS} WHERE target_id = ?", (target_id,)).fetchone())

    def list_targets(self, org_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM {TARGETS}"
        params: tuple[Any, ...]
        if org_id:
            sql += f" WHERE org_id = {self._ph}"
            params = (org_id, limit)
        else:
            params = (limit,)
        sql += f" ORDER BY last_seen_at DESC LIMIT {self._ph}"
        with self._connect() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return [self._row(row) or {} for row in cur.fetchall()]
            return [self._row(row) or {} for row in conn.execute(sql, params).fetchall()]

    def add_telemetry(self, telemetry: dict[str, Any]) -> dict[str, Any]:
        created = now_iso()
        sql = f"""
            INSERT INTO {TELEMETRY} (
                request_id, org_id, target_id, camera_id, zone, timestamp, bbox,
                movement_vector, reid_confidence, event_type, snapshot_ref,
                predicted_destination, qwen_status, raw, created_at
            )
            VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph})
        """
        params = (
            telemetry.get("request_id"),
            telemetry["org_id"],
            telemetry["target_id"],
            telemetry["camera_id"],
            telemetry.get("zone"),
            telemetry.get("timestamp") or created,
            self._json_param(telemetry.get("bbox") or {}),
            self._json_param(telemetry.get("movement_vector") or {}),
            float(telemetry.get("reid_confidence") or 0),
            telemetry.get("event_type", "tracking_update"),
            telemetry.get("snapshot_ref"),
            self._json_param(telemetry.get("predicted_destination") or {}),
            telemetry.get("qwen_status", "not_requested"),
            self._json_param(telemetry),
            created,
        )
        with self._connect() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(sql + " RETURNING id", params)
                    row = cur.fetchone()
                    telemetry_id = row["id"] if isinstance(row, dict) else row[0]
            else:
                cursor = conn.execute(sql, params)
                telemetry_id = cursor.lastrowid
        self.audit(telemetry["org_id"], "telemetry_ingested", "target", telemetry["target_id"], {"telemetry_id": telemetry_id, "camera_id": telemetry["camera_id"]})
        return {"id": telemetry_id, **telemetry, "created_at": created}

    def target_history(self, target_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT * FROM {TELEMETRY} WHERE target_id = %s ORDER BY timestamp DESC LIMIT %s", (target_id, limit))
                    return [self._row(row) or {} for row in cur.fetchall()]
            return [self._row(row) or {} for row in conn.execute(f"SELECT * FROM {TELEMETRY} WHERE target_id = ? ORDER BY timestamp DESC LIMIT ?", (target_id, limit)).fetchall()]

    def add_vision_event(self, event: dict[str, Any]) -> dict[str, Any]:
        created = now_iso()
        sql = f"""
            INSERT INTO {VISION_EVENTS} (request_id, org_id, target_id, camera_id, event_type, status, confidence, analysis, created_at)
            VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph})
        """
        params = (
            event.get("request_id"),
            event["org_id"],
            event.get("target_id"),
            event.get("camera_id"),
            event.get("event_type", "manual_verification"),
            event.get("status", "completed"),
            float(event.get("confidence") or 0),
            self._json_param(event.get("analysis") or {}),
            created,
        )
        with self._connect() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(sql + " RETURNING id", params)
                    row = cur.fetchone()
                    event_id = row["id"] if isinstance(row, dict) else row[0]
            else:
                cursor = conn.execute(sql, params)
                event_id = cursor.lastrowid
        self.audit(event["org_id"], "vision_verified", "vision_event", str(event_id), event)
        return {"id": event_id, **event, "created_at": created}

    def audit(self, org_id: str | None, action: str, resource_type: str, resource_id: str | None, details: dict[str, Any] | None = None, actor: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {AUDIT_LOG} (org_id, actor, action, resource_type, resource_id, details, created_at) VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph})",
                (org_id, actor, action, resource_type, resource_id, self._json_param(details or {}), now_iso()),
            )

    def audit_log(self, org_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM {AUDIT_LOG}"
        params: tuple[Any, ...]
        if org_id:
            sql += f" WHERE org_id = {self._ph}"
            params = (org_id, limit)
        else:
            params = (limit,)
        sql += f" ORDER BY id DESC LIMIT {self._ph}"
        with self._connect() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return [self._row(row) or {} for row in cur.fetchall()]
            return [self._row(row) or {} for row in conn.execute(sql, params).fetchall()]


def create_store(database_url: str | None, local_database_path: Path) -> CCTVStore:
    if database_url and database_url.startswith("sqlite:///"):
        return CCTVStore(None, Path(database_url.removeprefix("sqlite:///")))
    return CCTVStore(database_url, local_database_path)
