-- Lemtik CCTV AI Supabase/Postgres schema
-- Run this in the Supabase SQL editor for the project used by DATABASE_URL.

CREATE TABLE IF NOT EXISTS cctv_ai_cameras (
    camera_id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    name TEXT NOT NULL,
    zone TEXT,
    stream_url TEXT,
    status TEXT NOT NULL DEFAULT 'registered',
    topology JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cctv_ai_stream_sessions (
    session_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    org_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stopped_at TIMESTAMPTZ,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS cctv_ai_targets (
    target_id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_camera_id TEXT,
    last_zone TEXT,
    embedding JSONB NOT NULL DEFAULT '[]'::jsonb,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS cctv_ai_telemetry (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT,
    org_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    zone TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    bbox JSONB NOT NULL DEFAULT '{}'::jsonb,
    movement_vector JSONB NOT NULL DEFAULT '{}'::jsonb,
    reid_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    event_type TEXT NOT NULL,
    snapshot_ref TEXT,
    predicted_destination JSONB NOT NULL DEFAULT '{}'::jsonb,
    qwen_status TEXT NOT NULL DEFAULT 'not_requested',
    raw JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cctv_ai_vision_events (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT,
    org_id TEXT NOT NULL,
    target_id TEXT,
    camera_id TEXT,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    analysis JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cctv_ai_audit_log (
    id BIGSERIAL PRIMARY KEY,
    org_id TEXT,
    actor TEXT,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cctv_ai_cameras_org
    ON cctv_ai_cameras(org_id);

CREATE INDEX IF NOT EXISTS idx_cctv_ai_targets_org_last_seen
    ON cctv_ai_targets(org_id, last_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_cctv_ai_telemetry_target_time
    ON cctv_ai_telemetry(target_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_cctv_ai_telemetry_org_camera_time
    ON cctv_ai_telemetry(org_id, camera_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_cctv_ai_vision_events_org_target
    ON cctv_ai_vision_events(org_id, target_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_cctv_ai_audit_log_org_time
    ON cctv_ai_audit_log(org_id, created_at DESC);

-- Optional RLS posture:
-- This internal service should connect with the Supabase service role key
-- through DATABASE_URL. Do not expose these tables directly to anon clients.
ALTER TABLE cctv_ai_cameras ENABLE ROW LEVEL SECURITY;
ALTER TABLE cctv_ai_stream_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE cctv_ai_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE cctv_ai_telemetry ENABLE ROW LEVEL SECURITY;
ALTER TABLE cctv_ai_vision_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE cctv_ai_audit_log ENABLE ROW LEVEL SECURITY;
