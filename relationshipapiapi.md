# CCTV AI Relationship API Integration Guide

This service is an internal Render service. It should not be called directly by the frontend in production.

The frontend or CCTV viewer uploads/captures video frames through the Relationship API. The Relationship API validates org scope, actor permissions, incident context, and request shape, then forwards normalized JSON to this CCTV AI service.

## Service Role

CCTV AI handles visual perception and advisory judgement only:

- camera registry metadata
- frame or telemetry ingestion
- Re-ID continuity within the monitored environment
- dwell-time and loitering analysis
- camera topology prediction
- Qwen Vision verification
- combined visual + voice + telemetry judgement
- audit records for every visual event

It does not display CCTV video, dispatch officers, control devices, or write directly to the dashboard.

## Required Relationship API Env

Relationship API should know this service through:

```env
CCTV_AI_URL=https://lemtik-cctv-perception.onrender.com
CCTV_AI_INTERNAL_KEY=<same value as CCTV service INTERNAL_API_KEY>
```

For the CCTV service on Render:

```env
ENVIRONMENT=production
INTERNAL_API_KEY=<shared internal key>
DATABASE_URL=<supabase-postgres-connection-string>
REQUIRE_RELATIONSHIP_CLIENT=true
RELATIONSHIP_CLIENT_NAME=relationship-api
QWEN_API_KEY=<model-studio-key>
QWEN_API_PROTOCOL=dashscope
QWEN_BASE_URL=https://dashscope-us.aliyuncs.com/api/v1
QWEN_VISION_MODEL=qwen3-vl-flash-us
```

Run `schema.sql` in Supabase before deploying. CCTV AI uses the `cctv_ai_*` table namespace and should connect with a backend/service-role Postgres credential, not an anon client key.

## Required Headers From Relationship API

Every call from Relationship API to this service should include:

```http
Content-Type: application/json
X-Internal-Key: <CCTV_AI_INTERNAL_KEY>
X-Client-Name: relationship-api
X-Request-Id: <uuid>
X-Org-Id: <org_id>
X-Actor-Id: <user_or_service_id>
X-Actor-Role: <role>
X-Idempotency-Key: <uuid>
```

`X-Client-Name` is enforced in production when `REQUIRE_RELATIONSHIP_CLIENT=true`.

## Relationship API Routes To Add

The frontend should call Relationship API routes such as:

```http
POST /api/v1/cctv/cameras/register
GET  /api/v1/cctv/cameras
POST /api/v1/cctv/frames/ingest
POST /api/v1/cctv/telemetry/ingest
POST /api/v1/cctv/judgement/analyze
POST /api/v1/cctv/vision/verify
POST /api/v1/cctv/reid/analyze
POST /api/v1/cctv/reid/correlate
GET  /api/v1/cctv/reid/history/:target_id
POST /api/v1/cctv/topology/predict
GET  /api/v1/cctv/audit-log
```

Relationship API may expose these exact paths publicly/backend-authenticated, then proxy to this service using the same path.

## Downstream CCTV Service Routes

The CCTV service accepts:

```http
GET  /health
GET  /api/v1/cctv/health
GET  /api/v1/cctv/cameras
POST /api/v1/cctv/cameras/register
POST /api/v1/cctv/frames/ingest
POST /api/v1/cctv/telemetry/ingest
POST /api/v1/cctv/judgement/analyze
POST /api/v1/cctv/vision/verify
POST /api/v1/cctv/reid/analyze
POST /api/v1/cctv/reid/correlate
GET  /api/v1/cctv/reid/history/:target_id
POST /api/v1/cctv/topology/predict
GET  /api/v1/cctv/audit-log
```

`/api/v1/cctv/phone-test` is development-only and returns 404 in production.

## Frame Ingest Contract

Relationship API sends sampled frames from the frontend/CCTV viewer:

```json
{
  "request_id": "req_cctv_001",
  "org_id": "org_abc123",
  "incident_id": "INC-2026-001",
  "camera_id": "CCTV-EKO-NW-302",
  "target_id": "REID-000381",
  "zone": "North Wing Floor 3",
  "event_type": "manual_operator_verification",
  "event_confidence": 0.8,
  "frame_data": "data:image/jpeg;base64,...",
  "voice_transcript": "Person has been standing near the entrance for more than one minute.",
  "test_session_id": "session_or_stream_id",
  "verify_vision": false
}
```

Response:

```json
{
  "status": "success",
  "frame_hash": "string",
  "telemetry_result": {
    "target": { "target_id": "REID-..." },
    "telemetry": {
      "event_type": "manual_operator_verification",
      "qwen_status": "recommended"
    }
  },
  "vision_result": null
}
```

Use `verify_vision=false` for frame loops so Qwen is not called on every frame.

## Telemetry Ingest Contract

Use this when the frontend or another video worker already has detections:

```json
{
  "request_id": "req_cctv_002",
  "org_id": "org_abc123",
  "incident_id": "INC-2026-001",
  "camera_id": "CCTV-EKO-NW-302",
  "target_id": "REID-000381",
  "zone": "North Wing Floor 3",
  "event_type": "tailgating",
  "event_confidence": 0.91,
  "bbox": { "x": 110, "y": 50, "w": 80, "h": 180 },
  "movement_vector": { "direction": "east", "speed_mps": 1.4 },
  "snapshot_ref": "blob://snapshot-id",
  "attributes": {
    "clothing": "dark shirt",
    "carrying_object": true
  }
}
```

If `event_type` is one of the configured visual anomaly events and `event_confidence` crosses `QWEN_MIN_EVENT_CONFIDENCE`, CCTV AI publishes a best-effort trigger back to Relationship API:

```http
POST /api/v1/ai/analyze-image
```

Payload shape:

```json
{
  "request_id": "req_cctv_002",
  "org_id": "org_abc123",
  "image_url": "data:image/jpeg;base64,...",
  "incident": {
    "incident_id": "INC-2026-001"
  },
  "context": {
    "source": "cctv_perception_service",
    "event_type": "line_crossing",
    "event_confidence": 0.91,
    "target_id": "REID-000381",
    "camera_id": "CCTV-EKO-NW-302",
    "zone": "North Wing Floor 3",
    "visual_descriptors": {},
    "tracking_continuity": {},
    "identity_claim": false
  }
}
```

This trigger is advisory. Relationship API/Master AI may use the result as context, but CCTV AI does not decide, recommend response plans, or automate infrastructure.

## Combined Judgement Contract

Call this after a burst of frames, when a loitering threshold is crossed, or when the operator submits voice context.

```json
{
  "request_id": "judge_001",
  "org_id": "org_abc123",
  "incident_id": "INC-2026-001",
  "camera_id": "CCTV-EKO-NW-302",
  "target_id": "REID-000381",
  "voice_transcript": "Same person has been standing near the door and appears to be holding a knife.",
  "snapshots": ["data:image/jpeg;base64,..."],
  "use_qwen": true,
  "loitering_threshold_seconds": 30,
  "incident_context": {
    "incident_id": "INC-2026-001",
    "severity": 4,
    "location": "North Wing Floor 3"
  }
}
```

Response:

```json
{
  "status": "success",
  "target_id": "REID-000381",
  "judgement": {
    "event_type": "loitering_restricted_zone",
    "dwell_seconds": 45,
    "frames_seen": 22,
    "voice_transcript": "string",
    "local_confidence": 0.8
  },
  "vision_result": {
    "provider": "qwen",
    "vision_event": {
      "analysis": {
        "threat_summary": "string",
        "confidence": 0.75,
        "visual_explanation": "string",
        "recommended_follow_up_actions": [],
        "gaps": [],
        "identity_claim": false
      }
    }
  },
  "decision_boundary": "advisory_only"
}
```

## Recommended Frontend-To-Relationship Flow

1. Frontend obtains CCTV video or phone/camera preview.
2. Frontend samples frames at 1 frame every 1-3 seconds.
3. Frontend sends sampled frames to Relationship API, not directly to CCTV AI.
4. Relationship API forwards to CCTV AI `/frames/ingest`.
5. For loitering, frontend or Relationship API calls `/judgement/analyze` after the dwell threshold.
6. Relationship API attaches CCTV AI result to the incident timeline.
7. Master AI consumes the visual judgement as incident context.
8. Any infrastructure action still goes through approval and Autonomous Controller.

## Failure Handling

If CCTV AI is unavailable, Relationship API should return a degraded result:

```json
{
  "status": "degraded",
  "service": "cctvai",
  "message": "CCTV AI unavailable; manual CCTV review required.",
  "recommended_actions": ["Manual operator review", "Attach snapshot to incident"]
}
```

If Qwen fails but CCTV AI is alive, CCTV AI already returns `provider=heuristic-fallback`.

## Manual Input Needed

Before wiring Relationship API code, confirm:

- Final Relationship API public route names.
- Whether snapshots will be sent as base64 data URLs, blob IDs, or signed URLs.
- Whether voice arrives as browser transcript text or audio requiring transcription.
- Production `CCTV_AI_URL` after Render deployment.
