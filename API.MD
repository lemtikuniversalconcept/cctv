# CCTV Perception API

Base path may be either `/` or `/api/v1/cctv`.

## Auth

Set `X-Internal-Key` to `INTERNAL_API_KEY` for every non-health request.

## Register Camera

```http
POST /api/v1/cctv/cameras/register
```

```json
{
  "org_id": "org_abc123",
  "camera_id": "CCTV-EKO-NW-302",
  "name": "CCTV North Wing Floor 3",
  "zone": "North Wing Floor 3",
  "stream_url": "rtsp://camera.local/stream",
  "topology": {
    "neighbors": [
      {
        "camera_id": "CCTV-EKO-ELV-301",
        "zone": "Elevator Lobby B",
        "estimated_travel_seconds": 35,
        "transition_probability": 0.72,
        "direction_hint": "east"
      }
    ]
  }
}
```

## Ingest Telemetry

```http
POST /api/v1/cctv/telemetry/ingest
```

```json
{
  "request_id": "req_demo_1",
  "org_id": "org_abc123",
  "camera_id": "CCTV-EKO-NW-302",
  "target_id": "REID-000381",
  "zone": "North Wing Floor 3",
  "event_type": "tailgating",
  "event_confidence": 0.84,
  "bbox": { "x": 110, "y": 50, "w": 80, "h": 180 },
  "movement_vector": { "direction": "east", "speed_mps": 1.4 },
  "snapshot_ref": "s3://bucket/snapshot.jpg",
  "attributes": {
    "clothing": "dark shirt",
    "carrying_object": true
  }
}
```

Response includes the updated target, telemetry row, predicted destination, and whether Qwen verification is recommended.

The response also includes probabilistic tracking continuity and visual descriptor metadata:

```json
{
  "tracking_continuity": {
    "target_id": "REID-000381",
    "match_status": "continuous_track",
    "similarity": 0.91,
    "confidence": 0.91,
    "threshold": 0.82,
    "identity_claim": false,
    "candidates": []
  },
  "visual_descriptors": {
    "clothing": "red shirt",
    "body_shape": "average_upright",
    "dominant_colors": ["red", "black"],
    "accessories": ["backpack"],
    "descriptor_confidence": 0.84
  },
  "qwen_verification": {
    "status": "recommended",
    "trigger_event": "line_crossing",
    "relationship_api_delivery": {
      "status": "sent",
      "status_code": 200,
      "ok": true
    }
  },
  "safety": {
    "identity_claim": false,
    "decision_boundary": "perception_only"
  }
}
```

When `RELATIONSHIP_API_URL` is set and an anomaly event crosses `QWEN_MIN_EVENT_CONFIDENCE`, the service publishes a best-effort webhook to:

```http
POST <RELATIONSHIP_API_URL>/api/v1/ai/analyze-image
```

Supported trigger events include `line_crossing`, `loitering_restricted_zone`, `tailgating`, `unauthorized_access`, `forced_entry`, `object_abandonment`, `camera_obstruction`, `camera_freeze`, and `suspicious_behavior`.

## Vision Verify

```http
POST /api/v1/cctv/vision/verify
```

```json
{
  "request_id": "req_demo_2",
  "org_id": "org_abc123",
  "target_id": "REID-000381",
  "camera_id": "CCTV-EKO-NW-302",
  "event_type": "manual_operator_verification",
  "snapshots": ["https://example.com/snapshot.jpg"],
  "incident_context": {
    "incident_id": "INC-2024-001",
    "description": "Stabbing reported near North West Wing"
  }
}
```

The service returns model-backed analysis when a Qwen key is configured. Supported key environment variables are `QWEN_API_KEY`, `DASHSCOPE_API_KEY`, `ALIBABA_API_KEY`, and `ALIBABA_CLOUD_API_KEY`.

Default US Virginia DashScope multimodal endpoint:

```text
https://dashscope-us.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
```

Default model:

```text
qwen3-vl-flash-us
```

For global deployment scope in US Virginia, use `qwen3-vl-plus` or `qwen3-vl-flash`. Override `QWEN_BASE_URL` for workspace-specific Singapore, Japan, Hong Kong, or other Model Studio endpoints. Set `QWEN_API_PROTOCOL=openai` only when using the OpenAI-compatible Chat Completions API.

## Phone Snapshot / Frame Ingest

```http
POST /api/v1/cctv/frames/ingest
```

```json
{
  "request_id": "phone_001",
  "org_id": "org_abc123",
  "camera_id": "PHONE-CAMERA-TEST",
  "zone": "Phone Test Zone",
  "event_type": "manual_operator_verification",
  "event_confidence": 0.9,
  "verify_vision": true,
  "frame_data": "data:image/jpeg;base64,..."
}
```

Use `GET /api/v1/cctv/phone-test` for a simple browser camera test page.

## Combined Judgement

```http
POST /api/v1/cctv/judgement/analyze
```

```json
{
  "request_id": "judge_001",
  "org_id": "org_abc123",
  "camera_id": "PHONE-CAMERA-TEST",
  "target_id": "REID-000381",
  "voice_transcript": "Same person has been standing near the door for more than one minute and appears to be holding a knife.",
  "snapshots": ["data:image/jpeg;base64,..."],
  "use_qwen": true,
  "loitering_threshold_seconds": 30,
  "incident_context": {
    "zone": "Phone Test Zone"
  }
}
```

The response combines dwell time, frame count, telemetry history, optional voice transcript, and Qwen visual analysis. It remains advisory and does not dispatch or control infrastructure.

## Re-ID Analyze

```http
POST /api/v1/cctv/reid/analyze
```

```json
{
  "org_id": "org_abc123",
  "embedding": [0.12, -0.2, 0.44],
  "threshold": 0.82
}
```

The response returns similarity scores only. It never asserts personal identity.
