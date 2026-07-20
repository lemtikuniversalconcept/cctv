# Lemtik CCTV Perception Service

Standalone internal service for CCTV perception, Re-ID telemetry, camera topology, and selective Qwen Vision verification.

This service extends the existing Master AI CCTV behavior. Master AI still reasons and recommends. Autonomous Controller still executes approved actions. This service only produces visual perception and telemetry.

## Endpoints

- `GET /health`
- `GET /cameras`
- `POST /cameras/register`
- `POST /streams/start`
- `POST /streams/stop`
- `POST /telemetry/ingest`
- `POST /frames/ingest`
- `GET /phone-test`
- `POST /judgement/analyze`
- `POST /vision/verify`
- `POST /reid/analyze`
- `POST /reid/correlate`
- `GET /reid/history/{target_id}`
- `POST /topology/predict`
- `GET /audit-log`

All operational endpoints require `X-Internal-Key`.

## Behavior

- Registers cameras with RTSP URL, zone, and topology metadata.
- Accepts tracking telemetry from live adapters or downstream processors.
- Accepts browser/phone snapshots through `/frames/ingest` for no-CCTV testing.
- Supports a sampled frame loop and browser speech transcript in `/phone-test`.
- Combines dwell time, recent frames, telemetry, voice transcript, and Qwen Vision through `/judgement/analyze`.
- Maintains persistent `REID-*` target records without claiming real-world identity.
- Exports Re-ID similarity scores, continuity status, and visual descriptors such as clothing, body shape, colors, accessories, and motion characteristics.
- Predicts likely reappearance cameras from topology and movement direction.
- Invokes Qwen Vision only through `/vision/verify`, or marks telemetry as `recommended` when trigger events cross the configured confidence threshold.
- Publishes anomaly triggers to Relationship API `/api/v1/ai/analyze-image` when `RELATIONSHIP_API_URL` is configured.
- Stores telemetry, vision events, targets, camera transitions, and audit logs in Supabase/Postgres when `DATABASE_URL` is set.

## Runtime Notes

The committed service is adapter-ready and works without GPU packages. Real RTSP decoding, YOLO, ByteTrack, FastReID, ONVIF, CUDA, and TensorRT should be added as deployment-specific adapters behind the same endpoints.

Use [`.env.example`](./.env.example) as the template.

## Supabase

Production should set `DATABASE_URL` to the Supabase Postgres connection string. The service uses SQLite only when `DATABASE_URL` is not set, for local development.

Run [schema.sql](./schema.sql) in the Supabase SQL editor before deployment. The service writes to prefixed tables:

- `cctv_ai_cameras`
- `cctv_ai_stream_sessions`
- `cctv_ai_targets`
- `cctv_ai_telemetry`
- `cctv_ai_vision_events`
- `cctv_ai_audit_log`

Qwen Vision uses the official DashScope multimodal endpoint by default for US Virginia:

```text
https://dashscope-us.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
```

The default model is `qwen3-vl-flash-us`, which is the US service-scope Qwen-VL model. Set one of `QWEN_API_KEY`, `DASHSCOPE_API_KEY`, `ALIBABA_API_KEY`, or `ALIBABA_CLOUD_API_KEY`.

For global deployment scope in US Virginia, set `QWEN_VISION_MODEL=qwen3-vl-plus` or `qwen3-vl-flash`. To force the OpenAI-compatible gateway, set `QWEN_API_PROTOCOL=openai` and `QWEN_BASE_URL=https://dashscope-us.aliyuncs.com/compatible-mode/v1`.

## Phone Test

Start the service and open:

```text
http://127.0.0.1:8010/api/v1/cctv/phone-test
```

The page can use a browser camera, capture a frame, and send it to `/frames/ingest`.
For a physical phone on the same Wi-Fi, run the server on an address reachable from the phone and open the machine IP instead of `127.0.0.1`.
