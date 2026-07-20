from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import load_settings
from service import CCTVPerceptionService
from storage import create_store

try:
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    Header = None  # type: ignore
    HTTPException = None  # type: ignore
    Request = None  # type: ignore
    JSONResponse = None  # type: ignore


def _json_response(status: int, payload: dict[str, Any]) -> tuple[int, list[tuple[str, str]], bytes]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    return status, [("content-type", "application/json"), ("content-length", str(len(body)))], body


def _html_response(status: int, body: str) -> tuple[int, list[tuple[str, str]], bytes]:
    encoded = body.encode("utf-8")
    return status, [("content-type", "text/html; charset=utf-8"), ("content-length", str(len(encoded)))], encoded


def _relationship_caller_allowed(client_name: str | None) -> bool:
    if not settings.require_relationship_client:
        return True
    return (client_name or "").strip() == settings.relationship_client_name


def _canonical_path(path: str) -> str:
    normalized = path.rstrip("/") or "/"
    for prefix in ("/api/v1/cctv", "/api/v1/cctvai", "/api/v1"):
        if normalized.startswith(prefix):
            return normalized[len(prefix):] or "/"
    return normalized


def _parse_query(query_string: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for chunk in query_string.split("&"):
        if not chunk:
            continue
        if "=" in chunk:
            key, value = chunk.split("=", 1)
        else:
            key, value = chunk, ""
        result[key] = value
    return result


settings = load_settings(Path(__file__).resolve().parent)
store = create_store(settings.database_url, settings.local_database_path)
service = CCTVPerceptionService(settings, store)


class CCTVASGIApp:
    async def _receive_body(self, receive) -> bytes:
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        return bytes(body)

    async def handle(self, method: str, path: str, query_string: str, body: bytes, internal_key: str | None, client_name: str | None = None) -> tuple[int, list[tuple[str, str]], bytes]:
        normalized = _canonical_path(path)
        params = _parse_query(query_string)
        try:
            payload = json.loads(body.decode("utf-8") or "{}") if body else {}
            if method == "GET" and normalized in {"/", "/health", "/ready"}:
                return _json_response(200, service.health() if normalized != "/" else {"status": "ok", "service": "cctvai", "endpoints": self._endpoints()})
            if method == "GET" and normalized == "/phone-test":
                if settings.environment == "production":
                    return _json_response(404, {"status": "error", "message": "endpoint not found"})
                return _html_response(200, _phone_test_html())
            if not _relationship_caller_allowed(client_name):
                return _json_response(403, {"status": "error", "message": "requests must come through Relationship API"})
            if method == "GET" and normalized == "/cameras":
                return _json_response(200, service.list_cameras(params.get("org_id"), internal_key or params.get("x_internal_key")))
            if method == "POST" and normalized == "/cameras/register":
                return _json_response(200, service.register_camera(payload, internal_key))
            if method == "POST" and normalized == "/streams/start":
                return _json_response(200, service.start_stream(payload, internal_key))
            if method == "POST" and normalized == "/streams/stop":
                return _json_response(200, service.stop_stream(payload, internal_key))
            if method == "POST" and normalized == "/telemetry/ingest":
                return _json_response(200, service.ingest_telemetry(payload, internal_key))
            if method == "POST" and normalized == "/frames/ingest":
                return _json_response(200, await service.ingest_frame(payload, internal_key))
            if method == "POST" and normalized == "/judgement/analyze":
                return _json_response(200, await service.analyze_judgement(payload, internal_key))
            if method == "POST" and normalized == "/vision/verify":
                return _json_response(200, await service.verify_vision(payload, internal_key))
            if method == "POST" and normalized == "/reid/analyze":
                return _json_response(200, service.analyze_reid(payload, internal_key))
            if method == "POST" and normalized == "/reid/correlate":
                return _json_response(200, service.correlate_reid(payload, internal_key))
            if method == "GET" and normalized.startswith("/reid/history/"):
                return _json_response(200, service.target_history(normalized.rsplit("/", 1)[-1], internal_key or params.get("x_internal_key")))
            if method == "POST" and normalized == "/topology/predict":
                return _json_response(200, service.predict_destination(payload, internal_key))
            if method == "GET" and normalized == "/audit-log":
                return _json_response(200, service.audit_log(params.get("org_id"), internal_key or params.get("x_internal_key"), int(params.get("limit", "100"))))
            return _json_response(404, {"status": "error", "message": "endpoint not found"})
        except PermissionError as exc:
            return _json_response(401, {"status": "error", "message": str(exc)})
        except ValueError as exc:
            return _json_response(400, {"status": "error", "message": str(exc)})
        except json.JSONDecodeError:
            return _json_response(400, {"status": "error", "message": "invalid JSON body"})
        except Exception as exc:
            return _json_response(500, {"status": "error", "message": f"internal server error: {exc}"})

    def _endpoints(self) -> list[str]:
        return [
            "/health",
            "/cameras",
            "/cameras/register",
            "/streams/start",
            "/streams/stop",
            "/telemetry/ingest",
            "/frames/ingest",
            "/phone-test",
            "/judgement/analyze",
            "/vision/verify",
            "/reid/analyze",
            "/reid/correlate",
            "/reid/history/{target_id}",
            "/topology/predict",
            "/audit-log",
        ]

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await send({"type": "http.response.start", "status": 500, "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"Unsupported scope"})
            return
        headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
        status, response_headers, response_body = await self.handle(
            scope["method"].upper(),
            scope["path"],
            scope.get("query_string", b"").decode("utf-8"),
            await self._receive_body(receive),
            headers.get("x-internal-key"),
            headers.get("x-client-name"),
        )
        await send({"type": "http.response.start", "status": status, "headers": [(k.encode("latin1"), v.encode("latin1")) for k, v in response_headers]})
        await send({"type": "http.response.body", "body": response_body})


fallback_app = CCTVASGIApp()

if FastAPI is not None:
    api_app = FastAPI(title="Lemtik CCTV Perception Service", version="1.0.0")

    def require_key(
        x_internal_key: str | None = Header(default=None, alias="X-Internal-Key"),
        x_client_name: str | None = Header(default=None, alias="X-Client-Name"),
    ) -> str:  # type: ignore[valid-type]
        if x_internal_key != settings.internal_api_key:
            raise HTTPException(status_code=401, detail="invalid internal api key")
        if not _relationship_caller_allowed(x_client_name):
            raise HTTPException(status_code=403, detail="requests must come through Relationship API")
        return x_internal_key

    @api_app.get("/")
    @api_app.get("/api/v1")
    async def root() -> JSONResponse:  # type: ignore[valid-type]
        return JSONResponse({"status": "ok", "service": "cctvai", "endpoints": fallback_app._endpoints()})

    @api_app.get("/health")
    @api_app.get("/ready")
    @api_app.get("/api/v1/cctv/health")
    @api_app.get("/api/v1/cctvai/health")
    async def health() -> JSONResponse:  # type: ignore[valid-type]
        return JSONResponse(service.health())

    @api_app.get("/cameras")
    @api_app.get("/api/v1/cctv/cameras")
    async def cameras(org_id: str | None = None, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(service.list_cameras(org_id, x_internal_key))

    @api_app.post("/cameras/register")
    @api_app.post("/api/v1/cctv/cameras/register")
    async def register_camera(request: Request, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(service.register_camera(await request.json(), x_internal_key))

    @api_app.post("/streams/start")
    @api_app.post("/api/v1/cctv/streams/start")
    async def start_stream(request: Request, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(service.start_stream(await request.json(), x_internal_key))

    @api_app.post("/streams/stop")
    @api_app.post("/api/v1/cctv/streams/stop")
    async def stop_stream(request: Request, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(service.stop_stream(await request.json(), x_internal_key))

    @api_app.post("/telemetry/ingest")
    @api_app.post("/api/v1/cctv/telemetry/ingest")
    async def telemetry(request: Request, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(service.ingest_telemetry(await request.json(), x_internal_key))

    @api_app.post("/frames/ingest")
    @api_app.post("/api/v1/cctv/frames/ingest")
    async def frames(request: Request, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(await service.ingest_frame(await request.json(), x_internal_key))

    @api_app.post("/judgement/analyze")
    @api_app.post("/api/v1/cctv/judgement/analyze")
    async def judgement(request: Request, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(await service.analyze_judgement(await request.json(), x_internal_key))

    @api_app.get("/phone-test")
    @api_app.get("/api/v1/cctv/phone-test")
    async def phone_test() -> Any:
        from fastapi.responses import HTMLResponse

        if settings.environment == "production":
            raise HTTPException(status_code=404, detail="endpoint not found")
        return HTMLResponse(_phone_test_html())

    @api_app.post("/vision/verify")
    @api_app.post("/api/v1/cctv/vision/verify")
    async def vision(request: Request, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(await service.verify_vision(await request.json(), x_internal_key))

    @api_app.post("/reid/analyze")
    @api_app.post("/api/v1/cctv/reid/analyze")
    async def reid_analyze(request: Request, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(service.analyze_reid(await request.json(), x_internal_key))

    @api_app.post("/reid/correlate")
    @api_app.post("/api/v1/cctv/reid/correlate")
    async def reid_correlate(request: Request, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(service.correlate_reid(await request.json(), x_internal_key))

    @api_app.get("/reid/history/{target_id}")
    @api_app.get("/api/v1/cctv/reid/history/{target_id}")
    async def reid_history(target_id: str, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(service.target_history(target_id, x_internal_key))

    @api_app.post("/topology/predict")
    @api_app.post("/api/v1/cctv/topology/predict")
    async def topology_predict(request: Request, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(service.predict_destination(await request.json(), x_internal_key))

    @api_app.get("/audit-log")
    @api_app.get("/api/v1/cctv/audit-log")
    async def audit_log(org_id: str | None = None, limit: int = 100, x_internal_key: str = Header(default="", alias="X-Internal-Key"), x_client_name: str | None = Header(default=None, alias="X-Client-Name")) -> JSONResponse:  # type: ignore[valid-type]
        require_key(x_internal_key, x_client_name)
        return JSONResponse(service.audit_log(org_id, x_internal_key, limit))

    app = api_app
else:
    app = fallback_app


def _phone_test_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lemtik CCTV Phone Test</title>
  <style>
    body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; color: #e2e8f0; }
    main { max-width: 760px; margin: 0 auto; padding: 18px; }
    video, canvas, pre { width: 100%; border: 1px solid #334155; border-radius: 8px; background: #020617; }
    label { display: block; margin-top: 12px; font-size: 13px; color: #cbd5e1; }
    input, select { width: 100%; box-sizing: border-box; margin-top: 5px; padding: 10px; border-radius: 6px; border: 1px solid #475569; background: #111827; color: #e5e7eb; }
    button { margin-top: 12px; padding: 10px 12px; border: 0; border-radius: 6px; background: #22c55e; color: #04130a; font-weight: 700; }
    button.secondary { background: #38bdf8; color: #06121a; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    pre { min-height: 140px; padding: 12px; overflow: auto; white-space: pre-wrap; }
    .status { margin-top: 10px; color: #93c5fd; font-size: 13px; }
  </style>
</head>
<body>
  <main>
    <h1>CCTV Phone Test</h1>
    <video id="video" autoplay playsinline muted></video>
    <canvas id="canvas" hidden></canvas>
    <label>Internal key<input id="key" value="dev-internal-key"></label>
    <div class="row">
      <label>Org ID<input id="org" value="org_abc123"></label>
      <label>Camera ID<input id="camera" value="PHONE-CAMERA-TEST"></label>
    </div>
    <div class="row">
      <label>Zone<input id="zone" value="Phone Test Zone"></label>
      <label>Event
        <select id="event">
          <option>manual_operator_verification</option>
          <option>tailgating</option>
          <option>loitering_restricted_zone</option>
          <option>suspicious_behavior</option>
          <option>camera_obstruction</option>
        </select>
      </label>
    </div>
    <button id="start">Start Camera</button>
    <button id="capture" class="secondary">Capture And Send</button>
    <button id="loop">Start Frame Loop</button>
    <button id="stopLoop" class="secondary">Stop Loop</button>
    <button id="voice">Start Voice</button>
    <button id="judge" class="secondary">Analyze Judgement</button>
    <div class="status" id="status">Frames: 0 | Target: none | Voice: none</div>
    <pre id="output">Waiting...</pre>
  </main>
  <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const output = document.getElementById('output');
    const statusLine = document.getElementById('status');
    let activeTargetId = null;
    let frameCount = 0;
    let loopTimer = null;
    let recentFrames = [];
    let voiceTranscript = '';
    const testSessionId = 'phone_session_' + Date.now();

    function refreshStatus() {
      statusLine.textContent = `Frames: ${frameCount} | Target: ${activeTargetId || 'none'} | Voice: ${voiceTranscript || 'none'}`;
    }

    document.getElementById('start').onclick = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      video.srcObject = stream;
      output.textContent = 'Camera started.';
    };

    async function captureAndSend(showOutput = true) {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      const frame = canvas.toDataURL('image/jpeg', 0.72);
      recentFrames.push(frame);
      recentFrames = recentFrames.slice(-4);
      const body = {
        request_id: 'phone_' + Date.now(),
        org_id: document.getElementById('org').value,
        camera_id: document.getElementById('camera').value,
        target_id: activeTargetId,
        test_session_id: testSessionId,
        zone: document.getElementById('zone').value,
        event_type: document.getElementById('event').value,
        event_confidence: 0.9,
        verify_vision: false,
        voice_transcript: voiceTranscript,
        frame_data: frame
      };
      const response = await fetch('/api/v1/cctv/frames/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Internal-Key': document.getElementById('key').value },
        body: JSON.stringify(body)
      });
      const data = await response.json();
      activeTargetId = data.telemetry_result?.target?.target_id || activeTargetId;
      frameCount += 1;
      refreshStatus();
      if (showOutput) output.textContent = JSON.stringify(data, null, 2);
      return data;
    }

    document.getElementById('capture').onclick = async () => {
      await captureAndSend(true);
    };

    document.getElementById('loop').onclick = () => {
      if (loopTimer) return;
      loopTimer = setInterval(() => captureAndSend(false).catch((err) => output.textContent = err.message), 2000);
      output.textContent = 'Frame loop started. Capturing every 2 seconds.';
    };

    document.getElementById('stopLoop').onclick = () => {
      clearInterval(loopTimer);
      loopTimer = null;
      output.textContent = 'Frame loop stopped.';
    };

    document.getElementById('voice').onclick = () => {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        output.textContent = 'Browser speech recognition is not available. Type the voice note in the console request later or use Chrome.';
        return;
      }
      const recognition = new SpeechRecognition();
      recognition.lang = 'en-NG';
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.onresult = (event) => {
        let text = '';
        for (let i = 0; i < event.results.length; i++) text += event.results[i][0].transcript + ' ';
        voiceTranscript = text.trim();
        refreshStatus();
      };
      recognition.start();
      output.textContent = 'Voice capture started.';
    };

    document.getElementById('judge').onclick = async () => {
      if (!activeTargetId) {
        output.textContent = 'Capture at least one frame first.';
        return;
      }
      const body = {
        request_id: 'judge_' + Date.now(),
        org_id: document.getElementById('org').value,
        camera_id: document.getElementById('camera').value,
        target_id: activeTargetId,
        voice_transcript: voiceTranscript,
        snapshots: recentFrames,
        use_qwen: true,
        incident_context: {
          source: 'phone_test',
          test_session_id: testSessionId,
          zone: document.getElementById('zone').value
        }
      };
      const response = await fetch('/api/v1/cctv/judgement/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Internal-Key': document.getElementById('key').value },
        body: JSON.stringify(body)
      });
      output.textContent = JSON.stringify(await response.json(), null, 2);
    };
  </script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, reload=False)
