#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


BASE_URL = os.getenv("CCTV_GATEWAY_URL", "http://127.0.0.1:8010").rstrip("/")
INTERNAL_KEY = os.getenv("CCTV_INTERNAL_KEY", "dev-internal-key")
ORG_ID = os.getenv("CCTV_ORG_ID", "org_abc123")
TARGET_ID = os.getenv("CCTV_TARGET_ID", "REID-FAKE-0001")
INTERVAL_SECONDS = float(os.getenv("CCTV_FAKE_INTERVAL_SECONDS", "2"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def post_json(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Internal-Key": INTERNAL_KEY,
            "X-Client-Name": "relationship-api",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def register_cameras() -> None:
    cameras = [
        {
            "camera_id": "CCTV-1",
            "name": "Gate Entrance Camera",
            "zone": "Gate Entrance",
            "topology": {
                "neighbors": [
                    {
                        "camera_id": "CCTV-2",
                        "zone": "Walkway Blind Spot Exit",
                        "estimated_travel_seconds": 8,
                        "transition_probability": 0.78,
                        "direction_hint": "east",
                        "entry_zone": "walkway-east",
                    }
                ]
            },
        },
        {
            "camera_id": "CCTV-2",
            "name": "Lobby Walkway Camera",
            "zone": "Walkway Blind Spot Exit",
            "topology": {
                "neighbors": [
                    {
                        "camera_id": "CCTV-3",
                        "zone": "Lobby",
                        "estimated_travel_seconds": 10,
                        "transition_probability": 0.64,
                        "direction_hint": "east",
                    }
                ]
            },
        },
        {
            "camera_id": "CCTV-3",
            "name": "Lobby Camera",
            "zone": "Lobby",
            "topology": {"neighbors": []},
        },
    ]
    for camera in cameras:
        payload = {"org_id": ORG_ID, **camera}
        result = post_json("/api/v1/cctv/cameras/register", payload)
        print("registered", result.get("camera", {}).get("camera_id"))


def telemetry_sequence() -> list[dict]:
    base = {
        "org_id": ORG_ID,
        "target_id": TARGET_ID,
        "embedding_source": "lightweight_stable_fallback",
        "attributes": {
            "clothing": "dark hoodie",
            "body_shape": "average_upright",
            "dominant_colors": ["black", "gray"],
            "accessories": ["backpack"],
        },
    }
    return [
        {
            **base,
            "camera_id": "CCTV-1",
            "zone": "Gate Entrance",
            "event_type": "tracking_update",
            "event_confidence": 0.62,
            "bbox": {"x": 90, "y": 42, "w": 58, "h": 168},
            "movement_vector": {"direction": "east", "speed_mps": 1.1},
            "snapshot_ref": "fake://cctv-1-frame-001",
        },
        {
            **base,
            "camera_id": "CCTV-1",
            "zone": "Gate Entrance",
            "event_type": "blind_spot_entry",
            "event_confidence": 0.72,
            "entered_blind_spot": True,
            "last_exit_zone": "walkway-east",
            "bbox": {"x": 210, "y": 45, "w": 56, "h": 165},
            "movement_vector": {"direction": "east", "speed_mps": 1.3},
            "snapshot_ref": "fake://cctv-1-frame-002",
        },
        {
            **base,
            "camera_id": "CCTV-2",
            "zone": "Walkway Blind Spot Exit",
            "event_type": "tracking_update",
            "event_confidence": 0.68,
            "bbox": {"x": 20, "y": 52, "w": 55, "h": 166},
            "movement_vector": {"direction": "east", "speed_mps": 1.2},
            "snapshot_ref": "fake://cctv-2-frame-001",
        },
        {
            **base,
            "camera_id": "CCTV-2",
            "zone": "Walkway Blind Spot Exit",
            "event_type": "loitering_restricted_zone",
            "event_confidence": 0.88,
            "dwell_seconds": 42,
            "bbox": {"x": 80, "y": 53, "w": 55, "h": 166},
            "movement_vector": {"direction": "stationary", "speed_mps": 0.0},
            "snapshot_ref": "fake://cctv-2-frame-002",
        },
        {
            **base,
            "camera_id": "CCTV-3",
            "zone": "Lobby",
            "event_type": "tracking_update",
            "event_confidence": 0.66,
            "bbox": {"x": 140, "y": 48, "w": 59, "h": 169},
            "movement_vector": {"direction": "east", "speed_mps": 0.9},
            "snapshot_ref": "fake://cctv-3-frame-001",
        },
    ]


def main() -> None:
    print(f"posting fake telemetry to {BASE_URL}")
    register_cameras()
    for index, payload in enumerate(telemetry_sequence(), start=1):
        payload["request_id"] = f"fake_mcmot_{index}_{int(time.time())}"
        payload["timestamp"] = now_iso()
        result = post_json("/api/v1/cctv/telemetry/ingest", payload)
        telemetry = result.get("telemetry", {})
        continuity = result.get("tracking_continuity", {})
        blind_spot = continuity.get("blind_spot_prediction", {})
        print(
            json.dumps(
                {
                    "step": index,
                    "target_id": result.get("target", {}).get("target_id"),
                    "camera_id": telemetry.get("camera_id"),
                    "event_type": telemetry.get("event_type"),
                    "match_status": continuity.get("match_status"),
                    "similarity": continuity.get("similarity"),
                    "reappearance": blind_spot.get("estimated_reappearance", [])[:1],
                    "qwen_status": telemetry.get("qwen_status"),
                },
                indent=2,
            )
        )
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
