from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("INTERNAL_API_KEY", "dev-internal-key")
os.environ["LOCAL_DATABASE_PATH"] = str(Path(tempfile.gettempdir()) / "cctvai_test.db")

from config import load_settings
from service import CCTVPerceptionService
from storage import CCTVStore


def make_service() -> CCTVPerceptionService:
    db = Path(tempfile.gettempdir()) / "cctvai_unit_test.db"
    if db.exists():
        db.unlink()
    settings = load_settings(Path(__file__).resolve().parent)
    return CCTVPerceptionService(settings, CCTVStore(None, db))


def test_register_camera_and_predict_destination() -> None:
    service = make_service()
    key = service.settings.internal_api_key
    registered = service.register_camera(
        {
            "org_id": "org_abc123",
            "camera_id": "CCTV-1",
            "name": "Gate 1",
            "zone": "Gate",
            "topology": {
                "neighbors": [
                    {"camera_id": "CCTV-2", "zone": "Lobby", "estimated_travel_seconds": 30, "transition_probability": 0.7, "direction_hint": "east"}
                ]
            },
        },
        key,
    )
    assert registered["status"] == "success"
    prediction = service.predict_destination({"camera_id": "CCTV-1", "movement_vector": {"direction": "east"}}, key)
    assert prediction["prediction"]["likely_reappearance"][0]["camera_id"] == "CCTV-2"


def test_ingest_telemetry_creates_target_and_recommends_qwen() -> None:
    service = make_service()
    key = service.settings.internal_api_key
    service.register_camera({"org_id": "org_abc123", "camera_id": "CCTV-1", "name": "Gate 1"}, key)
    result = service.ingest_telemetry(
        {
            "org_id": "org_abc123",
            "camera_id": "CCTV-1",
            "event_type": "tailgating",
            "event_confidence": 0.9,
            "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
        },
        key,
    )
    assert result["status"] == "success"
    assert result["target"]["target_id"].startswith("REID-")
    assert result["qwen_verification"]["status"] == "recommended"


def test_reid_similarity_does_not_claim_identity() -> None:
    service = make_service()
    key = service.settings.internal_api_key
    embedding = [0.1, 0.2, 0.3]
    service.ingest_telemetry({"org_id": "org_abc123", "camera_id": "CCTV-1", "embedding": embedding}, key)
    result = service.analyze_reid({"org_id": "org_abc123", "embedding": embedding, "threshold": 0.7}, key)
    assert result["matches"]
    assert result["identity_claim"] is False


def test_tracking_continuity_descriptors_and_trigger_status() -> None:
    service = make_service()
    key = service.settings.internal_api_key
    first = service.ingest_telemetry(
        {
            "org_id": "org_abc123",
            "camera_id": "CCTV-1",
            "embedding": [0.2, 0.4, 0.6],
            "bbox": {"x": 10, "y": 20, "w": 50, "h": 150},
            "attributes": {"clothing": "red shirt", "accessories": ["backpack"]},
            "event_type": "line_crossing",
            "event_confidence": 0.95,
        },
        key,
    )
    assert first["target"]["target_id"].startswith("REID-")
    assert first["tracking_continuity"]["match_status"] == "new_track"
    assert first["visual_descriptors"]["clothing"] == "red shirt"
    assert first["qwen_verification"]["status"] == "recommended"
    assert first["qwen_verification"]["relationship_api_delivery"]["status"] == "not_configured"

    second = service.ingest_telemetry(
        {
            "org_id": "org_abc123",
            "camera_id": "CCTV-2",
            "embedding": [0.2, 0.4, 0.6],
            "event_type": "tracking_update",
        },
        key,
    )
    assert second["target"]["target_id"] == first["target"]["target_id"]
    assert second["tracking_continuity"]["similarity"] >= service.settings.default_reid_threshold
    assert second["safety"]["identity_claim"] is False


def test_mcmot_blind_spot_and_access_trigger_context() -> None:
    service = make_service()
    key = service.settings.internal_api_key
    service.register_camera(
        {
            "org_id": "org_abc123",
            "camera_id": "CCTV-A",
            "name": "Gate Camera",
            "topology": {
                "neighbors": [
                    {
                        "camera_id": "CCTV-B",
                        "zone": "Lobby",
                        "estimated_travel_seconds": 18,
                        "transition_probability": 0.74,
                        "direction_hint": "north",
                    }
                ]
            },
        },
        key,
    )
    result = service.ingest_telemetry(
        {
            "org_id": "org_abc123",
            "camera_id": "CCTV-A",
            "event_type": "denied_access_swipe",
            "event_confidence": 0.4,
            "access_denied": True,
            "entered_blind_spot": True,
            "movement_vector": {"direction": "north"},
            "snapshot_ref": "data:image/jpeg;base64,abc123",
            "door_controller_log": {"door_id": "DOOR-1", "access_granted": False, "badge_id": "redacted"},
            "embedding": [0.3, 0.2, 0.1],
        },
        key,
    )
    blind_spot = result["tracking_continuity"]["blind_spot_prediction"]
    assert blind_spot["entered_blind_spot"] is True
    assert blind_spot["estimated_reappearance"][0]["camera_id"] == "CCTV-B"
    assert result["qwen_verification"]["status"] == "recommended"
    assert result["qwen_verification"]["relationship_api_delivery"]["status"] == "not_configured"
    assert result["safety"]["decision_boundary"] == "perception_only"
