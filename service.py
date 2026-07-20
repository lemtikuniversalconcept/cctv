from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover - optional until Qwen calls are enabled
    httpx = None  # type: ignore

from config import Settings
from storage import CCTVStore, now_iso


QWEN_TRIGGER_EVENTS = {
    "unauthorized_access",
    "unauthorized_access_swipe",
    "invalid_access_swipe",
    "denied_access_swipe",
    "perimeter_boundary_violation",
    "line_crossing",
    "tailgating",
    "loitering_restricted_zone",
    "camera_obstruction",
    "camera_freeze",
    "forced_entry",
    "object_abandonment",
    "suspicious_behavior",
    "manual_operator_verification",
}

POSSIBLE_REID_THRESHOLD = 0.60
DEFAULT_LOITERING_DWELL_SECONDS = 30


def _stable_embedding(seed: str, size: int = 32) -> list[float]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    values = []
    for index in range(size):
        byte = digest[index % len(digest)]
        values.append(round((byte / 255.0) * 2 - 1, 6))
    return values


def _bbox_aspect_ratio(bbox: Any) -> float | None:
    if not isinstance(bbox, dict):
        return None
    width = float(bbox.get("w") or bbox.get("width") or 0)
    height = float(bbox.get("h") or bbox.get("height") or 0)
    if not height:
        return None
    return round(width / height, 4)


def fallback_embedding_seed(payload: dict[str, Any], camera: dict[str, Any] | None = None) -> str:
    attributes = payload.get("attributes") or {}
    bbox = payload.get("bbox") or {}
    visual_seed = {
        "camera_id": payload.get("camera_id") or (camera or {}).get("camera_id"),
        "zone": payload.get("zone") or (camera or {}).get("zone"),
        "bbox_aspect_ratio": _bbox_aspect_ratio(bbox),
        "clothing": attributes.get("clothing") or attributes.get("upper_clothing") or attributes.get("dominant_clothing"),
        "body_shape": attributes.get("body_shape"),
        "colors": attributes.get("colors") or attributes.get("dominant_colors") or [],
        "accessories": attributes.get("accessories") or [],
        "motion": attributes.get("motion_characteristics") or payload.get("movement_vector") or {},
    }
    return json.dumps(visual_seed, sort_keys=True, separators=(",", ":"), default=str)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    numerator = sum(a[i] * b[i] for i in range(length))
    left = math.sqrt(sum(a[i] * a[i] for i in range(length)))
    right = math.sqrt(sum(b[i] * b[i] for i in range(length)))
    if left == 0 or right == 0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left * right)))


class CCTVPerceptionService:
    def __init__(self, settings: Settings, store: CCTVStore) -> None:
        self.settings = settings
        self.store = store

    def _check_key(self, internal_key: str | None) -> None:
        if internal_key != self.settings.internal_api_key:
            raise PermissionError("invalid internal api key")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "cctvai",
            "environment": self.settings.environment,
            "capabilities": {
                "rtsp_registry": True,
                "tracking_telemetry": True,
                "reid_matching": True,
                "blind_spot_prediction": True,
                "qwen_vision": bool(self.settings.qwen_api_key),
                "ai_orchestrator_push": bool(self.settings.ai_orchestrator_url),
            },
            "models": {
                "detector": "adapter-ready",
                "tracker": "bytetrack-compatible",
                "reid": "embedding-compatible",
                "vision": self.settings.qwen_vision_model,
                "vision_protocol": self.settings.qwen_api_protocol,
                "vision_base_url": self.settings.qwen_base_url,
            },
            "storage": self.store.health(),
        }

    def register_camera(self, payload: dict[str, Any], internal_key: str | None) -> dict[str, Any]:
        self._check_key(internal_key)
        camera_id = str(payload.get("camera_id") or "").strip()
        org_id = str(payload.get("org_id") or "").strip()
        name = str(payload.get("name") or "").strip()
        if not camera_id or not org_id or not name:
            raise ValueError("camera_id, org_id, and name are required")
        camera = self.store.register_camera(
            {
                "camera_id": camera_id,
                "org_id": org_id,
                "name": name,
                "zone": payload.get("zone"),
                "stream_url": payload.get("stream_url"),
                "status": payload.get("status") or "registered",
                "topology": payload.get("topology") or {},
                "metadata": payload.get("metadata") or {},
            }
        )
        return {"status": "success", "camera": camera}

    def list_cameras(self, org_id: str | None, internal_key: str | None) -> dict[str, Any]:
        self._check_key(internal_key)
        return {"status": "success", "cameras": self.store.list_cameras(org_id)}

    def start_stream(self, payload: dict[str, Any], internal_key: str | None) -> dict[str, Any]:
        self._check_key(internal_key)
        camera_id = str(payload.get("camera_id") or "").strip()
        camera = self.store.get_camera(camera_id)
        if not camera:
            raise ValueError("camera not registered")
        session = {
            "session_id": payload.get("session_id") or f"stream_{uuid.uuid4().hex[:12]}",
            "camera_id": camera_id,
            "org_id": camera["org_id"],
            "status": "running",
            "started_at": now_iso(),
            "details": {
                "mode": payload.get("mode", "telemetry"),
                "frame_sample_rate": payload.get("frame_sample_rate", 5),
                "stream_url_configured": bool(camera.get("stream_url")),
                "note": "Runtime decoder is adapter-ready. Use telemetry ingestion for local/offline operation.",
            },
        }
        self.store.set_camera_status(camera_id, "streaming")
        return {"status": "success", "stream": self.store.create_stream_session(session)}

    def stop_stream(self, payload: dict[str, Any], internal_key: str | None) -> dict[str, Any]:
        self._check_key(internal_key)
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        stopped = self.store.stop_stream_session(session_id)
        if not stopped:
            raise ValueError("stream session not found")
        self.store.set_camera_status(stopped["camera_id"], "registered")
        return {"status": "success", "stream": stopped}

    def ingest_telemetry(self, payload: dict[str, Any], internal_key: str | None) -> dict[str, Any]:
        self._check_key(internal_key)
        return self._ingest_telemetry_authorized(payload)

    def _ingest_telemetry_authorized(self, payload: dict[str, Any]) -> dict[str, Any]:
        org_id = str(payload.get("org_id") or "").strip()
        camera_id = str(payload.get("camera_id") or "").strip()
        event_type = str(payload.get("event_type") or "tracking_update").strip()
        if not org_id or not camera_id:
            raise ValueError("org_id and camera_id are required")
        camera = self.store.get_camera(camera_id)
        if not camera:
            self.store.register_camera(
                {
                    "camera_id": camera_id,
                    "org_id": org_id,
                    "name": payload.get("camera_name") or camera_id,
                    "zone": payload.get("zone"),
                    "status": "observed",
                    "topology": payload.get("topology") or {},
                    "metadata": {"auto_registered": True},
                }
            )
            camera = self.store.get_camera(camera_id)
        target_id = str(payload.get("target_id") or "").strip()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            embedding = _stable_embedding(fallback_embedding_seed(payload, camera))
        descriptors = self._visual_descriptors(payload)
        continuity = self._tracking_continuity(org_id, embedding, target_id, payload=payload, camera=camera)
        if not target_id:
            target_id = continuity["target_id"]
        prediction = self.predict_destination({"camera_id": camera_id, "target_id": target_id, "movement_vector": payload.get("movement_vector") or {}}, internal_key=None, already_authorized=True)
        event_confidence = float(payload.get("event_confidence") or payload.get("reid_confidence") or continuity.get("confidence") or 0)
        qwen_status = "recommended" if self._should_request_qwen(event_type, event_confidence, payload=payload, continuity=continuity) else "not_requested"
        target = self.store.upsert_target(
            {
                "target_id": target_id,
                "org_id": org_id,
                "last_seen_at": payload.get("timestamp") or now_iso(),
                "last_camera_id": camera_id,
                "last_zone": payload.get("zone") or (camera or {}).get("zone"),
                "embedding": embedding,
                "attributes": {**(payload.get("attributes") or {}), "visual_descriptors": descriptors, "tracking_continuity": continuity},
                "status": payload.get("target_status", "active"),
            }
        )
        telemetry = self.store.add_telemetry(
            {
                "request_id": payload.get("request_id"),
                "org_id": org_id,
                "target_id": target_id,
                "camera_id": camera_id,
                "zone": payload.get("zone") or (camera or {}).get("zone"),
                "timestamp": payload.get("timestamp") or now_iso(),
                "bbox": payload.get("bbox") or {},
                "movement_vector": payload.get("movement_vector") or {},
                "reid_confidence": float(payload.get("reid_confidence") or continuity.get("similarity") or 0),
                "event_type": event_type,
                "snapshot_ref": payload.get("snapshot_ref"),
                "predicted_destination": prediction.get("prediction"),
                "qwen_status": qwen_status,
                "visual_descriptors": descriptors,
                "tracking_continuity": continuity,
                "similarity_score": continuity.get("similarity"),
                "identity_assertion": False,
            }
        )
        trigger_delivery = None
        if qwen_status == "recommended":
            trigger_delivery = self._publish_relationship_image_analysis(payload, target_id, event_type, event_confidence, descriptors, continuity)
        return {
            "status": "success",
            "target": target,
            "telemetry": telemetry,
            "tracking_continuity": continuity,
            "visual_descriptors": descriptors,
            "qwen_verification": {"status": qwen_status, "trigger_event": event_type, "relationship_api_delivery": trigger_delivery},
            "safety": {"identity_claim": False, "decision_boundary": "perception_only"},
        }

    async def ingest_frame(self, payload: dict[str, Any], internal_key: str | None) -> dict[str, Any]:
        self._check_key(internal_key)
        frame_data = str(payload.get("frame_data") or "").strip()
        if not frame_data:
            raise ValueError("frame_data is required")
        frame_hash = hashlib.sha256(frame_data.encode("utf-8")).hexdigest()[:16]
        telemetry = self._ingest_telemetry_authorized(
            {
                "request_id": payload.get("request_id"),
                "org_id": payload.get("org_id"),
                "camera_id": payload.get("camera_id") or "PHONE-CAMERA-TEST",
                "target_id": payload.get("target_id"),
                "camera_name": payload.get("camera_name") or "Phone Camera Test",
                "zone": payload.get("zone") or "Phone Test Zone",
                "event_type": payload.get("event_type") or "manual_operator_verification",
                "event_confidence": payload.get("event_confidence", 0.8),
                "bbox": payload.get("bbox") or {"source": "phone_snapshot", "frame_hash": frame_hash},
                "movement_vector": payload.get("movement_vector") or {},
                "snapshot_ref": f"phone-frame:{frame_hash}",
                "analysis_image_url": frame_data,
                "attributes": {
                    **(payload.get("attributes") or {}),
                    "source": "phone_camera",
                    "test_session_id": payload.get("test_session_id"),
                    "voice_transcript": payload.get("voice_transcript"),
                },
            }
        )
        vision = None
        if payload.get("verify_vision"):
            vision = await self.verify_vision(
                {
                    "request_id": payload.get("request_id"),
                    "org_id": payload.get("org_id"),
                    "target_id": telemetry["target"]["target_id"],
                    "camera_id": payload.get("camera_id") or "PHONE-CAMERA-TEST",
                    "event_type": payload.get("event_type") or "manual_operator_verification",
                    "snapshots": [frame_data],
                    "metadata": {"frame_hash": frame_hash, "source": "phone_test"},
                    "incident_context": payload.get("incident_context") or {},
                },
                internal_key,
            )
        return {"status": "success", "frame_hash": frame_hash, "telemetry_result": telemetry, "vision_result": vision}

    def _visual_descriptors(self, payload: dict[str, Any]) -> dict[str, Any]:
        attributes = payload.get("attributes") or {}
        bbox = payload.get("bbox") or {}
        width = float(bbox.get("w") or bbox.get("width") or 0) if isinstance(bbox, dict) else 0.0
        height = float(bbox.get("h") or bbox.get("height") or 0) if isinstance(bbox, dict) else 0.0
        aspect_ratio = round(width / height, 3) if height else None
        clothing = attributes.get("clothing") or attributes.get("upper_clothing") or attributes.get("dominant_clothing")
        body_shape = attributes.get("body_shape") or self._body_shape_from_bbox(aspect_ratio)
        colors = attributes.get("colors") or attributes.get("dominant_colors") or []
        accessories = attributes.get("accessories") or []
        descriptor_fields = [clothing, body_shape, colors, accessories, aspect_ratio]
        confidence = min(1.0, 0.2 + 0.16 * sum(1 for item in descriptor_fields if item not in (None, "", [], {})))
        return {
            "clothing": clothing,
            "body_shape": body_shape,
            "dominant_colors": colors,
            "accessories": accessories,
            "bbox_aspect_ratio": aspect_ratio,
            "motion_characteristics": attributes.get("motion_characteristics") or payload.get("movement_vector") or {},
            "descriptor_confidence": round(confidence, 3),
            "source": "provided_or_estimated",
        }

    def _body_shape_from_bbox(self, aspect_ratio: float | None) -> str | None:
        if aspect_ratio is None:
            return None
        if aspect_ratio < 0.35:
            return "tall_narrow"
        if aspect_ratio > 0.65:
            return "wide_or_crouched"
        return "average_upright"

    def _tracking_continuity(
        self,
        org_id: str,
        embedding: list[float],
        provided_target_id: str | None = None,
        payload: dict[str, Any] | None = None,
        camera: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = payload or {}
        camera = camera or {}
        blind_spot = self._blind_spot_transition_estimates(camera, payload)
        if provided_target_id:
            return {
                "target_id": provided_target_id,
                "match_status": "provided_target",
                "similarity": 1.0,
                "confidence": 1.0,
                "threshold": self.settings.default_reid_threshold,
                "embedding_source": payload.get("embedding_source") or "fastreid_or_adapter_embedding",
                "blind_spot_prediction": blind_spot,
                "mcmot_enabled": True,
                "identity_claim": False,
                "candidates": [],
            }
        candidates = []
        for target in self.store.list_targets(org_id, limit=100):
            score = cosine_similarity(embedding, target.get("embedding") or [])
            candidates.append(
                {
                    "target_id": target["target_id"],
                    "similarity": round(score, 4),
                    "last_camera_id": target.get("last_camera_id"),
                    "last_zone": target.get("last_zone"),
                    "match_confidence": round(score, 4),
                    "visual_descriptors": (target.get("attributes") or {}).get("visual_descriptors") or {},
                }
            )
        candidates.sort(key=lambda item: item["similarity"], reverse=True)
        best = candidates[0] if candidates else None
        if best and best["similarity"] >= self.settings.default_reid_threshold:
            target_id = best["target_id"]
            status = "continuous_track"
            confidence = best["similarity"]
        else:
            target_id = f"REID-{uuid.uuid4().hex[:8].upper()}"
            status = "new_track"
            confidence = 0.0
        possible = bool(best and POSSIBLE_REID_THRESHOLD <= best["similarity"] < self.settings.default_reid_threshold)
        return {
            "target_id": target_id,
            "match_status": status,
            "similarity": round(float(confidence), 4),
            "confidence": round(float(confidence), 4),
            "threshold": self.settings.default_reid_threshold,
            "possible_match_below_threshold": possible,
            "best_candidate": best,
            "candidates": candidates[:5],
            "embedding_source": payload.get("embedding_source") or "fastreid_or_adapter_embedding",
            "blind_spot_prediction": blind_spot,
            "mcmot_enabled": True,
            "identity_claim": False,
        }

    def _blind_spot_transition_estimates(self, camera: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        topology = camera.get("topology") or payload.get("proximity_map") or payload.get("camera_topology") or {}
        neighbors = topology.get("neighbors") or topology.get("neighboring_cameras") or []
        movement = payload.get("movement_vector") or {}
        in_blind_spot = bool(payload.get("entered_blind_spot") or payload.get("blind_spot") or payload.get("event_type") in {"blind_spot_entry", "target_lost"})
        ranked = []
        for neighbor in neighbors:
            probability = float(neighbor.get("transition_probability") or neighbor.get("probability") or 0.5)
            travel = int(neighbor.get("estimated_travel_seconds") or neighbor.get("travel_time_seconds") or 60)
            if movement.get("direction") and movement.get("direction") == neighbor.get("direction_hint"):
                probability += 0.15
            if payload.get("last_exit_zone") and payload.get("last_exit_zone") == neighbor.get("entry_zone"):
                probability += 0.1
            ranked.append(
                {
                    "camera_id": neighbor.get("camera_id"),
                    "zone": neighbor.get("zone"),
                    "estimated_transition_seconds": travel,
                    "transition_probability": round(min(probability, 0.99), 3),
                    "entry_zone": neighbor.get("entry_zone"),
                }
            )
        ranked.sort(key=lambda item: item["transition_probability"], reverse=True)
        return {
            "entered_blind_spot": in_blind_spot,
            "source_camera_id": camera.get("camera_id") or payload.get("camera_id"),
            "estimated_reappearance": ranked[:5],
            "confidence": ranked[0]["transition_probability"] if ranked else 0.0,
            "horizon_seconds": self.settings.topology_prediction_horizon_seconds,
        }

    async def analyze_judgement(self, payload: dict[str, Any], internal_key: str | None) -> dict[str, Any]:
        self._check_key(internal_key)
        org_id = str(payload.get("org_id") or "").strip()
        target_id = str(payload.get("target_id") or "").strip()
        if not org_id:
            raise ValueError("org_id is required")
        if not target_id:
            raise ValueError("target_id is required")
        history = self.store.target_history(target_id, self.settings.max_history_items)
        dwell_seconds = self._dwell_seconds(history)
        frames_seen = len(history)
        voice_transcript = str(payload.get("voice_transcript") or "").strip()
        event_type = self._classify_combined_event(dwell_seconds, voice_transcript, payload.get("operator_event_type"))
        local_assessment = {
            "event_type": event_type,
            "dwell_seconds": dwell_seconds,
            "frames_seen": frames_seen,
            "loitering_threshold_seconds": int(payload.get("loitering_threshold_seconds") or 30),
            "voice_transcript": voice_transcript,
            "telemetry_summary": self._telemetry_summary(history),
            "local_confidence": self._local_judgement_confidence(dwell_seconds, frames_seen, voice_transcript, event_type),
        }
        snapshots = payload.get("snapshots") or []
        vision_result = None
        if payload.get("use_qwen", True):
            vision_result = await self.verify_vision(
                {
                    "request_id": payload.get("request_id"),
                    "org_id": org_id,
                    "target_id": target_id,
                    "camera_id": payload.get("camera_id") or (history[0].get("camera_id") if history else None),
                    "event_type": event_type,
                    "snapshots": snapshots,
                    "metadata": {
                        "combined_judgement": True,
                        "dwell_seconds": dwell_seconds,
                        "frames_seen": frames_seen,
                        "voice_transcript": voice_transcript,
                    },
                    "incident_context": payload.get("incident_context") or {},
                },
                internal_key,
            )
        return {
            "status": "success",
            "target_id": target_id,
            "judgement": local_assessment,
            "vision_result": vision_result,
            "decision_boundary": "advisory_only",
        }

    def _dwell_seconds(self, history: list[dict[str, Any]]) -> int:
        if len(history) < 2:
            return 0
        timestamps = []
        for item in history:
            raw = item.get("timestamp") or item.get("created_at")
            if not raw:
                continue
            try:
                timestamps.append(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
            except ValueError:
                continue
        if len(timestamps) < 2:
            return 0
        return max(0, int((max(timestamps) - min(timestamps)).total_seconds()))

    def _classify_combined_event(self, dwell_seconds: int, voice_transcript: str, operator_event_type: Any) -> str:
        if operator_event_type:
            return str(operator_event_type)
        lowered = voice_transcript.lower()
        weapon_words = {"knife", "gun", "weapon", "machete", "blade", "armed"}
        if any(word in lowered for word in weapon_words):
            return "suspicious_behavior"
        if dwell_seconds >= 30:
            return "loitering_restricted_zone"
        return "manual_operator_verification"

    def _local_judgement_confidence(self, dwell_seconds: int, frames_seen: int, voice_transcript: str, event_type: str) -> float:
        confidence = 0.25
        if frames_seen >= 5:
            confidence += 0.2
        if dwell_seconds >= 30:
            confidence += 0.25
        if voice_transcript:
            confidence += 0.15
        if event_type in QWEN_TRIGGER_EVENTS:
            confidence += 0.1
        return round(min(confidence, 0.9), 2)

    def _telemetry_summary(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        cameras = []
        zones = []
        event_types = []
        for item in history:
            if item.get("camera_id") and item.get("camera_id") not in cameras:
                cameras.append(item["camera_id"])
            if item.get("zone") and item.get("zone") not in zones:
                zones.append(item["zone"])
            if item.get("event_type") and item.get("event_type") not in event_types:
                event_types.append(item["event_type"])
        return {"cameras": cameras, "zones": zones, "event_types": event_types}

    def _match_or_create_target(self, org_id: str, embedding: list[float]) -> str:
        best: tuple[str, float] | None = None
        for target in self.store.list_targets(org_id, limit=100):
            score = cosine_similarity(embedding, target.get("embedding") or [])
            if best is None or score > best[1]:
                best = (target["target_id"], score)
        if best and best[1] >= self.settings.default_reid_threshold:
            return best[0]
        return f"REID-{uuid.uuid4().hex[:8].upper()}"

    def analyze_reid(self, payload: dict[str, Any], internal_key: str | None) -> dict[str, Any]:
        self._check_key(internal_key)
        org_id = str(payload.get("org_id") or "").strip()
        embedding = payload.get("embedding")
        if not org_id or not isinstance(embedding, list):
            raise ValueError("org_id and embedding are required")
        candidates = []
        threshold = float(payload.get("threshold") or self.settings.default_reid_threshold)
        descriptors = self._visual_descriptors(payload)
        for target in self.store.list_targets(org_id, limit=100):
            score = cosine_similarity(embedding, target.get("embedding") or [])
            if score >= threshold:
                candidates.append(
                    {
                        "target_id": target["target_id"],
                        "similarity": round(score, 4),
                        "confidence": round(score, 4),
                        "last_camera_id": target.get("last_camera_id"),
                        "last_zone": target.get("last_zone"),
                        "visual_descriptors": (target.get("attributes") or {}).get("visual_descriptors") or {},
                        "identity_claim": False,
                    }
                )
        candidates.sort(key=lambda item: item["similarity"], reverse=True)
        return {
            "status": "success",
            "matches": candidates,
            "query_visual_descriptors": descriptors,
            "threshold": threshold,
            "identity_claim": False,
            "safety_note": "Similarity is probabilistic and must not be treated as absolute identity.",
        }

    def correlate_reid(self, payload: dict[str, Any], internal_key: str | None) -> dict[str, Any]:
        self._check_key(internal_key)
        target_id = str(payload.get("target_id") or "").strip()
        if not target_id:
            raise ValueError("target_id is required")
        history = self.store.target_history(target_id, self.settings.max_history_items)
        transitions = []
        previous = None
        for item in reversed(history):
            current = item.get("camera_id")
            if previous and current and previous != current:
                transitions.append({"from_camera_id": previous, "to_camera_id": current, "timestamp": item.get("timestamp"), "confidence": item.get("reid_confidence", 0)})
            previous = current
        return {"status": "success", "target_id": target_id, "history_count": len(history), "camera_transitions": transitions}

    def target_history(self, target_id: str, internal_key: str | None) -> dict[str, Any]:
        self._check_key(internal_key)
        return {"status": "success", "target_id": target_id, "history": self.store.target_history(target_id, self.settings.max_history_items)}

    def predict_destination(self, payload: dict[str, Any], internal_key: str | None, already_authorized: bool = False) -> dict[str, Any]:
        if not already_authorized:
            self._check_key(internal_key)
        camera_id = str(payload.get("camera_id") or "").strip()
        camera = self.store.get_camera(camera_id)
        topology = (camera or {}).get("topology") or {}
        neighbors = topology.get("neighbors") or []
        movement = payload.get("movement_vector") or {}
        ranked = []
        for neighbor in neighbors:
            probability = float(neighbor.get("transition_probability") or neighbor.get("probability") or 0.5)
            travel = int(neighbor.get("estimated_travel_seconds") or 60)
            if movement.get("direction") and movement.get("direction") == neighbor.get("direction_hint"):
                probability += 0.15
            ranked.append(
                {
                    "camera_id": neighbor.get("camera_id"),
                    "zone": neighbor.get("zone"),
                    "estimated_travel_seconds": travel,
                    "transition_probability": round(min(probability, 0.99), 3),
                }
            )
        ranked.sort(key=lambda item: item["transition_probability"], reverse=True)
        prediction = {
            "source_camera_id": camera_id,
            "target_id": payload.get("target_id"),
            "horizon_seconds": self.settings.topology_prediction_horizon_seconds,
            "likely_reappearance": ranked[:5],
            "confidence": round(ranked[0]["transition_probability"], 3) if ranked else 0.0,
        }
        return {"status": "success", "prediction": prediction}

    def _should_request_qwen(self, event_type: str, event_confidence: float, payload: dict[str, Any] | None = None, continuity: dict[str, Any] | None = None) -> bool:
        payload = payload or {}
        dwell_seconds = int(payload.get("dwell_seconds") or payload.get("loitering_dwell_seconds") or 0)
        access_denied = bool(payload.get("access_denied") or payload.get("door_controller_log", {}).get("access_granted") is False)
        perimeter_violation = bool(payload.get("perimeter_violation") or event_type in {"line_crossing", "perimeter_boundary_violation"})
        high_loitering = event_type == "loitering_restricted_zone" and dwell_seconds >= int(payload.get("loitering_threshold_seconds") or DEFAULT_LOITERING_DWELL_SECONDS)
        configured_event = event_type in QWEN_TRIGGER_EVENTS and event_confidence >= self.settings.qwen_min_event_confidence
        return bool(configured_event or high_loitering or access_denied or perimeter_violation)

    def _publish_relationship_image_analysis(
        self,
        payload: dict[str, Any],
        target_id: str,
        event_type: str,
        event_confidence: float,
        descriptors: dict[str, Any],
        continuity: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.settings.relationship_api_url:
            return {"status": "not_configured", "reason": "RELATIONSHIP_API_URL is not set"}
        if httpx is None:
            return {"status": "not_sent", "reason": "httpx is not installed"}
        snapshot_bundle = self._nearest_snapshot_bundle(payload)
        door_controller_log = payload.get("door_controller_log") or payload.get("access_control_log") or {}
        image_url = snapshot_bundle.get("image_url")
        if not image_url and isinstance(payload.get("snapshot_ref"), str) and payload["snapshot_ref"].startswith(("http://", "https://", "data:image")):
            image_url = payload["snapshot_ref"]
        if not image_url:
            return {"status": "not_sent", "reason": "no image URL or data URL available for /ai/analyze-image"}
        base = self.settings.relationship_api_url.rstrip("/")
        url = f"{base}/api/v1/ai/analyze-image"
        request_id = payload.get("request_id") or f"cctv_trigger_{uuid.uuid4().hex[:12]}"
        body = {
            "request_id": request_id,
            "org_id": payload.get("org_id"),
            "image_url": image_url,
            "incident": payload.get("incident_context") or {"incident_id": payload.get("incident_id")},
            "context": {
                "source": "cctv_perception_service",
                "event_type": event_type,
                "event_confidence": round(float(event_confidence), 4),
                "target_id": target_id,
                "camera_id": payload.get("camera_id"),
                "zone": payload.get("zone"),
                "visual_descriptors": descriptors,
                "tracking_continuity": continuity,
                "nearest_camera_snapshot": snapshot_bundle,
                "door_controller_log": door_controller_log,
                "identity_claim": False,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "X-Client-Name": "cctv-perception-service",
            "X-Request-Id": str(request_id),
        }
        if self.settings.relationship_api_key:
            headers["X-Internal-Key"] = self.settings.relationship_api_key
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(url, json=body, headers=headers)
            return {"status": "sent", "url": url, "status_code": response.status_code, "ok": 200 <= response.status_code < 300}
        except Exception as exc:
            return {"status": "failed", "url": url, "error": str(exc)}

    def _nearest_snapshot_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_url = payload.get("analysis_image_url") or payload.get("image_url") or payload.get("snapshot_url")
        snapshot_ref = payload.get("snapshot_ref")
        if not image_url and isinstance(snapshot_ref, str) and snapshot_ref.startswith(("http://", "https://", "data:image")):
            image_url = snapshot_ref
        return {
            "image_url": image_url,
            "snapshot_ref": snapshot_ref,
            "camera_id": payload.get("nearest_camera_id") or payload.get("camera_id"),
            "zone": payload.get("nearest_camera_zone") or payload.get("zone"),
            "timestamp": payload.get("timestamp") or now_iso(),
        }

    async def verify_vision(self, payload: dict[str, Any], internal_key: str | None) -> dict[str, Any]:
        self._check_key(internal_key)
        org_id = str(payload.get("org_id") or "").strip()
        if not org_id:
            raise ValueError("org_id is required")
        event_type = str(payload.get("event_type") or "manual_operator_verification")
        analysis: dict[str, Any]
        provider = "heuristic-fallback"
        if self.settings.qwen_api_key and payload.get("snapshots"):
            try:
                analysis = await self._call_qwen_vision(payload)
                provider = "qwen"
            except Exception as exc:
                analysis = self._fallback_vision(payload, f"Qwen vision unavailable: {exc}")
        else:
            analysis = self._fallback_vision(payload, "Qwen vision not configured or no snapshots supplied.")
        event = self.store.add_vision_event(
            {
                "request_id": payload.get("request_id"),
                "org_id": org_id,
                "target_id": payload.get("target_id"),
                "camera_id": payload.get("camera_id"),
                "event_type": event_type,
                "status": "completed",
                "confidence": analysis.get("confidence", 0),
                "analysis": analysis,
            }
        )
        return {"status": "success", "provider": provider, "vision_event": event}

    async def _call_qwen_vision(self, payload: dict[str, Any]) -> dict[str, Any]:
        if httpx is None:
            raise RuntimeError("httpx is not installed")
        if self.settings.qwen_api_protocol == "openai":
            return await self._call_qwen_vision_openai(payload)
        return await self._call_qwen_vision_dashscope(payload)

    async def _call_qwen_vision_dashscope(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshots = payload.get("snapshots") or []
        content: list[dict[str, Any]] = []
        for snapshot in snapshots[:4]:
            if isinstance(snapshot, str):
                content.append({"image": snapshot})
            elif isinstance(snapshot, dict) and snapshot.get("url"):
                content.append({"image": snapshot["url"]})
        content.append(
            {
                "text": (
                    "You are the vision verification layer for Lemtik Security. Return only JSON with "
                    "threat_summary, confidence, visual_explanation, recommended_follow_up_actions, gaps, and identity_claim=false. "
                    "Do not infer personal identity. Context: "
                    + json.dumps({"event": payload.get("event_type"), "metadata": payload.get("metadata"), "incident_context": payload.get("incident_context")}, default=str)
                )
            }
        )
        body = {
            "model": self.settings.qwen_vision_model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {"result_format": "message"},
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.settings.qwen_base_url}/services/aigc/multimodal-generation/generation",
                headers={"Authorization": f"Bearer {self.settings.qwen_api_key}", "Content-Type": "application/json"},
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            raw = data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(raw, list):
                raw = " ".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in raw)
            return self._parse_vision_json(raw, data)

    async def _call_qwen_vision_openai(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshots = payload.get("snapshots") or []
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are the vision verification layer for Lemtik Security. Return only JSON with "
                    "threat_summary, confidence, visual_explanation, recommended_follow_up_actions, gaps, and identity_claim=false. "
                    "Do not infer personal identity."
                ),
            }
        ]
        for snapshot in snapshots[:4]:
            if isinstance(snapshot, str):
                content.append({"type": "image_url", "image_url": {"url": snapshot}})
            elif isinstance(snapshot, dict) and snapshot.get("url"):
                content.append({"type": "image_url", "image_url": {"url": snapshot["url"]}})
        content.append({"type": "text", "text": json.dumps({"event": payload.get("event_type"), "metadata": payload.get("metadata"), "incident_context": payload.get("incident_context")}, default=str)})
        body = {
            "model": self.settings.qwen_vision_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.settings.qwen_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.qwen_api_key}", "Content-Type": "application/json"},
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            raw = data["choices"][0]["message"]["content"]
            return self._parse_vision_json(raw, data)

    def _parse_vision_json(self, raw: Any, response_data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return {"threat_summary": str(raw), "confidence": 0.5, "visual_explanation": str(raw), "recommended_follow_up_actions": [], "gaps": [], "identity_claim": False}
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            return {
                "threat_summary": raw[:500],
                "confidence": 0.5,
                "visual_explanation": raw,
                "recommended_follow_up_actions": ["Manual operator review"],
                "gaps": ["Model did not return strict JSON."],
                "identity_claim": False,
                "raw_response": response_data,
            }

    def _fallback_vision(self, payload: dict[str, Any], note: str) -> dict[str, Any]:
        event_type = payload.get("event_type") or "manual_operator_verification"
        confidence = 0.45 if event_type in QWEN_TRIGGER_EVENTS else 0.25
        return {
            "threat_summary": f"Manual review required for {event_type}.",
            "confidence": confidence,
            "visual_explanation": note,
            "recommended_follow_up_actions": ["Route snapshot to operator", "Confirm or reject target match", "Attach result to incident timeline"],
            "gaps": ["No model-backed visual verification was completed."],
            "identity_claim": False,
        }

    def audit_log(self, org_id: str | None, internal_key: str | None, limit: int = 100) -> dict[str, Any]:
        self._check_key(internal_key)
        return {"status": "success", "audit_log": self.store.audit_log(org_id, limit)}
