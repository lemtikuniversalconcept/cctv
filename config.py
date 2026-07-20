from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    local_database_path: Path
    internal_api_key: str
    relationship_api_url: str | None
    relationship_api_key: str | None
    ai_orchestrator_url: str | None
    ai_orchestrator_key: str | None
    qwen_api_key: str | None
    qwen_base_url: str
    qwen_vision_model: str
    qwen_api_protocol: str
    require_relationship_client: bool
    relationship_client_name: str
    environment: str
    host: str
    port: int
    default_reid_threshold: float
    topology_prediction_horizon_seconds: int
    qwen_min_event_confidence: float
    max_history_items: int


def load_settings(base_dir: Path | None = None) -> Settings:
    root = base_dir or Path(__file__).resolve().parent
    environment = os.getenv("ENVIRONMENT", "production").strip()
    internal_api_key = os.getenv("INTERNAL_API_KEY", "").strip()
    if environment == "production" and not internal_api_key:
        raise RuntimeError("INTERNAL_API_KEY is required in production.")
    database_url = os.getenv("DATABASE_URL", "").strip() or None
    if environment == "production" and not database_url:
        raise RuntimeError("DATABASE_URL is required in production. CCTV AI must use Supabase/Postgres, not local storage.")
    if environment == "production" and database_url and not database_url.startswith(("postgres://", "postgresql://")):
        raise RuntimeError("DATABASE_URL must be a Supabase/Postgres connection string in production.")
    return Settings(
        database_url=database_url,
        local_database_path=Path(os.getenv("LOCAL_DATABASE_PATH", root / "cctvai.db")),
        internal_api_key=internal_api_key or "dev-internal-key",
        relationship_api_url=os.getenv("RELATIONSHIP_API_URL", "").strip() or None,
        relationship_api_key=os.getenv("RELATIONSHIP_API_KEY", "").strip() or None,
        ai_orchestrator_url=os.getenv("AI_ORCHESTRATOR_URL", "").strip() or None,
        ai_orchestrator_key=os.getenv("AI_ORCHESTRATOR_KEY", "").strip() or None,
        qwen_api_key=(
            os.getenv("QWEN_API_KEY", "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip()
            or os.getenv("ALIBABA_API_KEY", "").strip()
            or os.getenv("ALIBABA_CLOUD_API_KEY", "").strip()
            or None
        ),
        qwen_base_url=os.getenv("QWEN_BASE_URL", "https://dashscope-us.aliyuncs.com/api/v1").strip().rstrip("/"),
        qwen_vision_model=os.getenv("QWEN_VISION_MODEL", "qwen3-vl-flash-us").strip(),
        qwen_api_protocol=os.getenv("QWEN_API_PROTOCOL", "dashscope").strip().lower(),
        require_relationship_client=os.getenv("REQUIRE_RELATIONSHIP_CLIENT", "true" if environment == "production" else "false").strip().lower() in {"1", "true", "yes"},
        relationship_client_name=os.getenv("RELATIONSHIP_CLIENT_NAME", "relationship-api").strip(),
        environment=environment,
        host=os.getenv("HOST", "0.0.0.0").strip(),
        port=int(os.getenv("PORT", "8000")),
        default_reid_threshold=float(os.getenv("DEFAULT_REID_THRESHOLD", "0.82")),
        topology_prediction_horizon_seconds=int(os.getenv("TOPOLOGY_PREDICTION_HORIZON_SECONDS", "180")),
        qwen_min_event_confidence=float(os.getenv("QWEN_MIN_EVENT_CONFIDENCE", "0.70")),
        max_history_items=int(os.getenv("MAX_HISTORY_ITEMS", "200")),
    )
