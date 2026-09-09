from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional until deployed
    np = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional until deployed
    Image = None  # type: ignore

try:
    import onnxruntime
except Exception:  # pragma: no cover - optional until deployed
    onnxruntime = None  # type: ignore


# YOLOX-Nano, Apache 2.0 licensed (Megvii-BaseDetection/YOLOX), ~3.6MB ONNX export from the
# project's own GitHub release - safe for a closed-source commercial product, unlike Ultralytics'
# YOLOv5/v8 weights and package (AGPL-3.0, which would require open-sourcing this service or a
# paid Enterprise license to use commercially over a network). Nano is the smallest variant
# specifically so this fits Render's free-tier 512MB RAM alongside the rest of the app.
MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolox_nano.onnx"
INPUT_SIZE = (416, 416)  # YOLOX-Nano's trained input resolution (not the 640 used by s/m/l/x)
NMS_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.35

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

# Security-relevant subset - detecting a "toaster" isn't useful telemetry for a perimeter camera.
# Anything not in this set is still returned in raw_detections but excluded from the primary
# tracked-object selection used for re-id embeddings.
TRACKABLE_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase"}

_DATA_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9+.-]+;base64,(.*)$", re.DOTALL)

_session: Any | None = None
_session_attempted = False


def detector_available() -> bool:
    return np is not None and Image is not None and onnxruntime is not None and MODEL_PATH.exists()


def _get_session() -> Any | None:
    global _session, _session_attempted
    if _session_attempted:
        return _session
    _session_attempted = True
    if not detector_available():
        return None
    try:
        # Single-threaded intra/inter-op: free-tier CPUs are 1 shared vCPU: letting onnxruntime
        # spawn a thread pool just adds contention overhead, not speed, on a box this small.
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        _session = onnxruntime.InferenceSession(str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception:
        _session = None
    return _session


def decode_image(frame_data: str) -> Any | None:
    """Decode a base64 or data-URL image string into an (H, W, 3) BGR uint8 numpy array,
    matching the channel order YOLOX-Nano was trained and exported with. Returns None on any
    decode failure - callers must treat that as "no real image available", not an error."""
    if np is None or Image is None or not frame_data:
        return None
    try:
        match = _DATA_URL_RE.match(frame_data.strip())
        raw = match.group(1) if match else frame_data.strip()
        image_bytes = base64.b64decode(raw, validate=False)
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = np.array(img.convert("RGB"))
        return rgb[:, :, ::-1].copy()  # RGB -> BGR
    except Exception:
        return None


def _preprocess(image: Any) -> tuple[Any, float]:
    height, width = INPUT_SIZE
    padded = np.full((height, width, 3), 114, dtype=np.uint8)
    ratio = min(height / image.shape[0], width / image.shape[1])
    new_h, new_w = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
    resized = np.array(Image.fromarray(image[:, :, ::-1]).resize((new_w, new_h), Image.BILINEAR))[:, :, ::-1]
    padded[:new_h, :new_w] = resized
    chw = padded.transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(chw), ratio


def _postprocess(output: Any, ratio: float) -> Any:
    grids, strides_expanded = [], []
    for stride in (8, 16, 32):
        h, w = INPUT_SIZE[0] // stride, INPUT_SIZE[1] // stride
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        strides_expanded.append(np.full((*grid.shape[:2], 1), stride))
    grids = np.concatenate(grids, 1)
    strides_expanded = np.concatenate(strides_expanded, 1)
    output[..., :2] = (output[..., :2] + grids) * strides_expanded
    output[..., 2:4] = np.exp(output[..., 2:4]) * strides_expanded
    predictions = output[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    return _multiclass_nms(boxes_xyxy, scores)


def _nms(boxes: Any, scores: Any, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


def _multiclass_nms(boxes: Any, scores: Any) -> Any | None:
    class_inds = scores.argmax(1)
    class_scores = scores[np.arange(len(class_inds)), class_inds]
    valid = class_scores > SCORE_THRESHOLD
    if not valid.sum():
        return None
    valid_boxes, valid_scores, valid_classes = boxes[valid], class_scores[valid], class_inds[valid]
    keep = _nms(valid_boxes, valid_scores, NMS_THRESHOLD)
    if not keep:
        return None
    return np.concatenate(
        [valid_boxes[keep], valid_scores[keep, None], valid_classes[keep, None]], axis=1
    )


def detect_objects(image: Any) -> list[dict[str, Any]]:
    """Runs real YOLOX-Nano inference. Returns [] on any failure (model missing, bad image,
    runtime not installed) rather than raising - detection is an enhancement, never a hard
    dependency for telemetry ingestion to keep working."""
    session = _get_session()
    if session is None or image is None:
        return []
    try:
        tensor, ratio = _preprocess(image)
        raw_output = session.run(None, {session.get_inputs()[0].name: tensor[None, :, :, :]})[0]
        detections = _postprocess(raw_output, ratio)
        if detections is None:
            return []
        results = []
        img_h, img_w = image.shape[0], image.shape[1]
        for x1, y1, x2, y2, score, class_idx in detections:
            class_name = COCO_CLASSES[int(class_idx)] if 0 <= int(class_idx) < len(COCO_CLASSES) else "unknown"
            x1c, y1c = max(0.0, float(x1)), max(0.0, float(y1))
            x2c, y2c = min(float(img_w), float(x2)), min(float(img_h), float(y2))
            if x2c <= x1c or y2c <= y1c:
                continue
            results.append(
                {
                    "class_name": class_name,
                    "confidence": round(float(score), 4),
                    "bbox": {"x": round(x1c, 1), "y": round(y1c, 1), "w": round(x2c - x1c, 1), "h": round(y2c - y1c, 1)},
                    "trackable": class_name in TRACKABLE_CLASSES,
                }
            )
        results.sort(key=lambda item: item["confidence"], reverse=True)
        return results
    except Exception:
        return []


def primary_subject(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The single highest-confidence trackable-class detection, used to drive the re-id crop.
    Prefers 'person' specifically since that's the security use case this pipeline serves."""
    trackable = [d for d in detections if d["trackable"]]
    if not trackable:
        return None
    people = [d for d in trackable if d["class_name"] == "person"]
    return (people or trackable)[0]


HISTOGRAM_BINS = 8  # per channel; 8x8x8 HSV histogram = 512-dim vector


def color_histogram_embedding(image: Any, bbox: dict[str, Any]) -> list[float] | None:
    """A real, cheap, honestly-modest visual feature: an HSV color-histogram of the detected
    subject's crop. This is NOT biometric re-identification - it will match "two people wearing
    similar-colored clothing" as similar, and won't survive a full outfit change. It is a real
    measurement of actual pixels, unlike the SHA-256-of-text-labels fallback it replaces for any
    caller that supplies a real frame. See fallback_embedding_seed() in service.py for the
    synthetic path used when no image is available at all."""
    if np is None or Image is None or image is None:
        return None
    try:
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        crop = image[int(y): int(y + h), int(x): int(x + w)]
        if crop.size == 0:
            return None
        with Image.fromarray(crop[:, :, ::-1]) as pil_crop:  # BGR -> RGB for PIL
            hsv = pil_crop.convert("HSV")
            arr = np.array(hsv)
        hist, _ = np.histogramdd(
            arr.reshape(-1, 3),
            bins=(HISTOGRAM_BINS, HISTOGRAM_BINS, HISTOGRAM_BINS),
            range=((0, 256), (0, 256), (0, 256)),
        )
        flat = hist.flatten()
        total = flat.sum()
        if total <= 0:
            return None
        normalized = flat / total
        return [round(float(v), 6) for v in normalized]
    except Exception:
        return None
