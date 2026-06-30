from collections import defaultdict
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHOLOGY_MODEL_PATH = PROJECT_ROOT / "runs" / "pathology" / "pathology_v1" / "weights" / "best.pt"

PATHOLOGY_CLASS_NAMES = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical lesion",
}

# OpenCV uses BGR colors
PATHOLOGY_COLORS = {
    0: (255, 0, 0),        # Blue
    1: (255, 255, 0),      # Cyan
    2: (255, 255, 255),    # White
}

# Use stricter thresholds to reduce clutter and false positives
PATHOLOGY_THRESHOLDS = {
    0: 0.45,   # impacted
    1: 0.55,   # caries
    2: 0.75,   # periapical lesion
}

# Keep the image readable
MAX_DETECTIONS_PER_CLASS = {
    0: 4,
    1: 6,
    2: 2,
}


@st.cache_resource
def load_pathology_model():
    if not PATHOLOGY_MODEL_PATH.exists():
        return None
    return YOLO(str(PATHOLOGY_MODEL_PATH))


def _clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2


def detect_pathology(image_bgr):
    model = load_pathology_model()
    if model is None:
        return []

    min_conf = min(PATHOLOGY_THRESHOLDS.values())

    results = model.predict(
        source=image_bgr,
        imgsz=1024,
        conf=min_conf,
        verbose=False,
    )

    if not results or results[0].boxes is None:
        return []

    h, w = image_bgr.shape[:2]
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        threshold = PATHOLOGY_THRESHOLDS.get(cls_id, 0.5)
        if conf < threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)

        detections.append(
            {
                "class_id": cls_id,
                "label": PATHOLOGY_CLASS_NAMES.get(cls_id, f"Class {cls_id}"),
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
            }
        )

    detections.sort(key=lambda d: d["conf"], reverse=True)

    grouped = defaultdict(list)
    for det in detections:
        grouped[det["class_id"]].append(det)

    final_detections = []
    for cls_id, dets in grouped.items():
        limit = MAX_DETECTIONS_PER_CLASS.get(cls_id, len(dets))
        final_detections.extend(dets[:limit])

    final_detections.sort(key=lambda d: d["conf"], reverse=True)
    return final_detections


def draw_pathology_boxes(image_bgr, detections, show_conf=False):
    output = image_bgr.copy()

    for det in detections:
        cls_id = det["class_id"]
        label = det["label"]
        conf = det["conf"]
        x1, y1, x2, y2 = det["bbox"]

        color = PATHOLOGY_COLORS.get(cls_id, (0, 255, 255))
        thickness = 2

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        text = label if not show_conf else f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        text_thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

        text_x = x1
        text_y = max(y1 - 8, text_h + 8)

        bg_x1 = text_x
        bg_y1 = text_y - text_h - 6
        bg_x2 = text_x + text_w + 8
        bg_y2 = text_y + baseline - 2

        cv2.rectangle(output, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(
            output,
            text,
            (text_x + 4, text_y - 2),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

    return output


def summarize_pathology(detections):
    grouped = defaultdict(list)
    for det in detections:
        grouped[det["label"]].append(det)

    summary = []
    for label, items in grouped.items():
        items = sorted(items, key=lambda d: d["conf"], reverse=True)
        summary.append(
            {
                "label": label,
                "count": len(items),
                "top_conf": items[0]["conf"],
                "detections": items,
            }
        )

    summary.sort(key=lambda x: x["top_conf"], reverse=True)
    return summary