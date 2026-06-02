# -*- coding: utf-8 -*-
"""Combined Base64 API for icon matching and single-class window detection.

Endpoints:
  POST /icon/locate
    Input : {"image_base64": "...", "software": "微信"}
    Output: {"center": [x, y]} or {"center": null}

  POST /window/detect
    Input : {"image_base64": "..."}
    Output: {
      "windows": [
        {
          "label": "0",
          "box": [x1, y1, x2, y2],
          "ocr": [{"text": "人体成分分析仪", "center": [x, y]}]
        }
      ]
    }
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np
from flask import Flask, jsonify, request
from ultralytics import YOLO

try:
    import icon_match_local as icon_core
except ModuleNotFoundError:
    icon_core = None


BASE_DIR = Path(__file__).resolve().parent
HOST = os.environ.get("VISION_API_HOST", "0.0.0.0")
PORT = int(os.environ.get("VISION_API_PORT", "5002"))

DEFAULT_WINDOW_MODEL_PATH = BASE_DIR / "runs" / "window_yolov8n_640_add9" / "weights" / "best.pt"
if not DEFAULT_WINDOW_MODEL_PATH.exists():
    DEFAULT_WINDOW_MODEL_PATH = BASE_DIR / "runs" / "window_yolov8n_640" / "weights" / "best.pt"
if not DEFAULT_WINDOW_MODEL_PATH.exists():
    DEFAULT_WINDOW_MODEL_PATH = BASE_DIR / "best.pt"
MODEL_PATH = os.environ.get("WINDOW_MODEL_PATH", str(DEFAULT_WINDOW_MODEL_PATH))
ICON_MODEL_PATH = os.environ.get("ICON_MODEL_PATH", str(BASE_DIR / "best.pt"))
IMG_SIZE = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.95
DEFAULT_MAX_DET = 300
OCR_DEFAULT_MIN_SCORE = 0.0
ICON_DEFAULT_CONF = 0.25
ICON_DEFAULT_IOU = 0.45
ICON_LABEL_PAD_X = 30
ICON_LABEL_PAD_TOP = 8
ICON_LABEL_PAD_BOTTOM = 90

app = Flask(__name__)


def json_response(payload, status=200):
    return app.response_class(
        response=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        status=status,
        mimetype="application/json",
    )


def strip_data_uri(image_base64: str) -> str:
    text = image_base64.strip()
    if "," in text and text.lower().startswith("data:"):
        return text.split(",", 1)[1].strip()
    return text


def decode_base64_image(image_base64: str):
    if not image_base64 or not image_base64.strip():
        raise ValueError("missing image_base64")

    raw_b64 = "".join(strip_data_uri(image_base64).split())
    try:
        image_bytes = base64.b64decode(raw_b64, validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid image_base64") from exc

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image")
    return img


def clamp_box_xyxy(box, width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def find_icon_center(img_bgr, software: str, sim_th: float):
    if icon_core is None:
        return find_icon_center_by_yolo_ocr(img_bgr, software)

    _, detections = icon_core.annotate_image_bgr(
        img_bgr,
        ICON_YOLO_MODEL,
        ICON_EMBEDDER,
        ICON_FEATS,
        ICON_LABELS,
        ICON_OUT_SIZE,
        sim_th=sim_th,
    )
    best = icon_core.find_best_by_label(detections, software)
    return best["center"] if best else None


def icon_text_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    return clamp_box_xyxy(
        [
            x1 - ICON_LABEL_PAD_X,
            y1 - ICON_LABEL_PAD_TOP,
            x2 + ICON_LABEL_PAD_X,
            y2 + ICON_LABEL_PAD_BOTTOM,
        ],
        width,
        height,
    )


def find_icon_center_by_yolo_ocr(img_bgr, software: str):
    if ICON_YOLO_MODEL is None:
        return find_text_center_by_ocr(img_bgr, software, OCR_DEFAULT_MIN_SCORE)

    target = compact_text(software)
    if not target:
        return None

    height, width = img_bgr.shape[:2]
    result = ICON_YOLO_MODEL.predict(
        img_bgr,
        imgsz=IMG_SIZE,
        conf=ICON_DEFAULT_CONF,
        iou=ICON_DEFAULT_IOU,
        max_det=DEFAULT_MAX_DET,
        verbose=False,
    )[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    for idx in np.argsort(-scores):
        box = clamp_box_xyxy(xyxy[idx], width, height)
        if box is None:
            continue
        search_box = icon_text_box(box, width, height)
        if search_box is None:
            continue
        texts = ocr_window_texts(img_bgr, search_box, OCR_DEFAULT_MIN_SCORE)
        joined = compact_text("".join(str(item.get("text", "")) for item in texts))
        if target in joined:
            x1, y1, x2, y2 = box
            return [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))]
    return None


def is_ocr_line(item):
    return (
        isinstance(item, (list, tuple))
        and len(item) >= 2
        and isinstance(item[1], (list, tuple))
        and len(item[1]) >= 2
        and isinstance(item[1][0], str)
    )


def collect_ocr_lines(obj, out):
    if is_ocr_line(obj):
        out.append(obj)
        return
    if isinstance(obj, (list, tuple)):
        for child in obj:
            collect_ocr_lines(child, out)


def polygon_center(poly):
    arr = np.array(poly, dtype=np.float32).reshape(-1, 2)
    if arr.size == 0:
        return None
    cx = int(round(float(arr[:, 0].mean())))
    cy = int(round(float(arr[:, 1].mean())))
    return [cx, cy]


def compact_text(text: str) -> str:
    return "".join(str(text).split())


def ocr_image_texts(img_bgr, min_score: float, offset_x: int = 0, offset_y: int = 0):
    if img_bgr.size == 0:
        return []

    result = OCR_ENGINE.ocr(img_bgr, cls=False)
    lines = []
    collect_ocr_lines(result, lines)

    items = []
    for line in lines:
        poly, rec = line[0], line[1]
        text = str(rec[0]).strip()
        score = float(rec[1]) if len(rec) > 1 else 1.0
        if not text or score < min_score:
            continue

        center = polygon_center(poly)
        if center is None:
            continue

        items.append(
            {
                "text": text,
                "center": [center[0] + offset_x, center[1] + offset_y],
            }
        )
    return items


def find_text_center_by_ocr(img_bgr, target: str, min_score: float):
    target_compact = compact_text(target)
    if not target_compact:
        return None
    for item in ocr_image_texts(img_bgr, min_score):
        text = str(item.get("text", ""))
        if target in text or target_compact in compact_text(text):
            return item.get("center")
    return None


def ocr_window_texts(img_bgr, box, min_score: float):
    x1, y1, x2, y2 = box
    crop = img_bgr[y1:y2, x1:x2]
    return ocr_image_texts(crop, min_score, x1, y1)


def detect_all_windows(img_bgr, conf: float, iou: float, max_det: int, ocr_min_score: float):
    result = WINDOW_MODEL.predict(
        img_bgr,
        imgsz=IMG_SIZE,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {"windows": []}

    height, width = img_bgr.shape[:2]
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()

    windows = []
    for idx in np.argsort(-scores):
        box = clamp_box_xyxy(xyxy[idx], width, height)
        if box is None:
            continue
        windows.append(
            {
                "label": "0",
                "box": box,
                "ocr": ocr_window_texts(img_bgr, box, ocr_min_score),
            }
        )

    return {"windows": windows}


if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"window model not found: {MODEL_PATH}")

WINDOW_MODEL = YOLO(MODEL_PATH)

if icon_core is not None:
    icon_core.load_runtime_deps()
    ICON_FEATS, ICON_LABELS, ICON_OUT_SIZE = icon_core.load_index(
        icon_core.INDEX_DIR,
        fallback_out_size=icon_core.EMBED_OUT_SIZE_FALLBACK,
    )
    DEVICE = "cuda" if icon_core.torch.cuda.is_available() else "cpu"
    ICON_EMBEDDER = icon_core.Embedder(device=DEVICE)
    ICON_YOLO_MODEL = icon_core.YOLO(icon_core.MODEL_PATH)
else:
    ICON_FEATS = None
    ICON_LABELS = []
    ICON_OUT_SIZE = None
    DEVICE = "yolo_ocr_fallback"
    ICON_EMBEDDER = None
    ICON_YOLO_MODEL = YOLO(ICON_MODEL_PATH) if Path(ICON_MODEL_PATH).exists() else None

from paddleocr import PaddleOCR

OCR_ENGINE = PaddleOCR(
    use_angle_cls=False,
    lang="ch",
    use_gpu=False,
    show_log=False,
)


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "model": MODEL_PATH,
            "model_names": WINDOW_MODEL.names,
            "response_label": "0",
            "icon_model": ICON_MODEL_PATH if ICON_YOLO_MODEL is not None else None,
            "icon_model_names": ICON_YOLO_MODEL.names if ICON_YOLO_MODEL is not None else None,
            "icon_device": DEVICE,
            "icon_index_n": int(ICON_FEATS.shape[0]) if ICON_FEATS is not None else 0,
            "icon_mode": "embedding" if icon_core is not None else ("yolo_ocr_fallback" if ICON_YOLO_MODEL is not None else "ocr_fallback"),
            "ocr": "paddleocr",
        }
    )


@app.post("/icon/locate")
def icon_locate():
    data = request.get_json(silent=True) or {}
    software = str(data.get("software") or data.get("label") or "").strip()

    if not software:
        return jsonify({"error": "missing software"}), 400

    default_sim_th = icon_core.DEFAULT_SIM_TH if icon_core is not None else 0.75
    try:
        sim_th = float(data.get("sim_th", default_sim_th))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid sim_th"}), 400

    try:
        img = decode_base64_image(data.get("image_base64") or data.get("image") or "")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return json_response({"center": find_icon_center(img, software, sim_th)})


@app.post("/window/detect")
def window_detect():
    data = request.get_json(silent=True) or {}

    try:
        conf = float(data.get("conf", DEFAULT_CONF))
        iou = float(data.get("iou", DEFAULT_IOU))
        max_det = int(data.get("max_det", DEFAULT_MAX_DET))
        ocr_min_score = float(data.get("ocr_min_score", OCR_DEFAULT_MIN_SCORE))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid conf, iou, max_det or ocr_min_score"}), 400

    try:
        img = decode_base64_image(data.get("image_base64") or data.get("image") or "")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return json_response(detect_all_windows(img, conf, iou, max_det, ocr_min_score))


@app.post("/locate_icon")
def locate_icon_alias():
    return icon_locate()


@app.post("/detect_window")
def detect_window_alias():
    return window_detect()


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
