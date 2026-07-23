import os
import time
import base64
import subprocess
import shutil
from pathlib import Path

import torch
from torchvision.ops import nms
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from ultralytics import YOLO

from db import init_db, log_detection, get_dashboard_stats

# ── Robust Path Alignment ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "static" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Performance and Threshold Configuration ────────────────────────────────
# MODEL_PATH can point to a .pt file OR an exported format directory
# (e.g. best_openvino_model/, best.onnx) — Ultralytics' YOLO() class
# auto-detects the format either way, no code change needed here.
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.45"))
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FRAME_SKIP_RATE = int(os.environ.get("FRAME_SKIP_RATE", "2"))

torch.set_num_threads(os.cpu_count())  # avoid CPU thread oversubscription

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running on [{DEVICE.upper()}]")
print(f"[INFO] Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("[INFO] Model successfully loaded.")

_is_end2end = bool(getattr(model.model, "end2end", False))
if _is_end2end:
    print("[WARNING] Model reports end2end=True (NMS-free architecture). "
          "The iou= parameter passed to predict() has NO EFFECT for this "
          "model. Manual NMS is applied in post-processing instead.")

init_db()
print(f"[INFO] Detection log database ready at: {(BASE_DIR / 'wasp_detections.db')}")


# ── Helpers ────────────────────────────────────────────────────────────────
def allowed_file(filename: str, allowed_extensions: set) -> bool:
    return Path(filename).suffix.lower() in allowed_extensions


def encode_image(image_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", image_bgr)
    return base64.b64encode(buf).decode("utf-8")


def suppress_duplicates(result, iou_threshold: float):
    """Manually apply IoU-based NMS on top of the model's raw output.
    Needed because NMS-free (end2end) architectures like YOLOv10/YOLO26
    ignore the iou= argument passed to predict()."""
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return result
    boxes = result.boxes.xyxy
    scores = result.boxes.conf
    keep_idx = nms(boxes, scores, iou_threshold)
    result.boxes = result.boxes[keep_idx]
    return result


def run_tracked_inference(frame, conf_param, iou_param, use_half):
    """Try model.track() first (gives persistent IDs across frames).
    Some exported formats (ONNX/OpenVINO/TensorRT) don't support the
    tracker hooks .track() relies on — fall back to plain predict() in
    that case so the app keeps working, just without cross-frame IDs."""
    try:
        results = model.track(
            frame, conf=conf_param, iou=iou_param, device=DEVICE,
            half=use_half, persist=True, tracker="bytetrack.yaml",
            verbose=False,
        )
        return results, True
    except Exception as e:
        print(f"[WARNING] model.track() failed ({e}); falling back to "
              f"model.predict() — track IDs will be unavailable this run.")
        results = model.predict(
            frame, conf=conf_param, iou=iou_param, device=DEVICE,
            half=use_half, verbose=False,
        )
        return results, False


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                            model_path=MODEL_PATH,
                            conf=CONF_THRESHOLD,
                            iou=IOU_THRESHOLD)


@app.route("/api/stats")
def api_stats():
    days = request.args.get("days", default=30, type=int)
    return jsonify(get_dashboard_stats(days=days))


@app.route("/detect-video", methods=["POST"])
def detect_video():
    if "video" not in request.files:
        return jsonify({"error": "No file field found under 'video'."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({"error": "Unsupported format."}), 400

    conf_param = float(request.form.get("conf", CONF_THRESHOLD))
    iou_param = float(request.form.get("iou", IOU_THRESHOLD))

    ext = Path(file.filename).suffix.lower()
    timestamp = int(time.time())

    input_path = PROCESSED_DIR / f"temp_in_{timestamp}{ext}"
    temp_output_path = PROCESSED_DIR / f"temp_out_{timestamp}.mp4"
    out_filename = f"out_{timestamp}.mp4"
    final_output_path = PROCESSED_DIR / out_filename

    file.save(str(input_path))

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (w, h))

    max_simultaneous = 0
    frames_with_wasps = 0
    total_frames = 0
    total_inference_ms = 0.0
    unique_track_ids = set()
    last_computed_result = None
    use_half = (DEVICE == "cuda")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if total_frames % FRAME_SKIP_RATE == 0:
                t_start = time.perf_counter()
                results, has_ids = run_tracked_inference(frame, conf_param, iou_param, use_half)
                total_inference_ms += (time.perf_counter() - t_start) * 1000

                last_computed_result = suppress_duplicates(results[0], iou_param)

                num_wasps = len(last_computed_result.boxes)
                if num_wasps > 0:
                    frames_with_wasps += 1
                    max_simultaneous = max(max_simultaneous, num_wasps)
                    if has_ids and last_computed_result.boxes.id is not None:
                        unique_track_ids.update(last_computed_result.boxes.id.int().tolist())
            else:
                if last_computed_result is not None and len(last_computed_result.boxes) > 0:
                    frames_with_wasps += 1

            if last_computed_result is not None:
                annotated_frame = last_computed_result.plot()
                out.write(annotated_frame)
            else:
                out.write(frame)

            total_frames += 1

    finally:
        cap.release()
        out.release()

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(temp_output_path),
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(final_output_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[WARNING] Transcoding failed: {e}. Falling back to standard container layout.")
        shutil.copy(str(temp_output_path), str(final_output_path))
    finally:
        if input_path.exists():
            input_path.unlink()
        if temp_output_path.exists():
            temp_output_path.unlink()

    log_detection(
        source_type="video",
        filename=file.filename,
        count=max_simultaneous,
        confidences=[],
        inference_ms=total_inference_ms,
        conf_threshold=conf_param,
        iou_threshold=iou_param,
        extra={"unique_wasps": len(unique_track_ids), "total_frames": total_frames},
    )

    return jsonify({
        "max_concurrent": max_simultaneous,
        "unique_wasps": len(unique_track_ids),
        "total_frames": total_frames,
        "frames_with_wasps": frames_with_wasps,
        "inference_ms": round(total_inference_ms, 1),
        "processed_fps": round(total_frames / (total_inference_ms / 1000.0), 1) if total_inference_ms > 0 else 0,
        "video_url": f"/static/processed/{out_filename}"
    })


@app.route("/detect-image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No file field found under 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({"error": f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"}), 400

    conf_param = float(request.form.get("conf", CONF_THRESHOLD))
    iou_param = float(request.form.get("iou", IOU_THRESHOLD))

    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return jsonify({"error": "Could not decode image."}), 400

    h, w = image_bgr.shape[:2]

    t_start = time.perf_counter()
    results = model.predict(
        image_bgr, conf=conf_param, iou=iou_param, device=DEVICE,
        half=(DEVICE == "cuda"), verbose=False,
    )
    inference_ms = round((time.perf_counter() - t_start) * 1000, 1)

    result = suppress_duplicates(results[0], iou_param)

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = round(float(box.conf[0]), 4)
        bw, bh = x2 - x1, y2 - y1
        detections.append({
            "confidence": conf,
            "confidence_pct": f"{conf:.0%}",
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "width": bw, "height": bh,
            "area_pct": round((bw * bh) / (w * h) * 100, 2),
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    annotated_bgr = result.plot()

    log_detection(
        source_type="image",
        filename=file.filename,
        count=len(detections),
        confidences=[d["confidence"] for d in detections],
        inference_ms=inference_ms,
        conf_threshold=conf_param,
        iou_threshold=iou_param,
    )

    return jsonify({
        "count": len(detections),
        "detections": detections,
        "inference_ms": inference_ms,
        "image_w": w, "image_h": h,
        "original_b64": encode_image(image_bgr),
        "annotated_b64": encode_image(annotated_bgr),
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH,
        "device": DEVICE,
        "end2end_nms_free": _is_end2end,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
