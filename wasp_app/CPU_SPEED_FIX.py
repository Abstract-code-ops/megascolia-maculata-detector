# ═══════════════════════════════════════════════════════════════════════════
# CPU SPEED FIX — apply these changes to app.py
# ═══════════════════════════════════════════════════════════════════════════
#
# Root cause: mammoth_wasp_train.ipynb (cell 9) trains yolo26m.pt (medium).
# On CPU with no GPU, medium is roughly 2-3x slower than small for near-
# identical accuracy on a single-class detector like yours. This is very
# likely your entire slowdown — not a Flask/OpenCV problem.
#
# Ranked by effort. Do #1 first; it alone will probably fix this.


# ─── FIX 1 (do this first, ~30 seconds) ─────────────────────────────────────
# Re-run inference with the SMALL model instead of medium. Point MODEL_PATH
# at your v5/v6 small-model weights, or just retrain one more short run
# with model=yolo26s.pt if you only kept the medium checkpoint. For a
# single class with a few hundred validation images, small vs. medium mAP
# difference is usually within a couple of points — a fair trade for 2-3x
# CPU throughput.
#
#   MODEL_PATH=/path/to/yolo26s_weights/best.pt python app.py


# ─── FIX 2 (do this second, ~5 minutes, biggest CPU-specific win) ───────────
# Export to OpenVINO. Ultralytics publishes ~40%+ CPU inference gains for
# YOLO26 specifically via OpenVINO vs. plain PyTorch — this is a bigger
# lever than swapping model size alone, and you can combine both.
#
# Run this ONCE, offline, to produce an optimized model directory:
"""
from ultralytics import YOLO

model = YOLO("best.pt")               # your existing trained weights
model.export(format="openvino")       # writes best_openvino_model/ next to best.pt
"""
#
# Then point your Flask app at the exported folder instead of the .pt file:
#
#   MODEL_PATH=best_openvino_model python app.py
#
# No other code changes needed — Ultralytics' YOLO() class auto-detects
# the OpenVINO format from the folder and routes inference through it.
# NOTE: OpenVINO models don't support .track() the same way .pt models do
# in all Ultralytics versions — test your /detect-video route after
# switching; if track() misbehaves, keep video on the .pt model and only
# swap the image route to OpenVINO.


# ─── FIX 3 (add to app.py directly — thread tuning) ─────────────────────────
# Add near the top of app.py, right after `import torch`:
"""
import torch
torch.set_num_threads(os.cpu_count())   # avoid CPU oversubscription/underuse
"""
# This matters more than people expect: PyTorch's default thread count is
# sometimes wrong for containerized environments (Colab, Docker, some VPS
# providers under-report core count), silently leaving cores idle.


# ─── FIX 4 (small code change — reduce imgsz for inference) ─────────────────
# You trained at imgsz=800. You do NOT have to run inference at 800.
# Add an INFER_IMGSZ env var and pass it to both model calls:
"""
INFER_IMGSZ = int(os.environ.get("INFER_IMGSZ", "640"))

# in detect_image():
results = model.predict(image_bgr, conf=conf_param, iou=iou_param,
                         device=DEVICE, half=(DEVICE=="cuda"),
                         imgsz=INFER_IMGSZ, verbose=False)

# in detect_video(), inside the model.track(...) call:
results = model.track(frame, conf=conf_param, iou=iou_param, device=DEVICE,
                       half=use_half, persist=True, tracker="bytetrack.yaml",
                       imgsz=INFER_IMGSZ, verbose=False)
"""
# Dropping 800 -> 640 is roughly a (640/800)^2 ≈ 0.64x compute reduction,
# i.e. a real, close-to-linear speedup on CPU, at a small accuracy cost.
# Your wasps are large-bodied relative to the frame in most of your
# training images, so 640 is unlikely to meaningfully hurt detection here
# — worth testing on your validation set before committing to it.


# ─── FIX 5 (video-only — cheapest lever you already have) ───────────────────
# You already built FRAME_SKIP_RATE. Just raise it for CPU deployment:
#
#   FRAME_SKIP_RATE = 3   # was 2 — inference runs on 1 in 3 frames instead
#                         # of 1 in 2, roughly another 33% fewer inference
#                         # calls for the same video length.


# ─── Priority order if you only do ONE thing ────────────────────────────────
# Fix 1 (small model) + Fix 3 (thread tuning) together are ~2 minutes of
# work and will likely resolve "the model takes a bit long" on their own.
# Only chase OpenVINO (Fix 2) if you need more after that.
