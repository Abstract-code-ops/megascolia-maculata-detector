# Megascolia maculata — Field Detection Ledger

A Flask web app for detecting the Mammoth Wasp (*Megascolia maculata*) in images and video using a YOLO26 model exported to OpenVINO.

**Author:** Adil Ali

---

## Requirements

- Python 3.10
- ffmpeg and CV2
- OpenVINO-exported model folder (e.g. `best_openvino_model/`)

```bash
pip install -r requirements.txt
```

---

## Running the app

1. Place your exported OpenVINO model folder in the project directory.
2. Start the server, pointing `MODEL_PATH` at that folder:

```bash
MODEL_PATH=best_openvino_model python app.py
```

3. Open **http://localhost:5000**

---

## Running on Google Colab

Since this is only run temporarily on Colab I recommend uploading the model to drive first then in colab:

```python
# 1. Unzip the app
!unzip wasp_app_v2.zip

# 2. Copy your OpenVINO model folder into the app directory
import shutil
shutil.copytree(
    "/content/drive/MyDrive/path/to/best_openvino_model",
    "/content/wasp_app_v2/best_openvino_model"
)

# 3. Install dependencies
!pip install -r /content/wasp_app_v2/requirements.txt -q

# 4. Run the app in a background thread
import os, threading
os.chdir("/content/wasp_app_v2")
os.environ["MODEL_PATH"] = "best_openvino_model"

from app import app
t = threading.Thread(target=lambda: app.run(port=5000))
t.daemon = True
t.start()

# 5. Open a public URL
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(5000)"))
```

Open the printed URL in a **new browser tab**.

---

## Using the app

- **Detect tab** — upload an image or video, adjust confidence/IoU sliders, run detection.
- **Ledger tab** — view logged detection history: KPIs, charts, and a records table. Fills in automatically as you run detections.

---

## Environment variables

| Variable          | Default   | Description                                  |
|--------------------|-----------|------------------------------------------------|
| `MODEL_PATH`       | `best.pt` | Path to OpenVINO model folder (or `.pt`/`.onnx`) |
| `CONF_THRESHOLD`   | `0.25`    | Default confidence threshold                  |
| `IOU_THRESHOLD`    | `0.45`    | Default NMS IoU threshold                     |
| `FRAME_SKIP_RATE`  | `2`       | Run inference on 1-in-N video 

## NOTES:

- Detection history is stored in `wasp_detections.db` (SQLite), created automatically on first run.

