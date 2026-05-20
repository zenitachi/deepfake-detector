"""
Deepfake Detection — Flask App
================================
Usage:
    python app.py

Then open  http://127.0.0.1:5000  in your browser.

Requirements (install once):
    pip install flask torch torchvision timm albumentations pillow numpy
"""

import os, io, base64
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from flask import Flask, request, jsonify, render_template

# ── CONFIG — change these two paths to match your setup ────────
CHECKPOINT = r"D:\DATASETS\deepfake detection\best_model.pth"
TEST_FOLDER = r"D:\DATASETS\deepfake detection\real_vs_fake\test"

# ── MODEL SETTINGS (must match training) ───────────────────────
BACKBONE  = "efficientnet_b4"
IMG_SIZE  = 224
DROPOUT   = 0.4
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── TRANSFORM (same as val_tf in training) ─────────────────────
val_tf = A.Compose([
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# ── MODEL DEFINITION (must match training exactly) ─────────────
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features  # 1792 for B4

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


# ── LOAD MODEL ONCE AT STARTUP ─────────────────────────────────
print(f"Loading model on {DEVICE} ...")
model = DeepfakeDetector().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()
print("Model ready.\n")


# ── HELPER: run inference on a PIL Image ───────────────────────
def predict_pil(pil_img: Image.Image, threshold: float = 0.5):
    img_np = np.array(pil_img.convert("RGB"))
    # Pad with reflect if image is smaller than 224×224
    h, w = img_np.shape[:2]
    if h < IMG_SIZE or w < IMG_SIZE:
        pad_h = max(0, IMG_SIZE - h)
        pad_w = max(0, IMG_SIZE - w)
        img_np = np.pad(img_np,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="reflect")

    tensor = val_tf(image=img_np)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad(), autocast():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()

    is_real   = prob >= threshold
    label     = "REAL" if is_real else "FAKE"
    confidence = prob if is_real else (1 - prob)
    return {
        "label":      label,
        "real_prob":  round(prob, 4),
        "fake_prob":  round(1 - prob, 4),
        "confidence": round(confidence * 100, 1),
        "is_real":    is_real,
    }


# ── HELPER: list test images ────────────────────────────────────
def list_test_images(limit=200):
    """
    Returns a list of dicts:
        { filename, rel_path, true_label }
    Walks TEST_FOLDER/real/ and TEST_FOLDER/fake/
    """
    images = []
    for true_label in ("real", "fake"):
        folder = os.path.join(TEST_FOLDER, true_label)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append({
                    "filename":   fname,
                    "rel_path":   os.path.join(true_label, fname),
                    "true_label": true_label.upper(),
                    "abs_path":   os.path.join(folder, fname),
                })
                if len(images) >= limit:
                    break
        if len(images) >= limit:
            break
    return images


# ── FLASK APP ──────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit


@app.route("/")
def index():
    test_images = list_test_images(limit=200)
    return render_template("index.html", test_images=test_images)


@app.route("/predict/upload", methods=["POST"])
def predict_upload():
    """Predict on a user-uploaded image file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        pil_img = Image.open(io.BytesIO(f.read()))
        result  = predict_pil(pil_img)

        # Return a thumbnail as base64 so the frontend can show it
        thumb = pil_img.convert("RGB")
        thumb.thumbnail((300, 300))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        result["thumbnail"] = "data:image/jpeg;base64," + \
                               base64.b64encode(buf.getvalue()).decode()
        result["filename"] = f.filename
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/test", methods=["POST"])
def predict_test():
    """Predict on an image from the test folder (selected by rel_path)."""
    data = request.get_json(force=True)
    rel  = data.get("rel_path", "")
    abs_path = os.path.normpath(os.path.join(TEST_FOLDER, rel))

    # Security: make sure the resolved path is still inside TEST_FOLDER
    if not abs_path.startswith(os.path.normpath(TEST_FOLDER)):
        return jsonify({"error": "Invalid path"}), 400

    if not os.path.isfile(abs_path):
        return jsonify({"error": "File not found"}), 404

    try:
        pil_img = Image.open(abs_path)
        result  = predict_pil(pil_img)

        thumb = pil_img.convert("RGB")
        thumb.thumbnail((300, 300))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        result["thumbnail"] = "data:image/jpeg;base64," + \
                               base64.b64encode(buf.getvalue()).decode()
        result["filename"]   = os.path.basename(abs_path)
        result["true_label"] = "REAL" if rel.startswith("real") else "FAKE"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/thumb/<path:rel>")
def thumbnail(rel):
    """Serve a small JPEG thumbnail of a test image."""
    abs_path = os.path.normpath(os.path.join(TEST_FOLDER, rel))
    if not abs_path.startswith(os.path.normpath(TEST_FOLDER)):
        return "Forbidden", 403
    if not os.path.isfile(abs_path):
        return "Not found", 404

    pil_img = Image.open(abs_path).convert("RGB")
    pil_img.thumbnail((160, 160))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    from flask import send_file
    return send_file(buf, mimetype="image/jpeg")


if __name__ == "__main__":
    print(f"Test folder : {TEST_FOLDER}")
    print(f"Checkpoint  : {CHECKPOINT}")
    print("Starting Flask server at http://127.0.0.1:5000\n")
    app.run(debug=False, host="127.0.0.1", port=5000)