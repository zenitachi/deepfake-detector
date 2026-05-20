# =============================================================
#  Deepfake Image Detection — EfficientNet-B4
#  Dataset : 140K Real & Fake Faces (verified by inspection)
#  Images  : 256×256 .jpg  |  Labels : folder name (real/fake)
#  Splits  : train=100k  valid=20k  test=20k  (50/50 balanced)
#  Hardware: RTX 3070 8GB  |  Python 3.10  |  PyTorch 2.3
# =============================================================

import os, time, warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve
)

# ── CONFIG ─────────────────────────────────────────────────────
# Verified paths from your inspection output
DATA_ROOT   = r"D:\DATASETS\deepfake detection\real_vs_fake"
CHECKPOINT  = r"D:\DATASETS\deepfake detection\best_model.pth"
PLOT_DIR    = r"D:\DATASETS\deepfake detection"

# Model
BACKBONE    = "efficientnet_b4"   # 19M params, ~5.5 GB VRAM with batch 32
IMG_SIZE    = 224    # crop from native 256 — gives free augmentation headroom

# Training
BATCH_SIZE  = 32     # safe for 8 GB VRAM + AMP; drop to 16 if OOM
EPOCHS      = 25
LR          = 2e-4
WEIGHT_DECAY= 1e-4
DROPOUT     = 0.4
NUM_WORKERS = 4      # 4 workers on Windows is stable
SEED        = 42
UNFREEZE_AT = 5      # epoch at which backbone unfreezes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)

print("=" * 58)
print(f"  Device  : {DEVICE}")
if DEVICE.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU     : {p.name}")
    print(f"  VRAM    : {p.total_memory / 1e9:.1f} GB")
print(f"  Backbone: {BACKBONE}")
print(f"  Batch   : {BATCH_SIZE}  |  Epochs: {EPOCHS}")
print("=" * 58 + "\n")

# ── DATASET ────────────────────────────────────────────────────
# Confirmed layout:
#   DATA_ROOT/train/real/  →  50,000 × 256×256 .jpg
#   DATA_ROOT/train/fake/  →  50,000 × 256×256 .jpg
#   DATA_ROOT/valid/real/  →  10,000 × 256×256 .jpg
#   DATA_ROOT/valid/fake/  →  10,000 × 256×256 .jpg
#   DATA_ROOT/test/real/   →  10,000 × 256×256 .jpg
#   DATA_ROOT/test/fake/   →  10,000 × 256×256 .jpg
#   Labels: real = 1,  fake = 0

class FaceDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.transform = transform
        self.samples   = []

        for label, cls in [(1, "real"), (0, "fake")]:
            folder = os.path.join(root, split, cls)
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(".jpg"):
                    self.samples.append((os.path.join(folder, fname), label))

        n_real = sum(1 for _, l in self.samples if l == 1)
        n_fake = sum(1 for _, l in self.samples if l == 0)
        print(f"  [{split:5s}]  real={n_real:,}  fake={n_fake:,}  "
              f"total={len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # PIL → numpy (albumentations expects HWC uint8)
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.float32)

# ── AUGMENTATIONS ──────────────────────────────────────────────
# Images are native 256×256 — we RandomCrop to 224 during training.
# This is free augmentation: position shift + the actual crop.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = A.Compose([
    A.RandomCrop(IMG_SIZE, IMG_SIZE),          # 256→224, random position
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(
        brightness=0.15, contrast=0.15,
        saturation=0.10, hue=0.03, p=0.5
    ),
    A.OneOf([
        A.GaussNoise(var_limit=(5, 20), p=1.0),
        A.GaussianBlur(blur_limit=3, p=1.0),
        A.ImageCompression(quality_lower=80, quality_upper=100, p=1.0),
    ], p=0.35),
    A.ShiftScaleRotate(
        shift_limit=0.03, scale_limit=0.05,
        rotate_limit=8, border_mode=0, p=0.25
    ),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# Validation / test: deterministic centre crop
val_tf = A.Compose([
    A.CenterCrop(IMG_SIZE, IMG_SIZE),          # 256→224, centred
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# ── MODEL ──────────────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE, pretrained=True, num_classes=0
        )
        feat_dim = self.backbone.num_features          # 1792 for B4

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
        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print(f"  [Epoch {UNFREEZE_AT+1}] Backbone unfrozen — full fine-tune begins\n")

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)

# ── ONE EPOCH ──────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, scaler, training):
    model.train() if training else model.eval()
    total_loss, probs_all, labels_all = 0.0, [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader,
                                  desc="  train" if training else "  eval ",
                                  ncols=88, leave=False):
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            probs_all.extend(torch.sigmoid(logits).detach().cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    probs_arr = np.array(probs_all)
    preds_arr = (probs_arr >= 0.5).astype(int)
    metrics = {
        "loss": total_loss / len(loader),
        "auc" : roc_auc_score(labels_all, probs_arr),
        "acc" : accuracy_score(labels_all, preds_arr),
        "f1"  : f1_score(labels_all, preds_arr),
    }
    return metrics, labels_all, probs_all

# ── PLOTS ──────────────────────────────────────────────────────
def save_training_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Training curves — {BACKBONE}", fontsize=12, fontweight="bold")
    keys = [("loss","Loss"), ("auc","AUC-ROC"), ("acc","Accuracy")]
    for ax, (k, title) in zip(axes, keys):
        ep = range(1, len(history[f"tr_{k}"]) + 1)
        ax.plot(ep, history[f"tr_{k}"], "b-o", ms=3, label="Train")
        ax.plot(ep, history[f"vl_{k}"], "r-o", ms=3, label="Val")
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "training_curves.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved → {out}")

def save_evaluation_plots(labels, probs):
    preds = (np.array(probs) >= 0.5).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Test set evaluation", fontsize=12, fontweight="bold")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Fake", "Real"],
                yticklabels=["Fake", "Real"], linewidths=0.5)
    axes[0].set_title("Confusion matrix")
    axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    axes[1].plot(fpr, tpr, "steelblue", lw=2, label=f"AUC = {auc:.4f}")
    axes[1].plot([0,1],[0,1],"k--", lw=1)
    axes[1].set_title("ROC curve")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "test_evaluation.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved → {out}")

# ── INFERENCE ──────────────────────────────────────────────────
def predict(image_path, threshold=0.5):
    """Run on any single .jpg image after training."""
    model = DeepfakeDetector().to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    img    = np.array(Image.open(image_path).convert("RGB"))
    tensor = val_tf(image=img)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad(), autocast():
        prob = torch.sigmoid(model(tensor)).item()

    verdict = "REAL ✓" if prob >= threshold else "FAKE ✗"
    conf    = prob if prob >= threshold else 1 - prob
    print(f"\n  File       : {os.path.basename(image_path)}")
    print(f"  Real prob  : {prob:.4f}")
    print(f"  Prediction : {verdict}   (confidence {conf*100:.1f}%)")
    return prob, verdict

# ── MAIN ───────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading datasets...")
    train_ds = FaceDataset(DATA_ROOT, "train", train_tf)
    val_ds   = FaceDataset(DATA_ROOT, "valid", val_tf)
    test_ds  = FaceDataset(DATA_ROOT, "test",  val_tf)

    # persistent_workers=True avoids re-spawning workers every epoch on Windows
    loader_kw = dict(num_workers=NUM_WORKERS, pin_memory=True,
                     persistent_workers=True)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              drop_last=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, **loader_kw)

    model     = DeepfakeDetector().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler()

    # Phase 1: only head trains (backbone frozen → low LR needed)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    history  = {k: [] for k in
                ["tr_loss","tr_auc","tr_acc","vl_loss","vl_auc","vl_acc"]}
    best_auc = 0.0

    print(f"\nTraining for {EPOCHS} epochs  "
          f"(backbone unfreezes at epoch {UNFREEZE_AT+1})\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Unfreeze backbone ──────────────────────────────────
        if epoch == UNFREEZE_AT + 1:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": LR * 0.05},
                {"params": model.head.parameters(),     "lr": LR},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS - UNFREEZE_AT, eta_min=1e-6
            )

        tr, *_ = run_epoch(model, train_loader, optimizer,
                           criterion, scaler, training=True)
        vl, vl_labels, vl_probs = run_epoch(
            model, val_loader, optimizer, criterion, scaler, training=False
        )
        scheduler.step()

        for k, v in [("tr_loss", tr["loss"]), ("tr_auc", tr["auc"]),
                     ("tr_acc",  tr["acc"]),  ("vl_loss", vl["loss"]),
                     ("vl_auc",  vl["auc"]),  ("vl_acc",  vl["acc"])]:
            history[k].append(v)

        flag = " ← best" if vl["auc"] > best_auc else ""
        print(
            f"Ep {epoch:02d}/{EPOCHS}  ({time.time()-t0:.0f}s) | "
            f"Train  loss={tr['loss']:.4f}  auc={tr['auc']:.4f}  "
            f"acc={tr['acc']:.4f} | "
            f"Val  loss={vl['loss']:.4f}  auc={vl['auc']:.4f}  "
            f"acc={vl['acc']:.4f}  f1={vl['f1']:.4f}{flag}"
        )

        if vl["auc"] > best_auc:
            best_auc = vl["auc"]
            torch.save(model.state_dict(), CHECKPOINT)

    # ── Final test evaluation ──────────────────────────────────
    print(f"\nBest val AUC: {best_auc:.4f}  — loading best checkpoint...\n")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))

    test_m, test_labels, test_probs = run_epoch(
        model, test_loader, None, criterion, scaler, training=False
    )
    preds = (np.array(test_probs) >= 0.5).astype(int)

    print("=" * 58)
    print("  TEST SET RESULTS")
    print("=" * 58)
    print(f"  AUC-ROC  : {test_m['auc']:.4f}")
    print(f"  Accuracy : {test_m['acc']:.4f}")
    print(f"  F1 Score : {test_m['f1']:.4f}")
    print()
    print(classification_report(test_labels, preds,
                                target_names=["Fake", "Real"]))

    save_training_curves(history)
    save_evaluation_plots(test_labels, test_probs)

    print("\nAll done! Output files:")
    print(f"  {CHECKPOINT}")
    print(f"  {PLOT_DIR}\\training_curves.png")
    print(f"  {PLOT_DIR}\\test_evaluation.png")

    # ── Predict a single image (uncomment after training) ──────
    # predict(r"D:\my_photo.jpg")