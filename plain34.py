# plain34.py
# Plain-34 (tanpa residual) + training + plotting (history & confusion matrix)
from dataclasses import dataclass, asdict
from pathlib import Path
import json, random, time, os

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from contextlib import nullcontext

# ===================== Config (inline hyperparams) =====================
@dataclass
class CFG:
    data_root: Path = Path("datasettubes")
    train_csv: str  = "train.csv"
    img_dir: str    = "train"
    outdir: Path    = Path("runs/plain34")

    # training
    epochs: int     = 15
    batch_size: int = 32
    lr: float       = 1e-4
    val_ratio: float= 0.2
    img_size: int   = 224
    seed: int       = 42
    workers: int    = 0        # set 0 kalau Windows-mu suka rewel
    opt: str        = "adamw"    # "adamw" atau "sgd"
    weight_decay: float = 1e-4

cfg = CFG()

# ===================== Model =====================
class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # dibiarkan agar mirip struktur resnet (tidak dijumlahkan)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)  # TANPA residual sum
        return out

class Plain34(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.stage1 = self._make_stage(64,  64,  3, stride=1)
        self.stage2 = self._make_stage(64,  128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_stage(self, in_c, out_c, num_blocks, stride):
        layers = []
        downsample = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c),
        ) if (stride != 1 or in_c != out_c) else None
        layers.append(PlainBlock(in_c, out_c, stride=stride, downsample=downsample))
        for _ in range(1, num_blocks):
            layers.append(PlainBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)

# ===================== Data =====================
class CSVImageDataset(Dataset):
    def __init__(self, csv_path: Path, root_dir: Path, label_idx_map: dict, transform=None):
        self.df = pd.read_csv(csv_path)
        assert {"filename","label"}.issubset(self.df.columns)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label_idx_map = label_idx_map

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / str(row["filename"])
        y = self.label_idx_map[str(row["label"])]
        with Image.open(img_path) as im: im = im.convert("RGB")
        if self.transform: im = self.transform(im)
        return im, y

# ===================== Utils =====================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def stratified_split(df: pd.DataFrame, val_ratio=0.2, seed=42):
    g = df.groupby("label")
    idx_train, idx_val = [], []
    rng = np.random.default_rng(seed)
    for _, group in g:
        n = len(group); n_val = max(1, int(round(n*val_ratio)))
        perm = rng.permutation(n)
        idx_val.extend(group.index[perm[:n_val]].tolist())
        idx_train.extend(group.index[perm[n_val:]].tolist())
    return df.loc[idx_train].reset_index(drop=True), df.loc[idx_val].reset_index(drop=True)

def accuracy_from_logits(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()

def amp_ctx():
    """Autocast baru. Aktif hanya saat CUDA ada. Pilih bf16 jika GPU support, else fp16."""
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()

# ---------- Plotting ----------
def plot_history_csv(history_csv: Path, outdir: Path):
    df = pd.read_csv(history_csv)
    epochs = df["epoch"].values
    plt.figure(figsize=(8,5))
    plt.plot(epochs, df["train_loss"], label="train_loss")
    plt.plot(epochs, df["val_loss"],   label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend(); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(outdir / "loss_curve.png", dpi=200); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, df["train_acc"], label="train_acc")
    plt.plot(epochs, df["val_acc"],   label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve"); plt.legend(); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(outdir / "acc_curve.png", dpi=200); plt.close()

def plot_confusion_matrix(cm: np.ndarray, classes: list, out_png: Path, normalize=True):
    fig, ax = plt.subplots(figsize=(6.5,6))
    mat = cm.astype(float)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True); row_sums[row_sums==0] = 1.0
        mat = mat / row_sums
    im = ax.imshow(mat, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel="Predicted", ylabel="True",
           title="Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".2f" if normalize else "d"
    thresh = mat.max() / 2.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout(); plt.savefig(out_png, dpi=220); plt.close(fig)

# ===================== Train/Eval =====================
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = total_acc = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device.type == "cuda":
            with amp_ctx():                      # <<< pakai API baru
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc  += (logits.argmax(1) == y).float().mean().item() * bs
        n += bs
    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds: bool = False):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    all_targets, all_preds = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # autocast baru (aktif hanya di CUDA)
        with amp_ctx():
            logits = model(x)
            loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(logits, y) * bs
        n += bs

        if return_preds:
            all_targets.append(y.detach().cpu().numpy())
            all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())

    avg_loss = total_loss / max(n, 1)
    avg_acc  = total_acc / max(n, 1)

    if return_preds:
        y_true = np.concatenate(all_targets) if all_targets else np.array([])
        y_pred = np.concatenate(all_preds)   if all_preds else np.array([])
        return avg_loss, avg_acc, y_true, y_pred
    return avg_loss, avg_acc


# ===================== Main =====================
def main():
    set_seed(cfg.seed)
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    print("Config:", json.dumps(asdict(cfg), indent=2, default=str))

    # CSV & classes
    csv_path = cfg.data_root / cfg.train_csv
    img_root = cfg.data_root / cfg.img_dir
    df = pd.read_csv(csv_path)
    assert {"filename","label"}.issubset(df.columns)
    classes = sorted(df["label"].astype(str).unique().tolist())
    label_to_idx = {c:i for i,c in enumerate(classes)}
    with open(cfg.outdir / "classes.json", "w") as f:
        json.dump({"classes": classes, "label_to_idx": label_to_idx}, f, indent=2)

    # split
    df_train, df_val = stratified_split(df, val_ratio=cfg.val_ratio, seed=cfg.seed)
    df_train.to_csv(cfg.outdir / "train_split.csv", index=False)
    df_val.to_csv(cfg.outdir / "val_split.csv", index=False)

    # transforms
    train_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # dataset & loader
    ds_train = CSVImageDataset(cfg.outdir / "train_split.csv", img_root, label_to_idx, transform=train_tfms)
    ds_val   = CSVImageDataset(cfg.outdir / "val_split.csv",   img_root, label_to_idx, transform=val_tfms)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.workers, pin_memory=True)

    # model/opt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Plain34(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    if cfg.opt.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                                    weight_decay=cfg.weight_decay, nesterov=True)

    # Pakai GradScaler dengan flag enabled agar aman di CPU/GPU
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


    # train
    history = []
    best_val_acc = 0.0
    best_path = cfg.outdir / "best_plain34.pth"
    start = time.time()
    for epoch in range(1, cfg.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, criterion, device, scaler)
        va_loss, va_acc = evaluate(model, dl_val, criterion, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"[{epoch:03d}/{cfg.epochs}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "label_to_idx": label_to_idx,
                "epoch": epoch,
                "val_acc": best_val_acc,
                "cfg": asdict(cfg),
            }, best_path)

        torch.save({"model_state": model.state_dict(), "epoch": epoch}, cfg.outdir / "last_plain34.pth")
        pd.DataFrame(history).to_csv(cfg.outdir / "history.csv", index=False)

    dur = (time.time() - start)/60
    print(f"Selesai {dur:.1f} menit. Best val_acc = {best_val_acc:.4f}")
    print(f"Best weight: {best_path}")

    # plotting
    plot_history_csv(cfg.outdir / "history.csv", cfg.outdir)
    print(f"Saved plots: {cfg.outdir/'loss_curve.png'}, {cfg.outdir/'acc_curve.png'}")

    # confusion matrix pada split validasi
    _, _, y_true, y_pred = evaluate(model, dl_val, criterion, device, return_preds=True)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    np.savetxt(cfg.outdir / "confusion_matrix_raw.csv", cm, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm, classes, cfg.outdir / "confusion_matrix.png", normalize=True)
    print(f"Saved confusion matrix: {cfg.outdir/'confusion_matrix.png'} and CSV")

if __name__ == "__main__":
    main()
