import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
parser.add_argument("--img-size", type=int, default=96)
parser.add_argument("--tta-n", type=int, default=5)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--amp", action="store_true")
args = parser.parse_args()

_default = os.environ.get("IIVP_DATA_DIR") or str(Path(__file__).resolve().parent)
DATA_DIR = Path(args.data_dir or _default)
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"
TEST_CSV = DATA_DIR / "test.csv"
OUT_CSV = DATA_DIR / "submission.csv"
NUM_CLASSES = 10
IMG_SIZE = args.img_size
BATCH = args.batch
EPOCHS = args.epochs
LR = args.lr
SEEDS = args.seeds
TTA_N = args.tta_n
VAL_SPLIT = 0.1
MIXUP_ALPHA = 0.3
LABEL_SMOOTH = 0.1
NUM_WORKERS = args.workers
WARMUP_EPOCHS = 3
USE_AMP = args.amp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda" and not USE_AMP:
    USE_AMP = True
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
print(f"device={device} img={IMG_SIZE} batch={BATCH} epochs={EPOCHS} seeds={SEEDS} amp={USE_AMP}", flush=True)


class TrainDS(Dataset):
    def __init__(self, root: Path, tf=None):
        self.tf = tf
        self.images = []
        self.labels = []
        print("loading train images...", flush=True)
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            lbl = int(cls_dir.name)
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() == ".png":
                    img = Image.open(p).convert("RGB")
                    img.load()
                    self.images.append(img)
                    self.labels.append(lbl)
        print(f"{len(self.images)} images loaded", flush=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.tf:
            img = self.tf(img)
        return img, self.labels[i]


class TestDS(Dataset):
    def __init__(self, csv_path: Path, root: Path, tf=None):
        self.tf = tf
        self.ids = []
        self.images = []
        with open(csv_path) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                self.ids.append(int(row[0]))
        print("loading test images...", flush=True)
        for img_id in self.ids:
            img = Image.open(root / f"{img_id}.png").convert("RGB")
            img.load()
            self.images.append(img)
        print(f"{len(self.images)} test images loaded", flush=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = self.images[i]
        if self.tf:
            img = self.tf(img)
        return img, self.ids[i]


# ImageNet normalization stats for the pretrained model
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def tta_tfs(n, img_size):
    angles = [0, 8, -8, 15, -15][:n]
    tfs = []
    for angle in angles:
        if angle == 0:
            tfs.append(eval_tf)
        else:
            tfs.append(transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation((angle, angle)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]))
    return tfs


def build_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return m.to(device)


def mixup_data(x, y, alpha=0.3):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = random.betavariate(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return x * lam + x[idx] * (1 - lam), y, y[idx], lam


def mixup_loss(criterion, preds, y_a, y_b, lam):
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


def train_one_epoch(model, loader, opt, criterion, sched_warmup, epoch):
    model.train()
    tot, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if sched_warmup and epoch <= WARMUP_EPOCHS:
            step = (epoch - 1) * len(loader) + i
            lr_scale = min(1.0, (step + 1) / (WARMUP_EPOCHS * len(loader)))
            for pg in opt.param_groups:
                pg["lr"] = LR * lr_scale

        xm, ya, yb, lam = mixup_data(x, y, MIXUP_ALPHA)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
            out = model(xm)
            loss = mixup_loss(criterion, out, ya, yb, lam)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        tot += x.size(0)

    dt = time.time() - t0
    return loss_sum / tot, correct / tot, dt


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        tot += x.size(0)
    return loss_sum / tot, correct / tot


@torch.no_grad()
def predict_tta(model, base_test_ds, n=5):
    model.eval()
    tta_transforms = tta_tfs(n, IMG_SIZE)
    all_probs = None
    for tta_idx, tf in enumerate(tta_transforms):
        orig_tf = base_test_ds.tf
        base_test_ds.tf = tf
        pm = device.type == "cuda"
        loader = DataLoader(base_test_ds, batch_size=BATCH * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pm)
        probs_list = []
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                logits = model(x)
            probs_list.append(F.softmax(logits.float(), dim=1).cpu())
        base_test_ds.tf = orig_tf
        probs_pass = torch.cat(probs_list, dim=0)
        all_probs = probs_pass if all_probs is None else all_probs + probs_pass
        print(f"TTA pass {tta_idx+1}/{n} done", flush=True)
    all_probs /= n
    return all_probs.argmax(1).tolist(), all_probs


class AugWrapper(Dataset):
    def __init__(self, base_ds, indices, tf):
        self.base = base_ds
        self.indices = indices
        self.tf = tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img = self.base.images[self.indices[i]]
        lbl = self.base.labels[self.indices[i]]
        return self.tf(img), lbl


def main():
    base_ds = TrainDS(TRAIN_DIR, tf=None)
    n_val = int(len(base_ds) * VAL_SPLIT)
    n_train = len(base_ds) - n_val

    test_ds_base = TestDS(TEST_CSV, TEST_DIR, tf=eval_tf)
    test_ids = test_ds_base.ids
    print(f"train={n_train} val={n_val} test={len(test_ids)}", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    ensemble_probs = None

    for seed in SEEDS:
        print(f"\n--- seed {seed} ---", flush=True)
        random.seed(seed)
        torch.manual_seed(seed)

        all_idx = list(range(len(base_ds)))
        random.shuffle(all_idx)
        train_idx = all_idx[:n_train]
        val_idx = all_idx[n_train:]

        train_ds = AugWrapper(base_ds, train_idx, train_tf)
        val_ds = AugWrapper(base_ds, val_idx, eval_tf)

        pm = device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=pm)
        val_loader = DataLoader(val_ds, batch_size=BATCH * 2, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=pm)

        model = build_model()
        print(f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS),
                                                      eta_min=LR * 0.01)

        best_val = 0.0
        ckpt_path = DATA_DIR / f"model_seed{seed}.pt"

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc, dt = train_one_epoch(model, train_loader, opt, criterion,
                                                   sched_warmup=(epoch <= WARMUP_EPOCHS),
                                                   epoch=epoch)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            if epoch > WARMUP_EPOCHS:
                sched.step()

            lr_now = opt.param_groups[0]["lr"]
            print(f"[{epoch}/{EPOCHS}] tr={tr_acc:.4f} val={val_acc:.4f} "
                  f"loss={tr_loss:.4f}/{val_loss:.4f} lr={lr_now:.2e} {dt:.1f}s", flush=True)
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), ckpt_path)

        print(f"seed {seed} best val={best_val:.4f}", flush=True)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        preds, probs = predict_tta(model, test_ds_base, n=TTA_N)
        ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs

    ensemble_probs /= len(SEEDS)
    final_preds = ensemble_probs.argmax(1).tolist()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for img_id, cat in zip(test_ids, final_preds):
            w.writerow([img_id, cat])
    print(f"done — {OUT_CSV} ({len(final_preds)} rows)", flush=True)


if __name__ == "__main__":
    main()
