import os
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

DATA_DIR = Path("D:/playground/image-video-processing-course")
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"
TEST_CSV = DATA_DIR / "test.csv"
OUT_CSV = DATA_DIR / "submission.csv"
CKPT = DATA_DIR / "model.pt"

NUM_CLASSES = 10
IMG_SIZE = 32
BATCH = 256
EPOCHS = 12
LR = 1e-3
NUM_WORKERS = 0
SEED = 42
VAL_SPLIT = 0.1

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device} torch={torch.__version__}")


class TrainDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples = []
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = int(cls_dir.name)
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() == ".png":
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


class TestDataset(Dataset):
    def __init__(self, csv_path: Path, root: Path, transform=None):
        self.transform = transform
        self.root = root
        self.ids = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.ids.append(int(row[0]))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        img = Image.open(self.root / f"{img_id}.png").convert("L")
        if self.transform:
            img = self.transform(img)
        return img, img_id


train_tf = transforms.Compose([
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


class CompactCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_one_epoch(model, loader, opt, loss_fn, epoch):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        if (i + 1) % 20 == 0:
            print(f"  epoch {epoch} step {i+1}/{len(loader)} "
                  f"loss={loss_sum/total:.4f} acc={correct/total:.4f}")
    dt = time.time() - t0
    return loss_sum / total, correct / total, dt


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for x, ids in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(1).cpu().tolist()
        out.extend(zip(ids.tolist(), preds))
    return out


def main():
    print("loading train data...")
    full_train = TrainDataset(TRAIN_DIR, transform=train_tf)
    print(f"total samples: {len(full_train)}")

    n_val = int(len(full_train) * VAL_SPLIT)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    eval_full = TrainDataset(TRAIN_DIR, transform=eval_tf)
    val_set_eval = torch.utils.data.Subset(eval_full, val_set.indices)

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set_eval, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    test_set = TestDataset(TEST_CSV, TEST_DIR, transform=eval_tf)
    test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    print(f"train={len(train_set)} val={len(val_set_eval)} test={len(test_set)}")

    model = CompactCNN(NUM_CLASSES).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, dt = train_one_epoch(model, train_loader, opt, loss_fn, epoch)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        sched.step()
        print(f"[{epoch}/{EPOCHS}] train={tr_acc:.4f} val={val_acc:.4f} {dt:.1f}s")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), CKPT)
            print(f"  saved ({best_val:.4f})")

    model.load_state_dict(torch.load(CKPT, map_location=device))
    preds = predict(model, test_loader)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for img_id, cat in preds:
            w.writerow([img_id, cat])
    print(f"wrote {OUT_CSV} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
