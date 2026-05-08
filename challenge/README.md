# IIVP 2026 Challenge — Group 9

10-class handwritten character classification on 32×32 grayscale images.
Kaggle public score: **1.0000000**

## Approach

EfficientNet-B0 pretrained on ImageNet, fine-tuned end-to-end on the challenge dataset. The grayscale images are replicated to 3 channels and upscaled to 96×96 to make better use of the pretrained weights.

**Training:**
- Optimizer: AdamW (weight decay 1e-4)
- Learning rate: 5e-4 with cosine annealing and 3-epoch linear warmup
- Label smoothing ε = 0.1, Mixup α = 0.3
- Augmentation: random rotation ±15°, affine, perspective, random erasing
- 40 epochs per seed, AMP (float16) on CUDA

**Inference:**
- 3 models trained with different seeds (42, 123, 456)
- 5-pass test-time augmentation per model (original + 4 slight rotations)
- Final prediction = argmax of averaged softmax probabilities (15 total passes per image)

## Setup

```bash
pip install -r requirements.txt
```

GPU recommended. For CUDA 12.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Usage

```bash
python train_v2.py --data-dir /path/to/iivp-2026-challenge
```

The dataset directory should contain `train/train/{0-9}/` and `test/test/`, plus `test.csv`. The script writes `submission.csv` to the same directory when done.

Available options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | script parent | Path to dataset root |
| `--epochs` | 40 | Epochs per seed |
| `--seeds` | 42 123 456 | Seeds for ensemble |
| `--tta-n` | 5 | TTA passes at inference |
| `--batch` | 256 | Batch size |
| `--img-size` | 96 | Input resolution |
| `--lr` | 5e-4 | Peak learning rate |

## Results

| Seed | Best validation accuracy |
|------|--------------------------|
| 42   | 99.94%                   |
| 123  | 100.00%                  |
| 456  | 100.00%                  |

Kaggle public leaderboard: **1.0000000**
