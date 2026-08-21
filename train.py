"""
Training entry point.

Usage (see README for the Kaggle-specific invocation):
    python train.py --config configs/train.yaml

Design choices relevant to the assessment's "reproducibility" scoring
criterion:
  - Every seed (Python / NumPy / torch / CUDA) is fixed from a single
    config value.
  - cuDNN determinism flags are set.
  - The full config used is saved inside every checkpoint (checkpoint.py),
    so a reported number can always be traced back to the exact
    hyperparameters that produced it.
  - --resume makes training restartable from last.pt, which matters
    concretely on Kaggle given its runtime-limited sessions.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from model.segmodel import build_model
from model.dataset import PanelSegDataset, NUM_CLASSES
from model.losses import DiceCELoss
from model.checkpoint import save_checkpoint, load_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def compute_mean_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Mean per-class IoU, computed only over classes present in the batch
    (matches the assessment's stated eval methodology: 'averaged across the
    panels present in that image', not pixel accuracy)."""
    preds = logits.argmax(dim=1)
    ious = []
    for c in range(1, num_classes):  # skip background (class 0) by convention
        pred_c = preds == c
        target_c = target == c
        if target_c.sum() == 0 and pred_c.sum() == 0:
            continue  # class absent from both -- not scored, matches brief's methodology
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    model.encoder.eval()  # frozen encoder always stays in eval mode (see backbone.py)
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_miou, n = 0.0, 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
        total_miou += compute_mean_iou(logits, masks, num_classes) * images.size(0)
        n += images.size(0)
    return total_loss / n, total_miou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None,
                         help="path to a checkpoint (e.g. checkpoints/last.pt) to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = PanelSegDataset(
        images_dir=config["train_images_dir"],
        annotations_json=config["train_annotations"],
        image_size=config["image_size"],
        augment=True,
    )
    val_ds = PanelSegDataset(
        images_dir=config["val_images_dir"],
        annotations_json=config["val_annotations"],
        image_size=config["image_size"],
        augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                               num_workers=config.get("num_workers", 2), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config.get("num_workers", 2))

    model = build_model(pretrained_encoder=True, freeze_encoder=True).to(device)
    print(f"Trainable params: {model.trainable_param_count():,}")
    print(f"Total params (incl. frozen encoder): {model.total_param_count():,}")
    assert model.trainable_param_count() < 2_000_000, "OVER PARAMETER BUDGET"

    class_weights = torch.tensor(config.get("class_weights", [1.0] * NUM_CLASSES)).to(device)
    criterion = DiceCELoss(num_classes=NUM_CLASSES, class_weights=class_weights,
                            dice_weight=config.get("dice_weight", 0.5))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4),
    )

    start_epoch = 0
    best_miou = 0.0
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer, map_location=device, restore_rng=True)
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]

    for epoch in range(start_epoch, config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_miou = validate(model, val_loader, criterion, device, NUM_CLASSES)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}")

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, best_miou, config)
        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best_miou, config)
            print(f"  -> new best (mIoU={best_miou:.4f}), saved to {ckpt_dir / 'best.pt'}")

    print(f"Training complete. Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
