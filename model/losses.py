"""
Combined Dice + weighted Cross-Entropy loss.

Why not plain cross-entropy alone: panel classes are heavily imbalanced
(collar is a small fraction of pixels compared to front_body/back_body;
background dominates every image). Plain CE lets the model minimize loss
by mostly predicting background/large-panel classes and ignoring collar
almost entirely, which is precisely the failure mode the assessment's
"mask accuracy... including panels your training data underrepresents"
scoring criterion is designed to catch.

Dice loss is computed per-class and is naturally robust to imbalance
because it directly optimizes overlap (intersection over union-like
signal) rather than per-pixel likelihood. We combine it with a
class-weighted CE for stable early-training gradients (Dice alone can be
unstable when a class is entirely absent from a batch).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int, class_weights=None, dice_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)

        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)
        dice_per_class = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        dice_loss = 1.0 - dice_per_class.mean()

        return (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss


def compute_class_weights(class_pixel_counts: dict, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weighting from a {class_id: pixel_count} dict gathered
    over the training set (see train.py --compute_class_weights)."""
    counts = torch.tensor(
        [class_pixel_counts.get(c, 1) for c in range(num_classes)], dtype=torch.float
    )
    weights = counts.sum() / (num_classes * counts)
    return weights
