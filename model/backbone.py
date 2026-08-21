"""
Frozen, ImageNet-pretrained MobileNetV3-Small encoder.

This is deliberately NOT trained. The 2,000,000 trainable-parameter budget
in the assessment is tight enough that spending it on re-learning generic
edge/shape/texture features from scratch on a small custom dataset would be
a poor use of capacity. A frozen pretrained encoder supplies those generic
features "for free" (they do not count against the trainable budget) and
the trainable budget is spent entirely on the decoder, which is the part
of the network that actually needs to be specialised to garment panels.

We expose four feature maps at strides 4, 8, 16 and 32 (an FPN-style tap
pattern) so the decoder can fuse fine spatial detail (for boundary
accuracy) with coarse global context (for panel-identity decisions like
front vs. back body, which depend on the whole garment, not local texture).
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class MobileNetV3SmallEncoder(nn.Module):
    """
    Returns feature maps at stride 4, 8, 16, 32 for a 3xHxW input.
    Channel counts: (16, 24, 48, 96)
    """

    FEATURE_INDICES = (1, 3, 8, 11)  # indices into m.features, see README for derivation
    OUT_CHANNELS = (16, 24, 48, 96)

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # nn.Sequential of 13 blocks

        self.freeze = freeze
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
            self.features.eval()

    def train(self, mode: bool = True):
        # Keep frozen backbone permanently in eval mode (BatchNorm stats fixed)
        # even when the parent module is switched to train() for the decoder.
        super().train(mode)
        if self.freeze:
            self.features.eval()
        return self

    def forward(self, x: torch.Tensor):
        outs = []
        h = x
        max_idx = self.FEATURE_INDICES[-1]
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.FEATURE_INDICES:
                outs.append(h)
            if i == max_idx:
                break
        return outs  # [stride4, stride8, stride16, stride32]

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
