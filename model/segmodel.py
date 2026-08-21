import torch
import torch.nn as nn

from .backbone import MobileNetV3SmallEncoder
from .decoder import PanelDecoder, NUM_INTERNAL_CLASSES

# Fixed internal label indices (what the network is trained to predict).
# NOTE: this is the INTERNAL set used during training/inference -- it is
# deliberately different from the 5 output names required by the
# assessment brief (front_body, back_body, left_sleeve, right_sleeve,
# collar). Left/right sleeve is derived deterministically from the single
# "sleeve" class in postprocess.py -- see README "Label set" section.
INTERNAL_CLASSES = ["background", "front_body", "back_body", "sleeve", "collar"]


class PanelSegModel(nn.Module):
    def __init__(self, pretrained_encoder: bool = True, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = MobileNetV3SmallEncoder(pretrained=pretrained_encoder, freeze=freeze_encoder)
        self.decoder = PanelDecoder(
            encoder_channels=self.encoder.OUT_CHANNELS,
            num_classes=NUM_INTERNAL_CLASSES,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        logits = self.decoder(feats, out_size=x.shape[-2:])
        return logits  # (B, 5, H, W)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(pretrained_encoder: bool = True, freeze_encoder: bool = True) -> PanelSegModel:
    return PanelSegModel(pretrained_encoder=pretrained_encoder, freeze_encoder=freeze_encoder)


if __name__ == "__main__":
    model = build_model(pretrained_encoder=False)  # avoid network call for a quick local check
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print("output shape:", tuple(out.shape))
    print("trainable params:", model.trainable_param_count())
    print("total params (incl. frozen encoder):", model.total_param_count())
    assert model.trainable_param_count() < 2_000_000, "OVER PARAMETER BUDGET"
    print("OK: under 2,000,000 trainable parameter budget")
