"""
Lightweight FPN-style decoder.

Design choices, and why:

- Depthwise-separable convolutions (depthwise 3x3 + pointwise 1x1) instead
  of full 3x3 convs everywhere. A full 3x3 conv from C_in to C_out costs
  9 * C_in * C_out parameters; the separable version costs
  9 * C_in + C_in * C_out, which is dramatically cheaper once channel
  counts climb into the tens. This is the single biggest lever for staying
  under the 2M trainable-parameter cap without gutting representational
  capacity.

- Narrow channel widths (32 throughout the decoder). The frozen encoder
  already supplies rich generic features; the decoder's job is fusion and
  upsampling, not re-learning representations, so it does not need to be
  wide.

- Top-down fusion (stride32 -> stride16 -> stride8 -> stride4), each stage
  upsampling the previous decoder output and adding a 1x1-projected skip
  connection from the encoder at that resolution. This is the standard
  FPN pattern: deep, low-resolution features carry semantic/global context
  (useful for e.g. front-vs-back-body, which depends on whole-garment
  context, not local texture); shallow, high-resolution features carry
  the spatial precision needed for clean panel boundaries.

- CoordConv (see coord_conv.py) is applied at the final, highest-resolution
  stage only, where absolute position is most useful and the spatial map
  is small enough that the two extra channels are cheap.

Internal class set predicted by this network: 5 classes --
  0 background, 1 front_body, 2 back_body, 3 sleeve, 4 collar.
Note "sleeve" is a single merged class, NOT left_sleeve/right_sleeve
separately. See README "Label set" section and DESIGN_NOTE-adjacent
reasoning in coord_conv.py / postprocess.py for why left/right is handled
as a deterministic post-processing step instead of a class the network
must learn to discriminate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .coord_conv import CoordConv2d

NUM_INTERNAL_CLASSES = 5  # background, front_body, back_body, sleeve, collar
DECODER_WIDTH = 160


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class FusionStage(nn.Module):
    """Upsamples the running decoder feature, projects + adds a skip connection,
    then refines with a depthwise-separable conv."""

    def __init__(self, skip_ch, dec_ch=DECODER_WIDTH):
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_ch, dec_ch, 1, bias=False)
        self.skip_bn = nn.BatchNorm2d(dec_ch)
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(dec_ch, dec_ch),
            DepthwiseSeparableConv(dec_ch, dec_ch),
        )

    def forward(self, dec_feat, skip_feat):
        dec_feat = F.interpolate(
            dec_feat, size=skip_feat.shape[-2:], mode="bilinear", align_corners=False
        )
        skip = self.skip_bn(self.skip_proj(skip_feat))
        return self.refine(dec_feat + skip)


class PanelDecoder(nn.Module):
    def __init__(self, encoder_channels=(16, 24, 48, 96), dec_ch=DECODER_WIDTH,
                 num_classes=NUM_INTERNAL_CLASSES):
        super().__init__()
        c4_ch, c8_ch, c16_ch, c32_ch = encoder_channels  # stride4,8,16,32

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c32_ch, dec_ch, 1, bias=False),
            nn.BatchNorm2d(dec_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse16 = FusionStage(c16_ch, dec_ch)
        self.fuse8 = FusionStage(c8_ch, dec_ch)
        self.fuse4 = FusionStage(c4_ch, dec_ch)

        # Final refinement with CoordConv for absolute-position awareness,
        # at the highest resolution decoder stage.
        self.coord_refine = CoordConv2d(dec_ch, dec_ch, kernel_size=3, padding=1)
        self.coord_bn = nn.BatchNorm2d(dec_ch)
        self.coord_act = nn.ReLU(inplace=True)

        self.classifier = nn.Conv2d(dec_ch, num_classes, 1)

    def forward(self, feats, out_size):
        f4, f8, f16, f32 = feats
        x = self.bottleneck(f32)
        x = self.fuse16(x, f16)
        x = self.fuse8(x, f8)
        x = self.fuse4(x, f4)

        x = self.coord_act(self.coord_bn(self.coord_refine(x)))
        logits = self.classifier(x)

        # Upsample from stride-4 to full input resolution.
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
