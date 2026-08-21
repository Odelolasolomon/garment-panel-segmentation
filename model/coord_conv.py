"""
CoordConv (Liu et al., "An Intriguing Failure of Convolutional Neural
Networks and the CoordConv Solution", NeurIPS 2018).

Plain convolutions are translation-equivariant: a kernel produces the same
response to a pattern no matter where in the image it appears. That is
usually a feature, but it is exactly the wrong property when the task
depends on *where* something is (e.g. "is this the panel on the left half
of the garment or the right half"). CoordConv concatenates two extra input
channels -- normalized x and y pixel coordinates -- so the network has
absolute position available as an explicit signal instead of relying on it
leaking in indirectly through zero-padding at the image borders.

In this codebase CoordConv is used in the decoder stages. The primary
left/right disambiguation mechanism is NOT this layer -- it is the
deterministic connected-component + centroid split done in
inference/postprocess.py. CoordConv is a secondary aid that gives the
decoder positional context for panels whose expected location is
informative (e.g. collar near top-center), it is not relied upon as the
sole source of left/right correctness. See DESIGN_NOTE.md / README.md for
the full reasoning.
"""
import torch
import torch.nn as nn


class AddCoords(nn.Module):
    """Concatenates normalized [-1, 1] x and y coordinate channels to the input."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        device, dtype = x.device, x.dtype

        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)

        y_grid = y_coords.view(1, 1, height, 1).expand(batch, 1, height, width)
        x_grid = x_coords.view(1, 1, 1, width).expand(batch, 1, height, width)

        return torch.cat([x, x_grid, y_grid], dim=1)


class CoordConv2d(nn.Module):
    """Drop-in replacement for nn.Conv2d that adds coordinate channels first."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super().__init__()
        self.add_coords = AddCoords()
        self.conv = nn.Conv2d(
            in_channels + 2, out_channels, kernel_size, padding=padding, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_coords(x)
        return self.conv(x)
