"""
Tests for the model architecture itself: parameter budget compliance,
forward pass correctness, and confirmation that the encoder is actually
frozen (not just intended to be).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.segmodel import build_model


@pytest.fixture(scope="module")
def model():
    m = build_model(pretrained_encoder=False, freeze_encoder=True)  # no network call in tests
    m.eval()
    return m


def test_trainable_params_under_budget(model):
    assert model.trainable_param_count() < 2_000_000, (
        f"Trainable params {model.trainable_param_count():,} exceeds the "
        f"2,000,000 hard constraint."
    )


def test_encoder_params_are_frozen(model):
    for p in model.encoder.parameters():
        assert not p.requires_grad, "Encoder parameter has requires_grad=True; encoder must be frozen"


def test_forward_pass_output_shape(model):
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5, 256, 256), f"Unexpected output shape {tuple(out.shape)}"


def test_forward_pass_handles_non_square_input(model):
    # production images may not be perfectly square; the model should
    # still run without shape errors as long as dims are conv-friendly
    x = torch.randn(1, 3, 224, 288)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5, 224, 288)


def test_deterministic_forward_pass_in_eval_mode(model):
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out_a = model(x)
        out_b = model(x)
    assert torch.equal(out_a, out_b), "Model in eval mode must be deterministic for identical input"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
