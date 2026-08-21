"""
Checkpoint utilities.

Saves full training state (model, optimizer, epoch, RNG states, config) so
a run can be resumed exactly, not just reloaded for inference. Kaggle
sessions have a runtime cap (~9-12h); training is written to be resumable
so an interrupted run does not lose progress -- see train.py --resume.

best.pt and last.pt are kept separate so predict.py / submission packaging
always has an unambiguous file to point at (best.pt, selected by
validation mean IoU) without depending on which epoch the run happened to
stop on.
"""
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, epoch, best_miou, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
            "config": config,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, map_location="cpu", restore_rng=False):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if restore_rng and "rng_state" in ckpt:
        rng = ckpt["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return ckpt


def load_weights_only(path, model, map_location="cpu"):
    """Minimal loader for predict.py / apply_fabric -- no optimizer, no RNG,
    just the trained weights."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model
