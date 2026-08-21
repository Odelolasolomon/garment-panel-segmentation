"""
predict.py -- required CLI entry point.

Usage:
    python predict.py --image path/to/garment.jpg --output path/to/mask.png [--weights weights/best.pt]

Writes a single-channel PNG where pixel values index into this fixed
mapping (also documented in README.md -- single source of truth is
model/postprocess.py:OUTPUT_INDEX, both docs reference it directly so they
cannot drift apart):

    0 = background
    1 = front_body
    2 = back_body
    3 = left_sleeve
    4 = right_sleeve
    5 = collar

A panel the model does not detect in a given image is absent from the mask
(no pixels carry that index) rather than guessed -- this file never raises
on an absent panel, per the brief's explicit requirement.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).parent))
from model.segmodel import build_model
from model.checkpoint import load_weights_only
from model.postprocess import split_left_right_sleeve, masks_to_indexed_png, OUTPUT_INDEX

IMAGE_SIZE = 256  # must match the size used in training/configs/train.yaml
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(weights_path: str, device: torch.device):
    model = build_model(pretrained_encoder=False, freeze_encoder=True)
    load_weights_only(weights_path, model, map_location=device)
    model.to(device)
    model.eval()
    return model


def preprocess(image_path: str, image_size: int = IMAGE_SIZE):
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size[::-1]  # (H, W)
    resized = TF.resize(image, [image_size, image_size])
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor.unsqueeze(0), orig_size


@torch.no_grad()
def predict_mask(model, image_path: str, device: torch.device, return_original_size: bool = True) -> np.ndarray:
    """
    Returns a single-channel uint8 numpy array using OUTPUT_INDEX values.
    Never raises for a garment with an absent panel -- absent panels simply
    contribute no pixels to the returned array.
    """
    tensor, orig_size = preprocess(image_path)
    tensor = tensor.to(device)

    logits = model(tensor)  # (1, 5, H, W) internal classes
    internal_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    panel_masks = split_left_right_sleeve(internal_mask)  # deterministic, see postprocess.py
    indexed = masks_to_indexed_png(panel_masks)

    if return_original_size:
        indexed_img = Image.fromarray(indexed, mode="L")
        indexed_img = indexed_img.resize(orig_size[::-1], resample=Image.NEAREST)
        indexed = np.array(indexed_img)

    return indexed


def main():
    parser = argparse.ArgumentParser(description="Predict garment panel segmentation mask.")
    parser.add_argument("--image", required=True, help="Path to input garment image")
    parser.add_argument("--output", required=True, help="Path to write output single-channel PNG mask")
    parser.add_argument("--weights", default="weights/best.pt", help="Path to trained model checkpoint")
    parser.add_argument("--cpu", action="store_true", default=True,
                         help="Force CPU inference (default: True, per assessment constraint)")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(args.weights, device)
    mask = predict_mask(model, args.image, device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(args.output)

    present = [name for name, idx in OUTPUT_INDEX.items() if idx != 0 and (mask == idx).any()]
    absent = [name for name, idx in OUTPUT_INDEX.items() if idx != 0 and name not in present]
    print(f"Saved mask to {args.output}")
    print(f"Panels detected:  {present}")
    print(f"Panels absent:    {absent}")


if __name__ == "__main__":
    main()
