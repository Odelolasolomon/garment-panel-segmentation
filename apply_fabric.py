"""
apply_fabric.py -- Part 2, deterministic fabric fill.

Given a panel mask (as produced by predict.py) and a fabric swatch image,
fills the named panel with a flat tile of the fabric. No model is involved
in this step at all -- it is pure array indexing and image tiling, which is
the entire point of separating this from Part 1 (see assessment context:
"compositing... deterministically" is explicitly framed as NOT a machine
learning problem).

Guarantees:
  - apply_fabric("left_sleeve", ...) only ever touches pixels labelled
    left_sleeve in the mask. Same for every other panel name.
  - If the requested panel is absent from the mask (no pixels carry that
    index), the function returns the original image unchanged and does
    NOT raise -- this mirrors predict.py's absent-panel contract exactly,
    per the brief's "a request for an absent panel must return an empty
    mask without raising" requirement extended consistently into Part 2.
  - Same mask + same swatch + same panel name -> byte-identical output,
    every time (no randomness anywhere in this path).
"""
from pathlib import Path

import numpy as np
from PIL import Image

from model.postprocess import OUTPUT_INDEX


def _tile_fabric_to_shape(swatch: np.ndarray, height: int, width: int) -> np.ndarray:
    """Tiles a (h, w, 3) swatch to cover a (height, width, 3) region via wraparound indexing."""
    sh, sw = swatch.shape[0], swatch.shape[1]
    row_idx = np.arange(height) % sh
    col_idx = np.arange(width) % sw
    tiled = swatch[row_idx][:, col_idx]
    return tiled


def apply_fabric(image: np.ndarray, mask: np.ndarray, panel_name: str, swatch: np.ndarray) -> np.ndarray:
    """
    image:  (H, W, 3) uint8 RGB garment render
    mask:   (H, W) uint8 single-channel mask, values per OUTPUT_INDEX
            (exactly what predict.py writes to disk)
    panel_name: one of "front_body", "back_body", "left_sleeve",
                "right_sleeve", "collar"
    swatch: (h, w, 3) uint8 RGB fabric tile, any size

    Returns a new (H, W, 3) uint8 array. Does not mutate the inputs.
    """
    if panel_name not in OUTPUT_INDEX or panel_name == "background":
        raise ValueError(
            f"Unknown panel name '{panel_name}'. Must be one of "
            f"{[n for n in OUTPUT_INDEX if n != 'background']}"
        )

    panel_index = OUTPUT_INDEX[panel_name]
    panel_pixels = mask == panel_index

    result = image.copy()

    if not panel_pixels.any():
        # Absent panel: no-op, matches predict.py's "absent panel, no
        # exception" contract. Caller can check equality with the input
        # to detect this, or inspect panel_pixels.any() beforehand.
        return result

    height, width = mask.shape
    tiled_fabric = _tile_fabric_to_shape(swatch, height, width)

    result[panel_pixels] = tiled_fabric[panel_pixels]
    return result


def apply_fabric_from_files(image_path: str, mask_path: str, panel_name: str,
                             swatch_path: str, output_path: str) -> np.ndarray:
    """Convenience file-based wrapper matching how this would be invoked in the
    real pipeline (image path in, image path out)."""
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    swatch = np.array(Image.open(swatch_path).convert("RGB"))

    result = apply_fabric(image, mask, panel_name, swatch)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output_path)
    return result


if __name__ == "__main__":
    # Worked example (also exercised by tests/test_apply_fabric.py):
    # a synthetic garment image + mask + checkerboard swatch, filling
    # left_sleeve only.
    h, w = 100, 100
    image = np.full((h, w, 3), 200, dtype=np.uint8)  # light gray "render"
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:80, 35:65] = OUTPUT_INDEX["front_body"]
    mask[30:60, 5:30] = OUTPUT_INDEX["left_sleeve"]
    mask[30:60, 70:95] = OUTPUT_INDEX["right_sleeve"]

    swatch = np.zeros((10, 10, 3), dtype=np.uint8)
    swatch[::2, ::2] = [255, 0, 0]
    swatch[1::2, 1::2] = [255, 0, 0]  # red/black checkerboard

    result = apply_fabric(image, mask, "left_sleeve", swatch)

    left_region_changed = not np.array_equal(
        result[30:60, 5:30], image[30:60, 5:30]
    )
    right_region_unchanged = np.array_equal(
        result[30:60, 70:95], image[30:60, 70:95]
    )
    print("left_sleeve region modified:", left_region_changed)
    print("right_sleeve region untouched:", right_region_unchanged)
    assert left_region_changed and right_region_unchanged
    print("OK: apply_fabric worked example passed")
