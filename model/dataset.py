"""
Fashionpedia -> internal 5-class panel dataset.

Internal classes: background(0), front_body(1), back_body(2), sleeve(3), collar(4)

Fashionpedia's attribute/category taxonomy does not map 1:1 onto these.
We derive them from Fashionpedia's garment-part / category annotations as
follows (documented here so the mapping is auditable, per the README
"label set and why" requirement):

  - sleeve            <- Fashionpedia "sleeve" part annotations (left/right
                          merged into one class deliberately -- see
                          postprocess.py and DESIGN_NOTE reasoning: local
                          sleeve texture is not a reliable learning signal
                          for laterality, so we do not ask the network to
                          discriminate it).
  - collar            <- Fashionpedia "collar" / neckline part annotations.
  - front_body/back_body <- Fashionpedia is photographed almost entirely
                          front-on (worn garments, camera facing the
                          person). We treat the main torso/bodice
                          annotation as front_body for the (large) majority
                          of images. back_body is systematically
                          underrepresented in Fashionpedia -- this is the
                          domain-gap point flagged in the assessment brief
                          and in README "what is broken". We supplement
                          with a small synthetic/hand-labeled back-facing
                          set (see data/synthetic_back/) rather than
                          pretending Fashionpedia solves this.

IMPORTANT -- flip augmentation and label correctness:
Horizontal flip is normally "free" augmentation. Here it is not free for
any laterally-asymmetric label, because flipping the image swaps which
side of the garment is which. Since "sleeve" is a single merged class in
our internal label set, a flip does not actually change any label (both
sleeves map to the same class either way) -- so flipping is safe and
label-preserving under our chosen label design. This is a direct
consequence of the merged-sleeve decision: it is also what makes flip
augmentation usable at all without a label-swap step. If a future version
of this dataset splits sleeve into left/right internally, flip
augmentation MUST swap those two labels or it will actively train the
model in the wrong direction -- this is called out explicitly so it is
never silently reintroduced.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

BACKGROUND, FRONT_BODY, BACK_BODY, SLEEVE, COLLAR = 0, 1, 2, 3, 4
NUM_CLASSES = 5

# Fashionpedia category IDs relevant to our mapping (from the official
# category list at https://github.com/cvdfoundation/fashionpedia -- verify
# against the category json you actually download, IDs are dataset-version
# dependent and MUST be checked, not assumed, before training).
# Placeholder mapping -- fill in from categories_attributes.json after download.
FASHIONPEDIA_CATEGORY_TO_INTERNAL = {
    # "sleeve": SLEEVE,
    # "collar": COLLAR,
    # "torso/bodice main panel": FRONT_BODY,
}


class PanelSegDataset(Dataset):
    """
    Expects a directory of images + a COCO-style panel annotation json
    (Fashionpedia's native format). Produces (image_tensor, mask_tensor)
    pairs where mask is a single-channel LongTensor of internal class ids.
    """

    def __init__(self, images_dir, annotations_json, image_size=256,
                 augment=True, category_map=None):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.augment = augment
        self.category_map = category_map or FASHIONPEDIA_CATEGORY_TO_INTERNAL

        with open(annotations_json) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.anns_by_image = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def _rasterize_mask(self, image_id, height, width):
        """Builds a single-channel internal-class mask from COCO polygon/RLE anns."""
        from pycocotools import mask as maskUtils

        mask = np.zeros((height, width), dtype=np.uint8)
        anns = self.anns_by_image.get(image_id, [])
        for ann in anns:
            cat_id = ann["category_id"]
            internal_cls = self.category_map.get(cat_id)
            if internal_cls is None:
                continue  # category not part of our internal label set, ignore
            if isinstance(ann["segmentation"], list):
                rles = maskUtils.frPyObjects(ann["segmentation"], height, width)
                rle = maskUtils.merge(rles)
            else:
                rle = ann["segmentation"]
            m = maskUtils.decode(rle).astype(bool)
            mask[m] = internal_cls  # later anns can overwrite earlier at overlaps
        return mask

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info = self.images[image_id]
        img_path = self.images_dir / info["file_name"]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        mask_np = self._rasterize_mask(image_id, h, w)
        mask = Image.fromarray(mask_np, mode="L")

        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size],
                          interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            # Horizontal flip: label-safe here because sleeve is a merged
            # class (see module docstring). front_body/back_body are not
            # laterally swapped by a horizontal flip (that would require a
            # front/back swap, which flipping does not do), so no relabeling
            # needed for those either.
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Mild photometric jitter -- helps close some of the domain gap
            # between Fashionpedia's real-world photographic lighting and
            # the clean, controlled lighting of production 3D renders (see
            # README "domain gap" section). This is a cheap mitigation, not
            # a claim of having solved the gap.
            if random.random() < 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        return image_tensor, mask_tensor
