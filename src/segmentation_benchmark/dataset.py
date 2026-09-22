"""The datasets every architecture reads, both built from one split manifest.

Section 18 asks for a reusable library rather than ten unrelated programs,
and Section 7 requires all ten to see identical partitions. Both datasets
here therefore take their image IDs from ``data/split_manifest.json`` and
have no way to choose images themselves - there is no ``shuffle``, no
``subset``, no sampling argument. A model that wanted a different split
would have to rebuild the manifest, which changes the checksum, which the
runner refuses to proceed past.

Section 18 also says not to force incompatible interfaces into one loop.
The two classes below share their split, their transform pipeline and their
mask decoding, and differ only in what they return: a dense label map for
the seven semantic models, a list of per-object records for the two instance
models. That is the seam the assignment describes, and it is the only place
the two tracks diverge before evaluation.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import mask_utils
from ._config import (
    ANNOTATIONS_DIR,
    DATA_DIR,
    IMAGES_DIR,
    INPUT_SIZE,
)
from .transforms import build_eval_transforms, build_train_transforms

MASKS_DIR = DATA_DIR / "masks"
MANIFEST_PATH = DATA_DIR / "split_manifest.json"


def load_manifest() -> dict:
    """Read the split manifest, with a message that says how to fix it."""
    if not MANIFEST_PATH.is_file():
        raise SystemExit(
            "No split manifest. Build the dataset first:\n"
            "  python scripts/build_coco_subset.py"
        )
    return json.loads(MANIFEST_PATH.read_text())


class SemanticSegmentationDataset(Dataset):
    """Image and dense class map, for the seven semantic architectures.

    Returns ``(image, mask)`` where image is a normalized float tensor of
    shape ``(3, size, size)`` and mask is an int64 tensor of shape
    ``(size, size)`` holding class IDs 0-5 or the ignore sentinel.

    Masks are read from the PNGs that ``build_coco_subset.py`` rendered
    rather than decoded from annotations here. Section 32 compares training
    time across architectures, and a per-epoch RLE decode would add a
    constant to every model's time that has nothing to do with the model.
    """

    def __init__(self, split: str, size: int = INPUT_SIZE, augment: bool = None):
        manifest = load_manifest()
        if split not in manifest["image_ids"]:
            raise ValueError(f"Unknown split {split!r}")

        # Augmentation defaults to on for training and off otherwise, which
        # is Section 14's rule. Passing it explicitly is allowed because the
        # Section 54 question about whether augmentation helped needs a run
        # with it disabled.
        self.augment = (split == "train") if augment is None else augment
        self.split = split
        self.size = size

        coco = json.loads((ANNOTATIONS_DIR / f"instances_{split}.json").read_text())
        self.records = sorted(coco["images"], key=lambda r: r["id"])

        self.image_dir = IMAGES_DIR / split
        self.mask_dir = MASKS_DIR / split
        self.transforms = (
            build_train_transforms(size) if self.augment
            else build_eval_transforms(size)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        stem = Path(record["file_name"]).stem

        image = np.array(
            Image.open(self.image_dir / record["file_name"]).convert("RGB")
        )
        mask = np.array(Image.open(self.mask_dir / f"{stem}.png"))

        sample = self.transforms({"image": image, "mask": mask})
        return sample["image"], sample["mask"]

    def describe(self) -> dict:
        return {
            "split": self.split,
            "images": len(self.records),
            "input_size": self.size,
            "augmented": self.augment,
            "transforms": self.transforms.describe(),
        }


class InstanceSegmentationDataset(Dataset):
    """Image and per-object targets, for Mask R-CNN.

    Returns ``(image, target)`` in the layout torchvision's detection models
    expect: ``boxes`` as (N, 4) float xyxy, ``labels`` as (N,) int64,
    ``masks`` as (N, H, W) uint8, plus ``image_id``, ``area`` and ``iscrowd``.

    ``iscrowd`` is all zeros by construction. Crowd annotations were dropped
    from the instance targets when the subset was built, because a crowd blob
    is one region covering many objects and is not a trainable instance. They
    survive in the evaluation annotation file, where COCOeval uses them to
    avoid penalizing a detection that lands on an unannotated group - so the
    information is not lost, it is used where it means something.

    Instance IDs are carried through so a prediction can be traced back to
    the COCO annotation it matched, which Section 9 requires to stay aligned
    through every transform.
    """

    def __init__(self, split: str, size: int = INPUT_SIZE, augment: bool = None):
        manifest = load_manifest()
        if split not in manifest["image_ids"]:
            raise ValueError(f"Unknown split {split!r}")

        self.augment = (split == "train") if augment is None else augment
        self.split = split
        self.size = size

        coco = json.loads((ANNOTATIONS_DIR / f"instances_{split}.json").read_text())
        self.records = sorted(coco["images"], key=lambda r: r["id"])
        self.annotations = {}
        for annotation in coco["annotations"]:
            self.annotations.setdefault(annotation["image_id"], []).append(annotation)

        # The subset file stores our contiguous IDs, but mask_utils keys its
        # mapping on COCO's original IDs. Inverting here keeps mask_utils as
        # the single decoder for both tracks instead of duplicating the RLE
        # handling with a different category convention.
        from ._config import COCO_CATEGORY_IDS
        self._to_coco = {v: k for k, v in COCO_CATEGORY_IDS.items()}

        self.image_dir = IMAGES_DIR / split
        self.transforms = (
            build_train_transforms(size) if self.augment
            else build_eval_transforms(size)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        height, width = record["height"], record["width"]

        image = np.array(
            Image.open(self.image_dir / record["file_name"]).convert("RGB")
        )

        annotations = [
            {**a, "category_id": self._to_coco[a["category_id"]]}
            for a in self.annotations.get(record["id"], [])
        ]
        targets = mask_utils.instance_targets(annotations, height, width)
        semantic = mask_utils.semantic_mask(annotations, height, width)

        sample = self.transforms({"image": image, "mask": semantic, **targets})

        target = {
            "boxes": sample["boxes"],
            "labels": sample["labels"],
            "masks": sample["masks"],
            "area": sample["areas"],
            "iscrowd": torch.zeros(len(sample["labels"]), dtype=torch.int64),
            "image_id": torch.tensor(record["id"], dtype=torch.int64),
            "instance_ids": sample["instance_ids"],
        }
        return sample["image"], target

    def describe(self) -> dict:
        return {
            "split": self.split,
            "images": len(self.records),
            "input_size": self.size,
            "augmented": self.augment,
            "transforms": self.transforms.describe(),
        }


def instance_collate(batch: list) -> tuple:
    """Batch instance samples without padding them into a common shape.

    Images stack; targets cannot. One image holds three objects and the next
    holds forty, and torchvision's detection models are written to take a
    list of per-image target dicts precisely so that nothing has to be padded
    to a fixed object count. Padding would add phantom zero-area objects to
    every loss term.
    """
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def build_loader(dataset, batch_size: int, shuffle: bool, workers: int = 4,
                 seed: int = None):
    """Wrap a dataset in a DataLoader with reproducible shuffling.

    The generator is seeded so that two runs of the same architecture see
    batches in the same order, which is what makes a rerun a rerun rather
    than a new experiment. Worker seeding is left to torch's default
    per-worker derivation from this generator.
    """
    from torch.utils.data import DataLoader
    from ._config import SEED

    generator = torch.Generator()
    generator.manual_seed(SEED if seed is None else seed)

    is_instance = isinstance(dataset, InstanceSegmentationDataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=instance_collate if is_instance else None,
        generator=generator,
        persistent_workers=workers > 0,
        pin_memory=False,
        # Training batches of one are dropped. PSPNet's pyramid pooling
        # reduces the feature map to 1x1 in its widest branch, and batch
        # normalization over a (1, C, 1, 1) tensor has a single value per
        # channel and raises. A benchmark that died in the last batch of an
        # epoch, for one architecture, after twenty minutes of training, is
        # a bad way to discover that.
        #
        # Only on the training loader: evaluation runs under model.eval(),
        # where batch norm uses its running statistics and a batch of one is
        # fine, and dropping evaluation images would mean the models were
        # scored on different test sets. At batch size 8 this discards
        # nothing anyway - all three splits divide exactly.
        drop_last=shuffle,
    )
