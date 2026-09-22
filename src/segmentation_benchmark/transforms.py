"""Mask-safe augmentation: geometry applied to the image and every target.

Section 15 states the rule this module exists to enforce - a flipped or
rotated image paired with an unchanged mask is invalid training data. Every
geometric transform here therefore takes the whole sample and rewrites all
of it: the image, the semantic mask, each instance mask, and each instance
box. Color transforms take the image alone.

A sample is a plain dict so that the semantic and instance tracks can share
one pipeline. It always carries ``image`` and ``mask``; it carries ``masks``,
``labels``, ``boxes``, ``areas`` and ``instance_ids`` as well when the
instance track built it. Transforms that would invalidate instance data
update it when present and ignore it when absent, so the same configured
pipeline can serve both tracks and cannot apply two different augmentations
to what is supposed to be one shared protocol.

Two decisions here are easy to get wrong and are worth stating.

**Boxes are recomputed from the transformed mask, never transformed
directly.** Rotating the four corners of an axis-aligned box gives a
rotated rectangle, and taking its extent gives a box strictly larger than
the object. Re-deriving the box from the mask that was just rotated is exact
by construction, and it is the mask that Section 25 scores anyway.

**Pixels that a transform invents are ignored, not background.** Rotation
and out-of-bounds cropping expose area the camera never saw. Labeling it
background would teach every model that image corners are background, and
it would enter the loss as if it were an observation. It is marked
IGNORE_INDEX instead, which the loss and the metrics already exclude.
"""

import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter as _ColorJitter

from ._config import IGNORE_INDEX

# ImageNet statistics. Six of the ten architectures initialize from ImageNet
# weights, and those weights expect inputs standardized this way; using a
# different normalization would quietly degrade exactly the models the
# assignment asks to be compared against the from-scratch ones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _recompute_instances(sample: dict) -> dict:
    """Re-derive boxes and areas from the instance masks, dropping empties.

    Section 9 requires image ID, instance ID, mask, class and box to stay
    aligned after every transformation. That alignment is only preserved if
    dropping an instance drops it from all five arrays at once, which is why
    this filters with a single index array rather than five separate
    conditions.

    An instance can legitimately vanish: a crop can cut it away entirely, and
    a rotation can push it out of frame. Keeping a zero-pixel instance would
    hand Mask R-CNN an empty mask target and an undefined box.
    """
    if "masks" not in sample:
        return sample

    masks = sample["masks"]
    if len(masks) == 0:
        sample["boxes"] = np.zeros((0, 4), dtype=np.float32)
        sample["areas"] = np.zeros((0,), dtype=np.float32)
        return sample

    keep, boxes, areas = [], [], []
    for index, binary in enumerate(masks):
        if not binary.any():
            continue
        rows = np.where(binary.any(axis=1))[0]
        cols = np.where(binary.any(axis=0))[0]
        boxes.append([
            float(cols[0]), float(rows[0]),
            float(cols[-1] + 1), float(rows[-1] + 1),
        ])
        areas.append(float(binary.sum()))
        keep.append(index)

    keep = np.array(keep, dtype=np.int64)
    sample["masks"] = masks[keep] if len(keep) else masks[:0]
    sample["labels"] = sample["labels"][keep]
    sample["instance_ids"] = sample["instance_ids"][keep]
    sample["boxes"] = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    sample["areas"] = np.array(areas, dtype=np.float32)
    return sample


class Compose:
    """Apply transforms in order, and be able to say what that order was.

    Section 14 requires the exact transform order and parameters to be
    recorded. ``describe()`` returns them, and the training code writes that
    description into each run's configuration file, so the augmentation a set
    of results was produced under is recoverable from the results themselves
    rather than from whatever the source looked like at the time.
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def describe(self) -> list:
        return [t.describe() for t in self.transforms]


class Resize:
    """Resize the image bilinearly and every mask by nearest neighbor.

    Section 15 requires label-preserving interpolation for discrete masks.
    Bilinear resizing of a mask averages neighboring class IDs, so a boundary
    between person (1) and bicycle (3) would produce car (2) - a class that
    was never annotated anywhere near those pixels. Nearest neighbor cannot
    invent a label that was not already there.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, sample: dict) -> dict:
        size = (self.size, self.size)
        sample["image"] = cv2.resize(
            sample["image"], size, interpolation=cv2.INTER_LINEAR
        )
        sample["mask"] = cv2.resize(
            sample["mask"], size, interpolation=cv2.INTER_NEAREST
        )
        if "masks" in sample and len(sample["masks"]):
            sample["masks"] = np.stack([
                cv2.resize(
                    m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                for m in sample["masks"]
            ])
        elif "masks" in sample:
            sample["masks"] = np.zeros((0, self.size, self.size), dtype=bool)
        return _recompute_instances(sample)

    def describe(self) -> dict:
        return {
            "transform": "Resize",
            "size": self.size,
            "image_interpolation": "bilinear",
            "mask_interpolation": "nearest",
        }


class RandomHorizontalFlip:
    """Mirror the image and every target together.

    The cheapest transform to get wrong and the easiest to verify, which is
    why scripts/verify_transforms.py renders it first.
    """

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.probability:
            return sample

        sample["image"] = np.ascontiguousarray(sample["image"][:, ::-1])
        sample["mask"] = np.ascontiguousarray(sample["mask"][:, ::-1])
        if "masks" in sample and len(sample["masks"]):
            sample["masks"] = np.ascontiguousarray(sample["masks"][:, :, ::-1])
        return _recompute_instances(sample)

    def describe(self) -> dict:
        return {"transform": "RandomHorizontalFlip", "p": self.probability}


class RandomRotation:
    """Rotate about the image center, filling exposed area with ignore.

    The fill is the reason this transform is not a one-liner. Rotating a
    512x512 image by 10 degrees exposes roughly 6% of the canvas that holds
    no observation. The image gets zeros there, which after normalization is
    simply a dark corner; the semantic mask gets IGNORE_INDEX so the loss
    never scores those pixels, and the instance masks get False so no object
    is credited with area it does not occupy.
    """

    def __init__(self, degrees: float = 10.0, probability: float = 0.5):
        self.degrees = degrees
        self.probability = probability

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.probability:
            return sample

        angle = random.uniform(-self.degrees, self.degrees)
        height, width = sample["mask"].shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        size = (width, height)

        sample["image"] = cv2.warpAffine(
            sample["image"], matrix, size,
            flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
        )
        sample["mask"] = cv2.warpAffine(
            sample["mask"], matrix, size,
            flags=cv2.INTER_NEAREST, borderValue=IGNORE_INDEX,
        )
        if "masks" in sample and len(sample["masks"]):
            sample["masks"] = np.stack([
                cv2.warpAffine(
                    m.astype(np.uint8), matrix, size,
                    flags=cv2.INTER_NEAREST, borderValue=0,
                ).astype(bool)
                for m in sample["masks"]
            ])
        return _recompute_instances(sample)

    def describe(self) -> dict:
        return {
            "transform": "RandomRotation",
            "degrees": self.degrees,
            "p": self.probability,
            "image_fill": 0,
            "mask_fill": IGNORE_INDEX,
        }


class RandomScaleCrop:
    """Scale by a random factor, then crop or pad back to the target size.

    This is the assignment's "crop" (Section 14) in the form that suits
    segmentation. A plain random crop only ever shows the model a subset of
    the frame; scaling first also varies apparent object size, which matters
    here because Section 54 asks which architecture performed best on small
    objects and bicycle occupies 0.26% of pixels in this dataset. Varying
    scale during training is the augmentation that question is sensitive to.

    When the scaled image is smaller than the target the remainder is padded,
    and the padding is ignore rather than background for the same reason
    rotation's fill is.
    """

    def __init__(self, size: int, scale_range=(0.75, 1.25), probability=0.5):
        self.size = size
        self.scale_range = scale_range
        self.probability = probability

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.probability:
            return sample

        scale = random.uniform(*self.scale_range)
        scaled = max(1, int(round(self.size * scale)))
        resized = Resize(scaled)(sample)

        if scaled >= self.size:
            top = random.randint(0, scaled - self.size)
            left = random.randint(0, scaled - self.size)
            stop_y, stop_x = top + self.size, left + self.size
            resized["image"] = resized["image"][top:stop_y, left:stop_x]
            resized["mask"] = resized["mask"][top:stop_y, left:stop_x]
            if "masks" in resized and len(resized["masks"]):
                resized["masks"] = resized["masks"][:, top:stop_y, left:stop_x]
        else:
            pad = self.size - scaled
            top = random.randint(0, pad)
            left = random.randint(0, pad)
            bottom, right = pad - top, pad - left
            resized["image"] = cv2.copyMakeBorder(
                resized["image"], top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
            resized["mask"] = cv2.copyMakeBorder(
                resized["mask"], top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=IGNORE_INDEX,
            )
            if "masks" in resized and len(resized["masks"]):
                resized["masks"] = np.stack([
                    cv2.copyMakeBorder(
                        m.astype(np.uint8), top, bottom, left, right,
                        cv2.BORDER_CONSTANT, value=0,
                    ).astype(bool)
                    for m in resized["masks"]
                ])

        if "masks" in resized and not len(resized["masks"]):
            resized["masks"] = np.zeros((0, self.size, self.size), dtype=bool)
        return _recompute_instances(resized)

    def describe(self) -> dict:
        return {
            "transform": "RandomScaleCrop",
            "size": self.size,
            "scale_range": list(self.scale_range),
            "p": self.probability,
            "pad_image_fill": 0,
            "pad_mask_fill": IGNORE_INDEX,
        }


class ColorJitter:
    """Perturb the image's color. Masks are not touched and must not be.

    Section 15 is explicit that color jitter applies to the image only. The
    masks are unchanged here not by oversight but because a color change
    moves no pixel, so every label still describes the pixel it labeled.
    """

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05,
                 probability=0.5):
        self.parameters = {
            "brightness": brightness, "contrast": contrast,
            "saturation": saturation, "hue": hue,
        }
        self.probability = probability
        self._jitter = _ColorJitter(**self.parameters)

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.probability:
            return sample
        jittered = self._jitter(Image.fromarray(sample["image"]))
        sample["image"] = np.array(jittered)
        return sample

    def describe(self) -> dict:
        return {
            "transform": "ColorJitter",
            "p": self.probability,
            "applies_to": "image only",
            **self.parameters,
        }


class ToTensor:
    """Convert to tensors, normalize the image, leave label values intact.

    The image becomes float and is standardized. The masks do not: a class
    ID is a label, not a measurement, and scaling it to [0, 1] would make
    255 - the ignore sentinel - indistinguishable from a class ID under
    rounding. The semantic mask becomes int64 because that is what
    cross-entropy expects as its target.
    """

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict) -> dict:
        image = torch.from_numpy(sample["image"]).permute(2, 0, 1).float().div(255)
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        sample["image"] = (image - mean) / std
        sample["mask"] = torch.from_numpy(sample["mask"].astype(np.int64))

        if "masks" in sample:
            sample["masks"] = torch.from_numpy(
                np.ascontiguousarray(sample["masks"])
            ).to(torch.uint8)
            sample["labels"] = torch.from_numpy(sample["labels"])
            sample["boxes"] = torch.from_numpy(sample["boxes"])
            sample["areas"] = torch.from_numpy(sample["areas"])
            sample["instance_ids"] = torch.from_numpy(sample["instance_ids"])
        return sample

    def describe(self) -> dict:
        return {
            "transform": "ToTensor+Normalize",
            "mean": list(self.mean),
            "std": list(self.std),
            "note": "image only; mask values are labels and are not scaled",
        }


def build_train_transforms(size: int) -> Compose:
    """The Section 14 training pipeline, in the order the assignment lists it.

    Resize, flip, crop, rotation, color jitter, tensor conversion,
    normalization. Geometry runs before color because color jitter's cost
    scales with pixel count and there is no reason to jitter pixels that the
    crop is about to discard.
    """
    return Compose([
        Resize(size),
        RandomHorizontalFlip(0.5),
        RandomScaleCrop(size, (0.75, 1.25), 0.5),
        RandomRotation(10.0, 0.5),
        ColorJitter(0.3, 0.3, 0.3, 0.05, 0.5),
        ToTensor(),
    ])


def build_eval_transforms(size: int) -> Compose:
    """The Section 14 validation and test pipeline: deterministic only.

    No randomness at all, so validation scores differ between epochs because
    the weights changed and for no other reason, and so the held-out test
    images are identical for all ten architectures.
    """
    return Compose([Resize(size), ToTensor()])
