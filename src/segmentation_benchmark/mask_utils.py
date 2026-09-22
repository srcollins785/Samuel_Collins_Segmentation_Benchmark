"""Turn COCO annotations into the two target formats the benchmark needs.

Section 8 wants one class ID per pixel for the seven semantic models.
Section 9 wants each object's class, box and binary mask kept separate for
the two instance models. Both come from the same annotation list, and this
module is the only place that converts between them, so the crowd and
overlap policies in ``_config`` are applied identically to both tracks.

The conversion runs at the image's native resolution. Resizing happens later,
in the transform pipeline, where the image and its masks are resized together
(Section 15) - doing it here would mean every model inherited one interpolation
decision made before anyone chose an input size.
"""

import numpy as np
from pycocotools import mask as coco_mask

from ._config import (
    COCO_CATEGORY_IDS,
    IGNORE_INDEX,
    NUM_CLASSES,
)


def decode_annotation(annotation: dict, height: int, width: int) -> np.ndarray:
    """Decode one COCO annotation into a binary mask at native resolution.

    COCO stores three segmentation formats and this has to handle all of
    them: a list of polygons for ordinary objects, an uncompressed RLE dict
    for some crowd regions, and a compressed RLE dict for others.
    ``frPyObjects`` normalizes the first two into the third.
    """
    segmentation = annotation["segmentation"]

    if isinstance(segmentation, list):
        # Polygons. An object split by occlusion has several, and merge
        # unions them into the single mask that object occupies.
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation["counts"], list):
        # Uncompressed RLE.
        rle = coco_mask.frPyObjects(segmentation, height, width)
    else:
        # Already compressed RLE.
        rle = segmentation

    return coco_mask.decode(rle).astype(bool)


def semantic_mask(annotations: list, height: int, width: int) -> np.ndarray:
    """Build the Section 8 semantic mask: one class ID per pixel.

    Returns a ``uint8`` array of shape ``(height, width)`` whose values are
    contiguous class IDs 0-5, or ``IGNORE_INDEX`` for pixels excluded from
    both training and evaluation.

    Three rules, all from ``_config`` so the two tracks cannot drift apart:

    1. Crowd regions are painted first as ``IGNORE_INDEX``. They mark real
       objects of a real class that COCO declined to annotate individually,
       so they must not become background, and they are not usable as
       instances either.
    2. Non-crowd instances are painted afterwards in descending area order,
       so a smaller object overwrites a larger one where they overlap. A cat
       on a sofa keeps its pixels.
    3. Anything untouched stays 0, background. That covers the 75 COCO
       categories outside our five and everything COCO never annotated.

    Because non-crowd instances paint over crowd regions, an individually
    annotated object standing inside a crowd blob keeps its own label rather
    than being ignored along with the crowd.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    selected = [a for a in annotations if a["category_id"] in COCO_CATEGORY_IDS]

    crowds = [a for a in selected if a.get("iscrowd", 0) == 1]
    objects = [a for a in selected if a.get("iscrowd", 0) == 0]

    for annotation in crowds:
        mask[decode_annotation(annotation, height, width)] = IGNORE_INDEX

    # Descending area: the largest is painted first and can be overwritten by
    # everything smaller. COCO's own 'area' field is the mask area, which is
    # what we want here rather than the box area.
    objects.sort(key=lambda a: a.get("area", 0.0), reverse=True)

    for annotation in objects:
        class_id = COCO_CATEGORY_IDS[annotation["category_id"]]
        mask[decode_annotation(annotation, height, width)] = class_id

    return mask


def instance_targets(annotations: list, height: int, width: int) -> dict:
    """Build the Section 9 instance targets: one entry per object.

    Returns aligned arrays - ``masks``, ``labels``, ``boxes``, ``areas``,
    ``instance_ids`` - where index *i* describes the same object in all five.
    Section 9 requires that alignment to survive every transformation, so the
    transform pipeline reorders all five together or none of them.

    Crowd annotations are dropped rather than ignored here. A crowd blob is
    not one object, so scoring it as an instance would be wrong in either
    direction: as a prediction target it teaches the model to emit one mask
    for a group, and as ground truth it is unmatchable. The COCO evaluator
    handles crowd regions separately through its own ``iscrowd`` flag, which
    is why they are preserved in the annotation file and excluded here.

    Boxes are derived from the decoded mask rather than copied from COCO's
    ``bbox`` field. The two agree for ordinary annotations, but a mask
    clipped at the image edge can leave COCO's box extending past it, and a
    box that does not bound its own mask breaks RoI Align's alignment
    assumption (Section 8, Model 8).
    """
    masks, labels, boxes, areas, instance_ids = [], [], [], [], []

    selected = [
        a for a in annotations
        if a["category_id"] in COCO_CATEGORY_IDS and a.get("iscrowd", 0) == 0
    ]
    selected.sort(key=lambda a: a["id"])

    for annotation in selected:
        binary = decode_annotation(annotation, height, width)
        if not binary.any():
            # A polygon degenerate at this resolution. Keeping it would give
            # Mask R-CNN an empty target and an undefined box.
            continue

        rows = np.where(binary.any(axis=1))[0]
        cols = np.where(binary.any(axis=0))[0]
        y1, y2 = float(rows[0]), float(rows[-1] + 1)
        x1, x2 = float(cols[0]), float(cols[-1] + 1)

        masks.append(binary)
        labels.append(COCO_CATEGORY_IDS[annotation["category_id"]])
        boxes.append([x1, y1, x2, y2])
        areas.append(float(binary.sum()))
        instance_ids.append(int(annotation["id"]))

    return {
        "masks": (
            np.stack(masks) if masks
            else np.zeros((0, height, width), dtype=bool)
        ),
        "labels": np.array(labels, dtype=np.int64),
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "areas": np.array(areas, dtype=np.float32),
        "instance_ids": np.array(instance_ids, dtype=np.int64),
    }


def class_pixel_counts(mask: np.ndarray) -> np.ndarray:
    """Count valid pixels per class, for dataset statistics and class balance.

    Ignored pixels are counted in neither, so the returned counts sum to the
    number of pixels the loss and the metrics will actually see.
    """
    valid = mask != IGNORE_INDEX
    return np.bincount(mask[valid].ravel(), minlength=NUM_CLASSES)


def has_foreground(mask: np.ndarray, minimum_pixels: int = 1) -> bool:
    """Whether a mask carries enough foreground to be worth including.

    Used when selecting the subset: an image annotated only with crowd
    regions, or whose sole annotation is a handful of pixels, contributes
    nothing to any of the five classes we are measuring.
    """
    counts = class_pixel_counts(mask)
    return int(counts[1:].sum()) >= minimum_pixels
