"""Section 25: COCO-style mask AP for Mask R-CNN and YOLO segmentation.

Section 25 requires the evaluator, class averaging, IoU thresholds, score
filtering and maximum detections all to be stated. They are, below and in
the ``conventions`` block of every result this module produces:

* **Evaluator** - pycocotools COCOeval, the reference implementation, with
  ``iouType="segm"``. Box AP is computed by the same evaluator with
  ``iouType="bbox"`` and reported only as a secondary figure, because
  Section 25 is explicit that mask quality is the primary objective.
* **IoU thresholds** - AP is averaged over 0.50 to 0.95 in steps of 0.05.
  AP50 and AP75 are single thresholds.
* **Class averaging** - COCOeval's own: AP is computed per category and
  averaged over categories, so the five classes count equally regardless of
  how many instances each contributes. Person supplies most of the objects
  in this dataset, and a detection-weighted average would be close to
  person's score alone.
* **Score filtering** - detections below INSTANCE_SCORE_THRESHOLD (0.05)
  are dropped before scoring. This is deliberately low. AP integrates
  precision over the whole recall curve, so discarding low-confidence
  detections can only remove recall the model had earned; a high threshold
  produces a tidier prediction set and a worse AP.
* **Maximum detections** - INSTANCE_MAX_DETECTIONS (100) per image, COCO's
  standard.

**Resolution.** Models predict at whatever input size the run configures,
and their masks are resized back to each image's native resolution before
scoring against the unmodified COCO ground truth. The alternative -
rescaling the ground truth down to the model's input size - would resample
the annotations and make these numbers incomparable with every published
COCO result. The consequence is that the instance track is scored at native
resolution while the semantic track is scored at the input size, which is
recorded in each result file rather than left to be inferred. Section 26
forbids comparing the two tracks' numbers directly in any case.
"""

import contextlib
import io

import numpy as np
import torch
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ._config import (
    ANNOTATIONS_DIR,
    CLASS_NAMES,
    INSTANCE_MAX_DETECTIONS,
    INSTANCE_SCORE_THRESHOLD,
    NUM_CLASSES,
)

# COCOeval's summary vector, in the order it returns them. Named here
# because indexing stats[0] through stats[11] at the call site is how a
# report ends up describing AP75 as AP50.
STAT_NAMES = [
    "ap",          # 0: AP @ IoU 0.50:0.95, all areas, maxDets 100
    "ap50",        # 1: AP @ IoU 0.50
    "ap75",        # 2: AP @ IoU 0.75
    "ap_small",    # 3: AP, area < 32^2
    "ap_medium",   # 4: AP, 32^2 <= area < 96^2
    "ap_large",    # 5: AP, area >= 96^2
    "ar_1",        # 6: AR given 1 detection per image
    "ar_10",       # 7: AR given 10 detections per image
    "ar_100",      # 8: AR given 100 detections per image
    "ar_small",    # 9
    "ar_medium",   # 10
    "ar_large",    # 11
]

CONVENTIONS = {
    "evaluator": "pycocotools COCOeval",
    "primary_metric": "mask AP averaged over IoU 0.50:0.05:0.95",
    "class_averaging": "per category, then averaged over categories",
    "score_threshold": INSTANCE_SCORE_THRESHOLD,
    "max_detections": INSTANCE_MAX_DETECTIONS,
    "mask_binarization": "sigmoid output thresholded at 0.5",
    "resolution": "predictions resized to native resolution; ground truth unmodified",
    "crowd_regions": "retained in ground truth; COCOeval uses them to suppress "
                     "false positives on unannotated groups",
    "box_ap": "secondary only (Section 25); mask quality is the objective",
    "reported_as": "fractions in [0, 1], not percentages",
}


def encode_mask(binary: np.ndarray) -> dict:
    """RLE-encode a binary mask in the layout COCOeval expects.

    Fortran ordering is not optional. pycocotools encodes column-major and
    rejects a C-contiguous array outright with "ndarray is not Fortran
    contiguous", so the conversion has to happen somewhere; doing it here
    means every caller gets it rather than each one rediscovering it.

    The failure is loud, which is worth knowing: a missing asfortranarray
    stops the evaluation instead of quietly scoring transposed masks. The
    uint8 cast is equally required - pycocotools will not encode a bool
    array.
    """
    encoded = coco_mask.encode(np.asfortranarray(binary.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def resize_mask(binary: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a binary mask to native resolution by nearest neighbor.

    Nearest rather than bilinear-then-threshold. Both are defensible, but
    the mask has already been thresholded once at 0.5 by the caller, and
    interpolating a binary array then thresholding it again applies two
    different roundings to one decision.
    """
    import cv2

    return cv2.resize(
        binary.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def detections_to_coco(detections: list, image_id: int,
                       native_size: tuple = None,
                       score_threshold: float = INSTANCE_SCORE_THRESHOLD,
                       max_detections: int = INSTANCE_MAX_DETECTIONS) -> list:
    """Convert one image's detections into COCO result records.

    ``detections`` is torchvision's output: a dict with ``boxes``,
    ``labels``, ``scores`` and ``masks``, where masks are float
    probabilities of shape (N, 1, H, W).
    """
    results = []
    scores = detections["scores"].detach().cpu().numpy()
    labels = detections["labels"].detach().cpu().numpy()
    boxes = detections["boxes"].detach().cpu().numpy()
    masks = detections["masks"].detach().cpu().numpy()

    keep = np.where(scores >= score_threshold)[0]
    # Highest scoring first, then truncate. Truncating before sorting would
    # keep an arbitrary 100 rather than the best 100.
    keep = keep[np.argsort(-scores[keep])][:max_detections]

    for index in keep:
        binary = masks[index]
        if binary.ndim == 3:
            binary = binary[0]
        binary = binary >= 0.5

        box = boxes[index]
        if native_size is not None:
            height, width = native_size
            predicted_height, predicted_width = binary.shape
            binary = resize_mask(binary, height, width)
            scale_x = width / predicted_width
            scale_y = height / predicted_height
            box = box * np.array([scale_x, scale_y, scale_x, scale_y])

        if not binary.any():
            # A mask that vanished under resizing has no segmentation to
            # score. Emitting it would add a false positive that the model
            # did not really make.
            continue

        results.append({
            "image_id": int(image_id),
            "category_id": int(labels[index]),
            "segmentation": encode_mask(binary),
            "score": float(scores[index]),
            # COCO boxes are [x, y, width, height], not corners.
            "bbox": [
                float(box[0]), float(box[1]),
                float(box[2] - box[0]), float(box[3] - box[1]),
            ],
        })
    return results


def _run_cocoeval(coco_gt: COCO, results: list, iou_type: str,
                  image_ids: list) -> dict:
    """Run COCOeval and return its twelve statistics, named.

    COCOeval prints a twelve-line table to stdout on every call. Ten models
    times two IoU types would bury the benchmark's own progress output, so
    it is captured and discarded; the numbers are returned instead.
    """
    if not results:
        # No detections at all is a real outcome, not an error - an
        # undertrained model can produce it. Zeros are the correct scores,
        # and a crash here would lose the rest of the benchmark.
        return {name: 0.0 for name in STAT_NAMES}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        evaluation = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        evaluation.params.imgIds = list(image_ids)
        evaluation.params.maxDets = [1, 10, INSTANCE_MAX_DETECTIONS]
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()

    return {
        name: float(value)
        for name, value in zip(STAT_NAMES, evaluation.stats)
    }


def _per_class_ap(coco_gt: COCO, results: list, image_ids: list) -> dict:
    """Mask AP for each class separately.

    COCOeval's summary averages over categories, which hides that a model
    may be excellent on person and useless on bicycle. Section 23 asks for
    the per-class breakdown on the semantic side; the same question is worth
    answering here.
    """
    if not results:
        return {name: 0.0 for name in CLASS_NAMES[1:]}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        evaluation = COCOeval(coco_gt, coco_dt, iouType="segm")
        evaluation.params.imgIds = list(image_ids)
        evaluation.evaluate()
        evaluation.accumulate()

    # precision has shape (thresholds, recall, category, area, maxDets).
    precision = evaluation.eval["precision"]
    per_class = {}
    for index, name in enumerate(CLASS_NAMES[1:]):
        values = precision[:, :, index, 0, 2]
        values = values[values > -1]
        per_class[name] = float(values.mean()) if values.size else 0.0
    return per_class


def _release_cache(device) -> None:
    """Hand cached blocks back, on whichever backend is in use."""
    device_type = getattr(device, "type", str(device))
    if device_type == "mps":
        torch.mps.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()


def evaluate_instance(model, loader, device, split: str = "test",
                      score_threshold: float = INSTANCE_SCORE_THRESHOLD,
                      include_box_ap: bool = True) -> dict:
    """Run an instance model over a loader and return Section 25's metrics.

    Used both for the per-epoch validation score that Section 21 selects
    checkpoints on and for the final test evaluation, so the number a
    checkpoint is chosen by is produced by the same code as the number
    reported.
    """
    annotation_path = ANNOTATIONS_DIR / f"instances_{split}.json"
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotation_path))

    model.eval()
    mask_results, image_ids = [], []

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader):
            images_on_device = [image.to(device) for image in images]
            outputs = model(images_on_device)

            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"])
                image_ids.append(image_id)
                record = coco_gt.imgs[image_id]
                mask_results += detections_to_coco(
                    output, image_id,
                    native_size=(record["height"], record["width"]),
                    score_threshold=score_threshold,
                )

            # Release the backend's cached blocks periodically.
            #
            # Metal's caching allocator does not return freed blocks to the
            # OS on its own, and its own accounting does not report what it
            # is holding: during a 25-epoch Mask R-CNN run this reported a
            # steady 863 MiB while the system accumulated 20.6 GB of wired
            # memory and began swapping. Epoch time went from 740 seconds to
            # 4,374 and the run had to be stopped at epoch 14.
            #
            # Detection models make this worse than semantic ones do,
            # because each image produces a variable number of differently
            # shaped mask tensors, so the allocator's pool fragments into
            # many size classes that are never reused.
            del outputs, images_on_device
            if batch_index % 25 == 24:
                _release_cache(device)

    _release_cache(device)

    return summarize_instance_results(
        coco_gt, mask_results, image_ids, include_box_ap
    )


def summarize_instance_results(coco_gt: COCO, mask_results: list,
                               image_ids: list,
                               include_box_ap: bool = True) -> dict:
    """Turn COCO-format detections into the Section 28 result row.

    Split out from ``evaluate_instance`` because YOLO produces its
    detections through ultralytics rather than through a torchvision
    forward pass. Both converge here, so Mask R-CNN and YOLO are scored by
    identical code - which is what makes the Section 28 comparison a
    comparison of the models rather than of two evaluation scripts.
    """
    segm = _run_cocoeval(coco_gt, mask_results, "segm", image_ids)

    summary = {
        "mask_ap": segm["ap"],
        "mask_ap50": segm["ap50"],
        "mask_ap75": segm["ap75"],
        "mask_ap_small": segm["ap_small"],
        "mask_ap_medium": segm["ap_medium"],
        "mask_ap_large": segm["ap_large"],
        "mask_ar_1": segm["ar_1"],
        "mask_ar_10": segm["ar_10"],
        "mask_ar_100": segm["ar_100"],
        "mask_ar_small": segm["ar_small"],
        "mask_ar_medium": segm["ar_medium"],
        "mask_ar_large": segm["ar_large"],
        "per_class_mask_ap": _per_class_ap(coco_gt, mask_results, image_ids),
        "detections": len(mask_results),
        "images": len(set(image_ids)),
        "conventions": CONVENTIONS,
    }

    if include_box_ap and mask_results:
        box_results = [
            {k: v for k, v in record.items() if k != "segmentation"}
            for record in mask_results
        ]
        bbox = _run_cocoeval(coco_gt, box_results, "bbox", image_ids)
        summary.update({
            "box_ap": bbox["ap"],
            "box_ap50": bbox["ap50"],
            "box_ap75": bbox["ap75"],
        })
    elif include_box_ap:
        summary.update({"box_ap": 0.0, "box_ap50": 0.0, "box_ap75": 0.0})

    return summary


def instance_to_semantic(detections: dict, height: int, width: int,
                         score_threshold: float = 0.5,
                         num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Collapse instance predictions into a semantic map, for Section 26.

    Section 26 permits an optional common semantic evaluation of instance
    outputs, and requires four things to be documented: score threshold,
    overlap resolution, class mapping, and background handling. It also
    requires the conversion settings to be selected on validation data only.

    * **Score threshold** - the caller's, defaulting to 0.5 and chosen on
      validation. This is a different threshold from the 0.05 used for AP,
      and for a different reason: AP wants every detection the model can
      offer, whereas a semantic map wants one confident label per pixel and
      is degraded by low-confidence masks painted over it.
    * **Overlap resolution** - masks are painted in ascending score order,
      so the highest-scoring instance is painted last and wins any contested
      pixel. This is the opposite convention from the ground-truth masks,
      which paint smallest-last, and that difference is itself a source of
      disagreement that a reader should know about.
    * **Class mapping** - identity. Both tracks were trained on the same six
      contiguous IDs, so no remapping is needed or performed.
    * **Background** - any pixel no accepted instance covers. An instance
      model has no background class and never predicts one, so background
      here means "nothing was detected", which is not the same statement a
      semantic model makes when it predicts class 0.

    That last difference is the reason Section 26 insists these numbers stay
    out of the main semantic table. This conversion is reported as a clearly
    labeled supplementary comparison, never as a Mask R-CNN mIoU set beside
    U-Net's.
    """
    semantic = np.zeros((height, width), dtype=np.uint8)

    scores = detections["scores"].detach().cpu().numpy()
    labels = detections["labels"].detach().cpu().numpy()
    masks = detections["masks"].detach().cpu().numpy()

    keep = np.where(scores >= score_threshold)[0]
    keep = keep[np.argsort(scores[keep])]  # ascending: best painted last

    for index in keep:
        binary = masks[index]
        if binary.ndim == 3:
            binary = binary[0]
        binary = binary >= 0.5
        if binary.shape != (height, width):
            binary = resize_mask(binary, height, width)
        label = int(labels[index])
        if 0 < label < num_classes:
            semantic[binary] = label

    return semantic
