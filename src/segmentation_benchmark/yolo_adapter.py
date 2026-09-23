"""Model 9: YOLO segmentation, driven through ultralytics.

Section 18 says not to force incompatible interfaces into one loop, and YOLO
is the clearest case in this benchmark. Ultralytics owns its training loop,
its optimizer schedule, its augmentation, its dataset format and its
checkpointing. Wrapping it in an ``nn.Module`` to satisfy the factory's
signature would mean either reimplementing that loop or pretending the
wrapper controls something it does not.

So YOLO trains through ultralytics, on the same images as every other model,
and its predictions are scored by the same ``summarize_instance_results``
that scores Mask R-CNN. The training path differs; the evaluation path is
identical. That is what makes the Section 28 comparison a comparison of two
models rather than of two evaluation scripts, and the differences that
remain are recorded rather than smoothed over:

* **Augmentation.** Ultralytics applies its own - mosaic, HSV jitter,
  translate, scale, flip - which is not the Section 14 pipeline the seven
  semantic models and Mask R-CNN share. It can be reduced but not removed
  without leaving YOLO training under a recipe its hyperparameters were
  never tuned for. Recorded as a documented deviation.
* **Optimizer.** Ultralytics selects and schedules its own. Section 17 is
  explicit that instance models keep their architecture-specific training
  configuration, so this is expected rather than a violation.
* **Label format.** YOLO wants normalized polygons, not masks. The
  conversion is below and is lossy in one direction: a mask with holes
  becomes an outer contour. That is a property of the format, and it is
  measured rather than assumed - ``export_split`` reports the mean IoU
  between each original mask and its polygon round-trip.
"""

import json
import shutil

import numpy as np

from ._config import (
    ANNOTATIONS_DIR,
    CLASS_NAMES,
    IMAGES_DIR,
    INPUT_SIZE,
    REPO_ROOT,
    SEED,
)

WORKSPACE = REPO_ROOT / "yolo_workspace"

# Ultralytics indexes classes from zero; our contiguous IDs reserve zero for
# background, which an instance model does not predict. YOLO class i is our
# class i + 1, and the conversion happens in exactly two places: writing
# labels and reading predictions.
YOLO_NAMES = CLASS_NAMES[1:]


def mask_to_polygons(binary: np.ndarray, min_points: int = 3) -> list:
    """Convert a binary mask to normalized polygons in YOLO's layout.

    Returns a list of flat ``[x1, y1, x2, y2, ...]`` lists with coordinates
    in [0, 1].

    Only external contours are kept. A mask with a hole - a person seen
    through a bicycle frame - becomes its outer boundary, which overstates
    the object slightly. The alternative, emitting the hole as a separate
    polygon, is worse: YOLO would read it as a second instance of the same
    class. This loss is inherent to the polygon format and is quantified by
    ``export_split`` rather than left as a caveat.
    """
    import cv2

    height, width = binary.shape
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        if len(contour) < min_points:
            continue
        points = contour.reshape(-1, 2).astype(np.float64)
        points[:, 0] /= width
        points[:, 1] /= height
        np.clip(points, 0.0, 1.0, out=points)
        polygons.append(points.reshape(-1).tolist())
    return polygons


def polygons_to_mask(polygons: list, height: int, width: int) -> np.ndarray:
    """Rasterize normalized polygons back to a binary mask, for round-trip checks."""
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.array(polygon, dtype=np.float64).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask.astype(bool)


def export_split(split: str, workspace=WORKSPACE, link_images: bool = True) -> dict:
    """Write one split in YOLO segmentation format.

    Images are symlinked rather than copied. Seven thousand JPEGs is 1.1 GB,
    and a second copy buys nothing except the chance for the two to diverge.

    Returns the conversion fidelity: the mean IoU between each instance's
    original mask and its polygon round-trip, so the report can state what
    the format conversion cost instead of assuming it cost nothing.
    """
    from . import mask_utils
    from ._config import COCO_CATEGORY_IDS

    coco = json.loads((ANNOTATIONS_DIR / f"instances_{split}.json").read_text())
    to_coco = {v: k for k, v in COCO_CATEGORY_IDS.items()}

    annotations_by_image = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    image_dir = workspace / "images" / split
    label_dir = workspace / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    ious, instances, dropped = [], 0, 0

    for record in sorted(coco["images"], key=lambda r: r["id"]):
        source = IMAGES_DIR / split / record["file_name"]
        destination = image_dir / record["file_name"]
        if not destination.exists():
            if link_images:
                destination.symlink_to(source.resolve())
            else:
                shutil.copy2(source, destination)

        height, width = record["height"], record["width"]
        raw = [
            {**a, "category_id": to_coco[a["category_id"]]}
            for a in annotations_by_image.get(record["id"], [])
        ]
        targets = mask_utils.instance_targets(raw, height, width)

        lines = []
        for binary, label in zip(targets["masks"], targets["labels"]):
            polygons = mask_to_polygons(binary)
            if not polygons:
                dropped += 1
                continue
            # Largest contour only: YOLO expects one polygon per instance,
            # and a fragmented mask would otherwise become several objects.
            polygon = max(polygons, key=len)

            restored = polygons_to_mask([polygon], height, width)
            union = (binary | restored).sum()
            if union:
                ious.append(float((binary & restored).sum() / union))

            coordinates = " ".join(f"{v:.6f}" for v in polygon)
            lines.append(f"{int(label) - 1} {coordinates}")
            instances += 1

        (label_dir / f"{record['file_name'].rsplit('.', 1)[0]}.txt").write_text(
            "\n".join(lines)
        )

    return {
        "split": split,
        "images": len(coco["images"]),
        "instances": instances,
        "dropped_instances": dropped,
        "polygon_roundtrip_mean_iou": float(np.mean(ious)) if ious else None,
        "polygon_roundtrip_min_iou": float(np.min(ious)) if ious else None,
        "conversion_note": (
            "masks converted to external contours; holes are filled and "
            "fragmented instances keep their largest component"
        ),
    }


def write_dataset_yaml(workspace=WORKSPACE) -> "Path":
    """The dataset descriptor ultralytics reads.

    ``val`` points at our validation split, not our test split. Ultralytics
    evaluates against ``val`` after every epoch and selects its own best
    checkpoint from it, so pointing this at test would be exactly the
    checkpoint-selection-on-test that Section 21 forbids - committed by the
    library rather than by our code, which makes it easier to miss and no
    less wrong.
    """
    path = workspace / "dataset.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(YOLO_NAMES))
    path.write_text(
        f"path: {workspace.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names}\n"
    )
    return path


def prepare(workspace=WORKSPACE, splits=("train", "val", "test")) -> dict:
    """Export every split and write the dataset descriptor."""
    workspace.mkdir(parents=True, exist_ok=True)
    report = {split: export_split(split, workspace) for split in splits}
    report["dataset_yaml"] = str(write_dataset_yaml(workspace))
    return report


def train_yolo(epochs: int = 25, input_size: int = INPUT_SIZE,
               batch_size: int = 8, model_name: str = "yolo11n-seg.pt",
               workspace=WORKSPACE, device: str = "mps",
               verbose: bool = True) -> dict:
    """Train YOLO segmentation through ultralytics on our split.

    The epoch budget, input size and batch size are forced to match the
    shared Section 13 protocol. Everything ultralytics decides for itself -
    optimizer, schedule, augmentation - is left alone and recorded, because
    overriding it piecemeal produces a configuration that is neither the
    shared recipe nor the one YOLO was tuned for.
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    results = model.train(
        data=str(workspace / "dataset.yaml"),
        epochs=epochs,
        imgsz=input_size,
        batch=batch_size,
        device=device,
        seed=SEED,
        project=str(workspace / "runs"),
        name="segment",
        exist_ok=True,
        verbose=verbose,
        val=True,
        plots=False,
    )

    directory = workspace / "runs" / "segment"
    return {
        "model": "yolo_seg",
        "variant": model_name,
        "epochs": epochs,
        "input_size": input_size,
        "batch_size": batch_size,
        "seed": SEED,
        "weights": str(directory / "weights" / "best.pt"),
        "results_csv": str(directory / "results.csv"),
        "checkpoint_selection": (
            "ultralytics selects best.pt on its own validation metric, "
            "computed over our validation split"
        ),
        "deviations": [
            "ultralytics augmentation (mosaic, HSV, translate, scale, flip) "
            "rather than the Section 14 pipeline",
            "ultralytics optimizer and learning-rate schedule rather than "
            "AdamW at 1e-4",
        ],
        "save_dir": str(getattr(results, "save_dir", directory)),
    }


def predict_and_score(weights, split: str = "test", input_size: int = INPUT_SIZE,
                      device: str = "mps", workspace=WORKSPACE,
                      score_threshold: float = None) -> dict:
    """Run trained YOLO over a split and score it with the shared evaluator.

    Predictions are resized to each image's native resolution and handed to
    the same ``summarize_instance_results`` Mask R-CNN goes through, so
    Section 28's comparison is between the two models under one evaluator.
    """
    import contextlib
    import io

    import torch
    from pycocotools.coco import COCO
    from ultralytics import YOLO

    from ._config import INSTANCE_SCORE_THRESHOLD
    from .instance_metrics import encode_mask, resize_mask, summarize_instance_results

    threshold = (
        INSTANCE_SCORE_THRESHOLD if score_threshold is None else score_threshold
    )

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(ANNOTATIONS_DIR / f"instances_{split}.json"))

    model = YOLO(str(weights))
    results_records, image_ids = [], []

    for record in sorted(coco_gt.imgs.values(), key=lambda r: r["id"]):
        image_id = record["id"]
        image_ids.append(image_id)
        path = IMAGES_DIR / split / record["file_name"]

        prediction = model.predict(
            str(path), imgsz=input_size, device=device,
            conf=threshold, verbose=False, retina_masks=True,
        )[0]

        if prediction.masks is None:
            continue

        masks = prediction.masks.data.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()
        classes = prediction.boxes.cls.cpu().numpy().astype(int)
        boxes = prediction.boxes.xyxy.cpu().numpy()

        for binary, score, class_index, box in zip(masks, scores, classes, boxes):
            binary = binary >= 0.5
            if binary.shape != (record["height"], record["width"]):
                binary = resize_mask(binary, record["height"], record["width"])
            if not binary.any():
                continue
            results_records.append({
                "image_id": int(image_id),
                # +1: back from YOLO's zero-based indexing to our contiguous IDs.
                "category_id": int(class_index) + 1,
                "segmentation": encode_mask(binary),
                "score": float(score),
                "bbox": [
                    float(box[0]), float(box[1]),
                    float(box[2] - box[0]), float(box[3] - box[1]),
                ],
            })

    summary = summarize_instance_results(coco_gt, results_records, image_ids)
    summary["model"] = "yolo_seg"
    summary["conventions"] = dict(summary["conventions"])
    summary["conventions"]["prediction_source"] = (
        "ultralytics predict with retina_masks=True, then resized to native "
        "resolution and scored by the shared COCOeval wrapper"
    )
    return summary
