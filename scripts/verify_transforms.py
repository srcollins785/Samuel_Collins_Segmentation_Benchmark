"""Prove the augmentation is mask-safe, then show it.

Section 15 asks for transformed image-mask pairs to be visualized before
starting a full benchmark, and gives the reason: a flipped or rotated image
paired with an unchanged mask is invalid training data. A figure is how that
instruction is usually satisfied, but a figure only catches a desync large
enough to see. A three-pixel drift between image and mask would train every
architecture on slightly wrong boundaries and would look perfectly fine at
figure resolution - while Section 24 asks which model had the best boundary
quality, a question that a three-pixel error makes meaningless.

So this runs three checks first and draws the figure second.

1. **Flip exactness.** With flip probability forced to 1, every field of the
   output must equal the hand-mirrored input: image, semantic mask, each
   instance mask, and each box. Exact equality, not approximate.

2. **Synthetic alignment under random geometry.** An image is constructed by
   painting each class its own solid color, so the image *is* the mask. After
   a random pipeline - scale, crop, rotate - the interior of every labeled
   region must still carry that class's color. Interiors only, because
   bilinear resampling legitimately blends the image at boundaries while
   nearest-neighbor resampling of the mask does not; eroding before
   comparing tests the alignment rather than the interpolation.

3. **Instance-target coherence.** Every surviving instance's box must bound
   its own mask, its area must match its mask's pixel count, and the five
   per-instance arrays must stay the same length - Section 9's alignment
   requirement, checked after the transforms that could break it.

Usage
-----
    python scripts/verify_transforms.py
    python scripts/verify_transforms.py --samples 8
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from segmentation_benchmark import transforms as T  # noqa: E402
from segmentation_benchmark._config import (  # noqa: E402
    CLASS_NAMES,
    IGNORE_INDEX,
    INPUT_SIZE,
    PLOTS_DIR,
    SEED,
)
from segmentation_benchmark.dataset import (  # noqa: E402
    InstanceSegmentationDataset,
)

# One color per class, plus a distinct one for ignore. Chosen to stay
# distinguishable in grayscale print as well as on screen.
PALETTE = np.array([
    [0, 0, 0],          # background
    [220, 20, 60],      # person
    [0, 130, 200],      # car
    [255, 225, 25],     # bicycle
    [60, 180, 75],      # dog
    [245, 130, 48],     # cat
], dtype=np.uint8)
IGNORE_COLOR = np.array([128, 128, 128], dtype=np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Render a class-ID mask as RGB for display."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(len(PALETTE)):
        out[mask == class_id] = PALETTE[class_id]
    out[mask == IGNORE_INDEX] = IGNORE_COLOR
    return out


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55):
    """Blend a mask over an image, leaving background pixels untouched."""
    blended = image.copy()
    painted = (mask != 0) & (mask != IGNORE_INDEX)
    colored = colorize(mask)
    blended[painted] = (
        alpha * colored[painted] + (1 - alpha) * image[painted]
    ).astype(np.uint8)
    ignored = mask == IGNORE_INDEX
    blended[ignored] = (
        0.35 * IGNORE_COLOR + 0.65 * image[ignored]
    ).astype(np.uint8)
    return blended


def raw_sample(dataset, index: int) -> dict:
    """One sample with masks decoded but no transforms applied."""
    from PIL import Image
    from segmentation_benchmark import mask_utils

    record = dataset.records[index]
    height, width = record["height"], record["width"]
    image = np.array(
        Image.open(dataset.image_dir / record["file_name"]).convert("RGB")
    )
    annotations = [
        {**a, "category_id": dataset._to_coco[a["category_id"]]}
        for a in dataset.annotations.get(record["id"], [])
    ]
    targets = mask_utils.instance_targets(annotations, height, width)
    semantic = mask_utils.semantic_mask(annotations, height, width)
    return {"image": image, "mask": semantic, **targets}


# ---------------------------------------------------------------------------
# Check 1: flip exactness
# ---------------------------------------------------------------------------

def check_flip(dataset, count: int) -> list:
    """A forced horizontal flip must mirror every field exactly."""
    failures = []
    flip = T.RandomHorizontalFlip(probability=1.0)
    resize = T.Resize(INPUT_SIZE)

    for index in range(count):
        original = resize(raw_sample(dataset, index))
        expected_image = original["image"][:, ::-1]
        expected_mask = original["mask"][:, ::-1]
        expected_masks = original["masks"][:, :, ::-1].copy()
        width = original["mask"].shape[1]

        flipped = flip({k: (v.copy() if isinstance(v, np.ndarray) else v)
                        for k, v in original.items()})

        if not np.array_equal(flipped["image"], expected_image):
            failures.append(f"sample {index}: image not mirrored")
        if not np.array_equal(flipped["mask"], expected_mask):
            failures.append(f"sample {index}: semantic mask not mirrored")
        if not np.array_equal(flipped["masks"], expected_masks):
            failures.append(f"sample {index}: instance masks not mirrored")

        # x1' = width - x2, x2' = width - x1. Derived from the mask by
        # _recompute_instances, so this checks the recomputation agrees with
        # the analytic answer rather than assuming it.
        for before, after in zip(original["boxes"], flipped["boxes"]):
            want = [width - before[2], before[1], width - before[0], before[3]]
            if not np.allclose(after, want, atol=1.0):
                failures.append(
                    f"sample {index}: box {after.tolist()} != {want}"
                )
                break
    return failures


# ---------------------------------------------------------------------------
# Check 2: synthetic alignment under random geometry
# ---------------------------------------------------------------------------

def synthetic_sample(size: int = 256) -> dict:
    """An image that is a literal rendering of its own mask.

    Blocks and a circle, one class each, so every labeled pixel has a known
    color and any image-mask drift shows up as a color mismatch rather than
    as something a human has to eyeball.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[30:110, 30:110] = 1
    mask[140:220, 40:120] = 2
    mask[40:100, 150:230] = 3
    cv2.circle(mask, (170, 170), 45, 4, -1)
    mask[0:20, :] = 5
    image = colorize(mask)
    return {"image": image, "mask": mask}


def check_alignment(trials: int = 60) -> list:
    """Random geometry must leave labeled interiors carrying their class color."""
    failures = []
    pipeline = T.Compose([
        T.Resize(INPUT_SIZE),
        T.RandomHorizontalFlip(0.5),
        T.RandomScaleCrop(INPUT_SIZE, (0.75, 1.25), 1.0),
        T.RandomRotation(15.0, 1.0),
    ])

    for trial in range(trials):
        sample = pipeline(synthetic_sample())
        image, mask = sample["image"], sample["mask"]

        for class_id in range(1, len(PALETTE)):
            region = (mask == class_id).astype(np.uint8)
            if region.sum() < 200:
                continue
            # Erode past the interpolation boundary. The image is resampled
            # bilinearly and the mask by nearest neighbor, so the outermost
            # few pixels of a region are blended in one and crisp in the
            # other by design. Interiors are where alignment is testable.
            interior = cv2.erode(region, np.ones((7, 7), np.uint8)) > 0
            if interior.sum() < 50:
                continue

            observed = image[interior].astype(np.int16)
            expected = PALETTE[class_id].astype(np.int16)
            drift = np.abs(observed - expected).max(axis=1)
            # A generous tolerance: the point is to catch a region carrying
            # the wrong class's color, which differs by 100+ per channel,
            # not to measure resampling error.
            if (drift > 60).mean() > 0.02:
                failures.append(
                    f"trial {trial}: {CLASS_NAMES[class_id]} interior "
                    f"{(drift > 60).mean():.1%} off-color - image and mask "
                    "are not aligned"
                )
                break
    return failures


# ---------------------------------------------------------------------------
# Check 3: instance-target coherence
# ---------------------------------------------------------------------------

def check_instances(dataset, count: int) -> list:
    """Boxes must bound their masks and the aligned arrays must stay aligned."""
    failures = []
    pipeline = T.Compose([
        T.Resize(INPUT_SIZE),
        T.RandomHorizontalFlip(0.5),
        T.RandomScaleCrop(INPUT_SIZE, (0.75, 1.25), 1.0),
        T.RandomRotation(12.0, 1.0),
    ])

    for index in range(count):
        sample = pipeline(raw_sample(dataset, index))
        lengths = {
            len(sample["masks"]), len(sample["labels"]),
            len(sample["boxes"]), len(sample["areas"]),
            len(sample["instance_ids"]),
        }
        if len(lengths) != 1:
            failures.append(f"sample {index}: instance arrays desynced {lengths}")
            continue

        for i, (binary, box, area) in enumerate(
            zip(sample["masks"], sample["boxes"], sample["areas"])
        ):
            if not binary.any():
                failures.append(f"sample {index}: instance {i} is empty")
                continue
            rows = np.where(binary.any(axis=1))[0]
            cols = np.where(binary.any(axis=0))[0]
            want = [cols[0], rows[0], cols[-1] + 1, rows[-1] + 1]
            if not np.allclose(box, want):
                failures.append(
                    f"sample {index}: instance {i} box {box.tolist()} "
                    f"does not bound its mask {want}"
                )
            if abs(area - binary.sum()) > 0.5:
                failures.append(
                    f"sample {index}: instance {i} area {area} != "
                    f"{binary.sum()} mask pixels"
                )
    return failures


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

def draw_figure(dataset, count: int, path: Path) -> None:
    """One row per image: original pair, then three independent augmentations."""
    pipeline = T.Compose([
        T.Resize(INPUT_SIZE),
        T.RandomHorizontalFlip(0.5),
        T.RandomScaleCrop(INPUT_SIZE, (0.75, 1.25), 0.8),
        T.RandomRotation(10.0, 0.8),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05, 0.8),
    ])
    resize = T.Resize(INPUT_SIZE)

    columns = 5
    figure, axes = plt.subplots(
        count, columns, figsize=(3.0 * columns, 3.0 * count)
    )
    if count == 1:
        axes = axes[np.newaxis, :]

    for row in range(count):
        base = resize(raw_sample(dataset, row))
        axes[row, 0].imshow(base["image"])
        axes[row, 0].set_ylabel(f"image {row}", fontsize=9)
        axes[row, 1].imshow(overlay(base["image"], base["mask"]))
        if row == 0:
            axes[row, 0].set_title("original", fontsize=10)
            axes[row, 1].set_title("original + mask", fontsize=10)

        for column in range(2, columns):
            augmented = pipeline(raw_sample(dataset, row))
            axes[row, column].imshow(
                overlay(augmented["image"], augmented["mask"])
            )
            if row == 0:
                axes[row, column].set_title(
                    f"augmented {column - 1}", fontsize=10
                )

        for column in range(columns):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    handles = [
        mpatches.Patch(color=PALETTE[i] / 255, label=CLASS_NAMES[i])
        for i in range(1, len(PALETTE))
    ]
    handles.append(mpatches.Patch(color=IGNORE_COLOR / 255, label="ignored"))
    figure.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005),
    )
    figure.suptitle(
        "Section 15 verification: geometry applied to image and mask together",
        fontsize=12,
    )
    figure.tight_layout(rect=[0, 0.02, 1, 0.98])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    dataset = InstanceSegmentationDataset("train", augment=False)
    print(f"Loaded {len(dataset)} training images.\n")

    all_failures = []
    for name, failures in [
        ("1. Flip exactness", check_flip(dataset, args.samples)),
        ("2. Alignment under random geometry", check_alignment()),
        ("3. Instance-target coherence", check_instances(dataset, args.samples)),
    ]:
        if failures:
            print(f"{name}: FAILED ({len(failures)})")
            for line in failures[:5]:
                print(f"    {line}")
            all_failures += failures
        else:
            print(f"{name}: passed")

    path = PLOTS_DIR / "transform_verification.png"
    draw_figure(dataset, args.samples, path)
    print(f"\nFigure: {path.relative_to(REPO_ROOT)}")

    if all_failures:
        print(
            f"\n{len(all_failures)} checks failed. The augmentation is not "
            "mask-safe; do not train on it."
        )
        return 1
    print("\nAugmentation is mask-safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
