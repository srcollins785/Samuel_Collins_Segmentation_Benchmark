"""Section 52 prediction figures for YOLO segmentation.

Separate from ``backfill_predictions.py`` for the same reason YOLO has its
own adapter: it does not load as a torchvision detection model, so it cannot
share the rendering path that takes one. It predicts through ultralytics and
returns its own result object.

The panels match the Mask R-CNN ones exactly - same test images, same order,
one color per instance rather than per class - so the two instance models
can be compared by eye as well as by mask AP.

Usage
-----
    python scripts/render_yolo_predictions.py
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pycocotools.coco import COCO  # noqa: E402

from segmentation_benchmark._config import (  # noqa: E402
    ANNOTATIONS_DIR, IMAGES_DIR, PREDICTIONS_DIR, RESULTS_DIR,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--score", type=float, default=0.5)
    args = parser.parse_args()

    from ultralytics import YOLO

    result_path = RESULTS_DIR / "raw" / "yolo_seg.json"
    if not result_path.is_file():
        raise SystemExit("No YOLO result yet; run the benchmark first.")

    payload = json.loads(result_path.read_text())
    weights = payload["config"]["weights"]
    if not Path(weights).is_file():
        raise SystemExit(f"No YOLO weights at {weights}")

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ANNOTATIONS_DIR / "instances_test.json"))

    model = YOLO(weights)
    output_dir = PREDICTIONS_DIR / "yolo_seg"
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_colors = plt.get_cmap("tab20")

    # Same images, in the same order, as every other model's panels: the
    # test loader sorts by image id, so this does too.
    records = sorted(coco.imgs.values(), key=lambda r: r["id"])[: args.count]
    saved = []

    for index, record in enumerate(records):
        path = IMAGES_DIR / "test" / record["file_name"]
        image = np.array(Image.open(path).convert("RGB").resize(
            (args.input_size, args.input_size), Image.BILINEAR
        ))

        truth_panel = image.copy()
        annotations = coco.loadAnns(
            coco.getAnnIds(imgIds=record["id"], iscrowd=False)
        )
        for order, annotation in enumerate(annotations):
            mask = coco.annToMask(annotation).astype(np.uint8)
            mask = np.array(Image.fromarray(mask).resize(
                (args.input_size, args.input_size), Image.NEAREST
            )).astype(bool)
            if not mask.any():
                continue
            color = np.array(instance_colors(order % 20)[:3]) * 255
            truth_panel[mask] = (
                0.55 * color + 0.45 * truth_panel[mask]
            ).astype(np.uint8)

        prediction = model.predict(
            str(path), imgsz=args.input_size, conf=args.score,
            verbose=False, retina_masks=True,
        )[0]

        prediction_panel = image.copy()
        kept = 0
        if prediction.masks is not None:
            for order, binary in enumerate(prediction.masks.data.cpu().numpy()):
                mask = np.array(Image.fromarray(
                    (binary >= 0.5).astype(np.uint8)
                ).resize((args.input_size, args.input_size), Image.NEAREST)).astype(bool)
                if not mask.any():
                    continue
                color = np.array(instance_colors(order % 20)[:3]) * 255
                prediction_panel[mask] = (
                    0.55 * color + 0.45 * prediction_panel[mask]
                ).astype(np.uint8)
                kept += 1

        figure, axes = plt.subplots(1, 3, figsize=(11, 3.9))
        axes[0].imshow(image)
        axes[0].set_title("image", fontsize=10)
        axes[1].imshow(truth_panel)
        axes[1].set_title(
            f"ground truth — {len(annotations)} instances", fontsize=10
        )
        axes[2].imshow(prediction_panel)
        axes[2].set_title(
            f"yolo_seg — {kept} above {args.score}", fontsize=10
        )
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle("One color per instance, not per class", fontsize=10)
        figure.tight_layout(rect=[0, 0, 1, 0.94])

        out = output_dir / f"example_{index:02d}.png"
        figure.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(figure)
        saved.append(str(out))

    payload["predictions"] = saved
    result_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {len(saved)} figure(s) to {output_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
