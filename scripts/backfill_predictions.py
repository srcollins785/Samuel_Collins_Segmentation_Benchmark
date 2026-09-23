"""Generate the Section 52 prediction figures for models that already ran.

The prediction figures were added after the benchmark had started, so the
models that finished first have results but no example panels. Retraining
them to produce pictures would be absurd and would also change their scores,
since Metal kernels are not bit-deterministic - the figures would then
illustrate weights that no longer match the reported numbers.

This loads each model's saved checkpoint instead, which is the exact
selected state the reported metrics came from, and renders the panels from
it. Models that already have figures are skipped.

Usage
-----
    python scripts/backfill_predictions.py
    python scripts/backfill_predictions.py --model segformer
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from segmentation_benchmark import visualize  # noqa: E402
from segmentation_benchmark._config import (  # noqa: E402
    CHECKPOINTS_DIR, NUM_CLASSES, PREDICTIONS_DIR, RESULTS_DIR,
)
from segmentation_benchmark.dataset import (  # noqa: E402
    InstanceSegmentationDataset, SemanticSegmentationDataset, build_loader,
)
from segmentation_benchmark.models import get_segmentation_model  # noqa: E402
from segmentation_benchmark.train import resolve_device  # noqa: E402

RAW_DIR = RESULTS_DIR / "raw"


def backfill_semantic(model_name: str, input_size: int, device) -> list:
    checkpoint = CHECKPOINTS_DIR / f"best_{model_name}.pt"
    if not checkpoint.is_file():
        print(f"  {model_name}: no checkpoint at {checkpoint.name}, skipping")
        return []

    model = get_segmentation_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                     weights_only=True))
    model = model.to(device)

    test_set = SemanticSegmentationDataset("test", size=input_size)
    loader = build_loader(test_set, 8, False, workers=2)

    saved = visualize.save_semantic_predictions(
        model, loader, device, model_name, count=8, normalized=True
    )
    closeup = visualize.save_boundary_closeup(
        model, loader, device, model_name, normalized=True
    )
    return saved + ([closeup] if closeup else [])


def backfill_instance(model_name: str, input_size: int, device) -> list:
    checkpoint = CHECKPOINTS_DIR / f"best_{model_name}.pt"
    if not checkpoint.is_file():
        print(f"  {model_name}: no checkpoint, skipping")
        return []

    model = get_segmentation_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                     weights_only=True))
    model = model.to(device)

    test_set = InstanceSegmentationDataset("test", size=input_size)
    loader = build_loader(test_set, 4, False, workers=2)
    return visualize.save_instance_predictions(
        model, loader, device, model_name, count=6
    )


def backfill_kmeans(input_size: int) -> list:
    """Re-fit the K-Means mapping from its saved file and render clusters.

    The mapping is loaded rather than refitted, so the figures show the same
    frozen descriptor-to-class mapping the reported metrics used.
    """
    import cv2
    import numpy as np
    from PIL import Image

    from segmentation_benchmark._config import IMAGES_DIR
    from segmentation_benchmark.dataset import MASKS_DIR
    from segmentation_benchmark.traditional import KMeansSegmenter

    mapping = CHECKPOINTS_DIR / "kmeans_mapping.json"
    if not mapping.is_file():
        print("  kmeans: no saved mapping, skipping")
        return []

    segmenter = KMeansSegmenter().load(mapping)
    test_set = SemanticSegmentationDataset("test", size=input_size,
                                           augment=False)

    images, masks = [], []
    for record in test_set.records[:6]:
        stem = record["file_name"].rsplit(".", 1)[0]
        image = np.array(
            Image.open(test_set.image_dir / record["file_name"]).convert("RGB")
        )
        mask = np.array(Image.open(MASKS_DIR / "test" / f"{stem}.png"))
        images.append(
            cv2.resize(image, (input_size, input_size), cv2.INTER_LINEAR)
        )
        masks.append(cv2.resize(
            mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST
        ))

    return visualize.save_kmeans_qualitative(segmenter, images, masks, count=6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if figures already exist.")
    args = parser.parse_args()

    if not RAW_DIR.is_dir():
        print("No results yet.")
        return 1

    device = resolve_device()
    print(f"Device: {device}\n")

    for path in sorted(RAW_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        model_name = payload.get("model", path.stem)
        if args.model and model_name != args.model:
            continue

        folder = PREDICTIONS_DIR / model_name
        existing = list(folder.glob("*.png")) if folder.is_dir() else []
        if existing and not args.force:
            print(f"  {model_name}: {len(existing)} figures already, skipping")
            continue

        input_size = payload.get("config", {}).get("input_size", 256)
        task = payload.get("task")

        print(f"  {model_name}: rendering from saved checkpoint")
        try:
            if task == "traditional":
                saved = backfill_kmeans(input_size)
            elif task == "instance" and model_name != "yolo_seg":
                saved = backfill_instance(model_name, input_size, device)
            elif task == "instance":
                print("    yolo_seg renders through ultralytics; skipping")
                continue
            else:
                saved = backfill_semantic(model_name, input_size, device)

            if saved:
                payload["predictions"] = [str(s) for s in saved]
                path.write_text(json.dumps(payload, indent=2, default=str))
                print(f"    {len(saved)} figure(s)")
        except Exception as error:
            print(f"    FAILED {type(error).__name__}: {error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
