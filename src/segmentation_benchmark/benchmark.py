"""Run one model end to end and save everything the report will need.

Each model produces exactly one file, ``results/raw/<model>.json``, holding
its configuration, its training history, its test metrics and its efficiency
measurements. The CSVs and figures are generated from those files and from
nothing else, which is what makes Section 29's "generate tables directly
from saved evaluation outputs" true rather than aspirational - there is no
path by which a number reaches a table without first being written down.

The order inside ``run_model`` is deliberate. Training and checkpoint
selection happen against validation only. The test set is not touched until
the selected checkpoint is loaded and frozen, and it is evaluated exactly
once. Section 21 forbids selecting on test results; doing the test pass last
and once is how that is enforced in practice rather than promised.
"""

import json
import time
import uuid

import torch

from ._config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    CONFUSION_DIR,
    EPOCHS,
    INPUT_SIZE,
    LEARNING_RATE,
    LOGS_DIR,
    NUM_CLASSES,
    OPTIMIZER,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
)
from .boundary import evaluate_boundary
from .dataset import (
    InstanceSegmentationDataset, SemanticSegmentationDataset, build_loader,
    load_manifest,
)
from .efficiency import profile_model, save_json
from .losses import CombinedSegmentationLoss
from .metrics import evaluate_semantic
from .models import MODEL_CARDS, describe_model, get_segmentation_model
from .train import resolve_device, train_instance_model, train_model

RAW_DIR = RESULTS_DIR / "raw"


def seed_everything(seed: int = SEED) -> None:
    """Seed every generator the run touches.

    Section 7 fixes the split at this seed; this fixes initialization,
    shuffling and augmentation too, so a rerun is a rerun. It does not make
    the result bit-identical - MPS kernels are not deterministic and torch
    offers no deterministic mode for them - which is recorded in the run
    configuration rather than claimed otherwise.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_configuration(model_name: str, task: str, input_size: int,
                      batch_size: int, epochs: int, pretrained: bool,
                      train_set, run_id: str) -> dict:
    """Everything Section 10 and Section 29 require stored with a measurement."""
    manifest = load_manifest()
    return {
        "run_id": run_id,
        "model": model_name,
        "task": task,
        "input_size": input_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "pretrained": pretrained,
        "num_classes": NUM_CLASSES,
        "split_checksum": manifest["checksum"],
        "split_sizes": manifest["sizes"],
        "mask_policies": manifest["policies"],
        "transforms": train_set.describe()["transforms"],
        "determinism": (
            "seeded initialization, shuffling and augmentation; MPS kernels "
            "are not bit-deterministic, so a rerun reproduces the protocol "
            "rather than the exact weights"
        ),
    }


def run_semantic(model_name: str, epochs: int = EPOCHS,
                 input_size: int = INPUT_SIZE, batch_size: int = BATCH_SIZE,
                 pretrained: bool = True, device=None, workers: int = 4,
                 verbose: bool = True) -> dict:
    """Train, select, and evaluate one semantic architecture."""
    device = device or resolve_device()
    run_id = uuid.uuid4().hex[:8]
    seed_everything()

    train_set = SemanticSegmentationDataset("train", size=input_size)
    val_set = SemanticSegmentationDataset("val", size=input_size)
    test_set = SemanticSegmentationDataset("test", size=input_size)

    train_loader = build_loader(train_set, batch_size, True, workers)
    val_loader = build_loader(val_set, batch_size, False, workers)
    test_loader = build_loader(test_set, batch_size, False, workers)

    model = get_segmentation_model(model_name, NUM_CLASSES, pretrained)
    criterion = CombinedSegmentationLoss()

    if verbose:
        print(f"  training {model_name} for {epochs} epochs at {input_size}px")

    started = time.perf_counter()
    history = train_model(
        model, train_loader, val_loader, epochs=epochs, device=device,
        model_name=model_name, criterion=criterion,
        checkpoint_dir=CHECKPOINTS_DIR, log_dir=LOGS_DIR, verbose=verbose,
    )
    wall_clock = time.perf_counter() - started

    # Test, once, with the selected weights already loaded by train_model.
    if verbose:
        print("  evaluating on the held-out test split")
    test_metrics, matrix = evaluate_semantic(model, test_loader, device)
    boundary_metrics, _ = evaluate_boundary(model, test_loader, device)
    test_metrics.update(boundary_metrics)

    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    (CONFUSION_DIR / f"{model_name}.json").write_text(
        json.dumps({"model": model_name, "matrix": matrix.to_list()}, indent=2)
    )

    checkpoint = CHECKPOINTS_DIR / f"best_{model_name}.pt"
    efficiency = profile_model(
        model, model_name, device, checkpoint_path=checkpoint,
        input_size=input_size,
    )

    training = history.to_dict()
    training["wall_clock_seconds"] = wall_clock
    training["peak_memory_mb"] = max(
        (e.get("device_memory_mb", 0.0) for e in training["epochs"]), default=0.0
    )

    payload = {
        "model": model_name,
        "task": "semantic",
        "run_id": run_id,
        "config": run_configuration(
            model_name, "semantic", input_size, batch_size, epochs,
            pretrained, train_set, run_id,
        ),
        "card": describe_model(model_name, model),
        "loss": criterion.describe(),
        "training": training,
        "test": test_metrics,
        "efficiency": efficiency,
    }
    save_json(payload, RAW_DIR / f"{model_name}.json")
    return payload


def run_instance(model_name: str = "maskrcnn", epochs: int = EPOCHS,
                 input_size: int = INPUT_SIZE, batch_size: int = BATCH_SIZE,
                 pretrained: bool = True, device=None, workers: int = 4,
                 verbose: bool = True) -> dict:
    """Train, select, and evaluate Mask R-CNN."""
    from .instance_metrics import evaluate_instance

    device = device or resolve_device()
    run_id = uuid.uuid4().hex[:8]
    seed_everything()

    train_set = InstanceSegmentationDataset("train", size=input_size)
    val_set = InstanceSegmentationDataset("val", size=input_size)
    test_set = InstanceSegmentationDataset("test", size=input_size)

    train_loader = build_loader(train_set, batch_size, True, workers)
    val_loader = build_loader(val_set, batch_size, False, workers)
    test_loader = build_loader(test_set, batch_size, False, workers)

    model = get_segmentation_model(model_name, NUM_CLASSES, pretrained)

    def validation_evaluator(current, loader, current_device):
        """Section 21's declared instance selection metric: validation mask AP."""
        summary = evaluate_instance(
            current, loader, current_device, split="val", include_box_ap=False
        )
        current.train()
        return {
            "mask_ap": summary["mask_ap"],
            "mask_ap50": summary["mask_ap50"],
            "mask_ap75": summary["mask_ap75"],
        }

    if verbose:
        print(f"  training {model_name} for {epochs} epochs at {input_size}px")

    started = time.perf_counter()
    history = train_instance_model(
        model, train_loader, val_loader, epochs=epochs, device=device,
        model_name=model_name, checkpoint_dir=CHECKPOINTS_DIR,
        log_dir=LOGS_DIR, evaluator=validation_evaluator, verbose=verbose,
    )
    wall_clock = time.perf_counter() - started

    if verbose:
        print("  evaluating on the held-out test split")
    test_metrics = evaluate_instance(model, test_loader, device, split="test")

    checkpoint = CHECKPOINTS_DIR / f"best_{model_name}.pt"
    efficiency = profile_model(
        model, model_name, device, checkpoint_path=checkpoint,
        input_size=input_size, instance=True,
    )

    training = history.to_dict()
    training["wall_clock_seconds"] = wall_clock
    training["peak_memory_mb"] = max(
        (e.get("device_memory_mb", 0.0) for e in training["epochs"]), default=0.0
    )

    payload = {
        "model": model_name,
        "task": "instance",
        "run_id": run_id,
        "config": run_configuration(
            model_name, "instance", input_size, batch_size, epochs,
            pretrained, train_set, run_id,
        ),
        "card": describe_model(model_name, model),
        "loss": {
            "note": (
                "Section 17: Mask R-CNN keeps its own four-term loss plus the "
                "RPN's two. The semantic CE+Dice recipe is not applied."
            ),
            "components": [
                "loss_classifier", "loss_box_reg", "loss_mask",
                "loss_objectness", "loss_rpn_box_reg",
            ],
        },
        "training": training,
        "test": test_metrics,
        "efficiency": efficiency,
    }
    save_json(payload, RAW_DIR / f"{model_name}.json")
    return payload


def run_yolo(epochs: int = EPOCHS, input_size: int = INPUT_SIZE,
             batch_size: int = BATCH_SIZE, device=None,
             verbose: bool = True) -> dict:
    """Prepare, train and score YOLO segmentation through ultralytics."""
    from . import yolo_adapter

    device = device or resolve_device()
    run_id = uuid.uuid4().hex[:8]
    seed_everything()

    if verbose:
        print("  exporting the shared split to YOLO format")
    export = yolo_adapter.prepare()

    started = time.perf_counter()
    training_record = yolo_adapter.train_yolo(
        epochs=epochs, input_size=input_size, batch_size=batch_size,
        device=str(device), verbose=verbose,
    )
    wall_clock = time.perf_counter() - started

    if verbose:
        print("  evaluating on the held-out test split")
    test_metrics = yolo_adapter.predict_and_score(
        training_record["weights"], split="test",
        input_size=input_size, device=str(device),
    )

    from ultralytics import YOLO
    weights = YOLO(training_record["weights"])
    torch_model = weights.model
    checkpoint = __import__("pathlib").Path(training_record["weights"])

    efficiency = {}
    try:
        from .efficiency import checkpoint_size_mb, environment
        from .models import count_parameters, weight_size_mb
        efficiency.update(count_parameters(torch_model))
        efficiency["weight_size_mb"] = round(weight_size_mb(torch_model), 3)
        efficiency["checkpoint_size_mb"] = round(
            checkpoint_size_mb(checkpoint) or 0.0, 3
        )
        efficiency.update(_time_yolo(weights, input_size, str(device)))
        efficiency["environment"] = environment()
    except Exception as error:
        efficiency["measurement_error"] = str(error)[:300]

    payload = {
        "model": "yolo_seg",
        "task": "instance",
        "run_id": run_id,
        "config": {
            "run_id": run_id,
            "model": "yolo_seg",
            "task": "instance",
            "input_size": input_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": SEED,
            "split_checksum": load_manifest()["checksum"],
            "export": export,
            **training_record,
        },
        "card": MODEL_CARDS["yolo_seg"],
        "loss": {
            "note": (
                "Section 17: ultralytics' own segmentation losses (box, "
                "class, DFL, mask). The semantic CE+Dice recipe is not applied."
            ),
        },
        "training": {
            "total_train_seconds": wall_clock,
            "wall_clock_seconds": wall_clock,
            "completed_epochs": epochs,
            "mean_epoch_seconds": wall_clock / max(epochs, 1),
            "epochs": [],
            "note": (
                "ultralytics owns the loop; per-epoch history is in its own "
                "results.csv, referenced from config.results_csv"
            ),
        },
        "test": test_metrics,
        "efficiency": efficiency,
    }
    save_json(payload, RAW_DIR / "yolo_seg.json")
    return payload


def _time_yolo(weights, input_size: int, device: str) -> dict:
    """Latency and throughput for YOLO, timed the same way as every other model."""
    import numpy as np

    from ._config import INFERENCE_WARMUP_ITERS

    blank = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    for _ in range(INFERENCE_WARMUP_ITERS):
        weights.predict(blank, imgsz=input_size, device=device, verbose=False)

    timed = 50
    started = time.perf_counter()
    for _ in range(timed):
        weights.predict(blank, imgsz=input_size, device=device, verbose=False)
    elapsed = time.perf_counter() - started

    return {
        "latency_ms": elapsed / timed * 1000,
        "latency_batch_size": 1,
        "throughput_images_per_second": timed / elapsed,
        "throughput_batch_size": 1,
        "timed_region": (
            "ultralytics predict end to end, which includes its own "
            "preprocessing and NMS -- not comparable pass-for-pass with the "
            "forward-only timing used for the torchvision models"
        ),
        "inference_memory_mb": None,
        "gmacs": None,
        "complexity_note": "N/A: ultralytics model not traced",
    }


def run_traditional(input_size: int = INPUT_SIZE, fit_images: int = 300,
                    verbose: bool = True) -> dict:
    """Fit and evaluate the K-Means baseline.

    The mapping is fitted on training images and frozen before any test
    image is read, which is the separation Section 10 requires.
    """
    import numpy as np

    from .metrics import ConfusionMatrix
    from .traditional import KMeansSegmenter

    run_id = uuid.uuid4().hex[:8]
    seed_everything()

    train_set = SemanticSegmentationDataset("train", size=input_size, augment=False)
    test_set = SemanticSegmentationDataset("test", size=input_size, augment=False)

    def raw_pairs(dataset, limit=None):
        from PIL import Image

        from ._config import IMAGES_DIR
        from .dataset import MASKS_DIR

        count = limit or len(dataset.records)
        for record in dataset.records[:count]:
            stem = record["file_name"].rsplit(".", 1)[0]
            image = np.array(
                Image.open(dataset.image_dir / record["file_name"]).convert("RGB")
            )
            mask = np.array(
                Image.open(MASKS_DIR / dataset.split / f"{stem}.png")
            )
            import cv2
            image = cv2.resize(image, (input_size, input_size), cv2.INTER_LINEAR)
            mask = cv2.resize(
                mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST
            )
            yield image, mask

    segmenter = KMeansSegmenter()
    if verbose:
        print(f"  fitting the cluster-to-class mapping on {fit_images} training images")

    started = time.perf_counter()
    images, masks = zip(*list(raw_pairs(train_set, fit_images)))
    fit_report = segmenter.fit(images, masks, max_images=fit_images)
    fit_seconds = time.perf_counter() - started

    if verbose:
        print("  evaluating on the held-out test split")
    matrix = ConfusionMatrix()
    from .boundary import BoundaryAccumulator
    boundary = BoundaryAccumulator()

    latencies = []
    for image, mask in raw_pairs(test_set):
        single = time.perf_counter()
        prediction = segmenter.predict(image)
        latencies.append(time.perf_counter() - single)
        matrix.update(prediction, mask)
        boundary.update(prediction, mask)

    test_metrics = matrix.summary()
    test_metrics.update(boundary.summary())

    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    (CONFUSION_DIR / "kmeans.json").write_text(
        json.dumps({"model": "kmeans", "matrix": matrix.to_list()}, indent=2)
    )

    mapping_path = CHECKPOINTS_DIR / "kmeans_mapping.json"
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    segmenter.save(mapping_path)

    import numpy as np
    mean_latency = float(np.mean(latencies)) * 1000

    from .efficiency import checkpoint_size_mb, environment
    payload = {
        "model": "kmeans",
        "task": "traditional",
        "run_id": run_id,
        "config": {
            "run_id": run_id, "model": "kmeans", "task": "traditional",
            "input_size": input_size, "batch_size": 1, "epochs": 0,
            "seed": SEED, "split_checksum": load_manifest()["checksum"],
            **segmenter.describe(), **fit_report,
        },
        "card": MODEL_CARDS["kmeans"],
        "training": {
            "total_train_seconds": fit_seconds,
            "mean_epoch_seconds": fit_seconds,
            "completed_epochs": 0,
            "best_epoch": None,
            "epochs": [],
            "note": "unsupervised; 'training' is fitting the frozen class mapping",
        },
        "test": test_metrics,
        "efficiency": {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "frozen_parameters": 0,
            "weight_size_mb": round(
                checkpoint_size_mb(mapping_path) or 0.0, 4
            ),
            "checkpoint_size_mb": round(checkpoint_size_mb(mapping_path) or 0.0, 4),
            "gmacs": None,
            "complexity_note": "N/A: no neural network to trace",
            "latency_ms": mean_latency,
            "latency_batch_size": 1,
            "throughput_images_per_second": 1000.0 / mean_latency,
            "throughput_batch_size": 1,
            "inference_memory_mb": None,
            "timed_region": "per-image K-Means fit and cluster labeling, on CPU",
            "environment": environment(),
        },
    }
    save_json(payload, RAW_DIR / "kmeans.json")
    return payload
