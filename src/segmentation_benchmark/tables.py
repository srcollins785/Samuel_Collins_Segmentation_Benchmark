"""Sections 27, 28, 29 and 23: the required CSVs, generated from saved results.

Section 29 forbids typing experimental results by hand or inventing missing
values, and requires tables to be generated directly from saved evaluation
outputs. This module reads ``results/raw/<model>.json`` - written by the
benchmark as each model finishes - and writes the CSVs. It imports no model
code and runs no inference, so a table can only contain a number some model
actually produced, and the tables can be rebuilt in a second without
retraining anything.

Section 29 also requires "N/A" where a measurement does not apply, with an
explanation. Missing values are written as ``N/A`` and the reason is carried
in the raw JSON next to the field, never silently replaced with zero - a
zero in a throughput column is a claim that the model processed no images,
which is different from not having measured it.
"""

import json

import pandas as pd

from ._config import CLASS_NAMES, RESULTS_DIR
from .models import ALL_MODELS, MODEL_CARDS

RAW_DIR = RESULTS_DIR / "raw"

NOT_APPLICABLE = "N/A"


def load_results(raw_dir=RAW_DIR) -> dict:
    """Read every saved per-model result file."""
    raw_dir = raw_dir or RAW_DIR
    if not raw_dir.is_dir():
        return {}
    results = {}
    for path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        results[payload.get("model", path.stem)] = payload
    return results


def _get(payload: dict, *keys, default=NOT_APPLICABLE):
    """Fetch a nested value, returning N/A rather than raising or zeroing."""
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def _order(names: list) -> list:
    """Order rows as the assignment's checklist lists the models."""
    index = {name: position for position, name in enumerate(ALL_MODELS)}
    return sorted(names, key=lambda n: index.get(n, len(index)))


def semantic_table(results: dict) -> pd.DataFrame:
    """Section 27: semantic_segmentation_results.csv.

    Pixel accuracy, mIoU, Dice, pixel precision and recall, plus parameter
    count, model size and throughput, for every semantic architecture.

    The traditional baseline appears here only if a class mapping was fitted
    on training data, which Section 10 requires before its output can be
    scored semantically at all. Its row is marked so no reader mistakes an
    unsupervised clustering for a trained segmenter.
    """
    rows = []
    for name in _order([
        n for n, p in results.items()
        if p.get("task") in {"semantic", "traditional"}
    ]):
        payload = results[name]
        test = payload.get("test", {})
        rows.append({
            "model": name,
            "task": payload.get("task", NOT_APPLICABLE),
            "pixel_accuracy": _get(test, "pixel_accuracy"),
            "mean_iou": _get(test, "mean_iou"),
            "mean_iou_foreground": _get(test, "mean_iou_foreground"),
            "dice": _get(test, "dice"),
            "precision": _get(test, "precision"),
            "recall": _get(test, "recall"),
            "boundary_precision": _get(test, "boundary_precision"),
            "boundary_recall": _get(test, "boundary_recall"),
            "boundary_f1": _get(test, "boundary_f1"),
            "total_parameters": _get(payload, "efficiency", "total_parameters"),
            "trainable_parameters": _get(
                payload, "efficiency", "trainable_parameters"
            ),
            "model_size_mb": _get(payload, "efficiency", "weight_size_mb"),
            "checkpoint_size_mb": _get(payload, "efficiency", "checkpoint_size_mb"),
            "latency_ms": _get(payload, "efficiency", "latency_ms"),
            "images_per_second": _get(
                payload, "efficiency", "throughput_images_per_second"
            ),
            "backbone": MODEL_CARDS.get(name, {}).get("backbone", NOT_APPLICABLE),
            "initialization": MODEL_CARDS.get(name, {}).get(
                "initialization", NOT_APPLICABLE
            ),
        })
    return pd.DataFrame(rows)


def instance_table(results: dict) -> pd.DataFrame:
    """Section 28: instance_segmentation_results.csv.

    Mask AP, AP50, AP75, Average Recall, optional box AP, parameters, model
    size, latency and throughput. The small/medium/large breakdown is
    included because Section 54 asks which architecture handled small
    objects best, and that question has a direct answer here.
    """
    rows = []
    for name in _order([
        n for n, p in results.items() if p.get("task") == "instance"
    ]):
        payload = results[name]
        test = payload.get("test", {})
        rows.append({
            "model": name,
            "mask_ap": _get(test, "mask_ap"),
            "mask_ap50": _get(test, "mask_ap50"),
            "mask_ap75": _get(test, "mask_ap75"),
            "mask_ar_100": _get(test, "mask_ar_100"),
            "mask_ap_small": _get(test, "mask_ap_small"),
            "mask_ap_medium": _get(test, "mask_ap_medium"),
            "mask_ap_large": _get(test, "mask_ap_large"),
            "box_ap": _get(test, "box_ap"),
            "box_ap50": _get(test, "box_ap50"),
            "total_parameters": _get(payload, "efficiency", "total_parameters"),
            "model_size_mb": _get(payload, "efficiency", "weight_size_mb"),
            "checkpoint_size_mb": _get(payload, "efficiency", "checkpoint_size_mb"),
            "latency_ms": _get(payload, "efficiency", "latency_ms"),
            "images_per_second": _get(
                payload, "efficiency", "throughput_images_per_second"
            ),
            "backbone": MODEL_CARDS.get(name, {}).get("backbone", NOT_APPLICABLE),
        })
    return pd.DataFrame(rows)


def efficiency_table(results: dict) -> pd.DataFrame:
    """Section 29: segmentation_efficiency_results.csv, every architecture.

    One row per model across all three tracks, because efficiency is the
    one axis on which a traditional baseline, a semantic network and an
    instance network are directly comparable - they all take an image and
    take time to do it. Quality is not comparable across tracks, which is
    why Section 26 keeps those tables apart and this one together.
    """
    rows = []
    for name in _order(list(results)):
        payload = results[name]
        efficiency = payload.get("efficiency", {})
        training = payload.get("training", {})
        environment = efficiency.get("environment", {})
        rows.append({
            "model": name,
            "task": payload.get("task", NOT_APPLICABLE),
            "total_parameters": _get(efficiency, "total_parameters"),
            "trainable_parameters": _get(efficiency, "trainable_parameters"),
            "frozen_parameters": _get(efficiency, "frozen_parameters"),
            "gmacs": _get(efficiency, "gmacs"),
            "model_size_mb": _get(efficiency, "weight_size_mb"),
            "checkpoint_size_mb": _get(efficiency, "checkpoint_size_mb"),
            "total_train_seconds": _get(training, "total_train_seconds"),
            "mean_epoch_seconds": _get(training, "mean_epoch_seconds"),
            "completed_epochs": _get(training, "completed_epochs"),
            "selected_epoch": _get(training, "best_epoch"),
            "latency_ms": _get(efficiency, "latency_ms"),
            "latency_batch_size": _get(efficiency, "latency_batch_size"),
            "images_per_second": _get(efficiency, "throughput_images_per_second"),
            "throughput_batch_size": _get(efficiency, "throughput_batch_size"),
            "train_peak_memory_mb": _get(training, "peak_memory_mb"),
            "inference_memory_mb": _get(efficiency, "inference_memory_mb"),
            "input_size": _get(payload, "config", "input_size"),
            "batch_size": _get(payload, "config", "batch_size"),
            "precision": environment.get("precision", NOT_APPLICABLE),
            "seed": environment.get("seed", NOT_APPLICABLE),
            "device": environment.get("device", NOT_APPLICABLE),
            "backend": environment.get("backend", NOT_APPLICABLE),
            "run_id": _get(payload, "run_id"),
        })
    return pd.DataFrame(rows)


def per_class_iou_table(results: dict) -> pd.DataFrame:
    """Section 23: one column per semantic model, one row per class.

    Section 23 says dashes indicate values to be measured, not zero scores,
    so a class a model never predicted and that never appeared is N/A rather
    than 0.0. The distinction matters here: bicycle is 0.26% of pixels, and
    a model scoring 0.0 on it has failed at something a model scoring N/A
    was never asked to do.
    """
    semantic = _order([
        n for n, p in results.items()
        if p.get("task") in {"semantic", "traditional"}
    ])
    rows = []
    for class_name in CLASS_NAMES:
        row = {"class": class_name}
        for name in semantic:
            value = _get(
                results[name], "test", "per_class_iou", class_name
            )
            row[name] = value
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_boundary_table(results: dict) -> pd.DataFrame:
    """Section 24's per-class boundary F1, alongside the per-class IoU."""
    semantic = _order([
        n for n, p in results.items()
        if p.get("task") in {"semantic", "traditional"}
    ])
    rows = []
    for class_name in CLASS_NAMES:
        row = {"class": class_name}
        for name in semantic:
            row[name] = _get(
                results[name], "test", "per_class_boundary_f1", class_name
            )
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_table(results: dict) -> pd.DataFrame:
    """Per-epoch history for every trained model, in one long-format table.

    Section 54 asks which model converged fastest and which overfit most.
    Both are properties of the curve rather than the final score, so the
    curves are exported rather than only their endpoints.
    """
    rows = []
    for name in _order(list(results)):
        for record in results[name].get("training", {}).get("epochs", []):
            rows.append({
                "model": name,
                "task": results[name].get("task", NOT_APPLICABLE),
                **{
                    key: value for key, value in record.items()
                    if not isinstance(value, (dict, list))
                },
            })
    return pd.DataFrame(rows)


def write_all(results: dict = None, results_dir=RESULTS_DIR) -> dict:
    """Write every required CSV and report where each went."""
    results = load_results() if results is None else results
    results_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "semantic_segmentation_results.csv": semantic_table(results),
        "instance_segmentation_results.csv": instance_table(results),
        "segmentation_efficiency_results.csv": efficiency_table(results),
        "per_class_iou.csv": per_class_iou_table(results),
        "per_class_boundary_f1.csv": per_class_boundary_table(results),
        "training_history.csv": convergence_table(results),
    }

    written = {}
    for filename, frame in outputs.items():
        path = results_dir / filename
        # N/A rather than an empty cell, so a reader can tell a missing
        # measurement from a parsing accident.
        frame.to_csv(path, index=False, na_rep=NOT_APPLICABLE)
        written[filename] = {"path": str(path), "rows": len(frame)}
    return written
