"""Tests for the Section 27-29 CSV generation.

Section 29 forbids inventing missing values. The tests that matter here are
the ones about what happens when a measurement is absent: it must come out
as N/A, never as zero, because a zero in a throughput column is a claim
about the model and an N/A is a statement about the measurement.
"""

import json

import pandas as pd
import pytest

from segmentation_benchmark import tables


def write_result(directory, name, task="semantic", **overrides):
    payload = {
        "model": name,
        "task": task,
        "run_id": "run1234",
        "config": {"input_size": 256, "batch_size": 8},
        "test": {
            "pixel_accuracy": 0.8, "mean_iou": 0.5, "mean_iou_foreground": 0.4,
            "dice": 0.6, "precision": 0.61, "recall": 0.59,
            "boundary_f1": 0.45,
            "per_class_iou": {
                "background": 0.9, "person": 0.6, "car": 0.4,
                "bicycle": None, "dog": 0.3, "cat": 0.35,
            },
            "per_class_boundary_f1": {
                "background": 0.8, "person": 0.5, "car": 0.3,
                "bicycle": None, "dog": 0.25, "cat": 0.3,
            },
        },
        "efficiency": {
            "total_parameters": 1000, "trainable_parameters": 1000,
            "frozen_parameters": 0, "weight_size_mb": 4.0,
            "checkpoint_size_mb": 4.1, "latency_ms": 10.0,
            "throughput_images_per_second": 50.0, "gmacs": 12.0,
            "environment": {"precision": "float32", "seed": 42,
                            "device": "test", "backend": "cpu"},
        },
        "training": {
            "total_train_seconds": 100.0, "mean_epoch_seconds": 4.0,
            "completed_epochs": 25, "best_epoch": 20, "peak_memory_mb": 500.0,
            "epochs": [{"epoch": 1, "train_loss": 1.0, "val_mean_iou": 0.2}],
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key].update(value)
        else:
            payload[key] = value
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(payload))
    return payload


class TestLoading:
    def test_reads_every_saved_result(self, tmp_path):
        write_result(tmp_path, "unet")
        write_result(tmp_path, "deeplabv3")
        assert set(tables.load_results(tmp_path)) == {"unet", "deeplabv3"}

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert tables.load_results(tmp_path / "absent") == {}


class TestSemanticTable:
    def test_one_row_per_semantic_model(self, tmp_path):
        write_result(tmp_path, "unet")
        write_result(tmp_path, "maskrcnn", task="instance")

        frame = tables.semantic_table(tables.load_results(tmp_path))

        assert list(frame["model"]) == ["unet"]

    def test_includes_every_section_27_column(self, tmp_path):
        write_result(tmp_path, "unet")
        frame = tables.semantic_table(tables.load_results(tmp_path))

        for column in ("pixel_accuracy", "mean_iou", "dice", "precision",
                       "recall", "total_parameters", "model_size_mb",
                       "images_per_second"):
            assert column in frame.columns, f"Section 27 requires {column}"

    def test_rows_follow_the_assignment_checklist_order(self, tmp_path):
        for name in ("segformer", "unet", "kmeans", "fcn_resnet50"):
            write_result(
                tmp_path, name,
                task="traditional" if name == "kmeans" else "semantic",
            )

        frame = tables.semantic_table(tables.load_results(tmp_path))

        assert list(frame["model"]) == [
            "kmeans", "fcn_resnet50", "unet", "segformer"
        ]

    def test_missing_measurement_becomes_na_not_zero(self, tmp_path):
        write_result(tmp_path, "unet", efficiency={"latency_ms": None})
        frame = tables.semantic_table(tables.load_results(tmp_path))
        assert frame.loc[0, "latency_ms"] == tables.NOT_APPLICABLE

    def test_absent_field_becomes_na(self, tmp_path):
        payload = write_result(tmp_path, "unet")
        del payload["efficiency"]["gmacs"]
        (tmp_path / "unet.json").write_text(json.dumps(payload))

        frame = tables.efficiency_table(tables.load_results(tmp_path))

        assert frame.loc[0, "gmacs"] == tables.NOT_APPLICABLE


class TestInstanceTable:
    def test_holds_only_instance_models(self, tmp_path):
        write_result(tmp_path, "unet")
        write_result(tmp_path, "maskrcnn", task="instance", test={
            "mask_ap": 0.3, "mask_ap50": 0.5, "mask_ap75": 0.32,
            "mask_ar_100": 0.4, "box_ap": 0.35,
        })

        frame = tables.instance_table(tables.load_results(tmp_path))

        assert list(frame["model"]) == ["maskrcnn"]

    def test_includes_the_small_object_breakdown(self, tmp_path):
        """Section 54 asks which architecture handled small objects best."""
        write_result(tmp_path, "maskrcnn", task="instance", test={
            "mask_ap": 0.3, "mask_ap_small": 0.1,
            "mask_ap_medium": 0.3, "mask_ap_large": 0.5,
        })
        frame = tables.instance_table(tables.load_results(tmp_path))
        assert frame.loc[0, "mask_ap_small"] == 0.1

    def test_no_semantic_metric_appears_in_it(self, tmp_path):
        """Section 26: the two tables stay separate."""
        write_result(tmp_path, "maskrcnn", task="instance", test={"mask_ap": 0.3})
        frame = tables.instance_table(tables.load_results(tmp_path))
        assert "mean_iou" not in frame.columns
        assert "dice" not in frame.columns


class TestEfficiencyTable:
    def test_covers_every_track(self, tmp_path):
        """Efficiency is the one axis all three tracks share."""
        write_result(tmp_path, "kmeans", task="traditional")
        write_result(tmp_path, "unet")
        write_result(tmp_path, "maskrcnn", task="instance")

        frame = tables.efficiency_table(tables.load_results(tmp_path))

        assert len(frame) == 3
        assert set(frame["task"]) == {"traditional", "semantic", "instance"}

    def test_carries_the_conditions_section_29_requires(self, tmp_path):
        write_result(tmp_path, "unet")
        frame = tables.efficiency_table(tables.load_results(tmp_path))

        for column in ("seed", "input_size", "batch_size", "precision",
                       "device", "run_id"):
            assert column in frame.columns, f"Section 29 requires {column} stored"

    def test_separates_total_from_trainable_parameters(self, tmp_path):
        write_result(tmp_path, "unet", efficiency={
            "total_parameters": 100, "trainable_parameters": 60,
            "frozen_parameters": 40,
        })
        frame = tables.efficiency_table(tables.load_results(tmp_path))
        assert frame.loc[0, "total_parameters"] == 100
        assert frame.loc[0, "trainable_parameters"] == 60


class TestPerClassTable:
    def test_one_row_per_class_one_column_per_model(self, tmp_path):
        write_result(tmp_path, "unet")
        write_result(tmp_path, "deeplabv3")

        frame = tables.per_class_iou_table(tables.load_results(tmp_path))

        assert list(frame["class"]) == [
            "background", "person", "car", "bicycle", "dog", "cat"
        ]
        assert {"unet", "deeplabv3"} <= set(frame.columns)

    def test_an_absent_class_is_na_not_zero(self, tmp_path):
        """Section 23: dashes mean to be measured, not a zero score."""
        write_result(tmp_path, "unet")
        frame = tables.per_class_iou_table(tables.load_results(tmp_path))

        bicycle = frame[frame["class"] == "bicycle"]["unet"].iloc[0]
        assert bicycle == tables.NOT_APPLICABLE


class TestWriteAll:
    def test_writes_every_required_file(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        write_result(raw, "unet")
        write_result(raw, "maskrcnn", task="instance", test={"mask_ap": 0.3})

        written = tables.write_all(tables.load_results(raw), tmp_path)

        for required in ("semantic_segmentation_results.csv",
                         "instance_segmentation_results.csv",
                         "segmentation_efficiency_results.csv"):
            assert required in written
            assert (tmp_path / required).is_file()

    def test_csv_spells_missing_values_as_na(self, tmp_path):
        raw = tmp_path / "raw"
        write_result(raw, "unet", efficiency={"checkpoint_size_mb": None})

        tables.write_all(tables.load_results(raw), tmp_path)
        text = (tmp_path / "segmentation_efficiency_results.csv").read_text()

        assert tables.NOT_APPLICABLE in text

    def test_history_is_exported_for_convergence_analysis(self, tmp_path):
        """Section 54 asks which model converged fastest and which overfit."""
        raw = tmp_path / "raw"
        write_result(raw, "unet")

        tables.write_all(tables.load_results(raw), tmp_path)
        frame = pd.read_csv(tmp_path / "training_history.csv")

        assert "epoch" in frame.columns
        assert "val_mean_iou" in frame.columns
