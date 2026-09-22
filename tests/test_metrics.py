"""Tests for the Section 22 semantic metrics.

The cases that matter are the conventions the assignment requires to be
declared: ignored pixels, absent classes, background inclusion, and the
claim that Dice and pixel F1 are the same quantity.
"""

import numpy as np
import pytest
import torch

from segmentation_benchmark._config import IGNORE_INDEX, NUM_CLASSES
from segmentation_benchmark.metrics import CONVENTIONS, ConfusionMatrix


def matrix_from(truth, prediction) -> ConfusionMatrix:
    return ConfusionMatrix().update(np.array(prediction), np.array(truth))


class TestCounts:
    def test_perfect_prediction(self):
        truth = [0, 0, 1, 1, 2, 2]
        matrix = matrix_from(truth, truth)

        assert matrix.pixel_accuracy() == 1.0
        assert matrix.mean_iou() == 1.0
        assert matrix.dice() == 1.0

    def test_rows_are_truth_and_columns_are_prediction(self):
        """Every metric's sign depends on this; an transposed matrix
        swaps precision and recall silently."""
        matrix = matrix_from(truth=[1, 1, 1], prediction=[2, 2, 2])

        assert matrix.matrix[1, 2] == 3
        assert matrix.matrix[2, 1] == 0
        assert matrix.false_negatives[1] == 3
        assert matrix.false_positives[2] == 3

    def test_iou_matches_the_definition(self):
        # 2 true positives for class 1, 1 false positive, 1 false negative.
        matrix = matrix_from(truth=[1, 1, 1, 0], prediction=[1, 1, 0, 1])

        iou = matrix.per_class_iou()
        assert iou[1] == pytest.approx(2 / (2 + 1 + 1))

    def test_dice_matches_the_definition(self):
        matrix = matrix_from(truth=[1, 1, 1, 0], prediction=[1, 1, 0, 1])

        dice = matrix.per_class_dice()
        assert dice[1] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


class TestConventions:
    def test_ignored_pixels_are_excluded(self):
        """A wrong prediction under an ignored pixel must not be counted."""
        truth = [1, 1, IGNORE_INDEX, IGNORE_INDEX]
        matrix = matrix_from(truth, prediction=[1, 1, 3, 4])

        assert matrix.matrix.sum() == 2
        assert matrix.pixel_accuracy() == 1.0

    def test_absent_classes_are_excluded_not_scored_zero(self):
        """Scoring an unseen class zero would punish a model for the split."""
        matrix = matrix_from(truth=[0, 0, 1, 1], prediction=[0, 0, 1, 1])

        iou = matrix.per_class_iou()
        assert np.isnan(iou[3]), "absent class should be undefined"
        assert matrix.mean_iou() == 1.0, "absent class dragged the mean down"

    def test_background_inclusion_is_reported_both_ways(self):
        # Background perfect, person entirely missed.
        matrix = matrix_from(truth=[0, 0, 0, 1], prediction=[0, 0, 0, 0])

        assert matrix.mean_iou() > matrix.mean_iou_foreground()
        assert matrix.mean_iou_foreground() == pytest.approx(0.0)

    def test_dice_and_pixel_f1_are_the_same_quantity(self):
        """Section 22 says so; this keeps two code paths from drifting."""
        matrix = matrix_from(truth=[0, 1, 1, 2], prediction=[0, 1, 2, 2])
        summary = matrix.summary()
        assert summary["dice"] == summary["pixel_f1"]

    def test_conventions_are_recorded_in_every_summary(self):
        summary = matrix_from([0, 1], [0, 1]).summary()
        assert summary["conventions"] == CONVENTIONS
        for key in ("ignored_pixels", "background", "absent_classes", "averaging"):
            assert key in summary["conventions"]


class TestAccumulation:
    def test_batches_accumulate_into_dataset_level_counts(self):
        """Not an average of per-batch scores; the two differ and only
        one is the figure the assignment's tables ask for."""
        matrix = ConfusionMatrix()
        matrix.update(np.array([1, 1]), np.array([1, 1]))
        matrix.update(np.array([0] * 100), np.array([1] * 100))

        # Per-batch averaging would give (1.0 + 0.0) / 2 for class 1.
        # Accumulated: 2 TP against 100 FN.
        iou = matrix.per_class_iou()
        assert iou[1] == pytest.approx(2 / (2 + 100))

    def test_merge_adds_counts(self):
        first = matrix_from([1, 1], [1, 1])
        second = matrix_from([2, 2], [2, 2])
        first.merge(second)
        assert first.matrix.sum() == 4

    def test_accepts_torch_tensors(self):
        matrix = ConfusionMatrix()
        matrix.update(torch.tensor([[1, 1], [0, 0]]), torch.tensor([[1, 1], [0, 0]]))
        assert matrix.pixel_accuracy() == 1.0

    def test_handles_multidimensional_batches(self):
        prediction = torch.zeros(2, 8, 8, dtype=torch.long)
        target = torch.zeros(2, 8, 8, dtype=torch.long)
        matrix = ConfusionMatrix().update(prediction, target)
        assert matrix.matrix.sum() == 2 * 8 * 8


class TestSummary:
    def test_reports_every_required_metric(self):
        summary = matrix_from([0, 1, 2], [0, 1, 2]).summary()
        for key in (
            "pixel_accuracy", "mean_iou", "dice", "precision", "recall",
            "per_class_iou", "support_pixels",
        ):
            assert key in summary, f"Section 22 requires {key}"

    def test_per_class_tables_are_named(self):
        """Section 23's table is per class; positional arrays invite
        off-by-one errors in the CSV writer."""
        summary = matrix_from([0, 1], [0, 1]).summary()
        assert set(summary["per_class_iou"]) == {
            "background", "person", "car", "bicycle", "dog", "cat"
        }

    def test_absent_classes_serialize_as_null_not_zero(self):
        summary = matrix_from([0, 1], [0, 1]).summary()
        assert summary["per_class_iou"]["bicycle"] is None

    def test_support_counts_valid_pixels_only(self):
        summary = matrix_from([1, 1, IGNORE_INDEX], [1, 1, 1]).summary()
        assert summary["support_pixels"]["person"] == 2

    def test_matrix_serializes_for_the_confusion_matrix_output(self):
        listed = matrix_from([0, 1], [0, 1]).to_list()
        assert len(listed) == NUM_CLASSES
        assert len(listed[0]) == NUM_CLASSES
