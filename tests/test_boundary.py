"""Tests for the Section 24 boundary metrics.

The property worth pinning down is that boundary F1 measures something IoU
does not. Several tests below construct predictions where the two disagree
on purpose.
"""

import numpy as np
import pytest

from segmentation_benchmark._config import IGNORE_INDEX
from segmentation_benchmark.boundary import BoundaryAccumulator, extract_boundary
from segmentation_benchmark.metrics import ConfusionMatrix


def square(size=64, top=10, bottom=40, left=10, right=40, value=1):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[top:bottom, left:right] = value
    return mask


class TestBoundaryExtraction:
    def test_perimeter_of_a_square(self):
        region = np.zeros((20, 20), dtype=bool)
        region[5:15, 5:15] = True

        boundary = extract_boundary(region)

        # A 10x10 square has 4*10 - 4 = 36 perimeter pixels.
        assert boundary.sum() == 36

    def test_boundary_lies_on_the_object_not_outside_it(self):
        """An inner boundary. An outer one would let a prediction that is
        uniformly one pixel too large score perfectly."""
        region = np.zeros((20, 20), dtype=bool)
        region[5:15, 5:15] = True

        boundary = extract_boundary(region)

        assert (boundary & region).sum() == boundary.sum()
        assert not boundary[10, 10], "interior pixel marked as boundary"
        assert boundary[5, 5], "corner pixel not marked as boundary"

    def test_empty_region_has_no_boundary(self):
        assert extract_boundary(np.zeros((10, 10), dtype=bool)).sum() == 0

    def test_full_region_has_boundary_only_at_the_image_edge(self):
        region = np.ones((10, 10), dtype=bool)
        boundary = extract_boundary(region)
        assert boundary.sum() == 36  # the 10x10 image border


class TestBoundaryScores:
    def test_perfect_prediction_scores_one(self):
        truth = square()
        accumulator = BoundaryAccumulator(tolerance=3).update(truth.copy(), truth)
        summary = accumulator.summary()

        assert summary["boundary_precision"] == pytest.approx(1.0)
        assert summary["boundary_recall"] == pytest.approx(1.0)
        assert summary["boundary_f1"] == pytest.approx(1.0)

    def test_a_shift_inside_the_tolerance_is_forgiven(self):
        truth = square()
        shifted = square(top=12, bottom=42, left=12, right=42)

        summary = BoundaryAccumulator(tolerance=3).update(shifted, truth).summary()

        assert summary["boundary_f1"] > 0.95

    def test_a_shift_beyond_the_tolerance_is_penalized(self):
        truth = square()
        shifted = square(top=18, bottom=48, left=18, right=48)

        summary = BoundaryAccumulator(tolerance=3).update(shifted, truth).summary()

        assert summary["boundary_f1"] < 0.6

    def test_tolerance_changes_the_score(self):
        """Section 24 requires the tolerance declared because it moves the
        number; this proves the knob is real and not decorative."""
        truth = square()
        shifted = square(top=16, bottom=46, left=16, right=46)

        tight = BoundaryAccumulator(tolerance=1).update(shifted, truth).summary()
        loose = BoundaryAccumulator(tolerance=9).update(shifted, truth).summary()

        assert loose["boundary_f1"] > tight["boundary_f1"]

    def test_measures_what_iou_cannot(self):
        """A small offset barely moves IoU and moves boundary F1 a lot,
        which is the reason Section 24 asks for boundary quality at all."""
        truth = square()
        near = square(top=12, bottom=42, left=12, right=42)
        far = square(top=18, bottom=48, left=18, right=48)

        near_iou = ConfusionMatrix().update(near, truth).per_class_iou()[1]
        far_iou = ConfusionMatrix().update(far, truth).per_class_iou()[1]
        near_f1 = BoundaryAccumulator(tolerance=3).update(near, truth).summary()
        far_f1 = BoundaryAccumulator(tolerance=3).update(far, truth).summary()

        # IoU degrades gradually; boundary F1 falls off a cliff at the
        # tolerance, which is the discrimination being bought.
        assert near_iou > far_iou
        assert near_f1["boundary_f1"] - far_f1["boundary_f1"] > near_iou - far_iou

    def test_a_predicted_class_absent_from_truth_scores_zero_precision(self):
        truth = np.zeros((64, 64), dtype=np.uint8)
        prediction = square(value=3)

        summary = BoundaryAccumulator(tolerance=3).update(prediction, truth).summary()

        assert summary["per_class_boundary_precision"]["bicycle"] == pytest.approx(0.0)


class TestConventions:
    def test_ignored_regions_are_excluded(self):
        """A crowd region's edge is not an annotated object boundary."""
        truth = square()
        truth[45:60, 45:60] = IGNORE_INDEX

        prediction = square()
        prediction[45:60, 45:60] = 1  # predict person over the ignored area

        summary = BoundaryAccumulator(tolerance=3).update(prediction, truth).summary()

        assert summary["boundary_f1"] == pytest.approx(1.0), (
            "prediction inside an ignored region affected the boundary score"
        )

    def test_absent_classes_are_excluded_from_the_mean(self):
        truth = square()
        summary = BoundaryAccumulator(tolerance=3).update(truth.copy(), truth).summary()

        assert summary["per_class_boundary_f1"]["cat"] is None
        assert summary["boundary_f1"] == pytest.approx(1.0)

    def test_counts_accumulate_across_images(self):
        accumulator = BoundaryAccumulator(tolerance=3)
        truth = square()
        accumulator.update(truth.copy(), truth)
        accumulator.update(truth.copy(), truth)

        assert accumulator.summary()["images"] == 2
        assert accumulator.total_truth[1] > 0

    def test_conventions_are_recorded(self):
        summary = BoundaryAccumulator(tolerance=3).update(square(), square()).summary()
        conventions = summary["conventions"]

        assert conventions["tolerance_px"] == 3
        for key in ("definition", "boundary", "distance", "ignored_pixels",
                    "evaluation_resolution"):
            assert key in conventions, f"Section 24 requires {key} declared"

    def test_reports_foreground_only_variant(self):
        summary = BoundaryAccumulator(tolerance=3).update(square(), square()).summary()
        assert "boundary_f1_foreground" in summary
