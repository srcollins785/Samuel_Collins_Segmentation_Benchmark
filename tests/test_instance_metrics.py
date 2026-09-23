"""Tests for the Section 25 instance metrics and the Section 26 separation.

These build a small COCO ground truth in memory rather than reading the
subset, so the suite runs before anything has been downloaded.
"""

import contextlib
import io
import json

import numpy as np
import pytest
import torch
from pycocotools.coco import COCO

from segmentation_benchmark.instance_metrics import (
    CONVENTIONS, STAT_NAMES, detections_to_coco, encode_mask,
    instance_to_semantic, summarize_instance_results,
)


def build_ground_truth(tmp_path, boxes_by_image):
    """A minimal COCO ground truth: one 100x100 image per entry."""
    images, annotations = [], []
    annotation_id = 1
    for image_id, boxes in boxes_by_image.items():
        images.append({
            "id": image_id, "file_name": f"{image_id}.jpg",
            "height": 100, "width": 100,
        })
        for (x1, y1, x2, y2, category) in boxes:
            mask = np.zeros((100, 100), dtype=bool)
            mask[y1:y2, x1:x2] = True
            annotations.append({
                "id": annotation_id, "image_id": image_id,
                "category_id": category, "iscrowd": 0,
                "area": float(mask.sum()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": encode_mask(mask),
            })
            annotation_id += 1

    document = {
        "images": images, "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in
                       enumerate(["person", "car", "bicycle", "dog", "cat"], 1)],
    }
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(document))
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def detection(x1, y1, x2, y2, category, score=1.0, size=100):
    mask = np.zeros((size, size), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return {
        "image_id": None, "category_id": category,
        "segmentation": encode_mask(mask), "score": score,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
    }


class TestEncoding:
    def test_round_trips(self):
        from pycocotools import mask as coco_mask
        original = np.zeros((40, 60), dtype=bool)
        original[5:20, 10:50] = True

        decoded = coco_mask.decode(encode_mask(original)).astype(bool)

        assert np.array_equal(decoded, original)

    def test_counts_is_a_string_not_bytes(self):
        """loadRes serializes the results; bytes are not JSON-encodable."""
        encoded = encode_mask(np.ones((10, 10), dtype=bool))
        assert isinstance(encoded["counts"], str)


class TestOracle:
    def test_ground_truth_as_predictions_scores_one(self, tmp_path):
        """The strongest available check on the whole evaluation path."""
        coco = build_ground_truth(tmp_path, {
            1: [(10, 10, 40, 40, 1), (50, 50, 80, 80, 2)],
            2: [(20, 20, 60, 60, 1)],
        })

        results = []
        for image_id in coco.imgs:
            for annotation in coco.loadAnns(coco.getAnnIds(imgIds=image_id)):
                record = detection(0, 0, 1, 1, annotation["category_id"])
                record["segmentation"] = annotation["segmentation"]
                record["bbox"] = annotation["bbox"]
                record["image_id"] = image_id
                results.append(record)

        summary = summarize_instance_results(coco, results, list(coco.imgs))

        assert summary["mask_ap"] == pytest.approx(1.0)
        assert summary["mask_ap50"] == pytest.approx(1.0)
        assert summary["mask_ap75"] == pytest.approx(1.0)

    def test_no_detections_scores_zero_without_crashing(self, tmp_path):
        """An undertrained model really can predict nothing; that is a
        result, not an error that should lose the rest of the benchmark."""
        coco = build_ground_truth(tmp_path, {1: [(10, 10, 40, 40, 1)]})

        summary = summarize_instance_results(coco, [], [1])

        assert summary["mask_ap"] == 0.0
        assert summary["detections"] == 0

    def test_wrong_class_scores_zero(self, tmp_path):
        coco = build_ground_truth(tmp_path, {1: [(10, 10, 40, 40, 1)]})
        record = detection(10, 10, 40, 40, category=4)
        record["image_id"] = 1

        summary = summarize_instance_results(coco, [record], [1])

        assert summary["mask_ap"] == pytest.approx(0.0)

    def test_ap50_is_more_forgiving_than_ap75(self, tmp_path):
        """A loose but overlapping mask should clear 0.50 and miss 0.75."""
        coco = build_ground_truth(tmp_path, {1: [(20, 20, 60, 60, 1)]})
        record = detection(20, 20, 70, 70, category=1)  # IoU ~ 0.64
        record["image_id"] = 1

        summary = summarize_instance_results(coco, [record], [1])

        assert summary["mask_ap50"] > summary["mask_ap75"]


class TestConventions:
    def test_stat_names_match_cocoeval_length(self):
        assert len(STAT_NAMES) == 12

    def test_every_required_convention_is_declared(self):
        for key in ("evaluator", "class_averaging", "score_threshold",
                    "max_detections", "resolution", "crowd_regions"):
            assert key in CONVENTIONS, f"Section 25 requires {key} stated"

    def test_summary_reports_the_required_metrics(self, tmp_path):
        coco = build_ground_truth(tmp_path, {1: [(10, 10, 40, 40, 1)]})
        record = detection(10, 10, 40, 40, category=1)
        record["image_id"] = 1

        summary = summarize_instance_results(coco, [record], [1])

        for key in ("mask_ap", "mask_ap50", "mask_ap75", "mask_ar_100", "box_ap"):
            assert key in summary, f"Section 28 table needs {key}"

    def test_small_medium_large_breakdown_is_reported(self, tmp_path):
        """Section 54 asks which architecture did best on small objects."""
        coco = build_ground_truth(tmp_path, {1: [(10, 10, 40, 40, 1)]})
        record = detection(10, 10, 40, 40, category=1)
        record["image_id"] = 1

        summary = summarize_instance_results(coco, [record], [1])

        for key in ("mask_ap_small", "mask_ap_medium", "mask_ap_large"):
            assert key in summary


class TestDetectionConversion:
    def make_output(self, size=64, score=0.9):
        mask = torch.zeros(1, 1, size, size)
        mask[0, 0, 10:30, 10:30] = 1.0
        return {
            "scores": torch.tensor([score]),
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            "masks": mask,
        }

    def test_filters_below_the_score_threshold(self):
        output = self.make_output(score=0.01)
        results = detections_to_coco(output, image_id=1, score_threshold=0.05)
        assert results == []

    def test_keeps_the_highest_scoring_when_truncating(self):
        """Truncating before sorting would keep an arbitrary subset."""
        count = 5
        output = {
            "scores": torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3]),
            "labels": torch.ones(count, dtype=torch.long),
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]] * count),
            "masks": torch.ones(count, 1, 32, 32),
        }

        results = detections_to_coco(output, 1, max_detections=2)

        assert [r["score"] for r in results] == [pytest.approx(0.9),
                                                 pytest.approx(0.8)]

    def test_emits_coco_style_xywh_boxes(self):
        results = detections_to_coco(self.make_output(), image_id=7)
        assert results[0]["bbox"] == [10.0, 10.0, 20.0, 20.0]
        assert results[0]["image_id"] == 7

    def test_rescales_masks_and_boxes_to_native_resolution(self):
        results = detections_to_coco(
            self.make_output(size=64), image_id=1, native_size=(128, 256)
        )

        from pycocotools import mask as coco_mask
        decoded = coco_mask.decode(results[0]["segmentation"])
        assert decoded.shape == (128, 256)
        # 64 -> 256 wide is 4x; the box must scale with the mask.
        assert results[0]["bbox"][0] == pytest.approx(40.0)


class TestSemanticConversion:
    """Section 26: the optional conversion, and its documented settings."""

    def make_two_overlapping(self):
        masks = torch.zeros(2, 1, 64, 64)
        masks[0, 0, 10:40, 10:40] = 1.0   # lower score
        masks[1, 0, 20:50, 20:50] = 1.0   # higher score, overlaps
        return {
            "scores": torch.tensor([0.6, 0.95]),
            "labels": torch.tensor([1, 4]),
            "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0],
                                   [20.0, 20.0, 50.0, 50.0]]),
            "masks": masks,
        }

    def test_highest_score_wins_an_overlap(self):
        semantic = instance_to_semantic(
            self.make_two_overlapping(), 64, 64, score_threshold=0.5
        )
        assert semantic[30, 30] == 4, "lower-scoring instance won the overlap"
        assert semantic[15, 15] == 1

    def test_uncovered_pixels_are_background(self):
        semantic = instance_to_semantic(
            self.make_two_overlapping(), 64, 64, score_threshold=0.5
        )
        assert semantic[60, 60] == 0

    def test_threshold_removes_low_confidence_instances(self):
        semantic = instance_to_semantic(
            self.make_two_overlapping(), 64, 64, score_threshold=0.9
        )
        assert semantic[15, 15] == 0, "instance below threshold was painted"
        assert semantic[30, 30] == 4
