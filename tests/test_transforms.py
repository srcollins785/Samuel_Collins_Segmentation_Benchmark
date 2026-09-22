"""Tests for mask-safe augmentation.

These run on synthetic samples rather than the COCO subset, so they pass on a
clean checkout before anything has been downloaded. The equivalent checks
against real images live in scripts/verify_transforms.py, which is what
Section 15 asks to be run before a full benchmark.
"""

import numpy as np
import pytest
import torch

from segmentation_benchmark import transforms as T
from segmentation_benchmark._config import IGNORE_INDEX


def make_sample(size=64, with_instances=True):
    """A sample whose mask is deliberately asymmetric, so a flip is detectable."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Off-center block: a symmetric shape would pass a broken flip.
    image[10:30, 5:20] = (220, 20, 60)
    mask[10:30, 5:20] = 1
    image[40:55, 35:60] = (0, 130, 200)
    mask[40:55, 35:60] = 2

    sample = {"image": image, "mask": mask}
    if with_instances:
        first = np.zeros((size, size), dtype=bool)
        first[10:30, 5:20] = True
        second = np.zeros((size, size), dtype=bool)
        second[40:55, 35:60] = True
        sample.update({
            "masks": np.stack([first, second]),
            "labels": np.array([1, 2], dtype=np.int64),
            "boxes": np.array([[5, 10, 20, 30], [35, 40, 60, 55]], dtype=np.float32),
            "areas": np.array([20 * 15, 15 * 25], dtype=np.float32),
            "instance_ids": np.array([101, 102], dtype=np.int64),
        })
    return sample


class TestHorizontalFlip:
    def test_mirrors_image_and_mask_together(self):
        sample = make_sample()
        expected_image = sample["image"][:, ::-1].copy()
        expected_mask = sample["mask"][:, ::-1].copy()

        out = T.RandomHorizontalFlip(probability=1.0)(sample)

        assert np.array_equal(out["image"], expected_image)
        assert np.array_equal(out["mask"], expected_mask)

    def test_leaves_sample_untouched_at_zero_probability(self):
        sample = make_sample()
        original = sample["mask"].copy()
        out = T.RandomHorizontalFlip(probability=0.0)(sample)
        assert np.array_equal(out["mask"], original)

    def test_boxes_are_mirrored(self):
        sample = make_sample()
        width = sample["mask"].shape[1]
        before = sample["boxes"].copy()

        out = T.RandomHorizontalFlip(probability=1.0)(sample)

        for old, new in zip(before, out["boxes"]):
            assert new[0] == pytest.approx(width - old[2], abs=1.0)
            assert new[2] == pytest.approx(width - old[0], abs=1.0)
            assert new[1] == pytest.approx(old[1], abs=1.0)
            assert new[3] == pytest.approx(old[3], abs=1.0)

    def test_instance_masks_mirror_with_the_image(self):
        sample = make_sample()
        expected = sample["masks"][:, :, ::-1].copy()
        out = T.RandomHorizontalFlip(probability=1.0)(sample)
        assert np.array_equal(out["masks"], expected)


class TestResize:
    def test_mask_never_gains_an_unannotated_class(self):
        """Nearest neighbor cannot interpolate class 1 and 3 into class 2."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:, :32] = 1
        mask[:, 32:] = 3
        sample = {"image": np.zeros((64, 64, 3), np.uint8), "mask": mask}

        out = T.Resize(37)(sample)  # deliberately not a clean ratio

        assert set(np.unique(out["mask"]).tolist()) <= {1, 3}

    def test_preserves_the_ignore_sentinel(self):
        mask = np.full((64, 64), IGNORE_INDEX, dtype=np.uint8)
        mask[20:40, 20:40] = 1
        sample = {"image": np.zeros((64, 64, 3), np.uint8), "mask": mask}

        out = T.Resize(128)(sample)

        assert set(np.unique(out["mask"]).tolist()) == {1, IGNORE_INDEX}

    def test_resizes_every_field_to_the_target(self):
        out = T.Resize(32)(make_sample(size=64))
        assert out["image"].shape == (32, 32, 3)
        assert out["mask"].shape == (32, 32)
        assert out["masks"].shape[1:] == (32, 32)


class TestRotation:
    def test_exposed_canvas_is_ignored_not_background(self):
        """The corner a rotation invents must not enter the loss as background."""
        sample = make_sample(size=64, with_instances=False)
        sample["mask"][:] = 1  # every real pixel is a class

        out = T.RandomRotation(degrees=30.0, probability=1.0)(sample)

        assert (out["mask"] == IGNORE_INDEX).any(), "no ignore fill after rotation"
        assert not (out["mask"] == 0).any(), "invented canvas labeled background"

    def test_instance_masks_do_not_gain_invented_area(self):
        """Rotation may clip an object away; it must not grow one.

        The tolerance is a few percent rather than exact. Rotating a mask by
        nearest neighbor resamples an axis-aligned rectangle onto a diagonal
        footprint, and aliasing along the new edges moves the pixel count by
        a handful in either direction. What would signal a real fault is the
        count climbing substantially, which is what happens if the fill value
        for exposed canvas is wrong.
        """
        sample = make_sample()
        before = sample["masks"].sum()
        out = T.RandomRotation(degrees=25.0, probability=1.0)(sample)
        assert out["masks"].sum() <= before * 1.05


class TestInstanceAlignment:
    def test_boxes_bound_their_masks_after_geometry(self):
        pipeline = T.Compose([
            T.RandomHorizontalFlip(1.0),
            T.RandomRotation(20.0, 1.0),
        ])
        out = pipeline(make_sample())

        for binary, box in zip(out["masks"], out["boxes"]):
            rows = np.where(binary.any(axis=1))[0]
            cols = np.where(binary.any(axis=0))[0]
            assert box.tolist() == [
                float(cols[0]), float(rows[0]),
                float(cols[-1] + 1), float(rows[-1] + 1),
            ]

    def test_arrays_stay_the_same_length(self):
        pipeline = T.Compose([
            T.RandomScaleCrop(32, (0.5, 0.7), 1.0),
            T.RandomRotation(30.0, 1.0),
        ])
        out = pipeline(make_sample())

        lengths = {
            len(out["masks"]), len(out["labels"]), len(out["boxes"]),
            len(out["areas"]), len(out["instance_ids"]),
        }
        assert len(lengths) == 1, f"instance arrays desynced: {lengths}"

    def test_a_vanished_instance_is_dropped_from_every_array(self):
        """Cropping an object away must remove it everywhere, not just from masks."""
        sample = make_sample(size=64)
        # Erase the second instance, simulating a crop that cut it away.
        sample["masks"][1][:] = False

        out = T.Resize(64)(sample)

        assert len(out["masks"]) == 1
        assert len(out["labels"]) == 1
        assert out["labels"][0] == 1
        assert out["instance_ids"][0] == 101


class TestColorJitter:
    def test_does_not_move_a_single_label(self):
        sample = make_sample()
        mask_before = sample["mask"].copy()
        masks_before = sample["masks"].copy()

        out = T.ColorJitter(probability=1.0)(sample)

        assert np.array_equal(out["mask"], mask_before)
        assert np.array_equal(out["masks"], masks_before)

    def test_actually_changes_the_image(self):
        sample = make_sample()
        before = sample["image"].copy()
        out = T.ColorJitter(0.5, 0.5, 0.5, 0.1, probability=1.0)(sample)
        assert not np.array_equal(out["image"], before)


class TestToTensor:
    def test_label_values_are_not_rescaled(self):
        """A class ID is a label; scaling it would collide with the sentinel."""
        sample = make_sample(size=32, with_instances=False)
        sample["mask"][0, 0] = IGNORE_INDEX

        out = T.ToTensor()(sample)

        assert out["mask"].dtype == torch.int64
        assert out["mask"][0, 0].item() == IGNORE_INDEX
        assert set(out["mask"].unique().tolist()) <= {0, 1, 2, IGNORE_INDEX}

    def test_image_is_normalized(self):
        out = T.ToTensor()(make_sample(size=32, with_instances=False))
        assert out["image"].dtype == torch.float32
        assert out["image"].shape == (3, 32, 32)
        # ImageNet normalization pushes a zeroed pixel well below zero.
        assert out["image"].min() < -1.0


class TestPipelines:
    def test_eval_pipeline_is_deterministic(self):
        """Two passes over one sample must be identical, or test is not held out."""
        pipeline = T.build_eval_transforms(48)
        first = pipeline(make_sample(size=64, with_instances=False))
        second = pipeline(make_sample(size=64, with_instances=False))
        assert torch.equal(first["image"], second["image"])
        assert torch.equal(first["mask"], second["mask"])

    def test_train_pipeline_is_not_deterministic(self):
        import random
        pipeline = T.build_train_transforms(48)

        random.seed(0)
        first = pipeline(make_sample(size=64, with_instances=False))
        random.seed(7)
        second = pipeline(make_sample(size=64, with_instances=False))

        assert not torch.equal(first["image"], second["image"])

    def test_describe_records_every_step_in_order(self):
        """Section 14 requires the exact transform order to be recoverable."""
        described = T.build_train_transforms(64).describe()
        names = [step["transform"] for step in described]
        assert names == [
            "Resize", "RandomHorizontalFlip", "RandomScaleCrop",
            "RandomRotation", "ColorJitter", "ToTensor+Normalize",
        ]

    def test_describe_records_that_jitter_is_image_only(self):
        described = T.build_train_transforms(64).describe()
        jitter = next(s for s in described if s["transform"] == "ColorJitter")
        assert jitter["applies_to"] == "image only"
