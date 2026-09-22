"""Tests for the Section 8 and Section 9 annotation conversion.

The policies these check are the ones the assignment asks to be documented
and applied consistently: what happens where annotations overlap, what
happens to crowd regions, and what happens to the 75 COCO categories outside
our five.
"""

import numpy as np

from segmentation_benchmark import mask_utils
from segmentation_benchmark._config import IGNORE_INDEX


def polygon_annotation(x1, y1, x2, y2, category_id, annotation_id=1, crowd=0):
    """A rectangular COCO polygon annotation."""
    return {
        "id": annotation_id,
        "category_id": category_id,
        "iscrowd": crowd,
        "area": float((x2 - x1) * (y2 - y1)),
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
    }


class TestSemanticMask:
    def test_maps_coco_categories_to_contiguous_ids(self):
        # COCO: person 1, bicycle 2, car 3, cat 17, dog 18.
        annotations = [
            polygon_annotation(0, 0, 10, 10, 1, 1),    # person -> 1
            polygon_annotation(20, 0, 30, 10, 3, 2),   # car -> 2
            polygon_annotation(40, 0, 50, 10, 2, 3),   # bicycle -> 3
            polygon_annotation(60, 0, 70, 10, 18, 4),  # dog -> 4
            polygon_annotation(80, 0, 90, 10, 17, 5),  # cat -> 5
        ]
        mask = mask_utils.semantic_mask(annotations, 100, 100)

        assert mask[5, 5] == 1
        assert mask[5, 25] == 2
        assert mask[5, 45] == 3
        assert mask[5, 65] == 4
        assert mask[5, 85] == 5

    def test_unselected_categories_become_background(self):
        # 62 is 'chair', not one of the five.
        annotations = [polygon_annotation(0, 0, 50, 50, 62, 1)]
        mask = mask_utils.semantic_mask(annotations, 100, 100)
        assert (mask == 0).all()

    def test_smaller_instance_wins_an_overlap(self):
        """A cat on a sofa-sized person must keep its own pixels."""
        big = polygon_annotation(0, 0, 100, 100, 1, 1)     # person, area 10000
        small = polygon_annotation(20, 20, 40, 40, 17, 2)  # cat, area 400

        # Order in the annotation list must not matter; the policy is by area.
        for annotations in ([big, small], [small, big]):
            mask = mask_utils.semantic_mask(annotations, 100, 100)
            assert mask[30, 30] == 5, "smaller instance lost the overlap"
            assert mask[5, 5] == 1, "larger instance lost its own pixels"

    def test_crowd_regions_become_ignore_not_background(self):
        annotations = [polygon_annotation(0, 0, 50, 50, 1, 1, crowd=1)]
        mask = mask_utils.semantic_mask(annotations, 100, 100)

        assert mask[10, 10] == IGNORE_INDEX
        assert mask[80, 80] == 0

    def test_an_annotated_object_inside_a_crowd_keeps_its_label(self):
        """Crowds are painted first so an individual annotation survives."""
        crowd = polygon_annotation(0, 0, 100, 100, 1, 1, crowd=1)
        individual = polygon_annotation(40, 40, 60, 60, 17, 2)

        mask = mask_utils.semantic_mask([crowd, individual], 100, 100)

        assert mask[50, 50] == 5
        assert mask[10, 10] == IGNORE_INDEX

    def test_output_dtype_and_shape(self):
        mask = mask_utils.semantic_mask([], 43, 71)
        assert mask.dtype == np.uint8
        assert mask.shape == (43, 71)


class TestInstanceTargets:
    def test_arrays_are_aligned(self):
        annotations = [
            polygon_annotation(0, 0, 10, 10, 1, 1),
            polygon_annotation(20, 20, 40, 40, 17, 2),
        ]
        targets = mask_utils.instance_targets(annotations, 100, 100)

        assert len(targets["masks"]) == 2
        assert len(targets["labels"]) == 2
        assert len(targets["boxes"]) == 2
        assert len(targets["instance_ids"]) == 2
        assert targets["labels"].tolist() == [1, 5]
        assert targets["instance_ids"].tolist() == [1, 2]

    def test_crowd_annotations_are_excluded(self):
        annotations = [
            polygon_annotation(0, 0, 10, 10, 1, 1),
            polygon_annotation(20, 20, 40, 40, 1, 2, crowd=1),
        ]
        targets = mask_utils.instance_targets(annotations, 100, 100)

        assert len(targets["labels"]) == 1
        assert targets["instance_ids"].tolist() == [1]

    def test_overlapping_instances_both_keep_full_masks(self):
        """Unlike the semantic mask, instances do not overwrite each other."""
        annotations = [
            polygon_annotation(0, 0, 100, 100, 1, 1),
            polygon_annotation(20, 20, 40, 40, 17, 2),
        ]
        targets = mask_utils.instance_targets(annotations, 100, 100)

        assert targets["masks"][0][30, 30], "large instance lost overlapped pixels"
        assert targets["masks"][1][30, 30]

    def test_boxes_bound_their_masks(self):
        annotations = [polygon_annotation(10, 20, 30, 50, 1, 1)]
        targets = mask_utils.instance_targets(annotations, 100, 100)

        box = targets["boxes"][0]
        binary = targets["masks"][0]
        rows = np.where(binary.any(axis=1))[0]
        cols = np.where(binary.any(axis=0))[0]
        assert box.tolist() == [
            float(cols[0]), float(rows[0]),
            float(cols[-1] + 1), float(rows[-1] + 1),
        ]

    def test_empty_annotations_give_well_shaped_empties(self):
        targets = mask_utils.instance_targets([], 64, 64)

        assert targets["masks"].shape == (0, 64, 64)
        assert targets["boxes"].shape == (0, 4)
        assert len(targets["labels"]) == 0

    def test_ordering_is_stable(self):
        """Sorted by annotation ID, so two runs pair the same instance IDs."""
        annotations = [
            polygon_annotation(0, 0, 10, 10, 1, 9),
            polygon_annotation(20, 20, 30, 30, 17, 3),
        ]
        targets = mask_utils.instance_targets(annotations, 100, 100)
        assert targets["instance_ids"].tolist() == [3, 9]


class TestPixelCounts:
    def test_ignored_pixels_are_counted_in_no_class(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 1
        mask[5:, :] = IGNORE_INDEX

        counts = mask_utils.class_pixel_counts(mask)

        assert counts.sum() == 50
        assert counts[1] == 50

    def test_has_foreground_ignores_background_only_masks(self):
        empty = np.zeros((10, 10), dtype=np.uint8)
        assert not mask_utils.has_foreground(empty)

        with_person = empty.copy()
        with_person[0, 0] = 1
        assert mask_utils.has_foreground(with_person)

    def test_crowd_only_mask_has_no_foreground(self):
        """Ignore is not foreground; an all-crowd image must not qualify."""
        mask = np.full((10, 10), IGNORE_INDEX, dtype=np.uint8)
        assert not mask_utils.has_foreground(mask)
