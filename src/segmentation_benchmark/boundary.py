"""Section 24: boundary precision, recall and F1.

Region metrics like IoU are dominated by an object's interior. A person
occupying 40,000 pixels whose outline is wrong by three pixels all the way
around loses roughly 1,500 of them - under 4% of the IoU - while looking
visibly sloppy everywhere it matters. Section 24 asks for boundary quality
separately because that is the failure IoU is least able to see, and
Section 54 asks which model had the best boundaries, which needs a number
rather than an impression.

The measure is the standard boundary F1 (Csurka et al., 2013), stated
explicitly because there are several things "boundary F1" can mean:

**Boundary precision** is the fraction of predicted boundary pixels lying
within the tolerance of some ground-truth boundary pixel. Low precision
means the model drew edges that are not there.

**Boundary recall** is the fraction of ground-truth boundary pixels lying
within the tolerance of some predicted boundary pixel. Low recall means the
model missed edges that are.

**Boundary F1** is their harmonic mean, computed per class and macro
averaged over classes present, matching how ``metrics.py`` averages
everything else.

Section 24 requires the tolerance and evaluation resolution to be declared.
The tolerance is BOUNDARY_TOLERANCE_PX = 3 pixels of Euclidean distance,
measured at the evaluation resolution of INPUT_SIZE x INPUT_SIZE - the same
resolution the semantic metrics are computed at, so a model's IoU and its
boundary F1 describe the same prediction rather than two different
resamplings of it.

Three pixels is not arbitrary, and it is also not neutral. At 512x512 it is
roughly 0.6% of the image width, which tolerates the disagreement between
two reasonable annotators and does not tolerate a systematically thick or
shifted edge. A larger tolerance flatters every model and compresses the
differences between them; a smaller one measures annotation noise. The
figure is reported alongside every boundary number so a reader can see
which knob produced it.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from ._config import BOUNDARY_TOLERANCE_PX, CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


def extract_boundary(binary: np.ndarray) -> np.ndarray:
    """One-pixel-wide inner boundary of a binary region.

    Computed as the region minus its erosion, which places the boundary on
    the object's own pixels rather than straddling the edge. The alternative,
    a dilation-minus-region outer boundary, sits on background pixels and
    would make a prediction that is uniformly one pixel too large score
    perfectly - exactly the error this metric exists to catch.

    Erosion is done by shifting rather than with a morphology call: a pixel
    is interior when all four of its orthogonal neighbors are inside the
    region, which is the 4-connected erosion and is measurably faster over
    the six class masks of a 512x512 image, repeated across 1,000 test
    images and ten architectures.
    """
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)

    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]   # up
        & padded[2:, 1:-1]    # down
        & padded[1:-1, :-2]   # left
        & padded[1:-1, 2:]    # right
    )
    return binary & ~interior


class BoundaryAccumulator:
    """Accumulate boundary matches per class across a whole test set.

    Counts are summed and the metric is computed once at the end, for the
    same reason ``ConfusionMatrix`` does it: averaging per-image boundary F1
    weights an image containing one small cat as heavily as an image
    containing eight people, and the dataset-level figure is what the
    Section 27 table reports.

    Images where a class has no boundary in either the prediction or the
    ground truth contribute nothing rather than a zero. A model cannot be
    credited or penalized for the boundary of an object that is not in the
    picture.
    """

    def __init__(self, tolerance: int = BOUNDARY_TOLERANCE_PX,
                 num_classes: int = NUM_CLASSES,
                 ignore_index: int = IGNORE_INDEX):
        self.tolerance = tolerance
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.matched_prediction = np.zeros(num_classes, dtype=np.int64)
        self.total_prediction = np.zeros(num_classes, dtype=np.int64)
        self.matched_truth = np.zeros(num_classes, dtype=np.int64)
        self.total_truth = np.zeros(num_classes, dtype=np.int64)
        self.images = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> "BoundaryAccumulator":
        """Add one image.

        Ignored pixels are handled by excluding boundary pixels that fall in
        an ignored region. A crowd region's edge is not a ground-truth
        object boundary - COCO never drew it - so scoring a model for
        matching or missing it would measure agreement with an annotation
        that does not exist.
        """
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        valid = target != self.ignore_index

        for class_id in range(self.num_classes):
            truth_region = (target == class_id) & valid
            predicted_region = (prediction == class_id) & valid

            truth_boundary = extract_boundary(truth_region) & valid
            predicted_boundary = extract_boundary(predicted_region) & valid

            has_truth = truth_boundary.any()
            has_prediction = predicted_boundary.any()
            if not has_truth and not has_prediction:
                continue

            # distance_transform_edt measures distance to the nearest zero,
            # so it is fed the complement of the boundary to get "distance
            # to the nearest boundary pixel".
            if has_truth:
                distance_to_truth = distance_transform_edt(~truth_boundary)
            else:
                distance_to_truth = None

            if has_prediction:
                distance_to_prediction = distance_transform_edt(~predicted_boundary)
            else:
                distance_to_prediction = None

            if has_prediction:
                self.total_prediction[class_id] += int(predicted_boundary.sum())
                if distance_to_truth is not None:
                    within = distance_to_truth[predicted_boundary] <= self.tolerance
                    self.matched_prediction[class_id] += int(within.sum())
                # With no ground-truth boundary at all, every predicted
                # boundary pixel is unmatched, which the zero increment
                # already represents.

            if has_truth:
                self.total_truth[class_id] += int(truth_boundary.sum())
                if distance_to_prediction is not None:
                    within = distance_to_prediction[truth_boundary] <= self.tolerance
                    self.matched_truth[class_id] += int(within.sum())

        self.images += 1
        return self

    def per_class_precision(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                self.total_prediction > 0,
                self.matched_prediction / self.total_prediction,
                np.nan,
            )

    def per_class_recall(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                self.total_truth > 0,
                self.matched_truth / self.total_truth,
                np.nan,
            )

    def per_class_f1(self) -> np.ndarray:
        precision = self.per_class_precision()
        recall = self.per_class_recall()
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = 2 * precision * recall / (precision + recall)
        # A class with boundaries on one side only scores zero rather than
        # NaN: the model did produce a definite, definitely wrong answer.
        both_absent = np.isnan(precision) & np.isnan(recall)
        f1 = np.where(both_absent, np.nan, np.nan_to_num(f1, nan=0.0))
        return f1

    def summary(self) -> dict:
        precision = self.per_class_precision()
        recall = self.per_class_recall()
        f1 = self.per_class_f1()

        def named(values):
            return {
                name: (None if np.isnan(value) else float(value))
                for name, value in zip(CLASS_NAMES, values)
            }

        return {
            "boundary_precision": float(np.nanmean(precision)),
            "boundary_recall": float(np.nanmean(recall)),
            "boundary_f1": float(np.nanmean(f1)),
            "boundary_f1_foreground": float(np.nanmean(f1[1:])),
            "per_class_boundary_precision": named(precision),
            "per_class_boundary_recall": named(recall),
            "per_class_boundary_f1": named(f1),
            "boundary_truth_pixels": {
                name: int(value)
                for name, value in zip(CLASS_NAMES, self.total_truth)
            },
            "images": self.images,
            "conventions": {
                "definition": "Csurka et al. 2013 boundary F1",
                "tolerance_px": self.tolerance,
                "evaluation_resolution": (
                    "same as the semantic metrics, so IoU and boundary F1 "
                    "describe one prediction rather than two resamplings"
                ),
                "boundary": "inner: region minus its 4-connected erosion",
                "distance": "Euclidean, via exact distance transform",
                "ignored_pixels": "boundary pixels inside ignored regions excluded",
                "absent_classes": "excluded from the mean, not scored zero",
                "averaging": "macro over classes present",
            },
        }


def evaluate_boundary(model, loader, device,
                      tolerance: int = BOUNDARY_TOLERANCE_PX) -> dict:
    """Run a semantic model over a loader and return boundary metrics.

    Separate from ``evaluate_semantic`` rather than folded into it. The
    distance transform is the expensive part of this benchmark's evaluation
    - six per image, on top of six more for the prediction - and charging
    every per-epoch validation pass for it would add more time than the
    training step for the smaller models. Boundary quality is a property of
    the final selected checkpoint, so it is measured once, at the end.
    """
    import torch

    model.eval()
    accumulator = BoundaryAccumulator(tolerance=tolerance)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, dict):
                logits = logits["out"]
            predictions = logits.argmax(dim=1).cpu().numpy()
            truths = targets.numpy()

            for prediction, truth in zip(predictions, truths):
                accumulator.update(prediction, truth)

    return accumulator.summary(), accumulator
