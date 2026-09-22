"""Section 22 semantic metrics, all derived from one confusion matrix.

Section 22 requires four conventions to be declared and held to throughout.
They are declared here, once, and every metric in the benchmark reads them
from this module:

**Which pixels count.** Pixels equal to IGNORE_INDEX are excluded before the
matrix is accumulated. They are crowd regions COCO declined to annotate and
canvas that augmentation invented, and they enter neither the numerator nor
the denominator of anything.

**Whether background is included.** Both. ``mean_iou`` covers all six
classes including background, and ``mean_iou_foreground`` covers the five
object classes only. Reporting only the first would flatter every model,
because background is 83% of pixels and trivially easy; reporting only the
second would discard a real prediction the models make. The report gives
both and says which is which, and the required CSV column ``mIoU`` is the
six-class figure.

**How absent classes are handled.** A class with no ground-truth pixels and
no predicted pixels has an undefined IoU - 0/0 - and is excluded from the
mean rather than scored zero or one. Scoring it zero would punish a model
for a class the test set never showed it; scoring it one would reward it.
Over a 1,000-image test set every class is present, so this matters only for
per-image or per-batch reporting, where it matters a lot.

**Averaging rule for Dice and F1.** Macro: computed per class, then
averaged, with the same present-class rule as IoU. Section 22 notes that
Dice and pixel F1 are the same quantity on the same counts, so one function
computes it and ``dice`` and ``pixel_f1`` are names for that function's
output rather than two separate calculations that could drift apart.
"""

import numpy as np
import torch

from ._config import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


class ConfusionMatrix:
    """Accumulate predictions against ground truth, one batch at a time.

    A confusion matrix rather than running metric averages, because the two
    are not the same and only one is right. Averaging per-batch IoU over
    batches weights every batch equally regardless of how many pixels of a
    class it held, so a batch containing four bicycle pixels influences the
    bicycle IoU as much as one containing forty thousand. Accumulating counts
    and computing the metric once at the end is the dataset-level figure the
    assignment's tables ask for.
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 ignore_index: int = IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction, target) -> "ConfusionMatrix":
        """Add a batch. Rows are ground truth, columns are prediction."""
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        prediction = np.asarray(prediction).reshape(-1)
        target = np.asarray(target).reshape(-1)

        valid = (target != self.ignore_index) & (target < self.num_classes)
        target = target[valid]
        prediction = prediction[valid]

        # bincount over a flattened (truth, prediction) index is materially
        # faster than np.add.at over 1.8 million pixels per 512x512 batch of
        # eight, which matters when it runs every validation epoch for every
        # one of seven architectures.
        flat = target.astype(np.int64) * self.num_classes + prediction.astype(np.int64)
        counts = np.bincount(flat, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)
        return self

    def merge(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        self.matrix += other.matrix
        return self

    # -- counts ------------------------------------------------------------

    @property
    def true_positives(self) -> np.ndarray:
        return np.diag(self.matrix).astype(np.float64)

    @property
    def false_positives(self) -> np.ndarray:
        return self.matrix.sum(axis=0).astype(np.float64) - self.true_positives

    @property
    def false_negatives(self) -> np.ndarray:
        return self.matrix.sum(axis=1).astype(np.float64) - self.true_positives

    @property
    def support(self) -> np.ndarray:
        """Ground-truth pixels per class."""
        return self.matrix.sum(axis=1).astype(np.float64)

    @property
    def present(self) -> np.ndarray:
        """Classes the evaluation actually saw, in truth or in prediction.

        A class absent from both is undefined, not zero. See the module
        docstring.
        """
        return (self.matrix.sum(axis=1) + self.matrix.sum(axis=0)) > 0

    # -- metrics -----------------------------------------------------------

    def pixel_accuracy(self) -> float:
        """Correctly classified valid pixels over all valid pixels."""
        total = self.matrix.sum()
        return float(self.true_positives.sum() / total) if total else 0.0

    def per_class_iou(self) -> np.ndarray:
        """TP / (TP + FP + FN) per class. NaN where the class is absent."""
        denominator = (
            self.true_positives + self.false_positives + self.false_negatives
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(denominator > 0, self.true_positives / denominator, np.nan)
        return iou

    def mean_iou(self) -> float:
        """Mean over all classes present, background included."""
        return float(np.nanmean(self.per_class_iou()))

    def mean_iou_foreground(self) -> float:
        """Mean over the five object classes, background excluded."""
        return float(np.nanmean(self.per_class_iou()[1:]))

    def per_class_dice(self) -> np.ndarray:
        """2TP / (2TP + FP + FN) per class. Identical to per-class pixel F1."""
        denominator = (
            2 * self.true_positives + self.false_positives + self.false_negatives
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                denominator > 0, 2 * self.true_positives / denominator, np.nan
            )

    def dice(self) -> float:
        """Macro Dice over present classes. Section 22's Dice / pixel F1."""
        return float(np.nanmean(self.per_class_dice()))

    def per_class_precision(self) -> np.ndarray:
        denominator = self.true_positives + self.false_positives
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                denominator > 0, self.true_positives / denominator, np.nan
            )

    def per_class_recall(self) -> np.ndarray:
        denominator = self.true_positives + self.false_negatives
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                denominator > 0, self.true_positives / denominator, np.nan
            )

    def precision(self) -> float:
        return float(np.nanmean(self.per_class_precision()))

    def recall(self) -> float:
        return float(np.nanmean(self.per_class_recall()))

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict:
        """Every Section 22 metric, plus the per-class table Section 23 needs."""
        iou = self.per_class_iou()
        dice = self.per_class_dice()
        precision = self.per_class_precision()
        recall = self.per_class_recall()

        def named(values):
            return {
                name: (None if np.isnan(value) else float(value))
                for name, value in zip(CLASS_NAMES, values)
            }

        return {
            "pixel_accuracy": self.pixel_accuracy(),
            "mean_iou": self.mean_iou(),
            "mean_iou_foreground": self.mean_iou_foreground(),
            "dice": self.dice(),
            "pixel_f1": self.dice(),  # same quantity, Section 22
            "precision": self.precision(),
            "recall": self.recall(),
            "per_class_iou": named(iou),
            "per_class_dice": named(dice),
            "per_class_precision": named(precision),
            "per_class_recall": named(recall),
            "support_pixels": {
                name: int(value)
                for name, value in zip(CLASS_NAMES, self.support)
            },
            "conventions": CONVENTIONS,
        }

    def to_list(self) -> list:
        """The raw matrix, for the Section 52 confusion_matrices/ output."""
        return self.matrix.tolist()


# Recorded into every set of results, so a CSV can be read years later
# without having to infer which convention produced it.
CONVENTIONS = {
    "ignored_pixels": f"target == {IGNORE_INDEX} excluded before accumulation",
    "background": "included in mean_iou; excluded from mean_iou_foreground",
    "absent_classes": "undefined (0/0) and excluded from means, not scored 0",
    "averaging": "macro over present classes",
    "dice_equals_f1": "same counts, same averaging (Section 22)",
    "reported_as": "fractions in [0, 1], not percentages",
}


def evaluate_semantic(model, loader, device, max_batches: int = None) -> dict:
    """Run a semantic model over a loader and return the Section 22 metrics.

    Used for the per-epoch validation score that Section 21 selects
    checkpoints on, and for the final test evaluation. One function for both,
    so the number a checkpoint was chosen by and the number reported are
    computed by identical code.
    """
    model.eval()
    matrix = ConfusionMatrix()

    with torch.no_grad():
        for index, (images, targets) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, dict):
                logits = logits["out"]
            matrix.update(logits.argmax(dim=1), targets)

    return matrix.summary(), matrix
