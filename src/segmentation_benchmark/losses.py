"""The shared semantic loss: 0.5 * CrossEntropy + 0.5 * Dice.

Section 16 names the combination and requires four things to be documented:
the Dice formulation, the class averaging, the ignore-label handling, and any
change from the stated recipe. All four are below, next to the code that
implements them.

Section 17 is the other half of the instruction, and it is a prohibition:
instance models do not use this loss. Mask R-CNN keeps its own four-term loss
and YOLO keeps its own, because both are trained against object proposals
rather than a dense label map, and forcing either into a per-pixel
cross-entropy would not be a harder version of the same task but a different
one. Their loss components are recorded by their own trainers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._config import CE_WEIGHT, DICE_WEIGHT, IGNORE_INDEX, NUM_CLASSES


class DiceLoss(nn.Module):
    """Soft multi-class Dice, macro-averaged over classes present in the batch.

    **Formulation.** For class *c*, with *p* the softmax probability and *g*
    the one-hot ground truth over valid pixels:

        Dice_c = (2 * sum(p_c * g_c) + eps) / (sum(p_c) + sum(g_c) + eps)

    The loss is ``1 - mean(Dice_c)``. Probabilities are used rather than
    thresholded predictions so the quantity is differentiable; this is the
    standard soft relaxation, and it means the reported training Dice is not
    directly comparable to the hard Dice that Section 22 evaluates with.

    **Averaging.** The mean runs over classes that actually occur in the
    batch's ground truth, not over all six. This matters here more than it
    usually would. Bicycle is 0.26% of pixels in this dataset, so most
    batches contain none, and a class with no ground-truth pixels scores
    (0 + eps) / (0 + 0 + eps) = 1 if the model correctly predicts none of it.
    Averaging that in would hand the model a free perfect score on every
    absent class, and the rarer the class the more often it collects the
    bonus - precisely inverting the pressure the Dice term exists to apply.

    **Ignore handling.** Pixels equal to IGNORE_INDEX are removed from both
    *p* and *g* before any sum. They contribute to neither numerator nor
    denominator, so a model is neither rewarded nor punished for whatever it
    predicts on a crowd region or on canvas that rotation invented.
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 ignore_index: int = IGNORE_INDEX, epsilon: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (N, C) and (B, H, W) -> (N,), keeping valid pixels.
        probabilities = F.softmax(logits, dim=1)
        batch, classes, height, width = probabilities.shape

        probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, classes)
        flat_target = target.reshape(-1)

        valid = flat_target != self.ignore_index
        if not valid.any():
            # Every pixel ignored. Returning zero rather than NaN keeps one
            # pathological batch from destroying the run; it cannot happen
            # with real data, but a heavy rotation on a tiny crop could.
            return logits.sum() * 0.0

        probabilities = probabilities[valid]
        flat_target = flat_target[valid]

        one_hot = F.one_hot(flat_target, num_classes=self.num_classes).float()

        intersection = (probabilities * one_hot).sum(dim=0)
        cardinality = probabilities.sum(dim=0) + one_hot.sum(dim=0)
        dice = (2 * intersection + self.epsilon) / (cardinality + self.epsilon)

        present = one_hot.sum(dim=0) > 0
        if not present.any():
            return logits.sum() * 0.0
        return 1.0 - dice[present].mean()


class CombinedSegmentationLoss(nn.Module):
    """The Section 16 recipe, and the only loss the semantic models train on.

    Returns the weighted total, and exposes both components so the training
    history can record them separately. Section 20 asks for training loss to
    be logged; logging only the sum would hide the case where cross-entropy
    is still falling while Dice has stopped moving, which is the signature of
    a model that is improving on background and has given up on the rare
    classes.

    Cross-entropy is unweighted. Class-frequency weighting is the obvious
    response to an 83% background dataset and it was deliberately not used:
    Section 13 asks for one shared recipe across architectures, weighting is
    a tuning decision that would have to be justified per model, and the Dice
    term already supplies the rare-class pressure that weighting would.
    Reported as a deviation only if it is ever changed.
    """

    def __init__(self, ce_weight: float = CE_WEIGHT,
                 dice_weight: float = DICE_WEIGHT,
                 ignore_index: int = IGNORE_INDEX,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        ce = self.cross_entropy(logits, target)
        dice = self.dice(logits, target)
        return {
            "loss": self.ce_weight * ce + self.dice_weight * dice,
            "cross_entropy": ce.detach(),
            "dice": dice.detach(),
        }

    def describe(self) -> dict:
        return {
            "loss": "0.5 * CrossEntropy + 0.5 * Dice",
            "ce_weight": self.ce_weight,
            "dice_weight": self.dice_weight,
            "class_weighting": "none (unweighted cross-entropy)",
            "dice_formulation": (
                "soft multi-class: (2 * sum(p*g) + eps) / (sum(p) + sum(g) + eps)"
            ),
            "dice_averaging": "macro over classes present in the batch ground truth",
            "ignore_index": IGNORE_INDEX,
            "ignore_handling": (
                "ignored pixels removed before both terms; they enter neither "
                "numerator nor denominator"
            ),
        }
