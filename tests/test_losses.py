"""Tests for the Section 16 semantic loss.

The behavior worth testing here is not that the loss decreases - it is that
the two properties the assignment asks to be documented are actually
implemented: ignored pixels are excluded from both terms, and absent classes
do not collect a free perfect Dice score.
"""

import torch

from segmentation_benchmark._config import IGNORE_INDEX, NUM_CLASSES
from segmentation_benchmark.losses import CombinedSegmentationLoss, DiceLoss


def perfect_logits(target, magnitude=20.0):
    """Logits that predict `target` with near-certainty."""
    batch, height, width = target.shape
    logits = torch.zeros(batch, NUM_CLASSES, height, width)
    safe = target.clone()
    safe[safe == IGNORE_INDEX] = 0
    return logits.scatter_(1, safe.unsqueeze(1), magnitude)


class TestDiceLoss:
    def test_perfect_prediction_is_near_zero(self):
        target = torch.zeros(2, 16, 16, dtype=torch.long)
        target[:, :8, :] = 1
        target[:, 8:, :] = 2

        loss = DiceLoss()(perfect_logits(target), target)

        assert loss.item() < 0.01

    def test_wrong_prediction_is_near_one(self):
        target = torch.ones(1, 16, 16, dtype=torch.long)
        wrong = torch.full_like(target, 2)

        loss = DiceLoss()(perfect_logits(wrong), target)

        assert loss.item() > 0.9

    def test_absent_classes_do_not_earn_a_free_score(self):
        """A batch with no bicycle must not score better for that fact.

        With averaging over all six classes, the four absent ones each score
        1.0 and drag the mean up regardless of what the model did on the two
        present ones. Averaging over present classes only is what makes the
        Dice term keep applying pressure on rare classes.
        """
        target = torch.zeros(1, 16, 16, dtype=torch.long)
        target[:, :8, :] = 1  # background and person only

        # Predict everything as background: half the pixels are wrong.
        logits = torch.zeros(1, NUM_CLASSES, 16, 16)
        logits[:, 0] = 20.0

        loss = DiceLoss()(logits, target)

        # Person is entirely missed, so with two classes present the mean
        # Dice is about 0.5 (background ~0.67, person 0) and the loss is
        # well away from zero. Averaging over all six would give roughly
        # 0.78 Dice and a loss near 0.2, hiding the total miss.
        assert loss.item() > 0.4, (
            "absent classes appear to be inflating the Dice average"
        )

    def test_ignored_pixels_do_not_affect_the_loss(self):
        """Two targets differing only in ignored regions must score the same."""
        base = torch.zeros(1, 16, 16, dtype=torch.long)
        base[:, :8, :] = 1
        logits = perfect_logits(base)

        first = base.clone()
        first[:, 12:, :] = IGNORE_INDEX
        second = base.clone()
        second[:, 12:, :] = IGNORE_INDEX

        # Change what the model predicts under the ignored region only.
        altered = logits.clone()
        altered[:, :, 12:, :] = 0.0
        altered[:, 4, 12:, :] = 20.0  # confidently predict 'dog' where ignored

        assert DiceLoss()(logits, first).item() == \
            DiceLoss()(altered, second).item()

    def test_all_ignored_returns_zero_rather_than_nan(self):
        target = torch.full((1, 8, 8), IGNORE_INDEX, dtype=torch.long)
        logits = torch.randn(1, NUM_CLASSES, 8, 8)

        loss = DiceLoss()(logits, target)

        assert torch.isfinite(loss)
        assert loss.item() == 0.0

    def test_is_differentiable(self):
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        target[:, :4, :] = 1
        logits = torch.randn(1, NUM_CLASSES, 8, 8, requires_grad=True)

        DiceLoss()(logits, target).backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestCombinedLoss:
    def test_returns_both_components_separately(self):
        """Section 20 logs them apart; the sum would hide a stalled Dice term."""
        target = torch.zeros(1, 16, 16, dtype=torch.long)
        target[:, :8, :] = 1

        out = CombinedSegmentationLoss()(perfect_logits(target), target)

        assert {"loss", "cross_entropy", "dice"} <= set(out)
        assert torch.isfinite(out["loss"])

    def test_weights_the_two_terms_as_documented(self):
        target = torch.zeros(1, 16, 16, dtype=torch.long)
        target[:, :8, :] = 2
        logits = torch.randn(1, NUM_CLASSES, 16, 16)

        criterion = CombinedSegmentationLoss()
        out = criterion(logits, target)

        expected = (
            criterion.ce_weight * out["cross_entropy"]
            + criterion.dice_weight * out["dice"]
        )
        assert out["loss"].item() == expected.item()

    def test_cross_entropy_honors_the_ignore_index(self):
        base = torch.zeros(1, 16, 16, dtype=torch.long)
        base[:, :8, :] = 1
        logits = perfect_logits(base)

        ignored = base.clone()
        ignored[:, 12:, :] = IGNORE_INDEX
        altered = logits.clone()
        altered[:, :, 12:, :] = 0.0
        altered[:, 5, 12:, :] = 20.0

        criterion = CombinedSegmentationLoss()
        assert criterion(logits, ignored)["cross_entropy"].item() == \
            criterion(altered, ignored)["cross_entropy"].item()

    def test_describe_states_every_documented_choice(self):
        described = CombinedSegmentationLoss().describe()
        for key in (
            "dice_formulation", "dice_averaging", "ignore_handling",
            "class_weighting", "ce_weight", "dice_weight",
        ):
            assert key in described, f"Section 16 requires {key} documented"
        assert described["ignore_index"] == IGNORE_INDEX
