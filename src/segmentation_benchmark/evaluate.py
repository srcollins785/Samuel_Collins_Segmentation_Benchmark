"""One entry point for evaluation, whichever track a model belongs to.

Section 18's suggested layout names an ``evaluate.py``, and the evaluation
code here genuinely is spread across three modules because the three things
it measures are unrelated: ``metrics`` accumulates a confusion matrix,
``boundary`` runs distance transforms, and ``instance_metrics`` drives
COCOeval. Splitting them keeps each honest about its own conventions.

What was missing is the front door. ``evaluate_model`` takes a model and a
loader and returns that model's complete test record without the caller
having to know which track it is on or which three modules to call in which
order. The benchmark uses it; so can anyone re-scoring a checkpoint.

Nothing here computes anything itself. It is composition, deliberately, so
there is exactly one implementation of each metric and no second code path
that could drift from the one that produced the reported numbers.
"""

from .boundary import evaluate_boundary
from .instance_metrics import evaluate_instance, instance_to_semantic
from .metrics import ConfusionMatrix, evaluate_semantic

__all__ = [
    "evaluate_model",
    "evaluate_semantic",
    "evaluate_boundary",
    "evaluate_instance",
    "instance_to_semantic",
    "ConfusionMatrix",
]


def evaluate_model(model, loader, device, task: str = "semantic",
                   split: str = "test", with_boundary: bool = True) -> tuple:
    """Evaluate one model and return ``(metrics, confusion_matrix_or_None)``.

    ``task`` selects the track:

    * ``"semantic"`` returns the Section 22 metrics, with the Section 24
      boundary metrics merged in when ``with_boundary``, plus the confusion
      matrix so a caller can write it to ``confusion_matrices/``.
    * ``"instance"`` returns the Section 25 COCO metrics and ``None``, since
      a confusion matrix is not defined for instance predictions.

    Boundary evaluation is optional because it is the expensive part - six
    distance transforms per image for the truth and six for the prediction -
    and it is a property of a final checkpoint rather than something worth
    paying for on every validation pass.
    """
    if task == "instance":
        return evaluate_instance(model, loader, device, split=split), None

    if task not in {"semantic", "traditional"}:
        raise ValueError(
            f"Unknown task {task!r}. Expected 'semantic', 'traditional' or "
            "'instance'."
        )

    metrics, matrix = evaluate_semantic(model, loader, device)
    if with_boundary:
        boundary, _ = evaluate_boundary(model, loader, device)
        metrics.update(boundary)
    return metrics, matrix
