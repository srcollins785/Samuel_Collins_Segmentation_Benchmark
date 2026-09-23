"""Finish a model whose training completed but whose evaluation crashed.

Training is the expensive part and it is checkpointed. When a run dies after
training - in evaluation, profiling or figure rendering - retraining is the
wrong recovery: it costs hours, and because Metal kernels are not
bit-deterministic the new weights would not be the ones the interrupted run
selected. This loads the saved checkpoint, which *is* the selected state,
and completes everything downstream of it.

The training history is read from ``logs/<model>_history.json``, which the
trainer writes after every epoch precisely so an interrupted run leaves its
curve behind.

Usage
-----
    python scripts/recover_model.py --model segnet --input-size 256
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from segmentation_benchmark import visualize  # noqa: E402
from segmentation_benchmark._config import (  # noqa: E402
    BATCH_SIZE, CHECKPOINTS_DIR, CONFUSION_DIR, EPOCHS, LOGS_DIR,
    NUM_CLASSES, RESULTS_DIR,
)
from segmentation_benchmark.benchmark import RAW_DIR, run_configuration  # noqa: E402
from segmentation_benchmark.boundary import evaluate_boundary  # noqa: E402
from segmentation_benchmark.dataset import (  # noqa: E402
    SemanticSegmentationDataset, build_loader,
)
from segmentation_benchmark.efficiency import profile_model, save_json  # noqa: E402
from segmentation_benchmark.losses import CombinedSegmentationLoss  # noqa: E402
from segmentation_benchmark.metrics import evaluate_semantic  # noqa: E402
from segmentation_benchmark.models import describe_model, get_segmentation_model  # noqa: E402
from segmentation_benchmark.train import resolve_device  # noqa: E402


def recover_instance(model_name: str, args, device, history: dict) -> dict:
    """Finish an instance model from its selected checkpoint.

    Mask R-CNN needed this for a different reason than SegNet did. Its
    training did not crash - it was stopped deliberately at epoch 14 of 25,
    because the host had exhausted memory (MPS's driver-side cache grows
    across rounds of COCOeval validation and is not reflected in the
    allocator's own accounting), epoch times had gone from 740 seconds to
    4,374 and were still climbing, and validation mask AP had not improved
    in six epochs. The saved checkpoint is epoch 8, which is what the
    Section 21 selection rule had already chosen.

    The shortfall against the Section 13 epoch budget is recorded in the
    result file so the report states it rather than leaving a reader to
    infer 25 epochs from the protocol table.
    """
    from segmentation_benchmark.dataset import InstanceSegmentationDataset
    from segmentation_benchmark.instance_metrics import evaluate_instance

    checkpoint = CHECKPOINTS_DIR / f"best_{model_name}.pt"
    model = get_segmentation_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model = model.to(device)

    train_set = InstanceSegmentationDataset("train", size=args.input_size)
    test_set = InstanceSegmentationDataset("test", size=args.input_size)
    test_loader = build_loader(test_set, args.batch_size, False, args.workers)

    print("  evaluating on the held-out test split")
    test_metrics = evaluate_instance(model, test_loader, device, split="test")

    print("  saving prediction examples")
    predictions = visualize.save_instance_predictions(
        model, test_loader, device, model_name, count=6
    )

    print("  measuring efficiency")
    efficiency = profile_model(
        model, model_name, device, checkpoint_path=checkpoint,
        input_size=args.input_size, instance=True,
    )

    history["peak_memory_mb"] = max(
        (e.get("device_memory_mb", 0.0) for e in history["epochs"]), default=0.0
    )

    completed = history["completed_epochs"]
    payload = {
        "model": model_name,
        "task": "instance",
        "run_id": f"recovered-{int(time.time()) % 100000:05d}",
        "config": {
            **run_configuration(
                model_name, "instance", args.input_size, args.batch_size,
                completed, True, train_set, "recovered",
            ),
            "epochs_planned": EPOCHS,
            "epochs_completed": completed,
            "early_stop_reason": (
                "Host memory exhausted. MPS retains a driver-side cache that "
                "its allocator accounting does not report; across rounds of "
                "COCOeval validation it grew until 20.6 GB was wired and the "
                "machine was swapping. Epoch time rose from 740 s to 4,374 s "
                "and was still climbing. Validation mask AP peaked at epoch "
                f"{history['best_epoch']} and did not improve in the "
                f"{completed - history['best_epoch']} epochs after it."
            ),
        },
        "card": describe_model(model_name, model),
        "loss": {
            "note": (
                "Section 17: Mask R-CNN keeps its own four-term loss plus the "
                "RPN's two. The semantic CE+Dice recipe is not applied."
            ),
            "components": [
                "loss_classifier", "loss_box_reg", "loss_mask",
                "loss_objectness", "loss_rpn_box_reg",
            ],
        },
        "training": history,
        "test": test_metrics,
        "efficiency": efficiency,
        "predictions": predictions,
        "recovery_note": (
            f"Stopped at {completed} of {EPOCHS} epochs; see "
            "config.early_stop_reason. Evaluated from the checkpoint the "
            "Section 21 rule selected (epoch "
            f"{history['best_epoch']}, validation mask AP "
            f"{history['best_score']:.4f})."
        ),
    }
    save_json(payload, RAW_DIR / f"{model_name}.json")
    print(
        f"\n  test mask AP {test_metrics['mask_ap']:.4f}  "
        f"AP50 {test_metrics['mask_ap50']:.4f}  "
        f"AP75 {test_metrics['mask_ap75']:.4f}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    model_name = args.model
    checkpoint = CHECKPOINTS_DIR / f"best_{model_name}.pt"
    history_path = LOGS_DIR / f"{model_name}_history.json"

    if not checkpoint.is_file():
        raise SystemExit(
            f"No checkpoint at {checkpoint}. Training did not complete; "
            f"rerun it:\n  python run_benchmark.py --model {model_name}"
        )
    if not history_path.is_file():
        raise SystemExit(f"No training history at {history_path}.")

    device = resolve_device()
    history = json.loads(history_path.read_text())
    print(
        f"Recovering {model_name}: {history['completed_epochs']} epochs "
        f"trained, best {history['selection_metric']}="
        f"{history['best_score']:.4f} at epoch {history['best_epoch']}"
    )

    if history.get("task") == "instance":
        recover_instance(model_name, args, device, history)
        print(f"  wrote {(RAW_DIR / f'{model_name}.json').relative_to(REPO_ROOT)}")
        return 0

    model = get_segmentation_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model = model.to(device)

    test_set = SemanticSegmentationDataset("test", size=args.input_size)
    train_set = SemanticSegmentationDataset("train", size=args.input_size)
    test_loader = build_loader(test_set, args.batch_size, False, args.workers)

    print("  evaluating on the held-out test split")
    test_metrics, matrix = evaluate_semantic(model, test_loader, device)
    boundary_metrics, _ = evaluate_boundary(model, test_loader, device)
    test_metrics.update(boundary_metrics)

    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    (CONFUSION_DIR / f"{model_name}.json").write_text(
        json.dumps({"model": model_name, "matrix": matrix.to_list()}, indent=2)
    )

    print("  saving prediction examples")
    predictions = visualize.save_semantic_predictions(
        model, test_loader, device, model_name, count=8, normalized=True
    )
    closeup = visualize.save_boundary_closeup(
        model, test_loader, device, model_name, normalized=True
    )

    print("  measuring efficiency")
    efficiency = profile_model(
        model, model_name, device, checkpoint_path=checkpoint,
        input_size=args.input_size,
    )

    history["peak_memory_mb"] = max(
        (e.get("device_memory_mb", 0.0) for e in history["epochs"]), default=0.0
    )

    payload = {
        "model": model_name,
        "task": "semantic",
        "run_id": f"recovered-{int(time.time()) % 100000:05d}",
        "config": run_configuration(
            model_name, "semantic", args.input_size, args.batch_size,
            history["completed_epochs"], True, train_set, "recovered",
        ),
        "card": describe_model(model_name, model),
        "loss": CombinedSegmentationLoss().describe(),
        "training": history,
        "test": test_metrics,
        "efficiency": efficiency,
        "predictions": predictions,
        "boundary_closeup": closeup,
        "recovery_note": (
            "Training completed normally; the original run died afterwards in "
            "profiling. Evaluation and measurement were redone from the saved "
            "checkpoint, which is the same selected state the interrupted run "
            "had chosen. The training history is the original."
        ),
    }
    save_json(payload, RAW_DIR / f"{model_name}.json")

    print(
        f"\n  test mIoU {test_metrics['mean_iou']:.4f}  "
        f"Dice {test_metrics['dice']:.4f}  "
        f"boundary F1 {test_metrics['boundary_f1']:.4f}"
    )
    print(f"  wrote {(RAW_DIR / f'{model_name}.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
