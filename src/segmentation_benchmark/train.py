"""The Section 20 training interface, in two adapters that do not pretend.

    history = train_model(model, train_loader, val_loader, optimizer,
                          epochs=25, device=device)

Section 18 asks for shared utilities and then warns against the failure mode
that instruction invites: "Reusability does not require forcing incompatible
model interfaces into one identical loop." Semantic and instance models
disagree about what a target is, what a forward pass returns, and what a
validation score means. So there are two loops here, sharing their timing,
memory accounting, history recording and checkpoint logic, and differing
where the task differs.

What they share is what makes the comparison fair: the same epoch budget,
the same optimizer family and learning rate, the same device, the same
timing conventions, and the same rule that a checkpoint is selected on
validation and never on test.
"""

import copy
import json
import time

import torch

from ._config import (
    CHECKPOINTS_DIR,
    EPOCHS,
    LEARNING_RATE,
    LOGS_DIR,
    OPTIMIZER,
    WEIGHT_DECAY,
)
from .losses import CombinedSegmentationLoss
from .metrics import evaluate_semantic


def resolve_device(requested: str = None) -> torch.device:
    """Pick a device, preferring Metal on this hardware.

    Section 29 asks for peak GPU memory and Section 33 asks for CUDA
    synchronization around timed regions. There is no CUDA here, so both map
    onto their MPS equivalents and the difference is recorded rather than
    glossed: MPS reports allocator totals rather than a driver-level peak,
    which is a weaker measurement and is labeled as such wherever it appears.
    """
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued work before reading a clock.

    Both MPS and CUDA dispatch asynchronously, so a timer stopped without
    this measures how long it took to enqueue the work, not to do it. Every
    epoch time and latency figure in the benchmark is bracketed by this.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_allocated_mb(device: torch.device) -> float:
    """Current device allocation in MB, or 0 where it cannot be measured."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return 0.0


def reset_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def build_optimizer(model, learning_rate: float = LEARNING_RATE):
    """AdamW at 1e-4, the Section 13 recipe, for every trainable model."""
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )


class TrainingHistory:
    """Everything Section 20 asks to be recorded, per epoch.

    Training loss, validation loss, training and validation mIoU and Dice,
    epoch time, learning rate and memory. Written to disk after every epoch
    rather than at the end, so a run killed at epoch 19 of 25 still leaves
    nineteen usable epochs and a recoverable best checkpoint.
    """

    def __init__(self, model_name: str, task: str):
        self.model_name = model_name
        self.task = task
        self.epochs = []
        self.best_epoch = None
        self.best_score = None
        self.best_metric = None

    def add(self, record: dict) -> None:
        self.epochs.append(record)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "task": self.task,
            "epochs": self.epochs,
            "completed_epochs": len(self.epochs),
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "selection_metric": self.best_metric,
            "total_train_seconds": sum(
                e.get("epoch_seconds", 0.0) for e in self.epochs
            ),
            "mean_epoch_seconds": (
                sum(e.get("epoch_seconds", 0.0) for e in self.epochs)
                / max(len(self.epochs), 1)
            ),
            "timing_note": (
                "epoch_seconds covers the training pass only: it excludes "
                "validation, checkpoint writing and dataset construction, "
                "which are timed separately so Section 32's comparison is "
                "between the models rather than between their I/O"
            ),
        }

    def save(self, path=None) -> None:
        path = path or (LOGS_DIR / f"{self.model_name}_history.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def train_model(model, train_loader, val_loader, optimizer=None,
                epochs: int = EPOCHS, device=None, model_name: str = "model",
                criterion=None, aux_weight: float = 0.4,
                checkpoint_dir=None, log_dir=None,
                verbose: bool = True) -> TrainingHistory:
    """Train a semantic model, selecting the checkpoint by validation mIoU.

    Section 21 requires semantic checkpoints to be selected by validation
    mIoU and forbids selecting on test results. Test data is not reachable
    from here: this function receives a training loader and a validation
    loader, and there is no third argument through which a test set could
    arrive.

    ``aux_weight`` applies only to PSPNet, whose auxiliary head is part of
    its published design. It is a documented deviation from the shared
    Section 13 recipe, recorded in the run configuration rather than left
    for a reader to discover in the loss curve.
    """
    device = device or resolve_device()
    model = model.to(device)
    optimizer = optimizer or build_optimizer(model)
    criterion = criterion or CombinedSegmentationLoss()

    checkpoint_dir = checkpoint_dir or CHECKPOINTS_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_{model_name}.pt"
    log_path = (log_dir or LOGS_DIR) / f"{model_name}_history.json"

    history = TrainingHistory(model_name, "semantic")
    history.best_metric = "val_mean_iou"
    # Negative infinity, not -1. Any finite score must beat the initial
    # value, and a sentinel chosen to suit one metric's range silently
    # stops saving checkpoints when a different metric is selected on.
    best_score = float("-inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        reset_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()

        totals = {"loss": 0.0, "cross_entropy": 0.0, "dice": 0.0}
        batches = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)

            if isinstance(outputs, dict):
                # PSPNet while training: main head plus auxiliary.
                computed = criterion(outputs["out"], targets)
                loss = computed["loss"]
                if "aux" in outputs:
                    loss = loss + aux_weight * criterion(
                        outputs["aux"], targets
                    )["loss"]
            elif isinstance(outputs, list):
                # U-Net++ with deep supervision: average over the heads.
                per_head = [criterion(o, targets) for o in outputs]
                loss = sum(c["loss"] for c in per_head) / len(per_head)
                computed = per_head[-1]
            else:
                computed = criterion(outputs, targets)
                loss = computed["loss"]

            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.detach())
            totals["cross_entropy"] += float(computed["cross_entropy"])
            totals["dice"] += float(computed["dice"])
            batches += 1

        synchronize(device)
        epoch_seconds = time.perf_counter() - started
        peak_memory = memory_allocated_mb(device)

        train_metrics, _ = evaluate_semantic(
            model, train_loader, device, max_batches=25
        )
        val_metrics, _ = evaluate_semantic(model, val_loader, device)
        val_loss = _validation_loss(model, val_loader, device, criterion)

        record = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(batches, 1),
            "train_cross_entropy": totals["cross_entropy"] / max(batches, 1),
            "train_dice_loss": totals["dice"] / max(batches, 1),
            "val_loss": val_loss,
            "train_mean_iou": train_metrics["mean_iou"],
            "train_dice": train_metrics["dice"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_mean_iou_foreground": val_metrics["mean_iou_foreground"],
            "val_dice": val_metrics["dice"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_per_class_iou": val_metrics["per_class_iou"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "device_memory_mb": peak_memory,
        }
        history.add(record)

        if val_metrics["mean_iou"] > best_score:
            best_score = val_metrics["mean_iou"]
            history.best_epoch = epoch
            history.best_score = best_score
            best_state = copy.deepcopy(model.state_dict())
            # Weights only. Section 31 compares equivalent artifacts, and a
            # file carrying AdamW's two moment tensors per parameter would be
            # roughly three times the size for reasons unrelated to the
            # architecture.
            torch.save(best_state, checkpoint_path)

        history.save(log_path)

        if verbose:
            print(
                f"  epoch {epoch:2d}/{epochs}  "
                f"loss {record['train_loss']:.4f}  "
                f"val_loss {val_loss:.4f}  "
                f"val_mIoU {val_metrics['mean_iou']:.4f}  "
                f"val_Dice {val_metrics['dice']:.4f}  "
                f"{epoch_seconds:.1f}s"
                f"{'  *' if epoch == history.best_epoch else ''}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _validation_loss(model, loader, device, criterion) -> float:
    """Validation loss under the same criterion the training used.

    Section 20 asks for it, and Section 54 asks which model overfit most -
    a question answered by training loss falling while this rises, which
    requires both to be measured the same way.
    """
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            elif isinstance(outputs, list):
                outputs = outputs[-1]
            total += float(criterion(outputs, targets)["loss"])
            batches += 1
    return total / max(batches, 1)


def train_instance_model(model, train_loader, val_loader, optimizer=None,
                         epochs: int = EPOCHS, device=None,
                         model_name: str = "maskrcnn", checkpoint_dir=None,
                         log_dir=None, evaluator=None,
                         verbose: bool = True) -> TrainingHistory:
    """Train Mask R-CNN, selecting the checkpoint by validation mask AP.

    Section 17 says instance models keep their own loss, and Mask R-CNN's is
    already four terms that torchvision returns from the forward pass in
    training mode. Section 20 asks for those components to be recorded rather
    than collapsed, because they fail in distinguishable ways: a rising
    ``loss_rpn_box_reg`` with a flat ``loss_mask`` means the proposals are
    degrading, which is a different problem from masks degrading.

    Section 21 requires a declared validation instance metric. That metric is
    mask AP, computed by the same COCO evaluator that produces the reported
    test numbers, passed in as ``evaluator`` so this module does not import
    the evaluation code that will import it back.
    """
    device = device or resolve_device()
    model = model.to(device)
    optimizer = optimizer or build_optimizer(model)

    checkpoint_dir = checkpoint_dir or CHECKPOINTS_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_{model_name}.pt"
    log_path = (log_dir or LOGS_DIR) / f"{model_name}_history.json"

    history = TrainingHistory(model_name, "instance")
    history.best_metric = "val_mask_ap" if evaluator else "train_total_loss"
    best_score = float("-inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        reset_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()

        component_totals, batches = {}, 0

        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            prepared = [
                {
                    "boxes": t["boxes"].to(device),
                    "labels": t["labels"].to(device),
                    "masks": t["masks"].to(device),
                }
                for t in targets
            ]

            # An image whose objects were all cropped away has no boxes, and
            # the RPN's loss is undefined without a positive anchor. Skipping
            # is correct rather than convenient: there is nothing to learn
            # from a target that specifies nothing.
            keep = [i for i, t in enumerate(prepared) if len(t["boxes"])]
            if not keep:
                continue
            images = [images[i] for i in keep]
            prepared = [prepared[i] for i in keep]

            optimizer.zero_grad(set_to_none=True)
            losses = model(images, prepared)
            total = sum(losses.values())
            total.backward()
            optimizer.step()

            for name, value in losses.items():
                component_totals[name] = (
                    component_totals.get(name, 0.0) + float(value.detach())
                )
            component_totals["total"] = (
                component_totals.get("total", 0.0) + float(total.detach())
            )
            batches += 1

        synchronize(device)
        epoch_seconds = time.perf_counter() - started
        peak_memory = memory_allocated_mb(device)

        record = {
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "device_memory_mb": peak_memory,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{
                f"train_{name}": value / max(batches, 1)
                for name, value in component_totals.items()
            },
        }

        if evaluator is not None:
            validation = evaluator(model, val_loader, device)
            record.update({f"val_{k}": v for k, v in validation.items()})
            score = validation.get("mask_ap", -1.0)
        else:
            # Without an evaluator the only available signal is the loss, and
            # lower is better, so it is negated to keep "higher wins" the one
            # selection rule in this file.
            score = -record.get("train_total", 0.0)

        history.add(record)

        if score > best_score:
            best_score = score
            history.best_epoch = epoch
            history.best_score = score
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)

        history.save(log_path)

        if verbose:
            components = "  ".join(
                f"{name.replace('loss_', '')} {value / max(batches, 1):.3f}"
                for name, value in component_totals.items()
                if name != "total"
            )
            extra = (
                f"  val_mAP {record.get('val_mask_ap', float('nan')):.4f}"
                if evaluator else ""
            )
            print(
                f"  epoch {epoch:2d}/{epochs}  "
                f"total {component_totals.get('total', 0) / max(batches, 1):.4f}  "
                f"{components}{extra}  {epoch_seconds:.1f}s"
                f"{'  *' if epoch == history.best_epoch else ''}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
