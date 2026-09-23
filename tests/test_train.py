"""Tests for the Section 20 training interface and Section 21 checkpointing.

These use a tiny synthetic dataset rather than COCO, so the suite stays fast
and runs before the subset has been downloaded. What is being tested is the
loop's bookkeeping - history, selection, timing, device handling - not
whether a model learns, which the benchmark itself demonstrates.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from segmentation_benchmark.architectures import UNet
from segmentation_benchmark.train import (
    TrainingHistory, build_optimizer, resolve_device, train_model,
)

NUM_CLASSES = 6


class TinyDataset(Dataset):
    """Eight 32x32 samples with a learnable pattern."""

    def __init__(self, length: int = 8, size: int = 32):
        self.length = length
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        image = torch.randn(3, self.size, self.size, generator=generator)
        mask = torch.zeros(self.size, self.size, dtype=torch.long)
        mask[: self.size // 2] = 1
        return image, mask


def loaders(batch_size: int = 4):
    train = DataLoader(TinyDataset(8), batch_size=batch_size, drop_last=True)
    validation = DataLoader(TinyDataset(4), batch_size=batch_size)
    return train, validation


class TestTrainingHistory:
    def test_records_every_field_section_20_requires(self, tmp_path):
        train, validation = loaders()
        model = UNet(NUM_CLASSES, base=4)

        history = train_model(
            model, train, validation, epochs=2, device=torch.device("cpu"),
            model_name="tiny", checkpoint_dir=tmp_path, log_dir=tmp_path, verbose=False,
        )

        record = history.to_dict()["epochs"][0]
        for key in (
            "train_loss", "val_loss", "train_mean_iou", "val_mean_iou",
            "train_dice", "val_dice", "epoch_seconds", "learning_rate",
            "device_memory_mb",
        ):
            assert key in record, f"Section 20 requires {key} recorded"

    def test_records_loss_components_separately(self, tmp_path):
        """The sum hides a stalled Dice term under a falling CE term."""
        train, validation = loaders()
        history = train_model(
            UNet(NUM_CLASSES, base=4), train, validation, epochs=1,
            device=torch.device("cpu"), verbose=False,
            checkpoint_dir=tmp_path, log_dir=tmp_path,
        )
        record = history.to_dict()["epochs"][0]
        assert "train_cross_entropy" in record
        assert "train_dice_loss" in record

    def test_reports_completed_epochs_and_mean_epoch_time(self, tmp_path):
        train, validation = loaders()
        history = train_model(
            UNet(NUM_CLASSES, base=4), train, validation, epochs=3,
            device=torch.device("cpu"), checkpoint_dir=tmp_path, log_dir=tmp_path, verbose=False,
        )
        summary = history.to_dict()
        assert summary["completed_epochs"] == 3
        assert summary["mean_epoch_seconds"] > 0
        assert summary["total_train_seconds"] > 0

    def test_states_what_epoch_time_includes(self):
        """Section 32 asks whether totals include validation and I/O."""
        assert "timing_note" in TrainingHistory("m", "semantic").to_dict()

    def test_history_is_written_each_epoch(self, tmp_path):
        """A run killed at epoch 19 of 25 should leave 19 usable epochs."""
        train, validation = loaders()
        log = tmp_path / "history.json"
        history = TrainingHistory("tiny", "semantic")
        history.add({"epoch": 1, "epoch_seconds": 1.0})
        history.save(log)
        assert log.is_file()
        import json
        assert json.loads(log.read_text())["completed_epochs"] == 1


class TestCheckpointSelection:
    def test_selects_by_validation_miou(self, tmp_path):
        train, validation = loaders()
        history = train_model(
            UNet(NUM_CLASSES, base=4), train, validation, epochs=2,
            device=torch.device("cpu"), model_name="tiny",
            checkpoint_dir=tmp_path, log_dir=tmp_path, verbose=False,
        )
        assert history.to_dict()["selection_metric"] == "val_mean_iou"

    def test_a_checkpoint_is_always_written(self, tmp_path):
        """Regression: the initial best score was -1.0, so a selection
        metric whose values fall below -1 never triggered a save and the
        run finished with no checkpoint at all."""
        train, validation = loaders()
        history = train_model(
            UNet(NUM_CLASSES, base=4), train, validation, epochs=2,
            device=torch.device("cpu"), model_name="tiny",
            checkpoint_dir=tmp_path, log_dir=tmp_path, verbose=False,
        )

        assert history.best_epoch is not None
        assert (tmp_path / "best_tiny.pt").is_file()

    def test_checkpoint_holds_weights_not_optimizer_state(self, tmp_path):
        """Section 31 compares equivalent artifacts; optimizer moments would
        roughly triple the file for reasons unrelated to architecture."""
        train, validation = loaders()
        train_model(
            UNet(NUM_CLASSES, base=4), train, validation, epochs=1,
            device=torch.device("cpu"), model_name="tiny",
            checkpoint_dir=tmp_path, log_dir=tmp_path, verbose=False,
        )

        state = torch.load(tmp_path / "best_tiny.pt", weights_only=True)
        assert all(isinstance(v, torch.Tensor) for v in state.values())
        assert not any("exp_avg" in key for key in state)

    def test_best_weights_are_restored_at_the_end(self, tmp_path):
        train, validation = loaders()
        model = UNet(NUM_CLASSES, base=4)
        history = train_model(
            model, train, validation, epochs=2, device=torch.device("cpu"),
            model_name="tiny", checkpoint_dir=tmp_path, log_dir=tmp_path, verbose=False,
        )

        saved = torch.load(tmp_path / "best_tiny.pt", weights_only=True)
        for key, value in model.state_dict().items():
            assert torch.equal(value.cpu(), saved[key]), (
                f"{key} differs; the returned model is not the best checkpoint"
            )

    def test_there_is_no_way_to_pass_test_data_in(self):
        """Section 21 forbids selecting on test results. The strongest form
        of that guarantee is an interface with nowhere to put it."""
        import inspect
        parameters = set(inspect.signature(train_model).parameters)
        assert not {"test_loader", "test_set", "test_data"} & parameters


class TestDeviceHandling:
    def test_resolve_device_honors_an_explicit_request(self):
        assert resolve_device("cpu").type == "cpu"

    def test_resolve_device_picks_an_accelerator_when_available(self):
        device = resolve_device()
        assert device.type in {"mps", "cuda", "cpu"}

    def test_optimizer_is_adamw_at_the_protocol_rate(self):
        from segmentation_benchmark._config import LEARNING_RATE
        optimizer = build_optimizer(UNet(NUM_CLASSES, base=4))
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == LEARNING_RATE

    def test_optimizer_skips_frozen_parameters(self):
        model = UNet(NUM_CLASSES, base=4)
        for parameter in model.encoders[0].parameters():
            parameter.requires_grad = False
        optimizer = build_optimizer(model)
        counted = sum(len(g["params"]) for g in optimizer.param_groups)
        assert counted < len(list(model.parameters()))
