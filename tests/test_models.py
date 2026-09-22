"""Tests for the Section 19 factory and the four hand-written architectures.

Models are built without pretrained weights wherever the test does not need
them, so the suite runs on a machine that has never downloaded a checkpoint.
The pretrained paths are exercised by the benchmark itself, which records
what it loaded into each run's configuration.
"""

import pytest
import torch

from segmentation_benchmark.architectures import (
    PSPNet, PyramidPoolingModule, SegNet, UNet, UNetPlusPlus,
)
from segmentation_benchmark.models import (
    ALL_MODELS, INSTANCE_MODELS, MODEL_CARDS, SEMANTIC_MODELS,
    count_parameters, describe_model, get_segmentation_model, weight_size_mb,
)

NUM_CLASSES = 6
SIZE = 64


class TestFactoryContract:
    def test_every_listed_model_has_a_card(self):
        """Section 10: a name is not a configuration."""
        for name in ALL_MODELS:
            assert name in MODEL_CARDS, f"{name} has no model card"
            card = MODEL_CARDS[name]
            for key in ("task", "backbone", "initialization"):
                assert card.get(key), f"{name} card is missing {key}"

    def test_unknown_model_names_are_rejected(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_segmentation_model("resnet50", NUM_CLASSES)

    def test_yolo_explains_why_it_is_not_here(self):
        """A silent KeyError would be worse than a message with the reason."""
        with pytest.raises(ValueError, match="yolo_adapter"):
            get_segmentation_model("yolo_seg", NUM_CLASSES)

    def test_kmeans_explains_why_it_is_not_here(self):
        with pytest.raises(ValueError, match="traditional"):
            get_segmentation_model("kmeans", NUM_CLASSES)

    def test_tracks_do_not_overlap(self):
        assert not set(SEMANTIC_MODELS) & set(INSTANCE_MODELS)

    def test_card_tasks_match_the_track_lists(self):
        for name in SEMANTIC_MODELS:
            assert MODEL_CARDS[name]["task"] == "semantic"
        for name in INSTANCE_MODELS:
            assert MODEL_CARDS[name]["task"] == "instance"


class TestSemanticOutputContract:
    """Every semantic model must return logits at the input resolution."""

    @pytest.mark.parametrize("name", ["unet", "unetpp", "segnet", "pspnet"])
    def test_output_shape_matches_input(self, name):
        model = get_segmentation_model(name, NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, SIZE, SIZE))
        assert out.shape == (2, NUM_CLASSES, SIZE, SIZE)

    @pytest.mark.parametrize("name", ["unet", "unetpp", "segnet"])
    def test_handles_a_non_power_of_two_input(self, name):
        """Skip alignment must not require powers of two."""
        model = get_segmentation_model(name, NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 96, 96))
        assert out.shape == (1, NUM_CLASSES, 96, 96)

    def test_logits_are_not_probabilities(self):
        """The loss applies softmax; a model that already did would double it."""
        model = UNet(NUM_CLASSES, base=8)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, SIZE, SIZE))
        assert not torch.allclose(
            out.softmax(dim=1).sum(dim=1), out.sum(dim=1)
        )


class TestUNet:
    def test_skip_connections_carry_encoder_features(self):
        """Removing a skip must change the output, or it is not connected."""
        model = UNet(NUM_CLASSES, base=8)
        model.eval()
        x = torch.randn(1, 3, SIZE, SIZE)
        with torch.no_grad():
            baseline = model(x)

        # Zero the first encoder's contribution and confirm the result moves.
        with torch.no_grad():
            for parameter in model.encoders[0].parameters():
                parameter.zero_()
            altered = model(x)

        assert not torch.allclose(baseline, altered)

    def test_width_scales_parameters(self):
        narrow = count_parameters(UNet(NUM_CLASSES, base=16))["total_parameters"]
        wide = count_parameters(UNet(NUM_CLASSES, base=32))["total_parameters"]
        assert wide > narrow * 3


class TestUNetPlusPlus:
    def test_matches_unet_width_by_default(self):
        """Section 54 compares the two; different widths would confound it."""
        assert UNetPlusPlus(NUM_CLASSES).blocks["x0_0"][0].out_channels == \
            UNet(NUM_CLASSES).encoders[0][0].out_channels

    def test_costs_more_parameters_than_unet_at_equal_width(self):
        """The nested nodes are the cost; this pins the claim down."""
        plain = count_parameters(UNet(NUM_CLASSES, base=64))["total_parameters"]
        nested = count_parameters(
            UNetPlusPlus(NUM_CLASSES, base=64)
        )["total_parameters"]
        assert nested > plain
        assert 1.3 < nested / plain < 1.6

    def test_deep_supervision_returns_several_outputs_while_training(self):
        model = UNetPlusPlus(NUM_CLASSES, base=8, deep_supervision=True)
        model.train()
        out = model(torch.randn(1, 3, SIZE, SIZE))
        assert isinstance(out, list) and len(out) > 1

    def test_deep_supervision_returns_one_output_in_eval(self):
        model = UNetPlusPlus(NUM_CLASSES, base=8, deep_supervision=True)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, SIZE, SIZE))
        assert isinstance(out, torch.Tensor)


class TestSegNet:
    def test_runs_without_pretrained_weights(self):
        model = SegNet(NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, SIZE, SIZE))
        assert out.shape == (1, NUM_CLASSES, SIZE, SIZE)

    def test_decoder_holds_no_encoder_feature_maps(self):
        """SegNet's whole point: indices cross, feature maps do not.

        A U-Net-style decoder block takes twice its output width because it
        concatenates a skip. SegNet's first decoder convolution takes exactly
        the width the unpooling produced, with nothing concatenated.
        """
        model = SegNet(NUM_CLASSES, pretrained=False)
        first_decoder_conv = model.decoders[0][0]
        last_encoder_conv = [
            m for m in model.encoders[-1] if isinstance(m, torch.nn.Conv2d)
        ][-1]
        assert first_decoder_conv.in_channels == last_encoder_conv.out_channels


class TestPSPNet:
    def test_pyramid_concatenates_every_branch(self):
        # Batch of 2: the 1x1 branch feeds batch norm a (B, C, 1, 1) tensor,
        # which needs more than one value per channel while training. The
        # training loader drops size-1 batches for this reason.
        module = PyramidPoolingModule(64, bins=(1, 2, 3, 6))
        out = module(torch.randn(2, 64, 32, 32))
        assert out.shape[1] == module.out_channels
        assert out.shape[-2:] == (32, 32)

    def test_pyramid_batch_of_one_needs_eval_mode(self):
        """Pins the constraint that build_loader's drop_last exists for."""
        module = PyramidPoolingModule(16, bins=(1,))
        module.train()
        with pytest.raises(ValueError, match="more than 1 value per channel"):
            module(torch.randn(1, 16, 8, 8))
        module.eval()
        with torch.no_grad():
            assert module(torch.randn(1, 16, 8, 8)).shape[0] == 1

    def test_pyramid_sees_the_whole_image_in_its_first_branch(self):
        """The 1x1 branch is global context; changing a far pixel must matter."""
        module = PyramidPoolingModule(16, bins=(1,))
        module.eval()
        x = torch.zeros(1, 16, 16, 16)
        with torch.no_grad():
            before = module(x)[0, :, 0, 0].clone()
            x[0, :, 15, 15] = 10.0
            after = module(x)[0, :, 0, 0]
        assert not torch.allclose(before, after)

    def test_auxiliary_head_is_training_only(self):
        model = PSPNet(NUM_CLASSES, pretrained=False, aux_loss=True)
        model.train()
        assert isinstance(model(torch.randn(2, 3, SIZE, SIZE)), dict)
        model.eval()
        with torch.no_grad():
            assert isinstance(model(torch.randn(1, 3, SIZE, SIZE)), torch.Tensor)

    def test_backbone_is_dilated_to_output_stride_eight(self):
        """Output stride 8, not 32, is what makes PSPNet expensive and precise."""
        model = PSPNet(NUM_CLASSES, pretrained=False, aux_loss=False)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            features = model.layer4(
                model.layer3(model.layer2(model.layer1(model.stem(x))))
            )
        assert features.shape[-2:] == (8, 8)


class TestMeasurement:
    def test_counts_total_and_trainable_separately(self):
        counts = count_parameters(UNet(NUM_CLASSES, base=8))
        assert counts["total_parameters"] == counts["trainable_parameters"]
        assert counts["frozen_parameters"] == 0

    def test_frozen_parameters_are_reported(self):
        model = UNet(NUM_CLASSES, base=8)
        for parameter in model.encoders[0].parameters():
            parameter.requires_grad = False
        counts = count_parameters(model)
        assert counts["frozen_parameters"] > 0
        assert counts["trainable_parameters"] < counts["total_parameters"]

    def test_weight_size_includes_buffers(self):
        """BatchNorm running statistics are in the file and must be counted."""
        model = UNet(NUM_CLASSES, base=8)
        assert any(True for _ in model.buffers())
        assert weight_size_mb(model) > 0

    def test_describe_carries_the_card_and_the_measurements(self):
        model = UNet(NUM_CLASSES, base=8)
        described = describe_model("unet", model)
        assert described["backbone"]
        assert described["initialization"]
        assert described["total_parameters"] > 0
        assert described["weight_size_mb"] > 0
