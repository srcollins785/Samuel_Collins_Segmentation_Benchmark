"""The Section 19 model factory: one call returns any of the ten models.

    model = get_segmentation_model("unet", 6)
    model = get_segmentation_model("deeplabv3", 6)
    model = get_segmentation_model("segformer", 6)

Section 19 asks for a reusable factory, and the value of one is that the
training loop never has to know which architecture it is holding. Reaching
that state takes real work, because the ten models disagree about almost
everything at their interface: torchvision's segmentation models return an
OrderedDict, SegFormer returns an object whose logits are at a quarter
resolution, PSPNet returns a second auxiliary output while training, and
Mask R-CNN returns a loss dict in training mode and a list of detections in
evaluation mode.

Each wrapper below normalizes one of those. After the factory, every
semantic model takes (B, 3, H, W) and returns (B, num_classes, H, W) at the
input resolution, and both instance models follow torchvision's detection
convention. Section 18 permits exactly this - separate adapters where the
interfaces genuinely differ - and warns against the opposite, forcing
incompatible models through one identical call.

Section 10 also requires each run to record its backbone, initialization
source and trainable layers rather than just a name. ``describe_model``
returns those, and the trainers write them into each run's configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._config import BYTES_PER_MB, NUM_CLASSES
from .architectures import PSPNet, SegNet, UNet, UNetPlusPlus

SEMANTIC_MODELS = [
    "fcn_resnet50", "unet", "unetpp", "segnet", "deeplabv3", "pspnet", "segformer",
]
INSTANCE_MODELS = ["maskrcnn", "yolo_seg"]
TRADITIONAL_MODELS = ["kmeans"]
ALL_MODELS = TRADITIONAL_MODELS + SEMANTIC_MODELS + INSTANCE_MODELS

# Section 10: an architecture name is not a configuration. Everything the
# report has to state about each model, recorded once, next to the code that
# builds it.
MODEL_CARDS = {
    "kmeans": {
        "task": "traditional", "backbone": "none",
        "initialization": "none (unsupervised, fitted per image)",
        "paper": "Lloyd 1982 / MacQueen 1967",
    },
    "fcn_resnet50": {
        "task": "semantic", "backbone": "ResNet50, output stride 8",
        "initialization": "torchvision FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1",
        "paper": "Long, Shelhamer & Darrell 2015",
    },
    "unet": {
        "task": "semantic", "backbone": "none (symmetric encoder-decoder)",
        "initialization": "random (Kaiming)",
        "paper": "Ronneberger, Fischer & Brox 2015",
    },
    "unetpp": {
        "task": "semantic", "backbone": "none (nested encoder-decoder)",
        "initialization": "random (Kaiming)",
        "paper": "Zhou et al. 2018",
    },
    "segnet": {
        "task": "semantic", "backbone": "VGG16-BN encoder",
        "initialization": "ImageNet VGG16_BN_Weights.IMAGENET1K_V1 (encoder only)",
        "paper": "Badrinarayanan, Kendall & Cipolla 2017",
    },
    "deeplabv3": {
        "task": "semantic", "backbone": "ResNet50, output stride 8, ASPP",
        "initialization": "torchvision DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1",
        "paper": "Chen et al. 2017",
    },
    "pspnet": {
        "task": "semantic", "backbone": "ResNet50 dilated, output stride 8",
        "initialization": "ImageNet ResNet50_Weights.IMAGENET1K_V2 (backbone only)",
        "paper": "Zhao et al. 2017",
    },
    "segformer": {
        "task": "semantic", "backbone": "MiT-B0 hierarchical transformer",
        "initialization": "nvidia/mit-b0 (ImageNet-1k pretrained encoder)",
        "paper": "Xie et al. 2021",
    },
    "maskrcnn": {
        "task": "instance", "backbone": "ResNet50-FPN",
        "initialization": "torchvision MaskRCNN_ResNet50_FPN_Weights.COCO_V1",
        "paper": "He et al. 2017",
    },
    "yolo_seg": {
        "task": "instance", "backbone": "YOLO11n-seg",
        "initialization": "ultralytics yolo11n-seg.pt (COCO pretrained)",
        "paper": "Jocher et al. / Ultralytics",
    },
}


class TorchvisionSemanticWrapper(nn.Module):
    """Unwrap torchvision's OrderedDict output to a plain logits tensor.

    FCN and DeepLabV3 return ``{"out": ..., "aux": ...}``. The auxiliary
    branch is dropped: torchvision's pretrained checkpoints carry an aux head
    sized for 21 VOC classes, keeping it would add a loss term that only two
    of the seven semantic models have, and Section 13 asks for one shared
    recipe. Recorded as a deviation.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out["out"] if isinstance(out, dict) else out


class PSPNetWrapper(nn.Module):
    """Expose PSPNet's auxiliary output only where the trainer looks for it.

    Returns a dict while training so the loop can add the auxiliary term, and
    a bare tensor in eval so every evaluation path is identical across the
    seven semantic models.
    """

    def __init__(self, model: PSPNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if self.training and isinstance(out, dict):
            return out
        return out["out"] if isinstance(out, dict) else out


class SegFormerWrapper(nn.Module):
    """Upsample SegFormer's quarter-resolution logits to the input size.

    This is not a quirk to paper over, it is the architecture. Section 10
    asks how the hierarchical encoder and lightweight decoder differ from the
    convolutional encoder-decoders, and this is the clearest difference:
    SegFormer's MLP decoder fuses multi-scale features and predicts at
    stride 4, then relies on bilinear upsampling for the last two octaves.
    U-Net spends a whole decoder path with learned transposed convolutions
    getting back to full resolution.

    That is why SegFormer's decoder is so cheap and why its parameter count
    is so low, and it is a reason to watch its boundary scores in Section 24
    specifically - the final 4x is interpolation, not learning.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


def _build_fcn(num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision.models.segmentation import (
        FCN_ResNet50_Weights, fcn_resnet50,
    )
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
    model = fcn_resnet50(weights=weights, aux_loss=True)
    # The pretrained head predicts 21 VOC classes; ours predicts six. Only
    # the final 1x1 convolution is replaced, so every learned feature below
    # it survives - which is the whole reason to start from these weights.
    model.classifier[4] = nn.Conv2d(512, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return TorchvisionSemanticWrapper(model)


def _build_deeplabv3(num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights, deeplabv3_resnet50,
    )
    weights = (
        DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
    )
    model = deeplabv3_resnet50(weights=weights, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return TorchvisionSemanticWrapper(model)


def _build_segformer(num_classes: int, pretrained: bool) -> nn.Module:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        model = SegformerForSemanticSegmentation(
            SegformerConfig(num_labels=num_classes)
        )
    return SegFormerWrapper(model)


def _build_maskrcnn(num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Both prediction heads are resized from COCO's 91 classes to our six.
    # The backbone, FPN, RPN and RoI Align all carry over untouched, which is
    # most of what the COCO weights are worth.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, num_classes)
    return model


def get_segmentation_model(model_name: str, num_classes: int = NUM_CLASSES,
                           pretrained: bool = True) -> nn.Module:
    """Return a configured model by name. The Section 19 interface.

    ``yolo_seg`` is deliberately absent. Ultralytics owns its own training
    loop, dataset format and checkpointing, and wrapping it in an nn.Module
    to satisfy a uniform signature would mean either reimplementing that loop
    or pretending to. It is built and driven by ``yolo_adapter`` instead,
    against the same split, and scored by the same evaluator as Mask R-CNN -
    so the comparison Section 28 asks for is still like-for-like even though
    the training path is not shared.

    ``kmeans`` is likewise absent: it has no parameters and nothing to train.
    It lives in ``traditional.py``.
    """
    name = model_name.lower()

    if name == "unet":
        return UNet(num_classes)
    if name == "unetpp":
        return UNetPlusPlus(num_classes)
    if name == "segnet":
        return SegNet(num_classes, pretrained=pretrained)
    if name == "pspnet":
        return PSPNetWrapper(PSPNet(num_classes, pretrained=pretrained))
    if name == "fcn_resnet50":
        return _build_fcn(num_classes, pretrained)
    if name == "deeplabv3":
        return _build_deeplabv3(num_classes, pretrained)
    if name == "segformer":
        return _build_segformer(num_classes, pretrained)
    if name == "maskrcnn":
        return _build_maskrcnn(num_classes, pretrained)

    if name == "yolo_seg":
        raise ValueError(
            "yolo_seg is not built here; it is driven by yolo_adapter, which "
            "owns ultralytics' training loop. See models.get_segmentation_model."
        )
    if name == "kmeans":
        raise ValueError(
            "kmeans has no parameters to configure; see traditional.py."
        )
    raise ValueError(
        f"Unknown model {model_name!r}. Available: {', '.join(ALL_MODELS)}"
    )


def count_parameters(model: nn.Module) -> dict:
    """Total and trainable parameter counts, as Section 30 requires both.

    They differ only where layers are frozen. Nothing is frozen in this
    benchmark - every model fine-tunes end to end - so the two match, and
    that fact is itself what Section 30 asks to be documented.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "frozen_layers": "none (all models fine-tuned end to end)",
    }


def weight_size_mb(model: nn.Module) -> float:
    """Size of the weights alone, in mebibytes.

    Section 31 asks for equivalent artifacts to be compared. This measures
    parameters plus buffers at their stored precision - the inference weight
    file - and deliberately not a training checkpoint, which would also carry
    AdamW's two moment tensors per parameter and roughly triple the figure
    for reasons that have nothing to do with the architecture.
    """
    parameters = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (parameters + buffers) / BYTES_PER_MB


def describe_model(model_name: str, model: nn.Module = None) -> dict:
    """The Section 10 configuration record for one model."""
    card = dict(MODEL_CARDS[model_name.lower()])
    card["model"] = model_name
    if model is not None:
        card.update(count_parameters(model))
        card["weight_size_mb"] = round(weight_size_mb(model), 3)
    return card
