"""The four architectures implemented here rather than imported.

U-Net, U-Net++, SegNet and PSPNet have no canonical torchvision
implementation, so they are written out. That is not incidental to the
assignment: Sections 10 and 53 ask for the mechanism of each one to be
explained, and a model whose skip connections you wrote yourself is one you
can describe precisely.

Each class carries the property the assignment asks it to demonstrate:

* U-Net        - encoder feature maps concatenated into the decoder
* U-Net++      - nested dense skip pathways between them
* SegNet       - max-pooling *indices* carried instead of feature maps
* PSPNet       - pyramid pooling over a dilated backbone

Every model here returns logits at the input resolution, shape
(B, num_classes, H, W), so one training loop and one metric implementation
serve all seven semantic architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, VGG16_BN_Weights, resnet50, vgg16_bn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 convolutions with batch norm, the unit U-Net repeats.

    Batch norm was not in the 2015 U-Net paper - it predates wide adoption -
    but it is in every practical implementation since, and training a
    from-scratch model at batch size 8 without it against six pretrained
    models would confound "no pretraining" with "no normalization" in the
    Section 54 comparison.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net: encoder, bottleneck, decoder, with skip connections.

    The skip connections are the point. Each decoder stage receives the
    encoder feature map at its own resolution, concatenated onto the
    upsampled features from below. The encoder path destroys spatial detail
    through four poolings - by the bottleneck a 512x512 input is 32x32 - and
    upsampling alone cannot invent back where a boundary was. The skip
    carries that information across unchanged, which is why Section 54 asks
    why skip connections matter for segmentation and not for classification:
    a classifier never has to put the detail back.

    Trained from scratch. There is no ImageNet U-Net to load, and Section 54
    asks how important pretrained initialization was, which needs models on
    both sides of that line.
    """

    def __init__(self, num_classes: int, base: int = 64, in_channels: int = 3):
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]

        self.encoders = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.encoders.append(conv_block(channels, width))
            channels = width

        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(widths[-1], widths[-1] * 2)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        channels = widths[-1] * 2
        for width in reversed(widths):
            self.upsamples.append(nn.ConvTranspose2d(channels, width, 2, stride=2))
            # in: upsampled (width) + skip (width)
            self.decoders.append(conv_block(width * 2, width))
            channels = width

        self.classifier = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips)
        ):
            x = upsample(x)
            # Odd input sizes leave the upsampled map a pixel short of its
            # skip. Aligning to the skip keeps the concatenation valid without
            # constraining the input to powers of two.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = decoder(torch.cat([skip, x], dim=1))

        return self.classifier(x)


class UNetPlusPlus(nn.Module):
    """U-Net++: the same backbone, with the skip connections made dense.

    Section 10 asks for the motivation, which is the semantic gap. In U-Net,
    decoder stage 1 concatenates the output of encoder stage 1 - features
    that have been through two convolutions - with features that have been
    through the entire network and back. Those two tensors describe the image
    at very different levels of abstraction, and the decoder has to reconcile
    them in one block.

    U-Net++ fills in the intermediate nodes. Node X[i][j] receives every
    earlier node at its own depth plus the upsampled node one level down, so
    the encoder feature reaches the decoder through a chain of convolutions
    of gradually increasing depth rather than in one jump. The cost is the
    extra nodes: roughly a third more parameters and noticeably more
    activation memory than plain U-Net at the same width.

    Deep supervision is available but off by default. Section 13 asks for a
    shared recipe, and deep supervision changes the loss, which would make
    U-Net++ the only model trained against a different objective and confound
    the direct U-Net comparison Section 10 asks for.

    The default width is 64 to match UNet's, and that matters more than it
    looks. U-Net++ is usually published at base 32, which here would give
    11.0M parameters against U-Net's 31.0M - so "U-Net++ scored lower" would
    be a statement about a model a third the size, not about nested skip
    pathways. At equal width the two are 31.0M and 44.0M, and that 41.6%
    difference *is* the cost of the nested nodes. Section 54 asks how the two
    compare; this is the configuration in which that question has an answer.
    """

    def __init__(self, num_classes: int, base: int = 64, in_channels: int = 3,
                 deep_supervision: bool = False):
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8, base * 16]
        self.depth = len(widths)
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.blocks = nn.ModuleDict()
        self.ups = nn.ModuleDict()

        for i in range(self.depth):
            for j in range(self.depth - i):
                if j == 0:
                    in_ch = in_channels if i == 0 else widths[i - 1]
                else:
                    # Every earlier node at this depth, plus the upsampled
                    # node from one level down.
                    in_ch = widths[i] * j + widths[i + 1]
                self.blocks[f"x{i}_{j}"] = conv_block(in_ch, widths[i])
                if j > 0:
                    self.ups[f"u{i}_{j}"] = nn.ConvTranspose2d(
                        widths[i + 1], widths[i + 1], 2, stride=2
                    )

        if deep_supervision:
            self.classifier = nn.ModuleList(
                [nn.Conv2d(widths[0], num_classes, 1) for _ in range(self.depth - 1)]
            )
        else:
            self.classifier = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x: torch.Tensor):
        nodes = {}
        for j in range(self.depth):
            for i in range(self.depth - j):
                if j == 0:
                    source = x if i == 0 else self.pool(nodes[(i - 1, 0)])
                    nodes[(i, j)] = self.blocks[f"x{i}_{j}"](source)
                else:
                    below = self.ups[f"u{i}_{j}"](nodes[(i + 1, j - 1)])
                    same = [nodes[(i, k)] for k in range(j)]
                    if below.shape[-2:] != same[0].shape[-2:]:
                        below = F.interpolate(
                            below, size=same[0].shape[-2:],
                            mode="bilinear", align_corners=False,
                        )
                    nodes[(i, j)] = self.blocks[f"x{i}_{j}"](
                        torch.cat(same + [below], dim=1)
                    )

        final = nodes[(0, self.depth - 1)]
        if self.deep_supervision:
            if self.training:
                return [
                    head(nodes[(0, j + 1)])
                    for j, head in enumerate(self.classifier)
                ]
            # In eval the heads are a ModuleList, which is not callable. The
            # deepest one is the full-depth prediction and the only one the
            # metrics should ever see; the shallower heads exist to put
            # gradient nearer the encoder during training, not to be scored.
            return self.classifier[-1](final)
        return self.classifier(final)


class SegNet(nn.Module):
    """SegNet: a VGG16 encoder whose decoder unpools with the saved indices.

    Section 10 asks for this to be contrasted with U-Net, and the contrast is
    about what crosses from encoder to decoder. U-Net sends the whole feature
    map - C x H x W floats per stage, concatenated into the decoder. SegNet
    sends only the argmax positions from each max pool: which of the four
    pixels in every 2x2 window was the largest.

    That is far less information, and it is deliberately less. The decoder
    unpools by scattering values back to exactly the positions the encoder
    pooled them from, so boundaries land where the encoder found them, and
    the convolutions that follow fill in the rest. The saving is memory: the
    indices are integers, not feature maps, so SegNet's decoder needs a small
    fraction of the activation memory U-Net's concatenations require. The
    cost is that the decoder has to reconstruct feature content that U-Net is
    handed outright.

    The encoder is VGG16 with batch norm, initialized from ImageNet, which is
    what the SegNet paper used.
    """

    # VGG16 stage layouts: channel widths per stage.
    STAGES = [
        (3, 64, 2), (64, 128, 2), (128, 256, 3), (256, 512, 3), (512, 512, 3),
    ]

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.encoders = nn.ModuleList()
        for in_ch, out_ch, repeats in self.STAGES:
            layers, channels = [], in_ch
            for _ in range(repeats):
                layers += [
                    nn.Conv2d(channels, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
                channels = out_ch
            self.encoders.append(nn.Sequential(*layers))

        self.decoders = nn.ModuleList()
        for index, (in_ch, out_ch, repeats) in enumerate(reversed(self.STAGES)):
            target = in_ch if index < len(self.STAGES) - 1 else out_ch
            layers, channels = [], out_ch
            for step in range(repeats):
                last = step == repeats - 1
                width = target if last else out_ch
                layers += [
                    nn.Conv2d(channels, width, 3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ]
                channels = width
            self.decoders.append(nn.Sequential(*layers))

        self.classifier = nn.Conv2d(self.STAGES[0][1], num_classes, 1)
        self.pretrained = pretrained
        if pretrained:
            self._load_vgg_encoder()

    def _load_vgg_encoder(self) -> None:
        """Copy VGG16-BN's convolutional weights into the encoder stages."""
        source = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).features
        donors = [m for m in source if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
        targets = [
            m for stage in self.encoders for m in stage
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))
        ]
        with torch.no_grad():
            for donor, target in zip(donors, targets):
                if donor.weight.shape == target.weight.shape:
                    target.weight.copy_(donor.weight)
                    if donor.bias is not None and target.bias is not None:
                        target.bias.copy_(donor.bias)
                    if isinstance(donor, nn.BatchNorm2d):
                        target.running_mean.copy_(donor.running_mean)
                        target.running_var.copy_(donor.running_var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices, sizes = [], []
        for encoder in self.encoders:
            x = encoder(x)
            sizes.append(x.shape)
            x, index = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(index)

        for decoder, index, size in zip(
            self.decoders, reversed(indices), reversed(sizes)
        ):
            x = F.max_unpool2d(x, index, 2, 2, output_size=size[-2:])
            x = decoder(x)

        return self.classifier(x)


class PyramidPoolingModule(nn.Module):
    """Pool the feature map at four scales, then concatenate all of them back.

    Section 10 asks how this captures information at different spatial
    scales. Each branch average-pools the whole feature map to a fixed grid -
    1x1, 2x2, 3x3, 6x6 - so the 1x1 branch produces a single vector
    summarizing the entire image and the 6x6 branch produces a coarse map of
    36 regional summaries. Each is projected to a narrow width, resized back
    to the input resolution, and concatenated onto the original features.

    The effect is that every output pixel sees global context alongside its
    own local features. A pixel in the middle of a large gray region is
    ambiguous on its own; knowing the image as a whole contains a road makes
    car far more likely than cat.
    """

    def __init__(self, in_channels: int, bins=(1, 2, 3, 6)):
        super().__init__()
        branch_channels = in_channels // len(bins)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for bin_size in bins
        ])
        self.out_channels = in_channels + branch_channels * len(bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = [x] + [
            F.interpolate(
                branch(x), size=size, mode="bilinear", align_corners=False
            )
            for branch in self.branches
        ]
        return torch.cat(pooled, dim=1)


class PSPNet(nn.Module):
    """PSPNet: a dilated ResNet50 backbone under a pyramid pooling module.

    The backbone's last two stages are dilated rather than strided, giving
    output stride 8 instead of 32. That keeps the feature map at 64x64 for a
    512x512 input - four times the linear resolution a stock ResNet50 would
    produce - without reducing the receptive field, because the dilation
    widens each kernel's reach to compensate for the stride it gave up. It
    is the same trade DeepLabV3 makes, and it is why both are more expensive
    per image than an encoder-decoder of similar depth.

    The auxiliary head on stage 3 is part of the original design and is used
    during training only. It is reported as a documented deviation from the
    shared Section 13 recipe, because it adds a second loss term that the
    other six semantic models do not have.
    """

    def __init__(self, num_classes: int, pretrained: bool = True,
                 aux_loss: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(
            weights=weights, replace_stride_with_dilation=[False, True, True]
        )

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.pyramid = PyramidPoolingModule(2048)
        self.head = nn.Sequential(
            nn.Conv2d(self.pyramid.out_channels, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1),
        )

        self.aux_loss = aux_loss
        if aux_loss:
            self.auxiliary = nn.Sequential(
                nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, 1),
            )

    def forward(self, x: torch.Tensor):
        size = x.shape[-2:]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        stage3 = self.layer3(x)
        x = self.layer4(stage3)

        out = self.head(self.pyramid(x))
        out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)

        if self.aux_loss and self.training:
            aux = F.interpolate(
                self.auxiliary(stage3), size=size,
                mode="bilinear", align_corners=False,
            )
            return {"out": out, "aux": aux}
        return out
