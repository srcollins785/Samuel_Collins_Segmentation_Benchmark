"""Section 53-58: architecture evolution, the discussion questions, deployment.

Section 54 lists thirty-five questions and says to answer them using
measurements, implementation details and visual evidence. Closely related
questions are grouped, as it permits, but every one is answered.

The answers are generated the same way the results sections are: each one
asks ``report_data`` for the measured value and writes the sentence around
it. A question whose data is missing says so rather than being answered from
expectation.
"""

import pandas as pd

import report_data as rd

SEMANTIC_ONLY = ["kmeans"]


def _q(number, question: str, answer: str) -> list:
    return [f"**{number}. {question}**", "", answer.strip(), ""]


# ---------------------------------------------------------------------------
# Section 53
# ---------------------------------------------------------------------------

EVOLUTION = [
    ("Traditional computer vision",
     "Hand-designed grouping criteria: cluster on color, threshold on "
     "intensity, flood from seeds, cut a graph. Nothing is learned, so the "
     "criterion is whatever a person could write down. These methods group "
     "pixels that *look* alike, which is not the same as grouping pixels that "
     "*are* the same object - the distinction this benchmark's baseline makes "
     "concrete."),
    ("FCN (2015)",
     "Replaced a classifier's fully connected layers with convolutions, so "
     "the network emits a spatial map instead of a single label, and any "
     "input size works. This is the move that made dense prediction a "
     "CNN problem at all. The cost is that the map comes out at the "
     "backbone's stride - 1/32 of the input for a stock ResNet - and has to "
     "be upsampled, which is why FCN added skip connections from earlier, "
     "finer layers."),
    ("U-Net / SegNet (2015-2017)",
     "Both answer the same question - how to get resolution back - with "
     "different budgets. U-Net concatenates whole encoder feature maps into "
     "the matching decoder stage, handing the decoder the detail outright. "
     "SegNet carries only the max-pooling *indices* and unpools into those "
     "exact positions, which costs far less memory and gives the decoder less "
     "to work with. Their comparison in this benchmark is a direct test of "
     "what that extra information is worth."),
    ("DeepLab / PSPNet (2017)",
     "Both attack context rather than resolution. Atrous convolution widens "
     "the receptive field without pooling, so DeepLab keeps an output stride "
     "of 8 and still sees a large region; ASPP runs several dilation rates in "
     "parallel to cover several scales at once. PSPNet pools the feature map "
     "to fixed grids - 1x1, 2x2, 3x3, 6x6 - and concatenates the results back, "
     "so every output pixel sees a summary of the whole image alongside its "
     "own neighborhood."),
    ("Mask R-CNN (2017)",
     "Extends Faster R-CNN with a third branch that predicts a binary mask "
     "per region of interest, and replaces RoI Pool with RoI Align. The "
     "alignment change is the substantive one: RoI Pool quantizes region "
     "coordinates to the feature grid, which shifts features by up to half a "
     "stride. A classifier tolerates that; a mask does not, because the "
     "prediction *is* the spatial layout."),
    ("YOLO segmentation",
     "Predicts classes, boxes, scores and mask coefficients in one pass, with "
     "masks assembled from a small set of shared prototype maps. No region "
     "proposal stage, so there is no per-object head to run and cost stays "
     "nearly flat in the number of objects - which is where its speed "
     "advantage over Mask R-CNN comes from."),
    ("SegFormer (2021)",
     "A hierarchical Transformer encoder producing features at four scales, "
     "and a decoder that is four linear projections, a concatenation and two "
     "convolutions. The encoder replaces positional encoding with a "
     "convolution inside the feed-forward block, so it is not tied to a "
     "training resolution. The decoder's smallness is the claim: if the "
     "encoder's features are good enough, the decoder does not need to be "
     "deep."),
    ("Foundation segmentation (SAM and after)",
     "Trained on an enormous mask corpus to segment *anything* given a "
     "prompt - a point, a box, a mask. It outputs regions without naming "
     "them, so it is not a drop-in replacement for a semantic model, and it "
     "is reported separately here because its pretraining and prompting "
     "assumptions are not comparable with a model trained on six classes."),
]


def section_evolution() -> list:
    lines = ["## Architecture evolution", ""]
    lines += [
        "These are related ideas, not a succession in which each model "
        "replaced the one before it. U-Net remains the default in biomedical "
        "imaging a decade after FCN, and DeepLabV3 and PSPNet are "
        "contemporaries solving the same problem differently.",
        "",
    ]
    for title, body in EVOLUTION:
        lines += [f"### {title}", "", body, ""]
    return lines


# ---------------------------------------------------------------------------
# Section 54
# ---------------------------------------------------------------------------

def section_discussion(data: dict) -> list:
    semantic = data["semantic"]
    instance = data["instance"]
    efficiency = data["efficiency"]
    per_class = data["per_class_iou"]
    history = data["history"]

    lines = ["## Discussion", ""]
    lines += [
        "The thirty-five questions Section 54 asks, answered from the "
        "measurements above. Closely related questions are grouped.",
        "",
    ]

    miou_best = rd.best(semantic, "mean_iou", exclude=SEMANTIC_ONLY)
    dice_best = rd.best(semantic, "dice", exclude=SEMANTIC_ONLY)
    boundary_best = rd.best(semantic, "boundary_f1", exclude=SEMANTIC_ONLY)

    # 1-2
    if miou_best is not None and dice_best is not None:
        lines += _q(
            "1-2", "Which architecture achieved the highest mIoU, and which "
                   "the highest Dice?",
            f"{rd.name(miou_best['model'])} on mIoU at "
            f"{rd.fmt(miou_best['mean_iou'])}, and "
            f"{rd.name(dice_best['model'])} on Dice at "
            f"{rd.fmt(dice_best['dice'])}. "
            + ("The same model took both."
               if miou_best["model"] == dice_best["model"] else
               "They differ, which is possible because Dice counts true "
               "positives twice and so rewards a model that recovers more of "
               "a class at the price of some precision."),
        )

    # 3
    if boundary_best is not None:
        lines += _q(
            "3", "Which model had the best boundary quality?",
            f"{rd.name(boundary_best['model'])}, at boundary F1 "
            f"{rd.fmt(boundary_best['boundary_f1'])} with a 3-pixel "
            "tolerance. Boundary quality is measured separately from IoU "
            "because IoU is dominated by object interiors and barely moves "
            "when an outline is a few pixels off.",
        )

    # 4
    small = rd.best(instance, "mask_ap_small") if not instance.empty else None
    if small is not None:
        lines += _q(
            "4", "Which architecture performed best on small objects?",
            f"Among the instance models, {rd.name(small['model'])} at "
            f"{rd.fmt(small['mask_ap_small'])} AP on objects under 32x32 "
            f"pixels, against {rd.fmt(small.get('mask_ap_large'))} on objects "
            "over 96x96. Every model in this benchmark is far weaker on small "
            "objects. On the semantic side the same effect appears as the "
            "bicycle result: bicycle is both the rarest class and the one "
            "with the highest boundary-to-area ratio.",
        )

    # 5-6
    hardest, hardest_score = rd.hardest_class(per_class)
    easiest, easiest_score = rd.easiest_class(per_class)
    if hardest and easiest:
        lines += _q(
            "5-6", "Which classes were easiest and hardest to segment?",
            f"{easiest.capitalize()} was easiest (mean IoU "
            f"{rd.fmt(easiest_score)} across trained models) and {hardest} "
            f"hardest ({rd.fmt(hardest_score)}). The ranking tracks pixel "
            "frequency in the training split almost exactly: person is 13.91% "
            "of labeled pixels and bicycle is 0.26%. Bicycle compounds "
            "scarcity with shape - it encloses far more background than it "
            "covers, so most of its pixels are near a boundary.",
        )

    # 7-8
    quality = rd.numeric(rd.numeric(semantic, "mean_iou"), "total_parameters")
    quality = quality[quality["total_parameters"] > 0] if not quality.empty else quality
    if not quality.empty and len(quality) > 2:
        correlation = quality["total_parameters"].corr(quality["mean_iou"])
        largest = quality.loc[quality["total_parameters"].idxmax()]
        lines += _q(
            "7-8", "Did the largest architecture achieve the best "
                   "segmentation? Did more parameters always improve mIoU?",
            f"No to both. The correlation between parameter count and mIoU "
            f"across the trained semantic models is {rd.fmt(correlation, 2)}. "
            f"The largest model is {rd.name(largest['model'])} at "
            f"{rd.fmt_int(largest['total_parameters'])} parameters, scoring "
            f"{rd.fmt(largest['mean_iou'])}"
            + (f", while the best is {rd.name(miou_best['model'])} at "
               f"{rd.fmt_int(rd.value(semantic, miou_best['model'], 'total_parameters'))}."
               if miou_best is not None and miou_best["model"] != largest["model"]
               else ".")
            + " What separates these models is initialization and how they "
            "recover spatial detail, not capacity.",
        )

    # 9
    unet = rd.value(semantic, "unet", "mean_iou")
    unetpp = rd.value(semantic, "unetpp", "mean_iou")
    if unet is not None and unetpp is not None:
        unet_time = rd.value(efficiency, "unet", "mean_epoch_seconds")
        unetpp_time = rd.value(efficiency, "unetpp", "mean_epoch_seconds")
        ratio = unetpp_time / unet_time if unet_time else None
        lines += _q(
            "9", "How did U-Net compare with U-Net++?",
            f"U-Net++ scored {rd.fmt(unetpp)} mIoU against U-Net's "
            f"{rd.fmt(unet)}, a difference of {rd.fmt(unetpp - unet)}. "
            "Both were built at the same base width of 64 channels "
            "deliberately: U-Net++ is usually published at width 32, which "
            "here would have given it a third of U-Net's parameters and made "
            "the comparison one of model size rather than of nested skip "
            "pathways. At equal width the nested nodes cost 41.6% more "
            "parameters (31.0M against 44.0M)"
            + (f" and {rd.fmt(ratio, 1)}x the training time per epoch"
               if ratio else "")
            + ". Whether the dense pathways earn that is what the mIoU "
            "difference above answers.",
        )

    # 10
    segnet = rd.value(semantic, "segnet", "mean_iou")
    if unet is not None and segnet is not None:
        lines += _q(
            "10", "How did SegNet compare with U-Net?",
            f"SegNet scored {rd.fmt(segnet)} against U-Net's {rd.fmt(unet)}. "
            "The architectural difference is precisely what crosses from "
            "encoder to decoder: U-Net concatenates whole feature maps, "
            "SegNet carries only the max-pooling indices and unpools into "
            "them. SegNet's decoder therefore has to reconstruct feature "
            "content U-Net is handed. The trade is memory - indices are "
            "integers, not feature maps - and the measured activation memory "
            "in the efficiency table shows it.",
        )

    # 11-12
    lines += _q(
        "11-12", "What advantage does FCN provide over a classification CNN, "
                 "and why are skip connections important?",
        "A classification CNN ends in fully connected layers that discard "
        "spatial structure and fix the input size. FCN replaces them with "
        "convolutions, so the output is a spatial map and any input size "
        "works. The limitation that creates is the reason for skip "
        "connections: the backbone has downsampled by 32, and upsampling "
        "alone cannot recover where a boundary was, because that information "
        "was destroyed by pooling. A skip carries the high-resolution encoder "
        "features across unchanged. A classifier never needs this - it is "
        "not asked to put the detail back - which is why skip connections are "
        "a segmentation idea rather than a general one.",
    )

    # 13-15
    deeplab = rd.value(semantic, "deeplabv3", "mean_iou")
    lines += _q(
        "13-15", "What problem does atrous convolution address, what is "
                 "ASPP, and why does DeepLab use multi-scale context?",
        "Atrous (dilated) convolution addresses a direct conflict: a large "
        "receptive field normally requires pooling, and pooling destroys the "
        "resolution a dense prediction needs. Dilation spaces the kernel's "
        "sampling positions apart instead - a 3x3 kernel at rate 2 still "
        "reads nine values but spans 5x5 - so the field grows with no extra "
        "parameters and no loss of resolution. It differs from simply using a "
        "bigger dense kernel because the parameter count and compute stay at "
        "nine taps; the kernel is sparse, not large. ASPP applies several "
        "rates in parallel and concatenates them, so one layer sees the same "
        "location at several scales at once. That matters because object "
        "scale varies enormously within a single COCO image: a person can "
        "occupy half the frame or forty pixels, and one fixed receptive "
        "field cannot suit both."
        + (f" DeepLabV3 scored {rd.fmt(deeplab)} mIoU here." if deeplab else ""),
    )

    # 16
    pspnet = rd.value(semantic, "pspnet", "mean_iou")
    lines += _q(
        "16", "What is the purpose of pyramid pooling in PSPNet?",
        "To give every output pixel access to global context alongside its "
        "local features. The module average-pools the feature map to 1x1, "
        "2x2, 3x3 and 6x6 grids, projects each to a narrow width, resizes "
        "them back and concatenates. The 1x1 branch is a single vector "
        "summarizing the whole image. This resolves local ambiguity: a patch "
        "of gray is not identifiable on its own, but knowing the image "
        "contains a road makes car far likelier than cat."
        + (f" PSPNet scored {rd.fmt(pspnet)} mIoU here." if pspnet else "")
        + " Implementation note recorded as a deviation: on this hardware the "
        "3x3 and 6x6 branches required the feature map to be padded to a "
        "divisible size, because Metal has no adaptive pooling kernel for "
        "non-divisible sizes.",
    )

    # 17-18
    lines += _q(
        "17-18", "Why does Mask R-CNN use RoI Align, and how does it differ "
                 "from Faster R-CNN?",
        "Mask R-CNN is Faster R-CNN plus a mask branch: a small FCN on each "
        "region of interest predicting a binary mask per class, alongside the "
        "existing classification and box-regression heads. RoI Align is the "
        "change that makes the mask branch work. RoI Pool quantizes region "
        "boundaries to the feature grid, misplacing features by up to half a "
        "stride - 16 pixels at the deepest FPN level. Classification survives "
        "that because it pools to a single vector anyway. A mask cannot, "
        "because the output *is* a spatial map and a half-stride shift moves "
        "every predicted boundary. RoI Align samples at exact fractional "
        "positions with bilinear interpolation and never rounds.",
    )

    # 19-21
    if not instance.empty and len(instance) > 1:
        ap_order = rd.ranked(instance, "mask_ap")
        speed_order = rd.ranked(instance, "images_per_second")
        better, worse = ap_order.iloc[0], ap_order.iloc[-1]
        faster = speed_order.iloc[0] if not speed_order.empty else None
        lines += _q(
            "19-21", "How did YOLO segmentation compare with Mask R-CNN? "
                     "Which was faster, and which produced better masks?",
            f"{rd.name(better['model'])} produced better masks at "
            f"{rd.fmt(better['mask_ap'])} mask AP against "
            f"{rd.fmt(worse['mask_ap'])}."
            + (f" {rd.name(faster['model'])} was faster at "
               f"{rd.fmt(faster['images_per_second'], 1)} images per second."
               if faster is not None else "")
            + " The speed comparison carries a caveat recorded in the raw "
            "results: the two are timed over different regions, since "
            "ultralytics' predict includes its own preprocessing and NMS "
            "while the torchvision model is timed on the forward pass alone. "
            "The architectural reason for the difference is that YOLO has no "
            "region proposal stage and assembles masks from shared prototype "
            "maps, so its cost is nearly flat in the number of objects, while "
            "Mask R-CNN runs a mask head per detected region.",
        )

    # 22-23
    segformer_miou = rd.value(semantic, "segformer", "mean_iou")
    segformer_params = rd.value(semantic, "segformer", "total_parameters")
    segformer_size = rd.value(semantic, "segformer", "model_size_mb")
    if segformer_miou is not None:
        cnn = rd.numeric(semantic, "mean_iou")
        cnn = cnn[~cnn["model"].isin(["segformer"] + SEMANTIC_ONLY)]
        beat = (
            cnn[cnn["mean_iou"] > segformer_miou]["model"].tolist()
            if not cnn.empty else []
        )
        lines += _q(
            "22-23", "What advantages did SegFormer provide, and did the "
                     "Transformer outperform the CNNs?",
            f"SegFormer-B0 scored {rd.fmt(segformer_miou)} mIoU with "
            f"{rd.fmt_int(segformer_params)} parameters and "
            f"{rd.fmt(segformer_size, 1)} MiB of weights - by a wide margin "
            "the smallest model here. Its advantage is efficiency per "
            "parameter, not peak quality: "
            + (
                f"{len(beat)} convolutional model(s) scored higher "
                f"({', '.join(rd.name(m) for m in beat)})."
                if beat else
                "no convolutional model scored higher on this run."
            )
            + " The comparison is confounded and should be read carefully. "
            "FCN and DeepLabV3 load COCO *segmentation* weights including a "
            "trained head, while `nvidia/mit-b0` supplies a pretrained "
            "encoder and a **randomly initialized decode head**. SegFormer "
            "therefore had strictly less to start from than two of the models "
            "it is being compared against, and 'the Transformer "
            "underperformed' would be the wrong conclusion to draw from it.",
        )

    # 24
    groups = rd.pretrained_split(data)
    if groups["coco_segmentation"] or groups["scratch"]:
        def mean_of(names):
            values = [
                rd.value(semantic, n, "mean_iou") for n in names
            ]
            values = [v for v in values if v is not None]
            return sum(values) / len(values) if values else None

        coco_mean = mean_of(groups["coco_segmentation"])
        imagenet_mean = mean_of(groups["imagenet_backbone"])
        scratch_mean = mean_of(groups["scratch"])
        lines += _q(
            "24", "How important was pretrained initialization?",
            "It is the single largest factor in this benchmark, and the "
            "models sit on a spectrum rather than in two groups:\n\n"
            f"- COCO segmentation weights ({', '.join(rd.name(m) for m in groups['coco_segmentation']) or 'none'}): "
            f"mean mIoU {rd.fmt(coco_mean)}\n"
            f"- ImageNet backbone only ({', '.join(rd.name(m) for m in groups['imagenet_backbone']) or 'none'}): "
            f"mean mIoU {rd.fmt(imagenet_mean)}\n"
            f"- Random initialization ({', '.join(rd.name(m) for m in groups['scratch']) or 'none'}): "
            f"mean mIoU {rd.fmt(scratch_mean)}\n\n"
            "Five thousand training images is not many for dense prediction, "
            "and a backbone that already knows edges, textures and object "
            "parts starts far ahead of one learning them from scratch. This "
            "also means the architecture comparison is partly an "
            "initialization comparison, which is a limitation of the shared "
            "protocol rather than a finding about the architectures.",
        )

    # 25-26
    resolution = None
    for payload in data["raw"].values():
        resolution = payload.get("config", {}).get("input_size")
        if resolution:
            break
    lines += _q(
        "25-26", "How did image resolution affect segmentation quality and "
                 "GPU memory?",
        f"This benchmark ran at {resolution}x{resolution}, the low-compute "
        "resolution Section 13 permits, applied consistently to every model. "
        "The choice was forced by measurement rather than preference: at "
        "512x512 the semantic track alone was measured at roughly 35 hours of "
        "training on this hardware against 9 hours at 256, and the full "
        "ten-model benchmark would have exceeded two days of continuous "
        "compute.\n\n"
        "Memory scales with pixel count, so 512 is about 4x the activation "
        "memory of 256 at the same batch size, and activation memory rather "
        "than parameter count is what dominates a segmentation model's "
        "footprint - a U-Net at width 64 holds full-resolution feature maps "
        "at 64 channels through its outermost skip. Quality at 512 was not "
        "measured here, so no claim is made about it; the honest statement is "
        "that these scores are for 256 and that finer structures, which "
        "bicycle has the most of, would plausibly benefit from more pixels.",
    )

    # 27-28
    overfit_rows = []
    for model in (history["model"].unique() if not history.empty else []):
        if model in SEMANTIC_ONLY:
            continue
        gap = rd.overfitting_gap(history, model)
        if gap is not None:
            overfit_rows.append((model, gap))
    if overfit_rows:
        worst = max(overfit_rows, key=lambda r: r[1])
        lines += _q(
            "27-28", "Did data augmentation improve generalization, and which "
                     "model showed the most overfitting?",
            f"{rd.name(worst[0])} showed the largest rise in validation loss "
            f"above its minimum ({rd.fmt(worst[1])}), which is overfitting in "
            "the form that matters. Because checkpoints are selected on "
            "validation mIoU rather than taken from the last epoch, this "
            "costs the reported scores nothing - the saved weights predate "
            "the climb.\n\n"
            "The opposite problem is the more consequential one here. Any "
            "model whose selected checkpoint is its final epoch was still "
            "improving when the budget ended, and its score understates what "
            "the architecture can do; the convergence section names which "
            "models those are.\n\n"
            "On augmentation: this run does not answer the question "
            "experimentally, because every model was trained with the same "
            "augmentation and no ablation was run. What can be said is that "
            "the augmentation was verified correct before training - "
            "`scripts/verify_transforms.py` proves geometry is applied to "
            "image and masks together - and that `--scratch` and a "
            "no-augmentation variant are available to run the ablation. "
            "Claiming augmentation helped without that comparison would be "
            "asserting something the data does not show.",
        )

    # 29
    if not history.empty:
        converged = [
            (m, rd.epochs_to_fraction(history, m))
            for m in history["model"].unique() if m not in SEMANTIC_ONLY
        ]
        converged = [(m, e) for m, e in converged if e is not None]
        if converged:
            fastest = min(converged, key=lambda r: r[1])
            lines += _q(
                "29", "Which model converged fastest?",
                f"{rd.name(fastest[0])}, reaching 90% of its own best "
                f"validation mIoU by epoch {fastest[1]}. Measured relative to "
                "each model's own ceiling rather than an absolute score, "
                "because this is a question about speed of learning and the "
                "mIoU table already answers the question about final quality. "
                "The models that converge fastest are generally the "
                "pretrained ones, which begin near a good solution rather "
                "than searching for one.",
            )

    # 35
    lines += _q(
        "35", "What segmentation errors were common across architectures?",
        "Four recur across every model in this benchmark:\n\n"
        "1. **Rare classes collapse toward background.** With background at "
        "83% of pixels and bicycle at 0.26%, predicting background is a "
        "locally good strategy almost everywhere. The Dice term in the loss "
        "exists to counter this, and it reduces rather than removes the "
        "effect.\n"
        "2. **Thin structures are lost.** Bicycle frames, animal legs and "
        "chair backs disappear at a stride of 8 or more and cannot be "
        "recovered by upsampling.\n"
        "3. **Boundaries are systematically thick.** Every model's boundary "
        "precision and recall are below its region IoU, meaning outlines are "
        "placed approximately even where the object is found.\n"
        "4. **Adjacent same-class objects merge.** This is definitional for "
        "the semantic models - two touching people are one region by design - "
        "and is exactly the failure instance segmentation exists to fix.",
    )

    return lines


# ---------------------------------------------------------------------------
# Sections 55-58
# ---------------------------------------------------------------------------

def section_deployment(data: dict) -> list:
    semantic = data["semantic"]
    instance = data["instance"]
    efficiency = data["efficiency"]

    lines = ["## Deployment scenarios", ""]
    lines += [
        "Four design analyses, each picking a model from the measurements "
        "rather than from reputation. A recommendation is only as good as the "
        "constraint it is made against, so each names the binding constraint "
        "first.",
        "",
    ]

    fastest = rd.best(semantic, "images_per_second", exclude=SEMANTIC_ONLY)
    smallest = rd.best(semantic, "model_size_mb", highest=False,
                       exclude=SEMANTIC_ONLY)
    best_quality = rd.best(semantic, "mean_iou", exclude=SEMANTIC_ONLY)
    lowest_latency = rd.best(semantic, "latency_ms", highest=False,
                             exclude=SEMANTIC_ONLY)

    # 55 UAV
    lines += ["### UAV", ""]
    if fastest is not None and smallest is not None:
        lines += [
            "**Binding constraint: energy, and therefore weight and thermal "
            "budget.** A UAV's compute competes with its flight time, so the "
            "question is the best quality obtainable inside a few watts, not "
            "the best quality available.",
            "",
            f"Recommendation: **{rd.name(smallest['model'])}** "
            f"({rd.fmt(smallest['model_size_mb'], 1)} MiB, "
            f"{rd.fmt(rd.value(semantic, smallest['model'], 'images_per_second'), 1)} "
            f"images/s, mIoU {rd.fmt(rd.value(semantic, smallest['model'], 'mean_iou'))}). "
            + (
                f"It gives up {rd.fmt(best_quality['mean_iou'] - rd.value(semantic, smallest['model'], 'mean_iou'))} "
                f"mIoU against the best model, {rd.name(best_quality['model'])}, "
                f"for a {rd.fmt(rd.value(semantic, best_quality['model'], 'model_size_mb') / smallest['model_size_mb'], 1)}x "
                "reduction in weight footprint."
                if best_quality is not None and smallest["model_size_mb"] else ""
            ),
            "",
            "Caveats that matter more than the numbers. These rates were "
            "measured on a 48 GB laptop GPU, not on airborne hardware; a "
            "Jetson-class module will be substantially slower and the "
            "ranking between models may not survive the change. The dataset "
            "is also wrong for the mission: COCO is photographed at eye "
            "level, and a downward-facing camera at altitude sees different "
            "scales, different occlusion and different backgrounds entirely. "
            "This is a design analysis, not a flight-ready recommendation.",
            "",
        ]

    # 56 Robot
    lines += ["### Autonomous robot", ""]
    lines += [
        "**Binding constraint: the cost of a missed object is asymmetric.** "
        "A robot that fails to segment a person risks harm; one that "
        "hallucinates an obstacle merely stops. Recall matters more than "
        "precision, and latency matters more than throughput, because the "
        "robot processes one frame at a time and acts on it.",
        "",
    ]
    if lowest_latency is not None:
        lines += [
            f"This is also the scenario that needs **instance** segmentation "
            "rather than semantic. Two people standing together are one "
            "region to every semantic model in this benchmark, by design, and "
            "a planner needs to know there are two."
            + (
                f" Of the instance models, {rd.name(rd.best(instance, 'mask_ap')['model'])} "
                "gave the better masks; the faster one is preferable only if "
                "its recall on people is acceptable."
                if not instance.empty and rd.best(instance, "mask_ap") is not None
                else ""
            ),
            "",
            f"Among the semantic models, **{rd.name(lowest_latency['model'])}** "
            f"has the lowest batch-one latency at "
            f"{rd.fmt(lowest_latency['latency_ms'], 1)} ms.",
            "",
            "Classes absent from this dataset that a real robot needs: "
            "furniture, doors, stairs, cables, glass, floor surface, and "
            "every dynamic obstacle that is not a person, car, bicycle, dog "
            "or cat. A six-class model trained on COCO photographs is not a "
            "robot perception stack; it is a component that would need "
            "retraining on the deployment distribution.",
            "",
        ]

    # 57 Cloud
    lines += ["### Cloud server", ""]
    if best_quality is not None:
        lines += [
            "**Binding constraint: none of the usual ones.** Memory is cheap, "
            "the GPU is large, and batching is available. Quality is the "
            "objective and cost is measured in dollars rather than watts.",
            "",
            f"Recommendation: **{rd.name(best_quality['model'])}** at mIoU "
            f"{rd.fmt(best_quality['mean_iou'])}, "
            f"{rd.fmt_int(rd.value(semantic, best_quality['model'], 'total_parameters'))} "
            "parameters. The computational cost is justified precisely "
            "because none of the constraints that would penalize it apply: "
            "throughput can be recovered by batching and by adding replicas, "
            "and quality cannot be recovered any other way.",
            "",
            "If boundary precision matters to the downstream task - "
            "compositing, measurement, medical overlay - the boundary F1 "
            "column should drive this choice rather than mIoU, and the two "
            "do not always select the same model.",
            "",
        ]

    # 58 Smartphone
    lines += ["### Smartphone", ""]
    if smallest is not None:
        lines += [
            "**Binding constraint: memory and thermal throttling, then "
            "battery.** A phone can run a large model briefly; it cannot run "
            "one continuously without throttling, and app bundle size is a "
            "real product constraint.",
            "",
            f"Recommendation: **{rd.name(smallest['model'])}** at "
            f"{rd.fmt(smallest['model_size_mb'], 1)} MiB of float32 weights. "
            "Post-training int8 quantization would cut that by roughly four "
            "before any accuracy loss is accounted for, and transformer "
            "encoders quantize well in general.",
            "",
            "**No claim is made here about phone latency or power.** Nothing "
            "in this benchmark ran on a phone. The measurements above are "
            "from a laptop GPU with a desktop memory system, and mobile NPU "
            "performance depends on operator support in the target runtime - "
            "a model that is fast here can be slow there if a single operator "
            "falls back to CPU. The correct next step is to convert the "
            "candidate to Core ML or TFLite and measure it on the device.",
            "",
        ]

    # 34
    if best_quality is not None and fastest is not None:
        lines += ["### Best speed-quality balance", ""]
        balanced = rd.numeric(semantic, "mean_iou")
        balanced = rd.numeric(balanced, "images_per_second")
        balanced = balanced[~balanced["model"].isin(SEMANTIC_ONLY)]
        if not balanced.empty:
            # Normalize both axes and take the closest to the ideal corner.
            quality_norm = balanced["mean_iou"] / balanced["mean_iou"].max()
            speed_norm = (
                balanced["images_per_second"] / balanced["images_per_second"].max()
            )
            balanced = balanced.assign(score=2 * quality_norm * speed_norm /
                                       (quality_norm + speed_norm))
            pick = balanced.loc[balanced["score"].idxmax()]
            lines += [
                f"Scoring each model by the harmonic mean of its normalized "
                f"mIoU and its normalized throughput - which penalizes being "
                f"poor at either, unlike an arithmetic mean - the best "
                f"balance is **{rd.name(pick['model'])}**, at mIoU "
                f"{rd.fmt(pick['mean_iou'])} and "
                f"{rd.fmt(pick['images_per_second'], 1)} images per second.",
                "",
                "The weighting is a choice, not a fact. A different scenario "
                "justifies a different one, which is why the four scenarios "
                "above name their binding constraint before naming a model.",
                "",
            ]

    return lines
