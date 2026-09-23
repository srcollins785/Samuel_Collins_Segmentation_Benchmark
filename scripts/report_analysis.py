"""The results, discussion and deployment sections of the report.

Split from ``generate_report.py`` because the sections here all share one
property: every sentence in them depends on a measurement. The sections in
the other file describe what was built and why, and would read the same
whatever the numbers turned out to be.

Nothing here hardcodes a winner. Each claim asks ``report_data`` for the
measured answer and writes the sentence around it, so a rerun that reorders
the models reorders the prose too. Where the data cannot support a claim the
text says so rather than filling the gap.
"""

import pandas as pd

import report_data as rd

SEMANTIC_ONLY = ["kmeans"]  # excluded from semantic superlatives; see Section 10


def _lead(text: str) -> list:
    return [text.strip(), ""]


# ---------------------------------------------------------------------------
# Section 27, 23: semantic results
# ---------------------------------------------------------------------------

def section_semantic_results(data: dict) -> list:
    semantic = data["semantic"]
    lines = ["## Semantic segmentation results", ""]

    if semantic.empty:
        return lines + ["_No semantic results have been generated yet._", ""]

    lines += _lead(
        "Every number in this section comes from `semantic_segmentation_"
        "results.csv`, which the benchmark generates from the per-model "
        "result files. All models were evaluated on the same 1,000 held-out "
        "test images, and no model saw them during training or checkpoint "
        "selection."
    )

    lines.append(rd.table(semantic, {
        "model": "Model",
        "pixel_accuracy": "Pixel acc.",
        "mean_iou": "mIoU",
        "mean_iou_foreground": "mIoU (fg)",
        "dice": "Dice",
        "precision": "Precision",
        "recall": "Recall",
    }))
    lines.append("")

    winner = rd.best(semantic, "mean_iou", exclude=SEMANTIC_ONLY)
    dice_winner = rd.best(semantic, "dice", exclude=SEMANTIC_ONLY)

    if winner is not None:
        order = rd.ranked(semantic, "mean_iou", exclude=SEMANTIC_ONLY)
        runner = order.iloc[1] if len(order) > 1 else None
        margin = (
            f", ahead of {rd.name(runner['model'])} at "
            f"{rd.fmt(runner['mean_iou'])}"
            if runner is not None else ""
        )
        lines += _lead(
            f"**{rd.name(winner['model'])} reached the highest mean IoU at "
            f"{rd.fmt(winner['mean_iou'])}**{margin}. The same model "
            + (
                "also took the best Dice score"
                if dice_winner is not None
                and dice_winner["model"] == winner["model"]
                else f"did not take Dice, which went to "
                     f"{rd.name(dice_winner['model'])} at "
                     f"{rd.fmt(dice_winner['dice'])}"
            )
            + ". Dice and mIoU rank the models "
            + (
                "identically here"
                if dice_winner is not None
                and dice_winner["model"] == winner["model"]
                else "differently, which is worth noting: Dice weights "
                     "true positives twice and so is more forgiving of a "
                     "model that over-predicts a class"
            )
            + "."
        )

    # Background is 83% of pixels, so the two means answer different questions.
    both = rd.numeric(semantic, "mean_iou")
    if not both.empty and "mean_iou_foreground" in both.columns:
        both = rd.numeric(both, "mean_iou_foreground")
    if not both.empty:
        gap = (both["mean_iou"] - both["mean_iou_foreground"]).mean()
        lines += _lead(
            f"The two mean-IoU columns differ by {rd.fmt(gap)} on average, "
            "and the gap is not noise. Background is 83% of all valid pixels "
            "in this dataset and is trivially easy, so including it lifts "
            "every model's mean. The six-class figure is the one the required "
            "table reports; the foreground figure is the one that describes "
            "how well the five classes anyone cares about were actually "
            "segmented. Both are given here so neither can be read as the "
            "other."
        )

    return lines


def section_per_class(data: dict) -> list:
    per_class = data["per_class_iou"]
    lines = ["## Per-class IoU", ""]

    if per_class.empty:
        return lines + ["_Not yet generated._", ""]

    lines += _lead(
        "Section 23 asks for one column per semantic model. The dashes in the "
        "required format mean a value to be measured; here an `N/A` means the "
        "class was absent from both the prediction and the ground truth for "
        "that model, which is not the same as a score of zero and is not "
        "averaged as one."
    )
    lines.append(rd.table(per_class, places=3))
    lines.append("")

    hardest, hardest_score = rd.hardest_class(per_class)
    easiest, easiest_score = rd.easiest_class(per_class)

    if hardest and easiest:
        lines += _lead(
            f"**{easiest.capitalize()} was the easiest foreground class** "
            f"(mean IoU {rd.fmt(easiest_score)} across the trained models) "
            f"and **{hardest} was the hardest** ({rd.fmt(hardest_score)}). "
            "That ordering was largely set before any model was trained. "
            "In the training split, person occupies 13.91% of labeled pixels "
            "and bicycle occupies 0.26% - a factor of 53 - and the classes "
            "rank on IoU close to the way they rank on how much of them "
            "there is to learn from."
        )
        lines += _lead(
            "Bicycle is also hard for a reason that is not only scarcity. A "
            "bicycle is mostly holes: the frame, wheels and handlebars enclose "
            "far more background than they cover, so a large share of its "
            "pixels lie within three pixels of a boundary. A model that gets "
            "the object roughly right still loses much of the intersection, "
            "which is exactly the failure IoU punishes hardest and the reason "
            "the boundary section below is worth reading next to this table."
        )

    lines.append(rd.figure("per_class_iou.png", "Per-class IoU across semantic models"))
    return lines


# ---------------------------------------------------------------------------
# Section 24: boundary quality
# ---------------------------------------------------------------------------

def section_boundary(data: dict) -> list:
    semantic = data["semantic"]
    lines = ["## Boundary quality", ""]

    if semantic.empty or "boundary_f1" not in semantic.columns:
        return lines + ["_Not yet generated._", ""]

    tolerance = 3
    resolution = None
    for payload in data["raw"].values():
        conventions = payload.get("test", {}).get("conventions", {})
        if "tolerance_px" in conventions:
            tolerance = conventions["tolerance_px"]
        resolution = payload.get("config", {}).get("input_size", resolution)

    lines += _lead(
        f"Boundary precision, recall and F1 follow Csurka et al. (2013), at a "
        f"tolerance of **{tolerance} pixels** and an evaluation resolution of "
        f"**{resolution}x{resolution}**. A predicted boundary pixel counts as "
        "matched when a ground-truth boundary pixel lies within that distance, "
        "and vice versa for recall."
    )
    lines += _lead(
        "This is reported separately from IoU because IoU cannot see it. A "
        "person occupying 40,000 pixels whose outline is wrong by three pixels "
        "all the way around loses under 4% of its IoU while looking visibly "
        "wrong everywhere it matters. Two models can tie on mIoU and differ "
        "substantially here."
    )

    lines.append(rd.table(semantic, {
        "model": "Model",
        "boundary_precision": "B-Precision",
        "boundary_recall": "B-Recall",
        "boundary_f1": "Boundary F1",
        "mean_iou": "mIoU (for reference)",
    }))
    lines.append("")

    winner = rd.best(semantic, "boundary_f1", exclude=SEMANTIC_ONLY)
    miou_winner = rd.best(semantic, "mean_iou", exclude=SEMANTIC_ONLY)
    if winner is not None and miou_winner is not None:
        if winner["model"] == miou_winner["model"]:
            lines += _lead(
                f"**{rd.name(winner['model'])} had the best boundaries** "
                f"(F1 {rd.fmt(winner['boundary_f1'])}), and it also had the "
                "best mIoU. The two metrics agreeing is the ordinary case "
                "rather than a guarantee: a model that segments regions well "
                "usually places their edges well too."
            )
        else:
            lines += _lead(
                f"**{rd.name(winner['model'])} had the best boundaries** "
                f"(F1 {rd.fmt(winner['boundary_f1'])}) even though "
                f"{rd.name(miou_winner['model'])} had the best mIoU "
                f"({rd.fmt(miou_winner['mean_iou'])}). This is the case "
                "Section 24 exists for. The two metrics disagree, so a "
                "ranking on mIoU alone would have recommended a model whose "
                "edges are measurably worse - which matters for any "
                "application where a mask is used to cut something out or to "
                "measure a distance."
            )

    segformer = rd.value(semantic, "segformer", "boundary_f1")
    if segformer is not None:
        lines += _lead(
            f"SegFormer's boundary F1 of {rd.fmt(segformer)} is worth reading "
            "against its architecture. Its MLP decoder predicts at stride 4 "
            "and relies on bilinear upsampling for the final factor of four, "
            "so the last two octaves of detail are interpolated rather than "
            "learned. U-Net spends an entire decoder path with transposed "
            "convolutions recovering that same detail."
        )

    lines.append(rd.figure("boundary_f1.png", "Boundary F1 at a 3-pixel tolerance"))
    return lines


# ---------------------------------------------------------------------------
# Section 28, 25: instance results
# ---------------------------------------------------------------------------

def section_instance_results(data: dict) -> list:
    instance = data["instance"]
    lines = ["## Instance segmentation results", ""]

    if instance.empty:
        return lines + ["_No instance results have been generated yet._", ""]

    lines += _lead(
        "Mask AP is averaged over IoU thresholds 0.50 to 0.95 in steps of "
        "0.05, computed by pycocotools' COCOeval with `iouType=\"segm\"`, "
        "averaged per category and then over categories. Detections below a "
        "score of 0.05 are dropped and at most 100 are kept per image. Box AP "
        "is secondary: Section 25 is explicit that mask quality is the "
        "objective."
    )

    lines.append(rd.table(instance, {
        "model": "Model",
        "mask_ap": "Mask AP",
        "mask_ap50": "AP50",
        "mask_ap75": "AP75",
        "mask_ar_100": "Mask AR",
        "box_ap": "Box AP",
    }))
    lines.append("")

    winner = rd.best(instance, "mask_ap")
    if winner is not None and len(instance) > 1:
        order = rd.ranked(instance, "mask_ap")
        loser = order.iloc[-1]
        lines += _lead(
            f"**{rd.name(winner['model'])} produced the better masks** at "
            f"{rd.fmt(winner['mask_ap'])} mask AP against "
            f"{rd.fmt(loser['mask_ap'])}."
        )

    speed = rd.best(instance, "images_per_second")
    if speed is not None:
        slow = rd.ranked(instance, "images_per_second").iloc[-1]
        ratio = (
            speed["images_per_second"] / slow["images_per_second"]
            if slow["images_per_second"] else None
        )
        lines += _lead(
            f"**{rd.name(speed['model'])} was the faster of the two** at "
            f"{rd.fmt(speed['images_per_second'], 1)} images per second"
            + (f", {rd.fmt(ratio, 1)}x {rd.name(slow['model'])}" if ratio else "")
            + ". That comparison carries a caveat the raw results record: "
            "the two are timed over different regions. The torchvision model "
            "is timed on its forward pass alone, while ultralytics' `predict` "
            "includes its own preprocessing and non-maximum suppression. The "
            "YOLO figure is therefore closer to end-to-end deployment cost "
            "and the Mask R-CNN figure closer to pure model cost, so the gap "
            "between them understates YOLO's advantage."
        )

    small = rd.numeric(instance, "mask_ap_small")
    if not small.empty:
        lines += _lead(
            "COCO's area breakdown answers the small-object question directly:"
        )
        lines.append(rd.table(instance, {
            "model": "Model",
            "mask_ap_small": "AP small (<32²)",
            "mask_ap_medium": "AP medium",
            "mask_ap_large": "AP large (≥96²)",
        }))
        lines.append("")
        best_small = rd.best(instance, "mask_ap_small")
        if best_small is not None:
            large = best_small.get("mask_ap_large")
            lines += _lead(
                f"**{rd.name(best_small['model'])} handled small objects "
                f"best** at {rd.fmt(best_small['mask_ap_small'])} AP, though "
                "every model scores far lower on small objects than on large "
                + (
                    f"ones ({rd.fmt(best_small['mask_ap_small'])} against "
                    f"{rd.fmt(large)} for the same model)"
                    if large is not None else "ones"
                )
                + ". Small objects survive fewer downsampling stages, and at "
                "an input of 256 pixels a COCO 'small' object can be a few "
                "dozen pixels across by the time it reaches the mask head."
            )

    lines.append(rd.figure(
        "instance_mask_ap.png", "Mask R-CNN against YOLO on mask AP, AP50 and AP75"
    ))

    lines += _lead(
        "**These numbers must not be compared with the semantic table.** "
        "Section 26 is explicit about it, and the reason is not pedantry: "
        "mask AP integrates precision over a recall curve at ten IoU "
        "thresholds for individually matched objects, while mIoU is a single "
        "pixel-set overlap over a whole dataset. A mask AP of 0.35 is not "
        "worse than an mIoU of 0.55; the two do not share units, a scale, or "
        "a task."
    )
    return lines


# ---------------------------------------------------------------------------
# Section 29-33: efficiency
# ---------------------------------------------------------------------------

def section_efficiency(data: dict) -> list:
    efficiency = data["efficiency"]
    lines = ["## Efficiency", ""]

    if efficiency.empty:
        return lines + ["_Not yet generated._", ""]

    environment = {}
    for payload in data["raw"].values():
        environment = payload.get("efficiency", {}).get("environment", {})
        if environment:
            break

    lines += _lead(
        "Efficiency is the one axis on which the traditional baseline, the "
        "semantic networks and the instance networks are directly comparable: "
        "they all take an image and all take time to do it. Quality is not "
        "comparable across tracks, which is why those tables are kept apart "
        "and this one is not."
    )

    if environment:
        lines += _lead(
            f"All measurements were taken on {environment.get('device', 'N/A')} "
            f"using the {environment.get('backend', 'N/A')} backend at "
            f"{environment.get('precision', 'N/A')} precision, torch "
            f"{environment.get('torch', 'N/A')}, seed "
            f"{environment.get('seed', 'N/A')}."
        )
        if environment.get("backend") == "mps":
            lines += _lead(
                "**A limitation worth stating plainly.** There is no CUDA on "
                "this hardware, so Section 29's peak GPU memory is measured "
                "with `torch.mps.current_allocated_memory`, which reports the "
                "allocator's current total rather than a driver-level peak. "
                "It is a weaker measurement than `torch.cuda.max_memory_"
                "allocated` and the memory column should be read as "
                "indicative rather than exact. Timing is unaffected: every "
                "timed region is bracketed by `torch.mps.synchronize`, "
                "without which an asynchronous backend reports how long it "
                "took to enqueue the work rather than to do it."
            )

    lines.append(rd.table(efficiency, {
        "model": "Model",
        "task": "Task",
        "total_parameters": "Params",
        "model_size_mb": "Size (MiB)",
        "mean_epoch_seconds": "s/epoch",
        "latency_ms": "Latency (ms)",
        "images_per_second": "Images/s",
        "train_peak_memory_mb": "Train mem (MiB)",
    }, places=2))
    lines.append("")

    lines += _lead(
        "Latency is batch-one; throughput is batched. Section 33 requires "
        "them distinguished because they answer different questions and the "
        "models do not rank the same way on both: a robot processing one "
        "frame at a time is bound by the first, a server scoring a queue by "
        "the second. The timed region is the forward pass, excluding file "
        "reading, preprocessing, host-device transfer and mask "
        "postprocessing - so these are model costs, not deployment estimates."
    )

    smallest = rd.best(efficiency, "model_size_mb", highest=False,
                       exclude=["kmeans"])
    fastest = rd.best(efficiency, "images_per_second", exclude=["kmeans"])
    if smallest is not None and fastest is not None:
        lines += _lead(
            f"**{rd.name(smallest['model'])} is the smallest network** at "
            f"{rd.fmt(smallest['model_size_mb'], 1)} MiB, and "
            f"**{rd.name(fastest['model'])} the fastest** at "
            f"{rd.fmt(fastest['images_per_second'], 1)} images per second. "
            "Size is weights and buffers at float32, excluding optimizer "
            "state; MiB means 1024*1024 bytes. A training checkpoint carrying "
            "AdamW's two moment tensors per parameter would be roughly three "
            "times these figures for reasons that have nothing to do with the "
            "architecture, which is why the inference artifact is what is "
            "compared."
        )

    for name, caption in [
        ("parameters.png", "Total parameters"),
        ("model_size.png", "Saved model size"),
        ("training_time.png", "Total training time"),
        ("inference_speed.png", "Inference throughput"),
    ]:
        lines.append(rd.figure(name, caption))

    return lines


def section_tradeoffs(data: dict) -> list:
    semantic = data["semantic"]
    lines = ["## Capacity, latency and quality", ""]

    if semantic.empty:
        return lines + ["_Not yet generated._", ""]

    quality = rd.numeric(semantic, "mean_iou")
    quality = rd.numeric(quality, "total_parameters")
    quality = quality[quality["total_parameters"] > 0]

    if not quality.empty and len(quality) > 2:
        correlation = quality["total_parameters"].corr(quality["mean_iou"])
        largest = quality.loc[quality["total_parameters"].idxmax()]
        best_quality = quality.loc[quality["mean_iou"].idxmax()]

        lines += _lead(
            f"**Did more parameters buy more quality? Not reliably.** Across "
            f"the trained semantic models the correlation between parameter "
            f"count and mIoU is {rd.fmt(correlation, 2)}. "
            + (
                f"The largest model, {rd.name(largest['model'])} at "
                f"{rd.fmt_int(largest['total_parameters'])} parameters, "
                f"scored {rd.fmt(largest['mean_iou'])}, while the best "
                f"model, {rd.name(best_quality['model'])}, has "
                f"{rd.fmt_int(best_quality['total_parameters'])}."
                if largest["model"] != best_quality["model"]
                else f"The largest model, {rd.name(largest['model'])}, also "
                     "scored highest, which on this evidence is a coincidence "
                     "of the set rather than a rule: the correlation across "
                     "all models is what it is."
            )
        )
        lines += _lead(
            "What separates these models is not capacity but what they were "
            "initialized from and how they recover spatial detail. A "
            "pretrained ResNet50 backbone with 35M parameters and a COCO "
            "segmentation head starts from a far better place than a "
            "randomly initialized encoder-decoder with 44M."
        )

    lines.append(rd.figure("miou_vs_parameters.png", "Quality against capacity"))
    lines.append(rd.figure("miou_vs_latency.png", "Quality against latency"))
    return lines


def section_convergence(data: dict) -> list:
    history = data["history"]
    lines = ["## Convergence and overfitting", ""]

    if history.empty:
        return lines + ["_Not yet generated._", ""]

    models = [
        m for m in history["model"].unique()
        if m not in SEMANTIC_ONLY
    ]

    rows = []
    for model in models:
        rows.append({
            "model": model,
            "epochs_to_90pct": rd.epochs_to_fraction(history, model),
            "val_loss_rise": rd.overfitting_gap(history, model),
        })
    frame = pd.DataFrame(rows)

    lines += _lead(
        "Two questions Section 54 asks are properties of the training curve "
        "rather than of the final score. **Epochs to 90%** is how many epochs "
        "a model needed to reach 90% of its own best validation mIoU - "
        "relative to its own ceiling, because this measures speed of "
        "learning and not final quality. **Validation loss rise** is how far "
        "validation loss climbed from its minimum by the last epoch, which "
        "is overfitting in the form that matters: a model still improving on "
        "training data while getting worse on held-out data."
    )
    lines.append(rd.table(frame, {
        "model": "Model",
        "epochs_to_90pct": "Epochs to 90% of own best",
        "val_loss_rise": "Val loss rise from minimum",
    }))
    lines.append("")

    converged = rd.numeric(frame, "epochs_to_90pct")
    if not converged.empty:
        fastest = converged.loc[converged["epochs_to_90pct"].idxmin()]
        lines += _lead(
            f"**{rd.name(fastest['model'])} converged fastest**, reaching 90% "
            f"of its best validation mIoU by epoch "
            f"{int(fastest['epochs_to_90pct'])}."
        )

    # A model whose selected checkpoint is its final epoch never stopped
    # improving, so its score is a floor rather than a ceiling. This is not a
    # detail: it changes what the pretrained-vs-scratch gap means.
    still_improving = []
    for model, payload in data["raw"].items():
        training = payload.get("training", {})
        best = training.get("best_epoch")
        completed = training.get("completed_epochs")
        if best and completed and best == completed:
            still_improving.append((model, best))

    if still_improving:
        names = ", ".join(rd.name(m) for m, _ in sorted(still_improving))
        lines += _lead(
            f"**{names} selected the final epoch as the best one**, which "
            "means validation mIoU was still rising when the 25-epoch budget "
            "ran out. Those scores are a floor, not a ceiling, and the honest "
            "reading of the gap between them and the pretrained models is "
            "that it combines two causes: the pretrained models start from "
            "better features, *and* the from-scratch models had not finished "
            "learning. A longer budget would narrow the gap by some unknown "
            "amount. Section 13 sets 25 epochs as a minimum rather than a "
            "sufficient number, and for the randomly initialized models on "
            "5,000 images it was evidently the former."
        )

    overfit = rd.numeric(frame, "val_loss_rise")
    if not overfit.empty:
        worst = overfit.loc[overfit["val_loss_rise"].idxmax()]
        if worst["val_loss_rise"] > 0:
            lines += _lead(
                f"**{rd.name(worst['model'])} overfit most**, its validation "
                f"loss rising {rd.fmt(worst['val_loss_rise'])} above its "
                "minimum by the final epoch. Because checkpoints are selected "
                "on validation mIoU rather than taken from the last epoch, "
                "that overfitting costs the reported scores nothing - the "
                "saved weights are from before the climb. It does mean the "
                "epoch budget was longer than that model needed."
            )
        else:
            lines += _lead(
                "No model's validation loss rose materially above its "
                "minimum by the final epoch, which means 25 epochs did not "
                "overtrain any of them on this dataset. Several were still "
                "improving when the budget ran out, so these scores are a "
                "floor rather than a ceiling."
            )

    return lines


# ---------------------------------------------------------------------------
# Section 52, 10, 24: what the numbers cannot show
# ---------------------------------------------------------------------------

def section_qualitative(data: dict) -> list:
    """Example predictions, the K-Means clusters, and boundary close-ups."""
    from pathlib import Path

    lines = ["## Qualitative results", ""]
    predictions_dir = Path(__file__).resolve().parent.parent / "predictions"

    if not predictions_dir.is_dir():
        return lines + ["_No prediction figures have been generated yet._", ""]

    lines += _lead(
        "Every model was rendered on the same test images in the same order. "
        "A panel built from whichever images a model happened to do well on "
        "is an advertisement rather than evidence, so the selection is the "
        "first images the deterministic test loader returns and is identical "
        "across models."
    )

    # Section 10: the traditional baseline, qualitatively.
    kmeans_figure = predictions_dir / "kmeans" / "clusters_00.png"
    if kmeans_figure.is_file():
        lines += ["### What K-Means actually produces", ""]
        lines.append(
            f"![K-Means clusters against the class mapping]"
            f"(../predictions/kmeans/clusters_00.png)\n"
        )
        lines += _lead(
            "The third panel is the one Section 10 asks for. K-Means finds "
            "coherent regions, and they are the wrong regions: horizontal "
            "bands of water, wave and sky, split by color and lighting rather "
            "than by object. The surfers - the only thing in the frame that "
            "is a labeled class - are absorbed into whichever band they "
            "overlap. After the frozen training-only mapping is applied, the "
            "fourth panel is almost entirely background.\n\n"
            "This is the difference between grouping pixels that *look* alike "
            "and grouping pixels that *are* the same object. No amount of "
            "tuning the clustering fixes it, because color similarity is not "
            "a proxy for object identity - which is the reason the nine "
            "learned models exist."
        )

    # Section 52: per-model predictions.
    models = [
        d.name for d in sorted(predictions_dir.iterdir())
        if d.is_dir() and d.name != "kmeans"
        and (d / "example_00.png").is_file()
    ]
    if models:
        lines += ["### Predictions by model", ""]
        for model in models:
            lines.append(f"**{rd.name(model)}**")
            lines.append("")
            lines.append(
                f"![{rd.name(model)} prediction]"
                f"(../predictions/{model}/example_00.png)\n"
            )

    # Section 24: boundary close-ups.
    closeups = [
        d.name for d in sorted(predictions_dir.iterdir())
        if d.is_dir() and (d / "boundary_closeup.png").is_file()
    ]
    if closeups:
        lines += ["### Boundary detail", ""]
        lines += _lead(
            "Each close-up is cropped on the densest concentration of "
            "disagreement between the prediction and the ground truth on that "
            "image, not on a region chosen to flatter the model. These are "
            "what the boundary F1 column summarizes."
        )
        for model in closeups:
            lines.append(f"**{rd.name(model)}**")
            lines.append("")
            lines.append(
                f"![{rd.name(model)} boundary detail]"
                f"(../predictions/{model}/boundary_closeup.png)\n"
            )

    return lines
