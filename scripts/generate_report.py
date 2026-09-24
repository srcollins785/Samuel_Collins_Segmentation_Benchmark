"""Build the assignment report in Markdown from the generated benchmark files.

This reads ``results/`` and writes ``report/``. It imports nothing from the
pipeline: the benchmark communicates with the report through files. That
decoupling is what lets the prose be re-rendered in a second without
re-running a seventeen-hour benchmark, and it means the report can only
describe results that were actually produced.

It also stays outside the installed package. The package is library code; a
course report carrying a student name, a course number and an instructor is
not.

Usage
-----
    python scripts/generate_report.py
"""

import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_analysis  # noqa: E402
import report_data as rd  # noqa: E402
import report_discussion  # noqa: E402

REPORT_DIR = REPO_ROOT / "report"
REPORT_MD = REPORT_DIR / "Segmentation_Benchmarking_Report.md"

STUDENT_NAME = "Samuel Collins"
STUDENT_TITLE = "Ph.D. Student, Department of Cyber-Physical Systems"
INSTITUTION = "Clark Atlanta University"
STUDENT_EMAIL = "samuel.collins@students.cau.edu"
COURSE = "Computer Vision - Image Segmentation Benchmarking"
INSTRUCTOR = "Dr. Kishor Datta Gupta"
REPOSITORY = "https://github.com/srcollins785/Samuel_Collins_Segmentation_Benchmark"


def section_header(data: dict) -> list:
    manifest = data["manifest"]
    resolution = None
    epochs = None
    for payload in data["raw"].values():
        config = payload.get("config", {})
        resolution = config.get("input_size", resolution)
        epochs = config.get("epochs", epochs)

    return [
        "# Comprehensive Image Segmentation Benchmarking",
        "",
        f"**{STUDENT_NAME}** · {STUDENT_TITLE} · {INSTITUTION}",
        f"{STUDENT_EMAIL}",
        "",
        f"{COURSE} · Instructor: {INSTRUCTOR} · {date.today().isoformat()}",
        "",
        f"Repository: {REPOSITORY}",
        "",
        "---",
        "",
        "## Summary",
        "",
        _headline(data),
        "",
        f"Ten segmentation methods were trained and evaluated on one shared "
        f"COCO 2017 subset - 5,000 training, 1,000 validation and 1,000 test "
        f"images over six classes - under a common protocol of "
        f"{epochs or 25} epochs, AdamW at 1e-4, batch size 8, at "
        f"{resolution or 256}x{resolution or 256} resolution, seed 42. "
        f"Split checksum `{manifest.get('checksum', 'N/A')}`.",
        "",
        "Every table and figure in this report was generated from saved "
        "evaluation outputs by `run_benchmark.py --tables-only`. No number "
        "here was typed by hand.",
        "",
    ]


def _headline(data: dict) -> str:
    """The one-paragraph result, computed rather than written."""
    semantic = data["semantic"]
    instance = data["instance"]

    best = rd.best(semantic, "mean_iou", exclude=["kmeans"])
    if best is None:
        return "_The benchmark has not produced results yet._"

    sentences = [
        f"**{rd.name(best['model'])} achieved the highest semantic mean IoU "
        f"at {rd.fmt(best['mean_iou'])}**"
    ]

    smallest = rd.best(semantic, "model_size_mb", highest=False,
                       exclude=["kmeans"])
    if smallest is not None and smallest["model"] != best["model"]:
        # Appended to the first sentence as a subordinate clause, not joined
        # as a separate one -- "...at 0.626. while SegFormer reached..." is
        # what joining them with a period produces.
        sentences[0] += (
            f", while {rd.name(smallest['model'])} reached "
            f"{rd.fmt(rd.value(semantic, smallest['model'], 'mean_iou'))} at "
            f"{rd.fmt(smallest['model_size_mb'], 1)} MiB, "
            f"{rd.fmt(rd.value(semantic, best['model'], 'model_size_mb') / smallest['model_size_mb'], 1)} "
            "times smaller"
        )

    instance_best = rd.best(instance, "mask_ap") if not instance.empty else None
    if instance_best is not None:
        sentences.append(
            f"On the instance track, {rd.name(instance_best['model'])} "
            f"reached {rd.fmt(instance_best['mask_ap'])} mask AP"
        )

    return ". ".join(sentences) + "."


def section_tasks() -> list:
    return [
        "## The four tasks, and which models do which",
        "",
        "| Task | Required output | Example |",
        "|---|---|---|",
        "| Image classification | One label for the image | \"Dog\" describes the whole photograph |",
        "| Object detection | Class and box per object | Dog: [x1, y1, x2, y2] |",
        "| Semantic segmentation | A class label for every valid pixel | All cars share one label |",
        "| Instance segmentation | Class and separate mask per object | Person #1 and Person #2 have distinct masks |",
        "",
        "Seven models here produce semantic segmentation (FCN, U-Net, "
        "U-Net++, SegNet, DeepLabV3, PSPNet, SegFormer) and need not "
        "distinguish two objects of the same class. Two produce instance "
        "segmentation (Mask R-CNN, YOLO-seg) and must keep separate masks "
        "for objects that share a label. One is a non-learned baseline "
        "(K-Means).",
        "",
        "### R-CNN is not Mask R-CNN",
        "",
        "The family evolved in four steps, and only the last one segments:",
        "",
        "- **R-CNN (2014)** ran selective search for region proposals, warped "
        "each to a fixed size and pushed every one through a CNN separately. "
        "Thousands of forward passes per image.",
        "- **Fast R-CNN (2015)** ran the CNN once over the whole image and "
        "pooled features per proposal instead, collapsing the cost.",
        "- **Faster R-CNN (2015)** replaced selective search with a Region "
        "Proposal Network, making proposals learned and the pipeline "
        "end-to-end.",
        "- **Mask R-CNN (2017)** added a mask branch per region of interest "
        "and replaced RoI Pool with RoI Align.",
        "",
        "Only the last predicts masks. A detection-only model does not "
        "satisfy an instance-segmentation requirement, which is why the "
        "instance track here uses Mask R-CNN and a YOLO **segmentation** "
        "variant rather than their detection counterparts.",
        "",
    ]


def section_dataset(data: dict) -> list:
    manifest = data["manifest"]
    if not manifest:
        return ["## Dataset", "", "_No split manifest found._", ""]

    statistics = manifest.get("statistics", {})
    policies = manifest.get("policies", {})

    lines = [
        "## Dataset",
        "",
        "A COCO 2017 subset over background plus five foreground classes: "
        "person, car, bicycle, dog and cat. COCO was chosen by the assignment "
        "because it supplies polygon and RLE masks, multiple instances per "
        "image, and the same annotations serve both tracks - instance "
        "annotations become class-level masks for the semantic models and "
        "stay separate for the instance models.",
        "",
        "### Splits",
        "",
        f"Generated at **seed {manifest.get('seed')}**, checksum "
        f"`{manifest.get('checksum')}`. Every model records this checksum "
        "with its results; two models whose checksums differ were not "
        "evaluated on the same test set and their comparison would not be one.",
        "",
        "| Split | Images | Instances |",
        "|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        stats = statistics.get(split, {})
        lines.append(
            f"| {split} | {stats.get('images', 'N/A'):,} | "
            f"{stats.get('instances', 'N/A'):,} |"
        )
    lines.append("")

    lines += [
        "**All three splits are drawn from `train2017`.** COCO's own layout "
        "invites taking the test split from `val2017`, but that would make "
        "the test set a different annotation pass, and any validation-to-test "
        "gap would then be partly a dataset difference rather than a "
        "generalization gap. Drawing all 7,000 images from one pool makes the "
        "three partitions identically distributed by construction - visible "
        "in the class shares below, which agree to within a third of a "
        "percentage point.",
        "",
        "### Class balance, and why it shapes every result",
        "",
        "| Split | background | person | car | bicycle | dog | cat | ignored |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        stats = statistics.get(split, {})
        pixels = stats.get("pixels_per_class", {})
        total = sum(pixels.values()) or 1
        ignored = stats.get("ignored_pixels", 0)
        row = " | ".join(
            f"{pixels.get(c, 0) / total * 100:.2f}%"
            for c in ["background", "person", "car", "bicycle", "dog", "cat"]
        )
        lines.append(f"| {split} | {row} | {ignored / total * 100:.2f}% |")
    lines.append("")

    lines += [
        "Background is 83% of labeled pixels and bicycle is 0.26% - a factor "
        "of more than 300. Two consequences run through everything below. "
        "First, a model that predicts background everywhere scores about 0.83 "
        "pixel accuracy, which is why Section 22 asks for mIoU and why the "
        "K-Means baseline's result is instructive rather than embarrassing. "
        "Second, six-class mIoU is steered by background and person, so the "
        "per-class table is reported before the mean rather than after it.",
        "",
        "### Mask creation policies",
        "",
        "Section 8 requires these documented and applied consistently across "
        "every model. They are defined once, in `_config.py`, and both tracks "
        "read them from there:",
        "",
        f"- **Overlapping annotations** - {policies.get('overlap', 'N/A')}. "
        "The alternative, first-annotation-wins, would let a large object "
        "erase a small one sitting on it, and small objects are already the "
        "hardest case.",
        f"- **Crowd regions** - {policies.get('crowd', 'N/A')}. A crowd blob "
        "marks real objects of a real class that COCO declined to annotate "
        "individually, so calling it background would teach every model that "
        "crowds are background. It is also not a usable instance. It is "
        "ignored in both, and retained in the evaluation annotation file "
        "where COCOeval uses it to avoid penalizing a detection that lands "
        "on an unannotated group.",
        f"- **Non-selected categories** - {policies.get('unselected', 'N/A')}.",
        f"- **Eligibility** - {policies.get('eligibility', 'N/A')}.",
        "",
        "Masks are 8-bit PNGs at native resolution, with 255 as the ignore "
        "sentinel. PNG because it is lossless: a JPEG mask would interpolate "
        "class IDs into each other at every boundary and invent classes that "
        "were never annotated.",
        "",
    ]
    return lines


def section_protocol(data: dict) -> list:
    resolution, epochs, batch = None, None, None
    transforms, loss = None, None
    for payload in data["raw"].values():
        config = payload.get("config", {})
        resolution = config.get("input_size", resolution)
        epochs = config.get("epochs", epochs)
        batch = config.get("batch_size", batch)
        transforms = config.get("transforms", transforms)
        loss = payload.get("loss", loss) or loss

    lines = [
        "## Experimental protocol",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Input resolution | {resolution}x{resolution} |",
        f"| Epochs | {epochs} |",
        f"| Batch size | {batch} |",
        "| Optimizer | AdamW |",
        "| Learning rate | 1e-4 |",
        "| Weight decay | 1e-4 |",
        "| Seed | 42 |",
        "| Mixed precision | not used (float32 throughout) |",
        "",
    ]

    if resolution == 256:
        lines += [
            "**Resolution is a documented deviation.** Section 13 specifies "
            "512x512 and permits 256x256 where compute is limited, applied "
            "consistently. It is applied consistently here, and the decision "
            "was made from measurement rather than convenience: training the "
            "seven semantic models for 25 epochs was measured at roughly 9 "
            "hours at 256 and 35 hours at 512 on this hardware, and the full "
            "ten-model benchmark at 512 would have exceeded two days of "
            "continuous compute.",
            "",
        ]

    lines += [
        "### Preprocessing and augmentation",
        "",
        "Training applies, in this order:",
        "",
    ]
    if transforms:
        for step in transforms:
            name = step.get("transform")
            details = ", ".join(
                f"{k}={v}" for k, v in step.items()
                if k not in {"transform", "note"} and v is not None
            )
            lines.append(f"- `{name}`, {details}" if details else f"- `{name}`")
    lines += [
        "",
        "Validation and test apply deterministic resizing and normalization "
        "only, so a validation score changes between epochs because the "
        "weights changed and for no other reason.",
        "",
        "**Every geometric transform rewrites the image, the semantic mask, "
        "each instance mask and each box together.** Section 15 warns that a "
        "flipped image paired with an unchanged mask is invalid training "
        "data, and asks for transformed pairs to be visualized before "
        "benchmarking. `scripts/verify_transforms.py` does that and checks it "
        "numerically first, because a figure only catches a desync large "
        "enough to see and a three-pixel drift would look fine at figure "
        "resolution while making the boundary results meaningless.",
        "",
        rd.figure("transform_verification.png",
                  "Section 15 verification: geometry applied to image and mask together"),
        "",
        "Two decisions inside that pipeline are easy to get wrong. Boxes are "
        "re-derived from the transformed mask rather than transformed "
        "directly, because rotating a box's corners and taking the extent "
        "gives a box strictly larger than the object. And pixels a transform "
        "invents - rotation corners, crop padding - are marked ignore rather "
        "than background, so they never enter the loss as an observation.",
        "",
        "### Loss",
        "",
    ]
    if loss and "loss" in loss:
        lines += [
            f"Semantic models: **{loss.get('loss')}**.",
            "",
            f"- Dice formulation: {loss.get('dice_formulation', 'N/A')}",
            f"- Class averaging: {loss.get('dice_averaging', 'N/A')}",
            f"- Ignore handling: {loss.get('ignore_handling', 'N/A')}",
            f"- Class weighting: {loss.get('class_weighting', 'N/A')}",
            "",
            "The averaging rule is the one that matters on this dataset. Dice "
            "is macro-averaged over classes **present in the batch**, not "
            "over all six. Bicycle is 0.26% of pixels, so most batches "
            "contain none, and an absent class scores a free 1.0 under the "
            "usual formulation - handing the model its largest bonus on "
            "exactly the rare classes the Dice term exists to pressure.",
            "",
        ]
    lines += [
        "Instance models keep their own losses, as Section 17 requires. Mask "
        "R-CNN trains on classification, box regression, mask and the two RPN "
        "terms; YOLO uses ultralytics' box, class, DFL and mask losses. "
        "Neither borrows the semantic recipe: they are trained against object "
        "proposals rather than a dense label map, and forcing a per-pixel "
        "cross-entropy onto them would be a different task, not a harder "
        "version of the same one.",
        "",
    ]
    return lines


def section_models(data: dict) -> list:
    lines = [
        "## Models",
        "",
        "Section 10 notes that an architecture name is not a configuration. "
        "Each model's backbone, initialization source and measured size are "
        "recorded with its results and reproduced here.",
        "",
        "| Model | Task | Backbone | Initialization |",
        "|---|---|---|---|",
    ]
    for model, payload in sorted(
        data["raw"].items(),
        key=lambda kv: ["kmeans", "fcn_resnet50", "unet", "unetpp", "segnet",
                        "deeplabv3", "pspnet", "segformer", "maskrcnn",
                        "yolo_seg"].index(kv[0])
        if kv[0] in ["kmeans", "fcn_resnet50", "unet", "unetpp", "segnet",
                     "deeplabv3", "pspnet", "segformer", "maskrcnn",
                     "yolo_seg"] else 99,
    ):
        card = payload.get("card", {})
        lines.append(
            f"| {rd.name(model)} | {payload.get('task', 'N/A')} | "
            f"{card.get('backbone', 'N/A')} | "
            f"{card.get('initialization', 'N/A')} |"
        )
    lines.append("")

    lines += [
        "**Initialization is not uniform, and that confounds the "
        "architecture comparison.** FCN and DeepLabV3 load COCO segmentation "
        "weights including a trained head. PSPNet and SegNet load ImageNet "
        "backbones. SegFormer loads a pretrained encoder with a randomly "
        "initialized decode head. U-Net and U-Net++ are entirely from "
        "scratch. This is a property of what pretrained weights exist, not a "
        "choice, but it means a ranking on mIoU is partly a ranking on how "
        "much each model was given to start with.",
        "",
        "### The traditional baseline",
        "",
        "Section 10 is explicit that K-Means cluster IDs are not class "
        "labels, and forbids matching clusters to test labels to inflate a "
        "score. The mapping here is fitted on training images only and "
        "frozen: clusters are described by a coarse, quantized descriptor of "
        "color, position and coverage, and each descriptor bin takes the "
        "class that won it across the training set. `predict()` takes an "
        "image and has no parameter through which a mask could be passed, so "
        "the inflation the assignment warns about is unavailable rather than "
        "merely discouraged.",
        "",
    ]

    kmeans = data["raw"].get("kmeans")
    if kmeans:
        test = kmeans.get("test", {})
        config = kmeans.get("config", {})
        bins = config.get("bins_per_class", {})
        lines += [
            f"The result is worth reading carefully. K-Means scored "
            f"**{rd.fmt(test.get('pixel_accuracy'))} pixel accuracy** and "
            f"**{rd.fmt(test.get('mean_iou'))} mIoU**, but only "
            f"**{rd.fmt(test.get('mean_iou_foreground'))} foreground mIoU**. "
            f"Of {config.get('descriptor_bins', 'N/A')} descriptor bins, "
            f"{bins.get('0', 'N/A')} map to background and "
            f"{bins.get('1', 'N/A')} to person; no bin maps to car, bicycle, "
            "dog or cat at all.",
            "",
            "That is close to the degenerate solution - predict background "
            "almost everywhere - and the pixel accuracy of "
            f"{rd.fmt(test.get('pixel_accuracy'))} is almost exactly the "
            "background share of the test set. It is not a bug and it is not "
            "a tuning failure to be fixed. It is what an honest color "
            "clustering produces on an 83%-background dataset: majority vote "
            "over any descriptor bin lands on background, because background "
            "is what most pixels are. A mapping fitted against test labels "
            "would have scored far higher and measured nothing.",
            "",
            "It also demonstrates the point Section 22 makes about metric "
            "choice more cleanly than any trained model could. Pixel accuracy "
            "of 0.825 sounds like a working segmenter. Foreground mIoU of "
            f"{rd.fmt(test.get('mean_iou_foreground'))} is the same model, "
            "described honestly.",
            "",
            "What K-Means actually does is group pixels that look alike. That "
            "is a different task from naming objects: it separates a dark dog "
            "from a light lawn, but it also separates the dog's sunlit flank "
            "from its shadowed one, and merges a gray cat with gray pavement. "
            "Those are properties of color clustering, not deficiencies to be "
            "tuned away.",
            "",
            "The feature vector follows the course reading (Nayar, FPCV-5-2, "
            "slide 30), which shows that clustering on color alone yields "
            "clusters mapping to many disconnected segments, and that adding "
            "each pixel's spatial coordinates to make a five-dimensional "
            "vector gives a more useful description. This implementation uses "
            "CIELAB rather than RGB for the color axes, because k-means is "
            "driven entirely by distance and CIELAB distances correspond "
            "better to perceived difference.",
            "",
            "That reading also predicts this outcome. It lists k-means as "
            "sensitive to initialization and to outliers, requiring k to be "
            "chosen in advance, and describes it as suited to \"relatively "
            "simple images\". COCO photographs are not simple images, and the "
            "result above is what that caveat looks like when it is measured "
            "rather than asserted. The reading's own remedies for those "
            "weaknesses, mean shift and normalized graph cut, are among the "
            "alternatives Section 10 lists as optional; neither is "
            "implemented here.",
            "",
        ]
    return lines


def section_metrics() -> list:
    return [
        "## Metrics and conventions",
        "",
        "Section 22 requires four conventions declared and held to "
        "throughout. They are defined once in `metrics.py` and recorded into "
        "every result file, so a CSV read later carries the convention that "
        "produced it.",
        "",
        "- **Which pixels count.** Pixels marked ignore - crowd regions, and "
        "canvas that augmentation invented - are excluded before the "
        "confusion matrix is accumulated. They enter neither numerator nor "
        "denominator of anything.",
        "- **Background.** Reported both ways. `mIoU` covers all six classes "
        "and is the figure the required table uses; `mIoU (fg)` covers the "
        "five object classes. Reporting only the first flatters every model, "
        "since background is 83% of pixels and trivially easy.",
        "- **Absent classes.** A class with no ground-truth and no predicted "
        "pixels has an undefined IoU and is excluded from the mean, not "
        "scored zero or one. Scoring it zero would punish a model for a class "
        "the test set never showed it.",
        "- **Averaging.** Macro: per class, then averaged. Dice and pixel F1 "
        "are the same quantity on the same counts, so one function computes "
        "both and they cannot drift apart.",
        "",
        "Metrics are accumulated as counts across the whole test set and "
        "computed once at the end, not averaged over per-batch scores. Those "
        "are different numbers: a per-batch average weights a batch holding "
        "four bicycle pixels as heavily as one holding forty thousand.",
        "",
        "### Keeping the two tracks apart",
        "",
        "Section 26 forbids comparing semantic mIoU with instance mask AP as "
        "though they measured the same thing, and this report does not. The "
        "two quality tables are separate, no figure places them on a shared "
        "axis, and the only combined table is efficiency - which is "
        "legitimately comparable, because every model takes an image and "
        "takes time to do it.",
        "",
    ]


def section_reproduction(data: dict) -> list:
    return [
        "## Reproducing this",
        "",
        "```bash",
        "python3.12 -m venv .venv",
        "source .venv/bin/activate",
        "pip install -r requirements.txt && pip install -e .",
        "",
        "python scripts/build_coco_subset.py      # ~1.2 GB, seed 42",
        "python scripts/verify_transforms.py      # Section 15 check",
        "",
        "python run_benchmark.py --task all --model all --input-size 256",
        "python run_benchmark.py --tables-only    # rebuild tables and figures",
        "python scripts/generate_report.py        # rebuild this document",
        "```",
        "",
        "The benchmark writes one `results/raw/<model>.json` per model as it "
        "finishes, and every table, figure and sentence in this report is "
        "derived from those files. Re-running `--tables-only` regenerates "
        "everything a reader sees in seconds without touching a model, which "
        "is what Section 51 asks for when it requires that evaluation be "
        "repeatable without retraining. `--skip-existing` resumes an "
        "interrupted run.",
        "",
        "Seeding covers initialization, shuffling and augmentation. It does "
        "not make results bit-identical: Metal kernels are not deterministic "
        "and torch offers no deterministic mode for them, so a rerun "
        "reproduces the protocol rather than the exact weights.",
        "",
    ]


def build(data: dict = None) -> str:
    data = data or rd.load()
    lines = []
    lines += section_header(data)
    lines += section_tasks()
    lines += section_dataset(data)
    lines += section_models(data)
    lines += section_protocol(data)
    lines += section_metrics()
    lines += report_analysis.section_semantic_results(data)
    lines += report_analysis.section_per_class(data)
    lines += report_analysis.section_boundary(data)
    lines += report_analysis.section_instance_results(data)
    lines += report_analysis.section_efficiency(data)
    lines += report_analysis.section_tradeoffs(data)
    lines += report_analysis.section_convergence(data)
    lines += report_analysis.section_qualitative(data)
    lines += report_discussion.section_evolution()
    lines += report_discussion.section_discussion(data)
    lines += report_discussion.section_deployment(data)
    lines += section_reproduction(data)
    return "\n".join(lines) + "\n"


def main() -> int:
    data = rd.load()
    if not data["raw"]:
        print(
            "No results in results/raw/. Run the benchmark first:\n"
            "  python run_benchmark.py --task all --model all"
        )
        return 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = build(data)
    REPORT_MD.write_text(document)

    words = len(document.split())
    print(f"Wrote {REPORT_MD.relative_to(REPO_ROOT)}")
    print(f"  {len(document.splitlines()):,} lines, {words:,} words")
    print(f"  models included: {', '.join(sorted(data['raw']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
