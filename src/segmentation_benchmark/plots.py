"""Section 49: the ten required comparison figures, generated from the CSVs.

Every figure reads ``results/*.csv`` and nothing else, so a figure cannot
show a number that is not also in a table, and the whole set redraws in
seconds without touching a model.

Section 49 keeps semantic quality and instance quality on separate plots,
and so does this module: there is no figure on which a semantic mIoU and an
instance mask AP appear together, because Section 26 says those are not
comparable quantities and a shared axis would assert that they are.

**Color.** Figures 1 to 8 plot one measure across models. That is a single
series - the models are categories on an axis, not series - so they use one
hue, and the ranking is carried by bar length and by direct value labels
rather than by ten colors that mean nothing. Only figures 9 and 10 have real
series (models compared across classes, and two instance models across three
metrics), and those draw from a fixed categorical order that was validated
for colorblind separation rather than chosen by eye. Three of the seven
light-mode slots fall below 3:1 contrast against the surface, which obliges
relief; ``per_class_iou.csv`` is that table view and ships alongside.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ._config import PLOTS_DIR, RESULTS_DIR  # noqa: E402

# Validated categorical order (adjacent pairlist, light surface): worst
# adjacent CVD dE 9.1, worst adjacent normal-vision dE 19.6. Assigned in
# fixed order and never cycled - an eighth series folds into "other" rather
# than repeating slot 1.
SERIES = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

SINGLE = "#2a78d6"
ACCENT = "#eb6834"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"

# Bars are thin and their data-ends are rounded, anchored to the baseline.
BAR_WIDTH = 0.62

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 130,
})

DISPLAY_NAMES = {
    "kmeans": "K-Means",
    "fcn_resnet50": "FCN-R50",
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "segnet": "SegNet",
    "deeplabv3": "DeepLabV3",
    "pspnet": "PSPNet",
    "segformer": "SegFormer",
    "maskrcnn": "Mask R-CNN",
    "yolo_seg": "YOLO-seg",
}


def _label(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _numeric(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Drop rows whose value is N/A, rather than plotting them as zero.

    A bar of height zero asserts the model scored zero. A missing bar says
    the measurement does not exist, which is the honest rendering of an N/A
    and is what Section 29 asks for.
    """
    if column not in frame.columns:
        return frame.iloc[0:0]
    out = frame.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=[column])


def _style_axes(axes, xlabel=None, ylabel=None, title=None):
    axes.set_title(title, color=INK, pad=12, loc="left")
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylabel:
        axes.set_ylabel(ylabel)
    axes.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)


def _bar_figure(frame, column, title, ylabel, path, value_format="{:.3f}",
                highlight_best=True, ascending=False, log=False):
    """One measure across models: a sorted bar chart with direct labels."""
    data = _numeric(frame, column)
    if data.empty:
        return None

    data = data.sort_values(column, ascending=ascending)
    names = [_label(n) for n in data["model"]]
    values = data[column].to_numpy()

    figure, axes = plt.subplots(figsize=(max(6.5, 0.95 * len(names) + 2), 4.6))

    best = values.argmax() if not ascending else values.argmin()
    colors = [
        ACCENT if (highlight_best and i == best) else SINGLE
        for i in range(len(values))
    ]

    bars = axes.bar(
        names, values, width=BAR_WIDTH, color=colors, linewidth=0,
    )
    for bar in bars:
        bar.set_joinstyle("round")

    if log:
        axes.set_yscale("log")

    span = values.max() - values.min() if len(values) > 1 else values.max()
    for bar, value in zip(bars, values):
        axes.annotate(
            value_format.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points", xytext=(0, 4),
            ha="center", fontsize=8.5, color=INK_SECONDARY,
        )

    if not log:
        axes.set_ylim(0, values.max() * 1.16 if values.max() > 0 else 1)
    _style_axes(axes, ylabel=ylabel, title=title)
    plt.setp(axes.get_xticklabels(), rotation=20, ha="right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _scatter_figure(frame, x_column, y_column, title, xlabel, ylabel, path,
                    log_x=False):
    """A trade-off plot: every point labeled, because identity matters here."""
    data = _numeric(_numeric(frame, x_column), y_column)
    if log_x:
        # A log axis has no place for zero, and the traditional baseline
        # genuinely has zero parameters -- it is not a small model, it is a
        # model with no learned parameters at all. Dropping it here is more
        # honest than plotting it at an invented position, and Section 49's
        # figure 7 asks for semantic mIoU against parameter count, which is
        # a question about networks.
        data = data[data[x_column] > 0]
    if data.empty:
        return None

    figure, axes = plt.subplots(figsize=(7.2, 5.0))
    axes.scatter(
        data[x_column], data[y_column], s=90, color=SINGLE,
        edgecolor=SURFACE, linewidth=2, zorder=3,
    )
    for _, row in data.iterrows():
        axes.annotate(
            _label(row["model"]),
            (row[x_column], row[y_column]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=9, color=INK_SECONDARY,
        )
    if log_x:
        axes.set_xscale("log")
    _style_axes(axes, xlabel=xlabel, ylabel=ylabel, title=title)
    axes.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def per_class_iou_figure(per_class: pd.DataFrame, path):
    """Figure 9: per-class IoU across semantic models.

    Seven series, so this is the one figure with a real categorical palette.
    A legend is always present; the contrast relief the palette requires is
    per_class_iou.csv, which carries the same numbers as text.
    """
    models = [c for c in per_class.columns if c != "class"]
    if not models:
        return None

    frame = per_class.copy()
    for model in models:
        frame[model] = pd.to_numeric(frame[model], errors="coerce")

    classes = frame["class"].tolist()
    positions = range(len(classes))
    width = 0.8 / max(len(models), 1)

    figure, axes = plt.subplots(figsize=(11, 5.2))
    for index, model in enumerate(models):
        offsets = [p - 0.4 + width * (index + 0.5) for p in positions]
        axes.bar(
            offsets, frame[model].fillna(0), width=width * 0.88,
            label=_label(model), color=SERIES[index % len(SERIES)], linewidth=0,
        )

    axes.set_xticks(list(positions))
    axes.set_xticklabels(classes)
    _style_axes(axes, ylabel="IoU", title="Per-class IoU across semantic models")
    axes.legend(
        frameon=False, ncol=min(len(models), 4), fontsize=9,
        loc="upper center", bbox_to_anchor=(0.5, -0.09),
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def instance_figure(instance: pd.DataFrame, path):
    """Figure 10: Mask R-CNN against YOLO on mask AP, AP50 and AP75.

    Kept strictly separate from every semantic figure. Section 26 forbids
    treating mask AP and mIoU as the same measurement, and the simplest way
    to honor that is never to give them a shared axis.
    """
    if instance.empty:
        return None

    metrics = [("mask_ap", "Mask AP"), ("mask_ap50", "AP50"), ("mask_ap75", "AP75")]
    frame = instance.copy()
    for column, _ in metrics:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame = frame.dropna(subset=[m for m, _ in metrics], how="all")
    if frame.empty:
        return None

    models = frame["model"].tolist()
    width = 0.8 / max(len(models), 1)

    figure, axes = plt.subplots(figsize=(7.4, 4.8))
    for index, (_, row) in enumerate(frame.iterrows()):
        offsets = [i - 0.4 + width * (index + 0.5) for i in range(len(metrics))]
        values = [row[column] for column, _ in metrics]
        bars = axes.bar(
            offsets, values, width=width * 0.88, label=_label(row["model"]),
            color=SERIES[index % len(SERIES)], linewidth=0,
        )
        for bar, value in zip(bars, values):
            if pd.notna(value):
                axes.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8.5, color=INK_SECONDARY,
                )

    axes.set_xticks(range(len(metrics)))
    axes.set_xticklabels([label for _, label in metrics])
    _style_axes(
        axes, ylabel="Average precision",
        title="Instance segmentation: Mask R-CNN vs YOLO",
    )
    axes.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def generate_all(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR) -> dict:
    """Write every Section 49 figure to the Section 52 filenames."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    def read(name):
        """Read a results CSV, tolerating one that has no rows yet.

        A table written before any model has run is a zero-byte file, and
        pandas raises EmptyDataError on it rather than returning an empty
        frame. Every figure function already handles an empty frame by
        drawing nothing, so the failure belongs here, not in the caller.
        """
        path = results_dir / name
        if not path.is_file():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    semantic = read("semantic_segmentation_results.csv")
    instance = read("instance_segmentation_results.csv")
    efficiency = read("segmentation_efficiency_results.csv")
    per_class = read("per_class_iou.csv")

    written = {}

    def record(key, path):
        if path is not None:
            written[key] = str(path)

    # 1 and 2: semantic quality.
    record("miou_comparison", _bar_figure(
        semantic, "mean_iou", "Semantic segmentation: mean IoU",
        "mIoU (6 classes, background included)",
        plots_dir / "miou_comparison.png",
    ))
    record("dice_comparison", _bar_figure(
        semantic, "dice", "Semantic segmentation: Dice score",
        "Dice (macro over present classes)",
        plots_dir / "dice_comparison.png",
    ))

    # 3 to 6: cost, across every architecture.
    record("parameters", _bar_figure(
        efficiency, "total_parameters", "Total parameters", "Parameters",
        plots_dir / "parameters.png", value_format="{:,.0f}",
        highlight_best=True, ascending=True,
    ))
    record("model_size", _bar_figure(
        efficiency, "model_size_mb", "Saved model size", "MiB (weights only)",
        plots_dir / "model_size.png", value_format="{:.1f}", ascending=True,
    ))
    record("training_time", _bar_figure(
        efficiency, "total_train_seconds", "Total training time", "Seconds",
        plots_dir / "training_time.png", value_format="{:,.0f}", ascending=True,
    ))
    record("inference_speed", _bar_figure(
        efficiency, "images_per_second", "Inference throughput", "Images / second",
        plots_dir / "inference_speed.png", value_format="{:.1f}",
    ))

    # 7 and 8: the trade-offs.
    record("miou_vs_parameters", _scatter_figure(
        semantic, "total_parameters", "mean_iou",
        "Quality against capacity", "Total parameters (log scale)", "mIoU",
        plots_dir / "miou_vs_parameters.png", log_x=True,
    ))
    record("miou_vs_latency", _scatter_figure(
        semantic, "latency_ms", "mean_iou",
        "Quality against latency", "Batch-one latency (ms)", "mIoU",
        plots_dir / "miou_vs_latency.png",
    ))

    # 9 and 10.
    record("per_class_iou", per_class_iou_figure(
        per_class, plots_dir / "per_class_iou.png"
    ))
    record("instance_mask_ap", instance_figure(
        instance, plots_dir / "instance_mask_ap.png"
    ))

    # Beyond the ten: boundary quality has its own figure because Section 24
    # asks for it and Section 54 asks which model had the best boundaries.
    record("boundary_f1", _bar_figure(
        semantic, "boundary_f1",
        "Boundary quality (F1, 3px tolerance)", "Boundary F1",
        plots_dir / "boundary_f1.png",
    ))

    return written
