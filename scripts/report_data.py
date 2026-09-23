"""Load the benchmark's saved outputs and answer questions about them.

The report is not allowed to state a number it did not read from disk, and
it is not allowed to import the pipeline. This module is the boundary: it
reads ``results/*.csv`` and ``results/raw/*.json`` and exposes small
query functions - who won, by how much, which class was hardest - so the
prose in the report can ask a question and get the measured answer rather
than a remembered one.

That indirection is the point. Section 29 forbids typing results by hand,
and the strongest version of that rule is a report that has no way to. Every
superlative in the generated document ("the highest mIoU", "the fastest",
"the smallest") comes from a function here, so if the numbers change the
sentences change with them, and a claim cannot survive the result that
justified it being revised away.
"""

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
PLOTS_DIR = REPO_ROOT / "plots"

NA = "N/A"


def _read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load() -> dict:
    """Everything the report reads, in one bundle."""
    raw = {}
    if RAW_DIR.is_dir():
        for path in sorted(RAW_DIR.glob("*.json")):
            payload = json.loads(path.read_text())
            raw[payload.get("model", path.stem)] = payload

    manifest_path = REPO_ROOT / "data" / "split_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    )

    return {
        "raw": raw,
        "manifest": manifest,
        "semantic": _read_csv("semantic_segmentation_results.csv"),
        "instance": _read_csv("instance_segmentation_results.csv"),
        "efficiency": _read_csv("segmentation_efficiency_results.csv"),
        "per_class_iou": _read_csv("per_class_iou.csv"),
        "per_class_boundary": _read_csv("per_class_boundary_f1.csv"),
        "history": _read_csv("training_history.csv"),
    }


# ---------------------------------------------------------------------------
# Queries the prose asks
# ---------------------------------------------------------------------------

def numeric(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rows with a real number in `column`, N/A dropped rather than zeroed."""
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=[column])


def best(frame: pd.DataFrame, column: str, highest: bool = True,
         exclude: list = None):
    """The winning row on one column, or None if nothing was measured."""
    data = numeric(frame, column)
    if exclude:
        data = data[~data["model"].isin(exclude)]
    if data.empty:
        return None
    return data.loc[data[column].idxmax() if highest else data[column].idxmin()]


def ranked(frame: pd.DataFrame, column: str, highest: bool = True,
           exclude: list = None) -> pd.DataFrame:
    data = numeric(frame, column)
    if exclude:
        data = data[~data["model"].isin(exclude)]
    return data.sort_values(column, ascending=not highest)


def value(frame: pd.DataFrame, model: str, column: str):
    """One model's value for one column, or None."""
    data = numeric(frame, column)
    row = data[data["model"] == model]
    return None if row.empty else float(row[column].iloc[0])


def fmt(number, places: int = 3, default: str = NA) -> str:
    """Format a measurement, or say plainly that there is not one."""
    if number is None:
        return default
    try:
        value = float(number)
    except (TypeError, ValueError):
        return str(number)
    if pd.isna(value):
        return default
    return f"{value:.{places}f}"


def fmt_int(number, default: str = NA) -> str:
    if number is None:
        return default
    try:
        return f"{int(float(number)):,}"
    except (TypeError, ValueError):
        return default


def fmt_seconds(seconds, default: str = NA) -> str:
    """Seconds as something a reader can hold in their head."""
    if seconds is None:
        return default
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return default
    if pd.isna(seconds):
        return default
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


DISPLAY = {
    "kmeans": "K-Means",
    "fcn_resnet50": "FCN-ResNet50",
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "segnet": "SegNet",
    "deeplabv3": "DeepLabV3-ResNet50",
    "pspnet": "PSPNet",
    "segformer": "SegFormer-B0",
    "maskrcnn": "Mask R-CNN",
    "yolo_seg": "YOLO11n-seg",
}


def name(model: str) -> str:
    return DISPLAY.get(model, model)


def table(frame: pd.DataFrame, columns: dict = None, places: int = 3) -> str:
    """Render a DataFrame as a Markdown table.

    Numbers are formatted to a fixed number of places so columns line up and
    so the report never shows a measurement at more precision than it has.
    """
    if frame.empty:
        return "_No results yet._\n"

    data = frame.copy()
    if columns:
        available = [c for c in columns if c in data.columns]
        data = data[available].rename(columns=columns)

    if "model" in data.columns:
        data["model"] = data["model"].map(lambda m: name(m))
    if "Model" in data.columns:
        data["Model"] = data["Model"].map(lambda m: name(m))

    def cell(value):
        if isinstance(value, float):
            if pd.isna(value):
                return NA
            if value.is_integer() and abs(value) >= 1000:
                return f"{int(value):,}"
            return f"{value:.{places}f}"
        return str(value)

    header = "| " + " | ".join(str(c) for c in data.columns) + " |"
    divider = "|" + "|".join("---" for _ in data.columns) + "|"
    rows = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in data.itertuples(index=False)
    ]
    return "\n".join([header, divider, *rows]) + "\n"


def figure(filename: str, caption: str) -> str:
    """Embed a figure, but only if it exists.

    A report that references a missing image is worse than one that says the
    figure is absent, because the reader cannot tell whether the finding was
    dropped or the file was.
    """
    if not (PLOTS_DIR / filename).is_file():
        return f"_Figure `{filename}` has not been generated._\n"
    return f"![{caption}](../plots/{filename})\n\n*{caption}*\n"


def hardest_class(per_class: pd.DataFrame, exclude=("kmeans",)) -> tuple:
    """The class with the lowest mean IoU across models, and its value."""
    if per_class.empty:
        return None, None
    models = [
        c for c in per_class.columns
        if c != "class" and c not in exclude
    ]
    if not models:
        return None, None
    data = per_class.copy()
    for model in models:
        data[model] = pd.to_numeric(data[model], errors="coerce")
    data["mean"] = data[models].mean(axis=1)
    foreground = data[data["class"] != "background"].dropna(subset=["mean"])
    if foreground.empty:
        return None, None
    row = foreground.loc[foreground["mean"].idxmin()]
    return row["class"], float(row["mean"])


def easiest_class(per_class: pd.DataFrame, exclude=("kmeans",)) -> tuple:
    if per_class.empty:
        return None, None
    models = [c for c in per_class.columns if c != "class" and c not in exclude]
    if not models:
        return None, None
    data = per_class.copy()
    for model in models:
        data[model] = pd.to_numeric(data[model], errors="coerce")
    data["mean"] = data[models].mean(axis=1)
    foreground = data[data["class"] != "background"].dropna(subset=["mean"])
    if foreground.empty:
        return None, None
    row = foreground.loc[foreground["mean"].idxmax()]
    return row["class"], float(row["mean"])


def overfitting_gap(history: pd.DataFrame, model: str):
    """How far validation loss rose from its minimum by the final epoch.

    Section 54 asks which model overfit most. A model whose validation loss
    bottoms at epoch 8 and climbs for seventeen more epochs is overfitting
    regardless of where its training loss went, and the size of that climb
    is the comparable quantity.
    """
    if history.empty or "val_loss" not in history.columns:
        return None
    rows = history[history["model"] == model].sort_values("epoch")
    losses = pd.to_numeric(rows.get("val_loss"), errors="coerce").dropna()
    if len(losses) < 3:
        return None
    return float(losses.iloc[-1] - losses.min())


def epochs_to_fraction(history: pd.DataFrame, model: str,
                       fraction: float = 0.9):
    """Epochs taken to reach a fraction of the model's own best val mIoU.

    Relative to its own ceiling rather than to an absolute score, because
    Section 54's convergence question is about speed of learning, not about
    final quality - which the mIoU table already answers.
    """
    if history.empty or "val_mean_iou" not in history.columns:
        return None
    rows = history[history["model"] == model].sort_values("epoch")
    scores = pd.to_numeric(rows.get("val_mean_iou"), errors="coerce")
    if scores.dropna().empty:
        return None
    target = scores.max() * fraction
    reached = rows[scores >= target]
    return None if reached.empty else int(reached["epoch"].iloc[0])


def pretrained_split(data: dict) -> dict:
    """Group models by what their weights were initialized from.

    Section 54 asks how important pretrained initialization was, and this
    benchmark cannot answer it as a clean two-group comparison: the models
    sit on a spectrum from fully pretrained segmentation heads to fully
    random. Grouping them honestly is the difference between an answer and
    an overclaim.
    """
    groups = {"coco_segmentation": [], "imagenet_backbone": [], "scratch": []}
    for model, payload in data["raw"].items():
        card = payload.get("card", {})
        initialization = str(card.get("initialization", "")).lower()
        if payload.get("task") not in {"semantic"}:
            continue
        if "coco" in initialization:
            groups["coco_segmentation"].append(model)
        elif "imagenet" in initialization or "mit-b0" in initialization:
            groups["imagenet_backbone"].append(model)
        else:
            groups["scratch"].append(model)
    return groups
