"""Section 52: example predictions, and the qualitative comparisons.

Three things the numbers cannot say on their own.

**Per-model prediction examples** (Section 52) go to ``predictions/<model>/``
on the same test images for every model, so the panels can be read side by
side. Same images, same order, chosen deterministically - a figure built from
whichever images a model did best on is an advertisement, not evidence.

**Boundary close-ups** (Section 24) zoom on a region where the prediction and
the ground truth disagree, because that is the comparison the boundary F1
number summarizes and the one a reader wants to check.

**K-Means cluster output** (Section 10) is rendered as raw cluster IDs
alongside its mapped class prediction. The assignment asks for a qualitative
comparison of the traditional baseline specifically because its cluster IDs
are not class labels, and showing the clusters makes visible what the method
actually produces before the frozen mapping is applied on top.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ._config import CLASS_NAMES, IGNORE_INDEX, PREDICTIONS_DIR  # noqa: E402

# Same palette as the verification figure, so a reader who has seen one can
# read the other without relearning the colors.
PALETTE = np.array([
    [0, 0, 0],          # background
    [220, 20, 60],      # person
    [0, 130, 200],      # car
    [255, 225, 25],     # bicycle
    [60, 180, 75],      # dog
    [245, 130, 48],     # cat
], dtype=np.uint8)
IGNORE_COLOR = np.array([128, 128, 128], dtype=np.uint8)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(len(PALETTE)):
        out[mask == class_id] = PALETTE[class_id]
    out[mask == IGNORE_INDEX] = IGNORE_COLOR
    return out


def denormalize(tensor: torch.Tensor, normalized: bool = True) -> np.ndarray:
    """Recover a displayable image from a model input."""
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if normalized:
        array = array * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(array * 255, 0, 255).astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55):
    blended = image.copy()
    painted = (mask != 0) & (mask != IGNORE_INDEX)
    colored = colorize(mask)
    blended[painted] = (
        alpha * colored[painted] + (1 - alpha) * image[painted]
    ).astype(np.uint8)
    return blended


def _legend(figure, extra=None):
    handles = [
        mpatches.Patch(color=PALETTE[i] / 255, label=CLASS_NAMES[i])
        for i in range(1, len(PALETTE))
    ]
    handles.append(mpatches.Patch(color=IGNORE_COLOR / 255, label="ignored"))
    if extra:
        handles += extra
    figure.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01),
    )


def save_semantic_predictions(model, loader, device, model_name: str,
                              count: int = 8, normalized: bool = True,
                              output_dir=None) -> list:
    """Image, ground truth and prediction for the first `count` test images.

    The first images in the loader, not a selection. The loader is
    deterministic for evaluation splits, so every model is shown on the same
    images in the same order and the panels compare.
    """
    output_dir = (output_dir or PREDICTIONS_DIR) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved, collected = [], 0

    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            if isinstance(logits, dict):
                logits = logits["out"]
            predictions = logits.argmax(dim=1).cpu().numpy()

            for index in range(len(images)):
                if collected >= count:
                    break
                image = denormalize(images[index], normalized)
                truth = targets[index].numpy()
                prediction = predictions[index]

                figure, axes = plt.subplots(1, 3, figsize=(11, 3.9))
                axes[0].imshow(image)
                axes[0].set_title("image", fontsize=10)
                axes[1].imshow(overlay(image, truth))
                axes[1].set_title("ground truth", fontsize=10)
                axes[2].imshow(overlay(image, prediction))
                axes[2].set_title(f"{model_name} prediction", fontsize=10)
                for axis in axes:
                    axis.set_xticks([])
                    axis.set_yticks([])
                _legend(figure)
                figure.tight_layout(rect=[0, 0.06, 1, 1])

                path = output_dir / f"example_{collected:02d}.png"
                figure.savefig(path, dpi=110, bbox_inches="tight")
                plt.close(figure)
                saved.append(str(path))
                collected += 1

            if collected >= count:
                break
    return saved


def save_boundary_closeup(model, loader, device, model_name: str,
                          normalized: bool = True, tolerance: int = 3,
                          output_dir=None):
    """Zoom on the region where prediction and truth disagree most.

    Section 24 asks for zoomed-in boundary comparison. The crop is chosen by
    finding the densest concentration of disagreement rather than by picking
    a pleasing region, so the figure shows the model's worst case on that
    image rather than its best.
    """
    from .boundary import extract_boundary

    output_dir = (output_dir or PREDICTIONS_DIR) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        images, targets = next(iter(loader))
        logits = model(images.to(device))
        if isinstance(logits, dict):
            logits = logits["out"]
        predictions = logits.argmax(dim=1).cpu().numpy()

    best_index, best_score, best_window = 0, -1, None
    window = 96

    for index in range(len(images)):
        truth = targets[index].numpy()
        prediction = predictions[index]
        disagreement = (truth != prediction) & (truth != IGNORE_INDEX)
        if not disagreement.any():
            continue

        # Coarse search for the densest window of disagreement.
        height, width = disagreement.shape
        step = max(16, window // 3)
        for top in range(0, max(1, height - window), step):
            for left in range(0, max(1, width - window), step):
                score = disagreement[top:top + window, left:left + window].sum()
                if score > best_score:
                    best_score, best_index = score, index
                    best_window = (top, left)

    if best_window is None:
        return None

    top, left = best_window
    image = denormalize(images[best_index], normalized)
    truth = targets[best_index].numpy()
    prediction = predictions[best_index]

    crop = slice(top, top + window), slice(left, left + window)

    figure, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    axes[0].imshow(image)
    axes[0].add_patch(mpatches.Rectangle(
        (left, top), window, window, fill=False, edgecolor="#eb6834", lw=2
    ))
    axes[0].set_title("image (crop marked)", fontsize=10)
    axes[1].imshow(image[crop])
    axes[1].set_title("crop", fontsize=10)
    axes[2].imshow(overlay(image[crop], truth[crop]))
    axes[2].set_title("ground-truth boundary", fontsize=10)
    axes[3].imshow(overlay(image[crop], prediction[crop]))
    axes[3].set_title(f"{model_name} boundary", fontsize=10)

    # Outline both boundaries so the comparison is about edges, not regions.
    for axis, mask in ((axes[2], truth[crop]), (axes[3], prediction[crop])):
        edges = np.zeros(mask.shape, dtype=bool)
        for class_id in range(1, len(PALETTE)):
            edges |= extract_boundary(mask == class_id)
        axis.contour(edges.astype(float), levels=[0.5],
                     colors="white", linewidths=0.8)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"Boundary detail, {model_name}, boundary F1 uses a "
        f"{tolerance}px tolerance",
        fontsize=11,
    )
    figure.tight_layout(rect=[0, 0.02, 1, 0.94])

    path = output_dir / "boundary_closeup.png"
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return str(path)


def save_kmeans_qualitative(segmenter, images, masks, count: int = 6,
                            output_dir=None) -> list:
    """Section 10's qualitative comparison for the traditional baseline.

    Four panels: image, ground truth, raw cluster IDs, mapped prediction.
    The third panel is the one the assignment is really asking for. Cluster
    IDs are not class labels, and seeing the clusters shows what the method
    genuinely produces - coherent regions that follow color and lighting
    rather than objects - before a frozen mapping is laid on top of them.
    """
    output_dir = (output_dir or PREDICTIONS_DIR) / "kmeans"
    output_dir.mkdir(parents=True, exist_ok=True)

    # A distinct, non-class palette for cluster IDs, so nobody reads a
    # cluster color as a class color.
    cluster_colors = plt.get_cmap("tab10")

    saved = []
    for index, (image, truth) in enumerate(zip(images, masks)):
        if index >= count:
            break
        clusters = segmenter.cluster_only(image)
        prediction = segmenter.predict(image)

        figure, axes = plt.subplots(1, 4, figsize=(14, 3.9))
        axes[0].imshow(image)
        axes[0].set_title("image", fontsize=10)
        axes[1].imshow(overlay(image, truth))
        axes[1].set_title("ground truth", fontsize=10)
        axes[2].imshow(clusters, cmap=cluster_colors, interpolation="nearest")
        axes[2].set_title("K-Means clusters (not classes)", fontsize=10)
        axes[3].imshow(overlay(image, prediction))
        axes[3].set_title("after the frozen class mapping", fontsize=10)
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        _legend(figure)
        figure.tight_layout(rect=[0, 0.06, 1, 1])

        path = output_dir / f"clusters_{index:02d}.png"
        figure.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(figure)
        saved.append(str(path))
    return saved


def save_instance_predictions(model, loader, device, model_name: str,
                              count: int = 6, score_threshold: float = 0.5,
                              output_dir=None) -> list:
    """Instance predictions, with each object in its own color.

    Colored by instance rather than by class, which is the whole point of the
    instance track: two people must be visibly two things. A class-colored
    rendering would look identical to a semantic prediction and would hide
    exactly what is being demonstrated.
    """
    output_dir = (output_dir or PREDICTIONS_DIR) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    instance_colors = plt.get_cmap("tab20")
    saved, collected = [], 0

    with torch.no_grad():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])

            for index, output in enumerate(outputs):
                if collected >= count:
                    break
                image = denormalize(images[index], normalized=False)

                truth_panel = image.copy()
                for order, binary in enumerate(targets[index]["masks"].numpy()):
                    color = np.array(instance_colors(order % 20)[:3]) * 255
                    mask = binary.astype(bool)
                    truth_panel[mask] = (
                        0.55 * color + 0.45 * truth_panel[mask]
                    ).astype(np.uint8)

                prediction_panel = image.copy()
                scores = output["scores"].cpu().numpy()
                masks = output["masks"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                kept = np.where(scores >= score_threshold)[0]

                for order, position in enumerate(kept):
                    binary = masks[position]
                    if binary.ndim == 3:
                        binary = binary[0]
                    binary = binary >= 0.5
                    color = np.array(instance_colors(order % 20)[:3]) * 255
                    prediction_panel[binary] = (
                        0.55 * color + 0.45 * prediction_panel[binary]
                    ).astype(np.uint8)

                figure, axes = plt.subplots(1, 3, figsize=(11, 3.9))
                axes[0].imshow(image)
                axes[0].set_title("image", fontsize=10)
                axes[1].imshow(truth_panel)
                axes[1].set_title(
                    f"ground truth, {len(targets[index]['masks'])} instances",
                    fontsize=10,
                )
                axes[2].imshow(prediction_panel)
                axes[2].set_title(
                    f"{model_name}, {len(kept)} above {score_threshold}",
                    fontsize=10,
                )
                for axis in axes:
                    axis.set_xticks([])
                    axis.set_yticks([])
                figure.suptitle(
                    "One color per instance, not per class", fontsize=10,
                )
                figure.tight_layout(rect=[0, 0, 1, 0.94])

                path = output_dir / f"example_{collected:02d}.png"
                figure.savefig(path, dpi=110, bbox_inches="tight")
                plt.close(figure)
                saved.append(str(path))
                collected += 1

            if collected >= count:
                break
    return saved
