"""Model 1: the non-deep-learning baseline, and the trap that comes with it.

Section 10 is unusually direct about the failure mode here:

    K-Means cluster IDs are not class labels. Use qualitative comparisons
    unless you define a class-mapping procedure using training data only. Do
    not match clusters to test labels to inflate semantic scores.

The tempting implementation is to run K-Means on a test image, then assign
each cluster whichever class it overlaps most in the ground truth. That
produces a respectable mIoU and is meaningless: it hands the algorithm the
answer and measures only whether the clusters happen to be shaped like
objects. Done that way K-Means can appear to beat U-Net, which is a result
about the evaluation and not about segmentation.

So the mapping here is fitted on training images only, frozen, and applied
unchanged to validation and test. Concretely: cluster every training image,
describe each cluster by its appearance and position, and record which class
the training ground truth says such regions usually are. At test time a
cluster is labeled by that frozen record, never by the test mask.

What this baseline actually does is group pixels that look alike. That is
not the same task as naming objects, and Section 10 asks for the difference
to be explained rather than papered over: K-Means separates a dark dog from
a light lawn, but also separates the dog's sunlit flank from its shadowed
one, and merges a gray cat with the gray pavement behind it. Those failures
are properties of color clustering, not deficiencies to be tuned away.
"""

import json

import numpy as np
from sklearn.cluster import KMeans

from ._config import IGNORE_INDEX, NUM_CLASSES, SEED

# Feature layout per pixel: L, a, b, then normalized row and column. Position
# is included because color alone has no notion of adjacency, and without it
# every scattered patch of similar color joins one cluster. It is weighted
# below color so that position nudges rather than dominates.
#
# This is the five-dimensional feature vector Nayar recommends in the course
# reading (FPCV-5-2, slide 30): clustering on color alone produces clusters
# that "map to many disconnected segments in the image", and adding the
# spatial coordinates encourages nearby pixels into the same segment. The
# deviation from that slide is the color space, CIELAB rather than RGB,
# because CIELAB distances track perceived difference and k-means is driven
# entirely by distance.
POSITION_WEIGHT = 0.35


def _features(image: np.ndarray) -> np.ndarray:
    """Per-pixel features: CIELAB color plus weighted normalized position.

    LAB rather than RGB because its distances correspond better to perceived
    difference, so a K-Means boundary falls closer to where a person would
    put one. That is the most defensible choice available to a method with
    no learned notion of an object.
    """
    import cv2

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    height, width = lab.shape[:2]

    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    rows = (rows / max(height - 1, 1)) * 255.0 * POSITION_WEIGHT
    cols = (cols / max(width - 1, 1)) * 255.0 * POSITION_WEIGHT

    return np.stack(
        [lab[..., 0], lab[..., 1], lab[..., 2], rows, cols], axis=-1
    ).reshape(-1, 5)


class KMeansSegmenter:
    """Cluster pixels per image, then label clusters with a frozen mapping.

    Two stages that must stay separate, because collapsing them is exactly
    what Section 10 forbids:

    ``fit`` reads training images and their ground truth, and learns a
    mapping from cluster descriptor to class.

    ``predict`` reads an image alone. It never sees a mask. Passing one in
    would be the inflation the assignment warns about, so ``predict`` has no
    parameter to pass it through.
    """

    def __init__(self, n_clusters: int = 6, seed: int = SEED,
                 num_classes: int = NUM_CLASSES):
        self.n_clusters = n_clusters
        self.seed = seed
        self.num_classes = num_classes
        # descriptor bin -> class ID, learned from training data only.
        self.mapping = {}
        self.fitted = False
        self._fit_images = 0

    def _cluster(self, image: np.ndarray) -> np.ndarray:
        """Cluster one image's pixels, returning a cluster-ID map.

        Seeded, and with a fixed number of initializations, so the same image
        gives the same clustering on every run. An unseeded K-Means would
        make the baseline's score change between runs for no reason.
        """
        features = _features(image)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            n_init=4,
            max_iter=100,
        )
        labels = kmeans.fit_predict(features)
        return labels.reshape(image.shape[:2])

    @staticmethod
    def _descriptor(image: np.ndarray, cluster_mask: np.ndarray) -> tuple:
        """A coarse, quantized description of one cluster's appearance.

        Deliberately coarse. The descriptor has to generalize from training
        images to unseen ones, and a fine-grained key would simply memorize
        individual training regions and miss at test time. Mean lightness and
        color opponency are quantized into a few bins each, along with the
        cluster's vertical center and how much of the frame it covers -
        sky-like regions sit high and cover a lot, a cat sits low and covers
        little.
        """
        import cv2

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        pixels = lab[cluster_mask]
        if len(pixels) == 0:
            return (0, 0, 0, 0, 0)

        lightness = int(np.clip(pixels[:, 0].mean() / 255 * 4, 0, 3))
        green_red = int(np.clip((pixels[:, 1].mean() - 128) / 64 + 1.5, 0, 2))
        blue_yellow = int(np.clip((pixels[:, 2].mean() - 128) / 64 + 1.5, 0, 2))

        rows = np.where(cluster_mask.any(axis=1))[0]
        center = int(np.clip(rows.mean() / cluster_mask.shape[0] * 3, 0, 2))
        coverage = int(np.clip(cluster_mask.mean() * 4, 0, 3))

        return (lightness, green_red, blue_yellow, center, coverage)

    def fit(self, images, masks, max_images: int = 300) -> dict:
        """Learn descriptor -> class from training images and their masks.

        For each cluster in each training image, the descriptor is computed
        and the ground-truth class histogram over that cluster's pixels is
        accumulated into that descriptor's bin. After all training images,
        each bin takes the class that won it overall.

        The accumulation is across images, not within one. A bin decided by a
        single image would be that image's idiosyncrasy; a bin summed over
        hundreds is a weak statement about what such regions usually are,
        which is the most a color clustering can support.
        """
        votes = {}
        used = 0

        for image, mask in zip(images, masks):
            if used >= max_images:
                break
            clusters = self._cluster(image)
            valid = mask != IGNORE_INDEX

            for cluster_id in range(self.n_clusters):
                selected = clusters == cluster_id
                if not selected.any():
                    continue
                key = self._descriptor(image, selected)
                counted = mask[selected & valid]
                if len(counted) == 0:
                    continue
                histogram = np.bincount(counted, minlength=self.num_classes)
                votes.setdefault(key, np.zeros(self.num_classes, dtype=np.int64))
                votes[key] += histogram
            used += 1

        self.mapping = {key: int(np.argmax(v)) for key, v in votes.items()}
        self.fitted = True
        self._fit_images = used

        assigned = np.bincount(
            list(self.mapping.values()) or [0], minlength=self.num_classes
        )
        return {
            "fit_images": used,
            "descriptor_bins": len(self.mapping),
            "bins_per_class": {str(i): int(c) for i, c in enumerate(assigned)},
        }

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Segment one image. Takes no mask, by design.

        Descriptors unseen during training fall back to background, which is
        the majority class at 83% of pixels. That is the honest default for
        a region the training data says nothing about, and it is decided
        without reference to the test label.
        """
        if not self.fitted:
            raise RuntimeError(
                "KMeansSegmenter.predict before fit. The cluster-to-class "
                "mapping must be learned from training data (Section 10)."
            )

        clusters = self._cluster(image)
        out = np.zeros(image.shape[:2], dtype=np.uint8)
        for cluster_id in range(self.n_clusters):
            selected = clusters == cluster_id
            if not selected.any():
                continue
            key = self._descriptor(image, selected)
            out[selected] = self.mapping.get(key, 0)
        return out

    def cluster_only(self, image: np.ndarray) -> np.ndarray:
        """Raw cluster IDs, for the qualitative figure Section 10 asks for.

        These are not class predictions and must never be scored against
        ground truth. They are what the method actually produces; the class
        mapping is a layer bolted on top so that a number exists at all.
        """
        return self._cluster(image)

    def describe(self) -> dict:
        return {
            "model": "kmeans",
            "n_clusters": self.n_clusters,
            "features": "CIELAB + normalized position",
            "position_weight": POSITION_WEIGHT,
            "seed": self.seed,
            "mapping_source": "training images only",
            "mapping_bins": len(self.mapping),
            "fit_images": self._fit_images,
            "unseen_descriptor_fallback": "background",
            "note": (
                "Cluster IDs are not class labels. The descriptor-to-class "
                "mapping is fitted on training data and frozen before any "
                "validation or test image is seen."
            ),
        }

    def save(self, path) -> None:
        """Persist the frozen mapping, so evaluation can rerun without refitting."""
        payload = {
            "config": self.describe(),
            "mapping": {",".join(map(str, k)): v for k, v in self.mapping.items()},
        }
        path.write_text(json.dumps(payload, indent=2))

    def load(self, path) -> "KMeansSegmenter":
        payload = json.loads(path.read_text())
        self.mapping = {
            tuple(int(p) for p in key.split(",")): value
            for key, value in payload["mapping"].items()
        }
        self.fitted = True
        self._fit_images = payload["config"].get("fit_images", 0)
        return self
