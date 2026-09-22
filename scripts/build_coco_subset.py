"""Build the COCO 2017 subset every model in this benchmark trains on.

Section 5 asks for roughly 5,000 training, 1,000 validation and 1,000 test
images carrying ground-truth masks for person, car, bicycle, dog and cat.
Section 7 requires one split, generated at SEED = 42, shared by all ten
architectures, with the manifest recorded. This script produces both, and
nothing else in the repository is allowed to partition data.

Three decisions worth stating, because each one could reasonably have gone
the other way:

**All three splits are drawn from train2017.** The obvious alternative is to
take the test split from val2017, which is what COCO's own layout invites.
That would make the test set a different draw from a different annotation
pass, and any gap between validation and test scores would then be partly a
dataset difference rather than a generalization gap. Drawing all 7,000 from
one pool means the three partitions are identically distributed by
construction, so the validation-to-test comparison in the report measures
what it claims to.

**Only the selected images are downloaded.** The full train2017 archive is
about 19 GB for 118,287 images, and this benchmark needs 7,000 of them.
Fetching them individually costs roughly 1.2 GB and a few minutes.

**Masks are rendered once, here, and saved as PNGs.** Decoding RLE is not
free, and doing it inside the training loop would charge every architecture
for it once per epoch - which would then show up in the Section 32 training
time comparison as a difference between models that is really a difference
in how often each one happened to re-decode the same annotation.

Usage
-----
    python scripts/build_coco_subset.py
    python scripts/build_coco_subset.py --limit 200   # small set, for tests
"""

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from segmentation_benchmark import mask_utils  # noqa: E402
from segmentation_benchmark._config import (  # noqa: E402
    ANNOTATIONS_DIR,
    CLASS_NAMES,
    COCO_CATEGORY_IDS,
    CROWD_POLICY,
    DATA_DIR,
    IGNORE_INDEX,
    IMAGES_DIR,
    MIN_FOREGROUND_PIXELS,
    OVERLAP_POLICY,
    SEED,
    SPLIT_SIZES,
    UNSELECTED_POLICY,
)

ANNOTATION_ZIP_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/train2017/{file_name}"
SOURCE_ANNOTATIONS = "instances_train2017.json"

MASKS_DIR = DATA_DIR / "masks"
MANIFEST_PATH = DATA_DIR / "split_manifest.json"

DOWNLOAD_WORKERS = 16
DOWNLOAD_RETRIES = 3


def download_annotations() -> Path:
    """Fetch and extract the COCO instance annotations if not already present."""
    target = ANNOTATIONS_DIR / SOURCE_ANNOTATIONS
    if target.is_file():
        print(f"  annotations already present: {target.name}")
        return target

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    archive = ANNOTATIONS_DIR / "annotations_trainval2017.zip"

    if not archive.is_file():
        print(f"  downloading {ANNOTATION_ZIP_URL} (~241 MB)")
        urllib.request.urlretrieve(ANNOTATION_ZIP_URL, archive)

    print("  extracting instance annotations")
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith(SOURCE_ANNOTATIONS):
                with z.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                break

    # The archive is 241 MB of which we keep one file; there is no reason to
    # leave it on a disk that also has to hold 7,000 images and ten sets of
    # checkpoints.
    archive.unlink()
    return target


def eligible_images(coco: dict) -> dict:
    """Images carrying at least one non-crowd annotation in our five classes.

    Returns ``{image_id: [annotation, ...]}`` where the annotation list holds
    every selected-category annotation for that image, crowd ones included.
    The crowd annotations are kept because the semantic mask needs them to
    mark ignore regions, even though they cannot make an image eligible on
    their own.
    """
    by_image = {}
    for annotation in coco["annotations"]:
        if annotation["category_id"] not in COCO_CATEGORY_IDS:
            continue
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    return {
        image_id: annotations
        for image_id, annotations in by_image.items()
        if any(a.get("iscrowd", 0) == 0 for a in annotations)
    }


def partition(image_ids: list, sizes: dict) -> dict:
    """Split the eligible pool at SEED = 42.

    Sorted before shuffling because the order COCO's JSON happens to produce
    is not guaranteed stable across releases, and a seeded shuffle of an
    unstable order is not reproducible. Sorting first makes the input to the
    shuffle deterministic, so the same seed gives the same split on any
    machine and any COCO download.
    """
    ordered = sorted(image_ids)
    rng = random.Random(SEED)
    rng.shuffle(ordered)

    total = sum(sizes.values())
    if len(ordered) < total:
        raise SystemExit(
            f"Only {len(ordered)} eligible images, need {total}. "
            "Check the category filter."
        )

    splits, cursor = {}, 0
    for name in ("train", "val", "test"):
        splits[name] = ordered[cursor:cursor + sizes[name]]
        cursor += sizes[name]
    return splits


def fetch_image(record: dict, destination: Path) -> tuple:
    """Download one image, retrying on transient failures.

    Returns ``(image_id, ok, note)``. A failure is reported rather than
    raised because one unreachable image should not discard 6,999 successful
    downloads; the caller decides whether the failure count is acceptable.
    """
    if destination.is_file() and destination.stat().st_size > 0:
        return record["id"], True, "cached"

    url = IMAGE_URL_TEMPLATE.format(file_name=record["file_name"])
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
            destination.write_bytes(payload)
            return record["id"], True, "downloaded"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == DOWNLOAD_RETRIES - 1:
                return record["id"], False, str(exc)
            time.sleep(2 ** attempt)
    return record["id"], False, "exhausted retries"


def download_split(records: list, split: str) -> list:
    """Download one split's images in parallel, returning the failures."""
    target_dir = IMAGES_DIR / split
    target_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    done = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(fetch_image, r, target_dir / r["file_name"]): r
            for r in records
        }
        for future in as_completed(futures):
            image_id, ok, note = future.result()
            done += 1
            if not ok:
                failures.append({"image_id": image_id, "reason": note})
            if done % 500 == 0 or done == len(records):
                print(f"    {split}: {done}/{len(records)}")
    return failures


def render_masks(records: list, annotations: dict, split: str) -> dict:
    """Render and save the semantic mask for each image in a split.

    Masks are saved as 8-bit PNGs at the image's native resolution, with
    IGNORE_INDEX = 255 for excluded pixels. PNG because it is lossless -
    a JPEG mask would interpolate class IDs into each other at every
    boundary, inventing classes that were never annotated.
    """
    mask_dir = MASKS_DIR / split
    mask_dir.mkdir(parents=True, exist_ok=True)

    class_pixels = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    ignored_pixels = 0
    instance_count = 0

    for index, record in enumerate(records, start=1):
        image_annotations = annotations[record["id"]]
        mask = mask_utils.semantic_mask(
            image_annotations, record["height"], record["width"]
        )

        stem = Path(record["file_name"]).stem
        Image.fromarray(mask).save(mask_dir / f"{stem}.png", optimize=True)

        class_pixels += mask_utils.class_pixel_counts(mask)
        ignored_pixels += int((mask == IGNORE_INDEX).sum())
        instance_count += sum(
            1 for a in image_annotations if a.get("iscrowd", 0) == 0
        )

        if index % 500 == 0 or index == len(records):
            print(f"    {split}: {index}/{len(records)}")

    return {
        "images": len(records),
        "instances": instance_count,
        "pixels_per_class": {
            name: int(count) for name, count in zip(CLASS_NAMES, class_pixels)
        },
        "ignored_pixels": ignored_pixels,
    }


def write_coco_subset(records: list, annotations: dict, split: str) -> Path:
    """Write a COCO-format annotation file for one split.

    Section 25 requires a COCO-style mask AP evaluator, and pycocotools'
    COCOeval needs ground truth in COCO's own format. Category IDs are
    remapped to our contiguous 1-5 so the evaluator's IDs match what the
    instance models predict; the original COCO IDs are recorded in the file's
    ``info`` block so the mapping is never guessed at later.

    Crowd annotations are kept here, with ``iscrowd=1`` intact. COCOeval uses
    them to suppress false positives that land on an unannotated group -
    which is exactly the behavior wanted, and the reason they are preserved
    in the annotation file while being excluded from the training targets.
    """
    selected_ids = {r["id"] for r in records}
    out_annotations = []
    for image_id in selected_ids:
        for annotation in annotations[image_id]:
            entry = dict(annotation)
            entry["category_id"] = COCO_CATEGORY_IDS[annotation["category_id"]]
            out_annotations.append(entry)

    document = {
        "info": {
            "description": f"COCO 2017 segmentation benchmark subset - {split}",
            "seed": SEED,
            "source": SOURCE_ANNOTATIONS,
            "category_id_mapping": {
                str(k): v for k, v in COCO_CATEGORY_IDS.items()
            },
            "overlap_policy": OVERLAP_POLICY,
            "crowd_policy": CROWD_POLICY,
            "unselected_policy": UNSELECTED_POLICY,
        },
        "images": records,
        "annotations": out_annotations,
        "categories": [
            {"id": index, "name": name, "supercategory": "object"}
            for index, name in enumerate(CLASS_NAMES)
            if index > 0
        ],
    }

    path = ANNOTATIONS_DIR / f"instances_{split}.json"
    path.write_text(json.dumps(document))
    return path


def manifest_checksum(splits: dict) -> str:
    """A short fingerprint of the split, for checking it has not drifted.

    Every downstream run records this. If a rebuilt split produces a
    different checksum, the comparison across models is no longer
    like-for-like and the run should stop rather than quietly report numbers
    from two different test sets.
    """
    payload = json.dumps(
        {name: sorted(ids) for name, ids in splits.items()}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Build a proportionally smaller subset, for tests and dry runs.",
    )
    args = parser.parse_args()

    sizes = dict(SPLIT_SIZES)
    if args.limit:
        scale = args.limit / sum(sizes.values())
        sizes = {k: max(1, int(v * scale)) for k, v in sizes.items()}
        print(f"Limited build: {sizes}")

    print("1. Annotations")
    annotation_path = download_annotations()

    print("2. Reading annotations")
    coco = json.loads(annotation_path.read_text())
    images_by_id = {image["id"]: image for image in coco["images"]}
    print(f"  {len(coco['images']):,} images, {len(coco['annotations']):,} annotations")

    print("3. Selecting eligible images")
    annotations = eligible_images(coco)
    print(
        f"  {len(annotations):,} carry a non-crowd annotation in "
        f"{', '.join(CLASS_NAMES[1:])}"
    )

    print(f"4. Partitioning at SEED = {SEED}")
    splits = partition(list(annotations), sizes)
    for name, ids in splits.items():
        print(f"  {name}: {len(ids):,}")

    records = {
        name: [images_by_id[i] for i in ids] for name, ids in splits.items()
    }

    print("5. Downloading images")
    all_failures = {}
    for name, split_records in records.items():
        failures = download_split(split_records, name)
        if failures:
            all_failures[name] = failures
            print(f"  {name}: {len(failures)} failed")

    if all_failures:
        total = sum(len(f) for f in all_failures.values())
        print(
            f"\n{total} images could not be downloaded. The split manifest "
            "records them; rerun to retry only the missing ones."
        )

    print("6. Rendering semantic masks")
    statistics = {
        name: render_masks(split_records, annotations, name)
        for name, split_records in records.items()
    }

    print("7. Writing COCO subsets for instance evaluation")
    for name, split_records in records.items():
        path = write_coco_subset(split_records, annotations, name)
        print(f"  {path.name}")

    print("8. Writing split manifest")
    manifest = {
        "seed": SEED,
        "source": SOURCE_ANNOTATIONS,
        "source_pool": "train2017 (all three splits drawn from one pool)",
        "generated": time.strftime("%Y-%m-%d"),
        "classes": CLASS_NAMES,
        "category_id_mapping": {str(k): v for k, v in COCO_CATEGORY_IDS.items()},
        "policies": {
            "overlap": OVERLAP_POLICY,
            "crowd": CROWD_POLICY,
            "unselected": UNSELECTED_POLICY,
            "eligibility": (
                f"at least one non-crowd selected-class annotation, "
                f"minimum {MIN_FOREGROUND_PIXELS} foreground pixel(s)"
            ),
        },
        "sizes": {name: len(ids) for name, ids in splits.items()},
        "statistics": statistics,
        "failures": all_failures,
        "image_ids": splits,
        "checksum": manifest_checksum(splits),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"\nSplit checksum: {manifest['checksum']}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
