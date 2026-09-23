"""Every constant the benchmark shares, and the reasoning behind each one.

Section 7 requires that all ten architectures see identical partitions and the
same held-out test images. That is only true if there is exactly one place
where the seed, the class mapping and the split sizes are written down, so
this module is that place and nothing else defines them.

The values here are the assignment's recommended protocol (Section 13). Where
a choice was left open - the overlap policy, the crowd policy, the byte
convention for model size - the choice is recorded here with its reason
rather than buried at the call site, because Section 8 requires the policy to
be documented and applied consistently across every model.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = REPO_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
IMAGES_DIR = DATA_DIR / "images"
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "plots"
PREDICTIONS_DIR = REPO_ROOT / "predictions"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
LOGS_DIR = REPO_ROOT / "logs"
CONFUSION_DIR = REPO_ROOT / "confusion_matrices"

# ---------------------------------------------------------------------------
# Section 7: dataset consistency
# ---------------------------------------------------------------------------

SEED = 42

# ---------------------------------------------------------------------------
# Section 8: semantic class mapping
# ---------------------------------------------------------------------------
#
# Six semantic labels: background plus the five recommended foreground
# classes. The contiguous IDs below are what the networks predict; COCO's own
# category IDs are sparse (person is 1, cat is 17, dog is 18) and are never
# used as network outputs.

CLASS_NAMES = ["background", "person", "car", "bicycle", "dog", "cat"]
NUM_CLASSES = len(CLASS_NAMES)

# COCO category ID -> our contiguous class ID.
COCO_CATEGORY_IDS = {
    1: 1,    # person
    3: 2,    # car
    2: 3,    # bicycle
    18: 4,   # dog
    17: 5,   # cat
}

# Pixels excluded from both training loss and evaluation. Section 8 requires
# that excluded pixels be ignored consistently, so one sentinel is used
# everywhere: in the saved masks on disk, in the loss, and in the metrics.
#
# 255 rather than -1 because masks are stored as 8-bit PNGs, and 255 is the
# only value in that range guaranteed not to collide with a class ID.
IGNORE_INDEX = 255

# Section 8 asks how overlapping annotations and crowd regions are treated.
#
# Overlaps: instances are painted largest-area-first, so a smaller object
# drawn later wins the contested pixels. The alternative - first annotation
# wins - would let a large sofa erase a cat sitting on it, and small objects
# are already the hardest case (Section 54 asks which architecture performed
# best on them). Losing them to draw order before training even starts would
# make that question unanswerable.
OVERLAP_POLICY = "smaller-instance-wins (paint in descending area order)"

# Crowd regions: COCO marks dense groups it did not annotate individually
# with iscrowd=1 and gives them one RLE blob covering the whole group. That
# blob is a real region of the correct class, so calling it background would
# teach every model that crowds are background. It is also not a usable
# instance, so it cannot be scored as one. Both tracks therefore mark crowd
# pixels IGNORE_INDEX and drop the annotation from the instance targets.
CROWD_POLICY = "iscrowd=1 regions marked IGNORE_INDEX; excluded from instance targets"

# Non-selected classes (the other 75 COCO categories) become background. They
# are genuinely "not one of the five", which is what background means here,
# and COCO's exhaustive annotation of its 80 categories does not extend to
# everything else in the frame anyway.
UNSELECTED_POLICY = "non-selected COCO categories are background"

# ---------------------------------------------------------------------------
# Section 5: split sizes
# ---------------------------------------------------------------------------

SPLIT_SIZES = {"train": 5000, "val": 1000, "test": 1000}

# An image qualifies for the subset only if it carries at least one
# non-crowd annotation in the five foreground classes. Images with nothing
# but background would train every model toward the majority class and tell
# us nothing about the five that matter.
MIN_FOREGROUND_PIXELS = 1

# ---------------------------------------------------------------------------
# Section 13: common experimental protocol
# ---------------------------------------------------------------------------

INPUT_SIZE = 512          # Section 13; 256 is the documented low-compute fallback
EPOCHS = 25               # Section 13 minimum
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 1e-4

# Section 16: Loss = 0.5 * CrossEntropy + 0.5 * Dice, for every semantic
# model. Instance models keep their own architecture-specific losses
# (Section 17) and never borrow this one.
CE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# Sections 31 and 33: measurement conventions
# ---------------------------------------------------------------------------

# Section 31 requires the byte convention to be stated. MB here means
# mebibytes, 1024 * 1024 bytes, matching what `ls -l` and Finder report for
# a checkpoint file.
BYTES_PER_MB = 1024 * 1024

# Section 33: warm-up iterations before any timed region, and the number of
# timed images. Batch-one latency and batched throughput are measured
# separately because the assignment requires them distinguished.
INFERENCE_WARMUP_ITERS = 10
INFERENCE_TIMED_IMAGES = 200
INFERENCE_BATCH_SIZES = (1, 8)

# Section 24: boundary evaluation tolerance, in pixels at whatever resolution
# the run evaluates at. A predicted boundary pixel counts as matched if a
# ground-truth boundary pixel lies within this distance.
#
# The absolute value is what every model is held to, so the ranking between
# them does not depend on it; the absolute scores do. Three pixels is 0.6% of
# image width at 512 and 1.2% at 256, so boundary F1 from a 256 run reads
# slightly more forgiving than the same models would score at 512. Both the
# tolerance and the resolution are recorded with every boundary result, which
# is what Section 24 asks for, and comparing boundary numbers across runs at
# different resolutions is not valid without accounting for it.
BOUNDARY_TOLERANCE_PX = 3

# Section 25: COCO-style instance evaluation settings.
INSTANCE_SCORE_THRESHOLD = 0.05
INSTANCE_MAX_DETECTIONS = 100
