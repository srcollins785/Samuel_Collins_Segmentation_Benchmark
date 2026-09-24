# Samuel_Collins_Segmentation_Benchmark

Benchmark ten segmentation methods on one shared COCO 2017 subset, under one
protocol, with every table and figure generated from saved evaluation outputs.

| Track | Methods |
|---|---|
| Traditional | K-Means color segmentation |
| Semantic | FCN-ResNet50, U-Net, U-Net++, SegNet, DeepLabV3-ResNet50, PSPNet, SegFormer-B0 |
| Instance | Mask R-CNN (ResNet50-FPN), YOLO11n-seg |

Six classes: background, person, car, bicycle, dog, cat. One split at seed
42, shared by every model, with a recorded checksum so a comparison across
models can be shown to be a comparison rather than assumed to be one.

---

## Installation

The virtual environment matters on macOS, where the system `python3` and a
Homebrew `python3` are different interpreters with different packages.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the exact versions the reported numbers were produced
with. `pyproject.toml` declares looser ranges, because a library should not
force an exact dependency set on its users.

One environment runs all ten models. That is worth a note: on macOS,
ultralytics excluded all of numpy 2.x before version 8.4, which forces a
second virtual environment and means the YOLO results are produced under
different library versions from everything else. ultralytics 8.4 permits
numpy ≥ 2.3.5, so the split is unnecessary here. **Downgrading ultralytics
below 8.4 reintroduces it.**

## Building the dataset

```bash
python scripts/build_coco_subset.py
```

Downloads the COCO 2017 instance annotations (~241 MB), selects 5,000 / 1,000
/ 1,000 images at seed 42 from the 73,519 `train2017` images that carry a
non-crowd annotation in the five foreground classes, fetches **only those
images** (~1.2 GB, against 19 GB for the full archive), renders the semantic
masks once to disk, and writes `data/split_manifest.json`.

All three splits are drawn from `train2017` rather than taking test from
`val2017`, so the three partitions are identically distributed by
construction and a validation-to-test gap measures generalization rather than
a difference between two annotation passes.

`--limit N` builds a proportionally smaller subset for a dry run.

## Verifying the augmentation

```bash
python scripts/verify_transforms.py
```

Section 15 asks for transformed image–mask pairs to be visualized before
benchmarking. This does that **and** checks three properties numerically
first, because a figure only catches a desync large enough to see and a
three-pixel drift would look fine at figure resolution while making the
boundary results meaningless:

1. A forced flip mirrors the image, the semantic mask, every instance mask
   and every box exactly.
2. A synthetic image that *is* a rendering of its own mask keeps each class's
   color through random scale, crop and rotation.
3. Instance boxes still bound their own masks afterwards, and the five
   per-instance arrays stay the same length.

Writes `plots/transform_verification.png`. Exits non-zero if any check fails.

## Running the benchmark

```bash
python run_benchmark.py --task semantic --model unet
python run_benchmark.py --task semantic --model deeplabv3
python run_benchmark.py --task semantic --model all
python run_benchmark.py --task instance --model maskrcnn
python run_benchmark.py --task instance --model yolo_seg
python run_benchmark.py --task all      --model all
```

Useful flags:

| Flag | Effect |
|---|---|
| `--input-size 256` | Section 13's low-compute resolution (what the reported run used) |
| `--epochs N` | Epoch budget, default 25 |
| `--skip-existing` | Skip models that already have a saved result, resumes an interrupted run |
| `--scratch` | Train without pretrained weights |
| `--tables-only` | Rebuild every CSV and figure from saved results, without training |

Or run everything cheapest-first, which is what produced the reported
results:

```bash
./scripts/run_full_benchmark.sh 256 25
```

### What each command does, and where output goes

Each model writes exactly one file, `results/raw/<model>.json`, holding its
configuration, training history, test metrics and efficiency measurements.
Everything a reader sees is derived from those files:

```
results/
    raw/<model>.json                        one per model, the source of truth
    semantic_segmentation_results.csv       Section 27
    instance_segmentation_results.csv       Section 28
    segmentation_efficiency_results.csv     Section 29
    per_class_iou.csv                       Section 23
    per_class_boundary_f1.csv               Section 24
    training_history.csv                    per-epoch curves
plots/
    miou_comparison.png       dice_comparison.png
    parameters.png            model_size.png
    training_time.png         inference_speed.png
    miou_vs_parameters.png    miou_vs_latency.png
    per_class_iou.png         instance_mask_ap.png
    boundary_f1.png           transform_verification.png
confusion_matrices/<model>.json
checkpoints/best_<model>.pt
logs/<model>_history.json
report/Segmentation_Benchmarking_Report.md
```

### Repeating the evaluation without retraining

This is the point of the split between `results/raw/` and everything else:

```bash
python run_benchmark.py --tables-only    # every CSV and figure, in seconds
python scripts/generate_report.py        # the full report
```

Neither touches a model. A model that has already run never runs again to
redraw a chart or reword a paragraph.

## Reading the results

Three things about this dataset shape every number, and are worth knowing
before reading a table.

**Background is 83% of labeled pixels.** A model that predicts background
everywhere scores about 0.83 pixel accuracy. The K-Means baseline scores
almost exactly that, with a foreground mIoU near zero, which is why the
tables lead with mIoU and per-class IoU rather than pixel accuracy.

**Bicycle is 0.26% of pixels**, against person's 13.9%, a factor of more
than 50. Six-class mIoU is steered by background and person, so the per-class
table is reported before the mean rather than after it.

**Initialization is not uniform across models.** FCN and DeepLabV3 load COCO
*segmentation* weights including a trained head; PSPNet and SegNet load
ImageNet backbones; SegFormer loads a pretrained encoder with a **randomly
initialized decode head**; U-Net and U-Net++ are entirely from scratch. A
ranking on mIoU is therefore partly a ranking on how much each model was
given to start with, which the report states rather than glosses.

## Documented deviations from the protocol

Section 13 asks for deviations to be recorded rather than smoothed over.

- **Input resolution 256×256, not 512×512.** Section 13 permits this where
  compute is limited, applied consistently, and it is applied consistently.
  Measured on this hardware: the seven semantic models at 25 epochs take
  roughly 9 hours at 256 and 35 hours at 512, and the full ten-model
  benchmark at 512 would exceed two days of continuous compute.
- **Mask R-CNN completed 14 of 25 epochs.** The host ran out of memory:
  Metal's allocator retains a driver-side cache its own accounting does not
  report, and across rounds of COCOeval validation it grew until 20.6 GB was
  wired and the machine was swapping. Epoch time went from 740 s to 4,374 s
  and was still climbing. Validation mask AP had peaked at epoch 8 and had
  not improved in the six epochs since, so the run was stopped and the model
  evaluated from the epoch-8 checkpoint that the Section 21 rule had already
  selected. `evaluate_instance` now releases the cache periodically, so a
  rerun should reach 25. Recorded in `results/raw/maskrcnn.json` under
  `config.early_stop_reason`.
- **SegNet was evaluated from its checkpoint, not in one pass.** Its training
  completed normally for all 25 epochs; the run then died in profiling,
  because `thop` registers float64 buffers on every submodule and MPS cannot
  hold float64. Profiling now runs on a copy. The model was recovered with
  `scripts/recover_model.py` rather than retrained, since retraining would
  have produced different weights from the ones the run selected.
- **PSPNet's pyramid pooling pads its feature map.** Metal has no adaptive
  average pooling kernel for non-divisible sizes
  ([pytorch#96056](https://github.com/pytorch/pytorch/issues/96056)), and the
  3×3 and 6×6 branches do not divide a 32×32 feature map. The map is
  replicate-padded to a divisible size, which keeps the published bins and
  the pooling semantics. On CUDA this is a no-op in effect.
- **PSPNet has an auxiliary loss** on stage 3, weighted 0.4. It is part of
  the published design; no other semantic model has a second loss term.
- **FCN and DeepLabV3 have their auxiliary heads dropped.** torchvision's
  pretrained checkpoints size them for 21 VOC classes, and keeping them would
  give two of seven models a loss term the others lack.
- **YOLO trains under ultralytics' own recipe**, its optimizer, schedule and
  augmentation, not the Section 14 pipeline. Section 17 is explicit that
  instance models keep their architecture-specific configuration. Its
  predictions are scored by the same COCOeval wrapper as Mask R-CNN, so the
  evaluation is shared even though the training is not.
- **Instance latency is not measured over the same region as semantic
  latency.** The torchvision models are timed on the forward pass alone;
  ultralytics' `predict` includes its own preprocessing and NMS.
- **Peak GPU memory is an allocator total, not a driver peak.** There is no
  CUDA on this hardware, so `torch.mps.current_allocated_memory` stands in
  for `torch.cuda.max_memory_allocated`. Timing is unaffected, every timed
  region is bracketed by `torch.mps.synchronize`.
- **Reruns are not bit-identical.** Seeding covers initialization, shuffling
  and augmentation, but Metal kernels are not deterministic and torch has no
  deterministic mode for them.

## Checkpoints

Trained weights are not committed, ten checkpoints exceed what a repository
should carry. They are reproducible from the committed manifest at the same
seed by rerunning the benchmark. `data/split_manifest.json` **is** committed,
so the exact 7,000 images can be rebuilt without guessing.

## Layout

```
src/segmentation_benchmark/
    _config.py          seed, class mapping, split sizes, every convention
    mask_utils.py       COCO annotations -> semantic masks and instance targets
    dataset.py          the two datasets, both from one split manifest
    transforms.py       mask-safe augmentation
    losses.py           0.5 * CrossEntropy + 0.5 * Dice
    architectures.py    U-Net, U-Net++, SegNet, PSPNet, written out
    models.py           the Section 19 factory
    traditional.py      K-Means with a training-only class mapping
    train.py            semantic and instance training adapters
    metrics.py          Section 22 semantic metrics
    boundary.py         Section 24 boundary F1
    instance_metrics.py Section 25 COCO mask AP
    evaluate.py         one entry point over the three above
    efficiency.py       Sections 30-33 measurements
    benchmark.py        one model, end to end, to one JSON file
    tables.py           the required CSVs
    plots.py            the required figures
    yolo_adapter.py     ultralytics, driven rather than wrapped
scripts/
    build_coco_subset.py    the dataset
    verify_transforms.py    the Section 15 check
    run_full_benchmark.sh   every model, cheapest first
    write_configuration.py  regenerates configuration.yaml
    generate_report.py      the report
    build_report_pdf.py     renders it to PDF
    recover_model.py        finish a model whose run died after training
    backfill_predictions.py prediction figures from saved checkpoints
    report_data.py          queries the report asks of the results
    report_analysis.py      results sections
    report_discussion.py    Sections 53-58
tests/                      163 tests
run_benchmark.py            the Section 51 command line
configuration.yaml          the protocol, generated from _config.py
```

## Tests

```bash
python -m pytest tests/ -q
```

163 tests, no network access and no trained model required, they run on a
clean checkout before the dataset has been downloaded.

## License

MIT.
