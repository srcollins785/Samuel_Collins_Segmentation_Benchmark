"""Run the segmentation benchmark.

Section 51 names these commands, and this file implements them:

    python run_benchmark.py --task semantic --model unet
    python run_benchmark.py --task semantic --model deeplabv3
    python run_benchmark.py --task semantic --model all
    python run_benchmark.py --task instance --model maskrcnn
    python run_benchmark.py --task instance --model yolo_seg
    python run_benchmark.py --task all --model all

Plus the one the assignment implies but does not name. Section 51 asks that
the README explain how evaluation can be repeated without retraining every
model, and that is only true if a command exists for it:

    python run_benchmark.py --tables-only

which rebuilds every CSV and figure from the saved per-model results in
seconds. The separation is the point: training writes results/raw/*.json,
and everything a reader sees is derived from those files. A model that has
already run never runs again to redraw a chart.

Lives at the repository root because the assignment names this path. The
work is in the package; this file is argument parsing, a preflight, and
progress reporting.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from segmentation_benchmark import benchmark, plots, tables  # noqa: E402
from segmentation_benchmark._config import (  # noqa: E402
    BATCH_SIZE, EPOCHS, INPUT_SIZE, PLOTS_DIR, RESULTS_DIR,
)
from segmentation_benchmark.dataset import load_manifest  # noqa: E402
from segmentation_benchmark.models import (  # noqa: E402
    INSTANCE_MODELS, SEMANTIC_MODELS, TRADITIONAL_MODELS,
)
from segmentation_benchmark.train import resolve_device  # noqa: E402

TASKS = {
    "traditional": TRADITIONAL_MODELS,
    "semantic": SEMANTIC_MODELS,
    "instance": INSTANCE_MODELS,
    "all": TRADITIONAL_MODELS + SEMANTIC_MODELS + INSTANCE_MODELS,
}


def preflight(input_size: int) -> dict:
    """Refuse to start a benchmark that cannot produce a valid comparison.

    Section 7 requires every architecture to see the same partitions. This
    checks the split exists and reports its checksum, which each model then
    records. A model whose recorded checksum differs from another's was not
    evaluated on the same test set, and the comparison between them is not
    one - so the checksum is printed here and stored there rather than
    assumed to match.
    """
    from segmentation_benchmark._config import IMAGES_DIR

    manifest = load_manifest()

    missing = []
    for split, count in manifest["sizes"].items():
        directory = IMAGES_DIR / split
        if not directory.is_dir():
            missing.append(f"{split}: directory absent")
            continue
        present = sum(1 for _ in directory.glob("*.jpg"))
        if present < count:
            missing.append(f"{split}: {present} images on disk, {count} expected")

    if missing:
        raise SystemExit(
            "The dataset is incomplete:\n  "
            + "\n  ".join(missing)
            + "\n\nRebuild it:\n  python scripts/build_coco_subset.py"
        )

    device = resolve_device()
    print(f"Split checksum : {manifest['checksum']}")
    print(f"Split sizes    : {manifest['sizes']}")
    print(f"Device         : {device}")
    print(f"Input size     : {input_size}x{input_size}")
    return {"manifest": manifest, "device": device}


def run_one(model_name: str, args, device) -> dict:
    """Dispatch one model to the runner that suits its track."""
    if model_name in TRADITIONAL_MODELS:
        return benchmark.run_traditional(
            input_size=args.input_size, verbose=not args.quiet
        )
    if model_name == "yolo_seg":
        return benchmark.run_yolo(
            epochs=args.epochs, input_size=args.input_size,
            batch_size=args.batch_size, device=device, verbose=not args.quiet,
        )
    if model_name in INSTANCE_MODELS:
        return benchmark.run_instance(
            model_name, epochs=args.epochs, input_size=args.input_size,
            batch_size=args.batch_size, pretrained=not args.scratch,
            device=device, workers=args.workers, verbose=not args.quiet,
        )
    return benchmark.run_semantic(
        model_name, epochs=args.epochs, input_size=args.input_size,
        batch_size=args.batch_size, pretrained=not args.scratch,
        device=device, workers=args.workers, verbose=not args.quiet,
    )


def build_outputs() -> None:
    """Regenerate every CSV and figure from the saved results."""
    from segmentation_benchmark.tables import load_results

    if not load_results():
        print(
            "\nNo saved results in results/raw/, so there is nothing to "
            "tabulate.\nRun at least one model first, for example:\n"
            "  python run_benchmark.py --task semantic --model unet"
        )
        return

    print("\nBuilding tables")
    for filename, info in tables.write_all().items():
        print(f"  {filename:42s} {info['rows']:>4d} rows")

    print("Building figures")
    for name, path in plots.generate_all().items():
        print(f"  {name:42s} {Path(path).name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", default="all", choices=sorted(TASKS),
        help="Which track to run (default: all).",
    )
    parser.add_argument(
        "--model", default="all",
        help="A model name, or 'all' for every model in the task.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--scratch", action="store_true",
        help="Train without pretrained weights (Section 54's initialization question).",
    )
    parser.add_argument(
        "--tables-only", action="store_true",
        help="Rebuild CSVs and figures from saved results without training.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip models that already have a saved result.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.tables_only:
        build_outputs()
        print(f"\nTables: {RESULTS_DIR}\nFigures: {PLOTS_DIR}")
        return 0

    context = preflight(args.input_size)
    device = context["device"]

    if args.model == "all":
        models = TASKS[args.task]
    else:
        models = [args.model]
        known = TASKS["all"]
        if args.model not in known:
            raise SystemExit(
                f"Unknown model {args.model!r}. Available: {', '.join(known)}"
            )

    if args.skip_existing:
        remaining = []
        for name in models:
            if (benchmark.RAW_DIR / f"{name}.json").is_file():
                print(f"  skipping {name}: already has a saved result")
            else:
                remaining.append(name)
        models = remaining

    print(f"\nRunning {len(models)} model(s): {', '.join(models) or '(none)'}")
    print(f"Protocol: {args.epochs} epochs, batch {args.batch_size}, "
          f"{args.input_size}px\n")

    started = time.perf_counter()
    completed, failed = [], {}

    for position, name in enumerate(models, start=1):
        print(f"[{position}/{len(models)}] {name}")
        model_started = time.perf_counter()
        try:
            run_one(name, args, device)
            elapsed = time.perf_counter() - model_started
            completed.append(name)
            print(f"  done in {elapsed / 60:.1f} min\n")
        except KeyboardInterrupt:
            print("\nInterrupted. Results for completed models are saved.")
            break
        except Exception as error:
            # One architecture failing should not discard the other nine.
            # The failure is recorded and reported at the end rather than
            # ending the run, and the summary says which models are missing
            # so no table is read as complete when it is not.
            failed[name] = f"{type(error).__name__}: {error}"
            print(f"  FAILED {failed[name]}\n")
            import traceback
            traceback.print_exc()

    build_outputs()

    total = time.perf_counter() - started
    print(f"\n{'=' * 64}")
    print(f"Completed {len(completed)}/{len(models)} in {total / 60:.1f} min")
    if failed:
        print(f"\n{len(failed)} model(s) failed and are absent from the tables:")
        for name, reason in failed.items():
            print(f"  {name}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
