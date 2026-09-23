#!/bin/bash
# Run every model at the resolution the run was configured for.
#
# The order is cheapest-first rather than the checklist order. Each model
# writes its own results/raw/<model>.json as it finishes, so a run that is
# interrupted leaves every completed model usable, and --skip-existing makes
# the whole thing resumable. Putting the 3.5-hour U-Net++ last means a
# failure in it costs nothing that came before.
set -u
cd "$(dirname "$0")/.."
SIZE="${1:-256}"
EPOCHS="${2:-25}"

for model in kmeans segformer fcn_resnet50 segnet deeplabv3 pspnet unet unetpp maskrcnn yolo_seg; do
  echo "=============================================================="
  echo "=== $model  ($(date '+%H:%M:%S'))"
  echo "=============================================================="
  .venv/bin/python run_benchmark.py \
    --task all --model "$model" \
    --input-size "$SIZE" --epochs "$EPOCHS" \
    --skip-existing --workers 4
done

echo "=== rebuilding tables and figures ==="
.venv/bin/python run_benchmark.py --tables-only
