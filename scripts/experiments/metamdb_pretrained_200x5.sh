#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-metAMDB}"
MODEL="${MODEL:-LocalMapper_pretrained}"
SEED="${SEED:-0}"
GPU="${GPU:-cuda:0}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-data/checkpoints/LocalMapper_202403.pth}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-200}"
ITERATIONS="${ITERATIONS:-5}"
SPLIT="${SPLIT:-train}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
PATIENCE="${PATIENCE:-5}"
CONFIDENT_PER_TEMPLATE="${CONFIDENT_PER_TEMPLATE:-100}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/localmapper-mpl}"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

for ITERATION in $(seq 1 "$ITERATIONS"); do
  echo "=== ${DATASET} pretrained iteration ${ITERATION}/${ITERATIONS} ==="
  "$PYTHON" -m scripts.Sample \
    --dataset="$DATASET" \
    --model="$MODEL" \
    --seed="$SEED" \
    --iteration="$ITERATION" \
    --split="$SPLIT" \
    --sample_limit="$SAMPLE_LIMIT" \
    --val_fraction="$VAL_FRACTION" \
    --test_fraction="$TEST_FRACTION"

  TRAIN_ARGS=(
    -m scripts.Train
    --dataset="$DATASET"
    --model="$MODEL"
    --seed="$SEED"
    --iteration="$ITERATION"
    --split="$SPLIT"
    --gpu="$GPU"
    --batch_size="$BATCH_SIZE"
    --num_epochs="$NUM_EPOCHS"
    --patience="$PATIENCE"
    --val_fraction="$VAL_FRACTION"
    --confident_per_template="$CONFIDENT_PER_TEMPLATE"
  )
  if [[ "$ITERATION" == "1" ]]; then
    TRAIN_ARGS+=(--init=checkpoint --checkpoint="$PRETRAINED_CHECKPOINT")
  else
    TRAIN_ARGS+=(--init=auto)
  fi
  "$PYTHON" "${TRAIN_ARGS[@]}"

  "$PYTHON" -m scripts.Test \
    --dataset="$DATASET" \
    --model="$MODEL" \
    --seed="$SEED" \
    --iteration="$ITERATION" \
    --split="$SPLIT" \
    --gpu="$GPU" \
    --batch_size="$BATCH_SIZE" \
    --val_fraction="$VAL_FRACTION" \
    --test_fraction="$TEST_FRACTION"
done
