#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

read -r -a PYTHON_CMD <<< "${PYTHON:-uv run python}"

DATASET="${DATASET:-metAMDB}"
INIT_MODE="${INIT_MODE:-scratch}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${RANDOM}}"
MODEL="${MODEL:-LocalMapper_${INIT_MODE}_${RUN_ID}}"
SEED="${SEED:-0}"
GPU="${GPU:-cuda:0}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-data/checkpoints/LocalMapper_202403.pth}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-200}"
SAMPLE_CANDIDATE_FACTOR="${SAMPLE_CANDIDATE_FACTOR:-20}"
ITERATIONS="${ITERATIONS:-5}"
SPLIT="${SPLIT:-train}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
PATIENCE="${PATIENCE:-5}"
CONFIDENT_PER_TEMPLATE="${CONFIDENT_PER_TEMPLATE:-100}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
BUDGET_LABEL="${BUDGET_LABEL:-}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/localmapper-mpl}"
LOCALMAPPER_CGRTOOLS_IGNORE="${LOCALMAPPER_CGRTOOLS_IGNORE:-1}"
LOCALMAPPER_CGR_MP_CONTEXT="${LOCALMAPPER_CGR_MP_CONTEXT:-fork}"
LOCALMAPPER_CGR_TIMEOUT_SECONDS="${LOCALMAPPER_CGR_TIMEOUT_SECONDS:-2}"
export MPLCONFIGDIR
export LOCALMAPPER_CGRTOOLS_IGNORE
export LOCALMAPPER_CGR_MP_CONTEXT
export LOCALMAPPER_CGR_TIMEOUT_SECONDS
mkdir -p "$MPLCONFIGDIR"

echo "Using DATASET=${DATASET} INIT_MODE=${INIT_MODE} MODEL=${MODEL}"
echo "Using CGRTools equivalence"
echo "Using LOCALMAPPER_CGRTOOLS_IGNORE=${LOCALMAPPER_CGRTOOLS_IGNORE}"
echo "Using LOCALMAPPER_CGR_MP_CONTEXT=${LOCALMAPPER_CGR_MP_CONTEXT}"
echo "Using LOCALMAPPER_CGR_TIMEOUT_SECONDS=${LOCALMAPPER_CGR_TIMEOUT_SECONDS}"

"${PYTHON_CMD[@]}" -m scripts.RunMetadata \
  --dataset="$DATASET" \
  --model="$MODEL" \
  --init_mode="$INIT_MODE" \
  --seed="$SEED" \
  --split="$SPLIT" \
  --sample_limit="$SAMPLE_LIMIT" \
  --sample_candidate_factor="$SAMPLE_CANDIDATE_FACTOR" \
  --iterations="$ITERATIONS" \
  --batch_size="$BATCH_SIZE" \
  --num_epochs="$NUM_EPOCHS" \
  --patience="$PATIENCE" \
  --confident_per_template="$CONFIDENT_PER_TEMPLATE" \
  --val_fraction="$VAL_FRACTION" \
  --test_fraction="$TEST_FRACTION" \
  --gpu="$GPU" \
  --pretrained_checkpoint="$PRETRAINED_CHECKPOINT" \
  --run_id="$RUN_ID" \
  --budget_label="$BUDGET_LABEL"

for ITERATION in $(seq 1 "$ITERATIONS"); do
  echo "=== ${DATASET} ${INIT_MODE} iteration ${ITERATION}/${ITERATIONS} ==="

  echo "--- Sampling annotations ---"
  "${PYTHON_CMD[@]}" -m scripts.Sample \
    --dataset="$DATASET" \
    --model="$MODEL" \
    --seed="$SEED" \
    --iteration="$ITERATION" \
    --split="$SPLIT" \
    --sample_limit="$SAMPLE_LIMIT" \
    --sample_candidate_factor="$SAMPLE_CANDIDATE_FACTOR" \
    --val_fraction="$VAL_FRACTION" \
    --test_fraction="$TEST_FRACTION"

  echo "--- Training mapper ---"
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
  if [[ "$INIT_MODE" == "pretrained" && "$ITERATION" == "1" ]]; then
    TRAIN_ARGS+=(--init=checkpoint --checkpoint="$PRETRAINED_CHECKPOINT")
  else
    TRAIN_ARGS+=(--init=auto)
  fi
  "${PYTHON_CMD[@]}" "${TRAIN_ARGS[@]}"

  echo "--- Testing mapper ---"
  "${PYTHON_CMD[@]}" -m scripts.Test \
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
