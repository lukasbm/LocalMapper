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
TEMPLATE_LIBRARY="${TEMPLATE_LIBRARY:-}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-200}"
SAMPLE_CANDIDATE_FACTOR="${SAMPLE_CANDIDATE_FACTOR:-20}"
ITERATIONS="${ITERATIONS:-5}"
TRAIN_SPLIT="${TRAIN_SPLIT:-${SPLIT:-train}}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
PATIENCE="${PATIENCE:-5}"
CONFIDENT_PER_TEMPLATE="${CONFIDENT_PER_TEMPLATE:-100}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
BUDGET_LABEL="${BUDGET_LABEL:-}"
EEQUAAM_CHUNK_SIZE="${EEQUAAM_CHUNK_SIZE:-25}"
EEQUAAM_BASE_TIMEOUT_SECONDS="${EEQUAAM_BASE_TIMEOUT_SECONDS:-10}"
EEQUAAM_TIMEOUT_SECONDS_PER_REACTION="${EEQUAAM_TIMEOUT_SECONDS_PER_REACTION:-2}"
LOCALMAPPER_TMPDIR="${LOCALMAPPER_TMPDIR:-/scratch/lukas/tmp/localmapper}"
LOCALMAPPER_EEQUAAM_TMPDIR="${LOCALMAPPER_EEQUAAM_TMPDIR:-$LOCALMAPPER_TMPDIR/eequaam}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$LOCALMAPPER_TMPDIR/mpl}"
LOCALMAPPER_AAM_BACKEND="${LOCALMAPPER_AAM_BACKEND:-eequaam_its}"
export MPLCONFIGDIR
export LOCALMAPPER_TMPDIR
export LOCALMAPPER_AAM_BACKEND
export LOCALMAPPER_EEQUAAM_TMPDIR
mkdir -p "$MPLCONFIGDIR" "$LOCALMAPPER_EEQUAAM_TMPDIR"

echo "Using DATASET=${DATASET} INIT_MODE=${INIT_MODE} MODEL=${MODEL}"
echo "Using TRAIN_SPLIT=${TRAIN_SPLIT} EVAL_SPLIT=${EVAL_SPLIT}"
echo "Using AAM equivalence backend ${LOCALMAPPER_AAM_BACKEND}"

"${PYTHON_CMD[@]}" -m scripts.RunMetadata \
  --dataset="$DATASET" \
  --model="$MODEL" \
  --init_mode="$INIT_MODE" \
  --seed="$SEED" \
  --train_split="$TRAIN_SPLIT" \
  --eval_split="$EVAL_SPLIT" \
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
  --template_library="$TEMPLATE_LIBRARY" \
  --eequaam_chunk_size="$EEQUAAM_CHUNK_SIZE" \
  --eequaam_base_timeout_seconds="$EEQUAAM_BASE_TIMEOUT_SECONDS" \
  --eequaam_timeout_seconds_per_reaction="$EEQUAAM_TIMEOUT_SECONDS_PER_REACTION" \
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
    --split="$TRAIN_SPLIT" \
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
    --split="$TRAIN_SPLIT"
    --gpu="$GPU"
    --batch_size="$BATCH_SIZE"
    --num_epochs="$NUM_EPOCHS"
    --patience="$PATIENCE"
    --val_fraction="$VAL_FRACTION"
    --confident_per_template="$CONFIDENT_PER_TEMPLATE"
  )
  if [[ ( "$INIT_MODE" == "finetune" || "$INIT_MODE" == "pretrained" ) && "$ITERATION" == "1" ]]; then
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
    --split="$EVAL_SPLIT" \
    --gpu="$GPU" \
    --batch_size="$BATCH_SIZE" \
    --val_fraction="$VAL_FRACTION" \
    --test_fraction="$TEST_FRACTION" \
    --template_library="$TEMPLATE_LIBRARY" \
    --eequaam_chunk_size="$EEQUAAM_CHUNK_SIZE" \
    --eequaam_base_timeout_seconds="$EEQUAAM_BASE_TIMEOUT_SECONDS" \
    --eequaam_timeout_seconds_per_reaction="$EEQUAAM_TIMEOUT_SECONDS_PER_REACTION"
done
