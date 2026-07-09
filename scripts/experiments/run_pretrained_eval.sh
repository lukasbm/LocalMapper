#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

read -r -a PYTHON_CMD <<< "${PYTHON:-uv run python}"

DATASET="${DATASET:-ringreactions}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${RANDOM}}"
MODEL="${MODEL:-LocalMapper_pretrained_eval_${RUN_ID}}"
SEED="${SEED:-0}"
GPU="${GPU:-cuda:0}"
EVAL_SPLIT="${EVAL_SPLIT:-all}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-data/checkpoints/LocalMapper_202403.pth}"
TEMPLATE_LIBRARY="${TEMPLATE_LIBRARY:-data/checkpoints/templates_202403.pkl}"
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

echo "Using DATASET=${DATASET} MODEL=${MODEL}"
echo "Using pretrained checkpoint ${PRETRAINED_CHECKPOINT}"
echo "Using EVAL_SPLIT=${EVAL_SPLIT}"

"${PYTHON_CMD[@]}" -m scripts.RunMetadata \
  --dataset="$DATASET" \
  --model="$MODEL" \
  --init_mode="paper_checkpoint" \
  --seed="$SEED" \
  --train_split="none" \
  --eval_split="$EVAL_SPLIT" \
  --iterations=1 \
  --batch_size="$BATCH_SIZE" \
  --gpu="$GPU" \
  --pretrained_checkpoint="$PRETRAINED_CHECKPOINT" \
  --template_library="$TEMPLATE_LIBRARY" \
  --eequaam_chunk_size="$EEQUAAM_CHUNK_SIZE" \
  --eequaam_base_timeout_seconds="$EEQUAAM_BASE_TIMEOUT_SECONDS" \
  --eequaam_timeout_seconds_per_reaction="$EEQUAAM_TIMEOUT_SECONDS_PER_REACTION" \
  --run_id="$RUN_ID"

"${PYTHON_CMD[@]}" -m scripts.Test \
  --dataset="$DATASET" \
  --model="$MODEL" \
  --seed="$SEED" \
  --iteration=0 \
  --split="$EVAL_SPLIT" \
  --gpu="$GPU" \
  --batch_size="$BATCH_SIZE" \
  --checkpoint="$PRETRAINED_CHECKPOINT" \
  --template_library="$TEMPLATE_LIBRARY" \
  --eequaam_chunk_size="$EEQUAAM_CHUNK_SIZE" \
  --eequaam_base_timeout_seconds="$EEQUAAM_BASE_TIMEOUT_SECONDS" \
  --eequaam_timeout_seconds_per_reaction="$EEQUAAM_TIMEOUT_SECONDS_PER_REACTION"
