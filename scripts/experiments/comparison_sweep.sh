#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INCLUDE_USPTO="${INCLUDE_USPTO:-0}"
DEFAULT_DATASETS="ringreactions metAMDB"
if [[ "$INCLUDE_USPTO" == "1" ]]; then
  DEFAULT_DATASETS="USPTO_50K ${DEFAULT_DATASETS}"
fi
DATASETS="${DATASETS:-$DEFAULT_DATASETS}"
MODES="${MODES:-scratch pretrained}"
SEEDS="${SEEDS:-0 1 2 3 4}"
RUN_ID="${RUN_ID:-comparison_$(date +%Y%m%d_%H%M%S)_${RANDOM}}"
COMPARISON_NUM_EPOCHS="${COMPARISON_NUM_EPOCHS:-100}"
PLAN_ONLY="${PLAN_ONLY:-0}"
SKIP_RUNS="${SKIP_RUNS:-0}"
RUN_PAPER_CHECKPOINT="${RUN_PAPER_CHECKPOINT:-1}"
PAPER_CHECKPOINT_EVAL_SPLIT="${PAPER_CHECKPOINT_EVAL_SPLIT:-all}"
ANALYZE_AFTER_RUN="${ANALYZE_AFTER_RUN:-1}"
ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-outputs/analyze_metrics_summaries.py}"
LOCALMAPPER_TMPDIR="${LOCALMAPPER_TMPDIR:-/scratch/lukas/tmp/localmapper}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$LOCALMAPPER_TMPDIR/mpl}"
DEFAULT_UV_CACHE_DIR="/scratch/lukas/tmp/uv-cache"
UV_CACHE_DIR="${UV_CACHE_DIR:-$DEFAULT_UV_CACHE_DIR}"
if [[ "$UV_CACHE_DIR" == /scratch/lukas/uv/cache* ]]; then
  echo "UV_CACHE_DIR=${UV_CACHE_DIR} is known to contain read-only entries; using ${DEFAULT_UV_CACHE_DIR}" >&2
  UV_CACHE_DIR="$DEFAULT_UV_CACHE_DIR"
fi
if [[ -e "$UV_CACHE_DIR" && ! -w "$UV_CACHE_DIR" ]]; then
  echo "UV_CACHE_DIR=${UV_CACHE_DIR} is not writable; using ${DEFAULT_UV_CACHE_DIR}" >&2
  UV_CACHE_DIR="$DEFAULT_UV_CACHE_DIR"
fi

# Format: label:sample_limit:iterations.
BUDGET_PROFILES="${BUDGET_PROFILES:-standard:200:5}"
RINGREACTIONS_BUDGET_PROFILES="${RINGREACTIONS_BUDGET_PROFILES:-standard:10:10}"

export RUN_ID
export LOCALMAPPER_TMPDIR
export MPLCONFIGDIR
export UV_CACHE_DIR
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$UV_CACHE_DIR"

budget_profiles_for_dataset() {
  case "$1" in
    ringreactions) echo "$RINGREACTIONS_BUDGET_PROFILES" ;;
    *) echo "$BUDGET_PROFILES" ;;
  esac
}

num_epochs_for_dataset() {
  case "$1" in
    metAMDB) echo "${METAMDB_NUM_EPOCHS:-$COMPARISON_NUM_EPOCHS}" ;;
    ringreactions) echo "${RINGREACTIONS_NUM_EPOCHS:-$COMPARISON_NUM_EPOCHS}" ;;
    USPTO_50K) echo "${USPTO50K_NUM_EPOCHS:-$COMPARISON_NUM_EPOCHS}" ;;
    Golden) echo "${GOLDEN_NUM_EPOCHS:-$COMPARISON_NUM_EPOCHS}" ;;
    *) echo "$COMPARISON_NUM_EPOCHS" ;;
  esac
}

RUN_COUNT=0
ITERATION_COUNT=0
TOTAL_RUNS=0
TOTAL_ACTIVE_LEARNING_ITERATIONS=0

word_count() {
  local count=0
  local _
  for _ in $1; do
    count=$((count + 1))
  done
  echo "$count"
}

compute_totals() {
  local dataset_name profile mode seed_value _budget_label _sample_limit iterations_value
  if [[ "$RUN_PAPER_CHECKPOINT" == "1" ]]; then
    TOTAL_RUNS=$((TOTAL_RUNS + $(word_count "$DATASETS") * $(word_count "$SEEDS")))
  fi
  for dataset_name in $DATASETS; do
    for profile in $(budget_profiles_for_dataset "$dataset_name"); do
      IFS=: read -r _budget_label _sample_limit iterations_value <<< "$profile"
      for mode in $MODES; do
        for seed_value in $SEEDS; do
          TOTAL_RUNS=$((TOTAL_RUNS + 1))
          TOTAL_ACTIVE_LEARNING_ITERATIONS=$((TOTAL_ACTIVE_LEARNING_ITERATIONS + iterations_value))
        done
      done
    done
  done
}

print_plan_header() {
  echo "### Comparison sweep plan ###"
  echo "RUN_ID=${RUN_ID}"
  echo "DATASETS=${DATASETS}"
  echo "INCLUDE_USPTO=${INCLUDE_USPTO}"
  echo "MODES=${MODES}"
  echo "SEEDS=${SEEDS}"
  echo "COMPARISON_NUM_EPOCHS=${COMPARISON_NUM_EPOCHS}"
  echo "RUN_PAPER_CHECKPOINT=${RUN_PAPER_CHECKPOINT}"
  echo "PAPER_CHECKPOINT_EVAL_SPLIT=${PAPER_CHECKPOINT_EVAL_SPLIT}"
  echo "ANALYZE_AFTER_RUN=${ANALYZE_AFTER_RUN}"
  echo "SKIP_RUNS=${SKIP_RUNS}"
  echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
  echo "PLAN_ONLY=${PLAN_ONLY}"
}

print_run_line() {
  echo "RUN dataset=$1 mode=$2 budget=$3 sample_limit=$4 iterations=$5 seed=$6 num_epochs=$7 model=$8"
}

print_totals() {
  echo "TOTAL_RUNS=${RUN_COUNT}"
  echo "TOTAL_ACTIVE_LEARNING_ITERATIONS=${ITERATION_COUNT}"
}

print_progress_banner() {
  local completed="$1"
  local total="$2"
  local dataset="$3"
  local mode="$4"
  local seed="$5"
  local model="$6"
  local percent=0
  if [[ "$total" -gt 0 ]]; then
    percent=$((completed * 100 / total))
  fi
  printf '\033[1;37;44m%*s\033[0m\n' 96 ''
  printf '\033[1;37;44m  COMPLETED RUN %3d/%-3d  %3d%%  dataset=%s mode=%s seed=%s\033[0m\n' \
    "$completed" "$total" "$percent" "$dataset" "$mode" "$seed"
  printf '\033[1;37;44m  model=%s\033[0m\n' "$model"
  printf '\033[1;37;44m%*s\033[0m\n' 96 ''
}

should_run_current() {
  if [[ "$RUN_COUNT" -le "$SKIP_RUNS" ]]; then
    echo "SKIP completed/assumed run ${RUN_COUNT}/${TOTAL_RUNS}; resume starts after SKIP_RUNS=${SKIP_RUNS}"
    return 1
  fi
  return 0
}

run_analysis() {
  if [[ "$PLAN_ONLY" == "1" || "$ANALYZE_AFTER_RUN" != "1" ]]; then
    return
  fi
  read -r -a PYTHON_CMD <<< "${PYTHON:-uv run python}"
  echo "Running metrics summary analysis for RUN_ID=${RUN_ID}"
  "${PYTHON_CMD[@]}" "$ROOT_DIR/$ANALYSIS_SCRIPT" --run-id "$RUN_ID"
}

compute_totals
print_plan_header

if [[ "$RUN_PAPER_CHECKPOINT" == "1" ]]; then
  for DATASET_NAME in $DATASETS; do
    for SEED_VALUE in $SEEDS; do
      export DATASET="$DATASET_NAME"
      export SEED="$SEED_VALUE"
      export EVAL_SPLIT="$PAPER_CHECKPOINT_EVAL_SPLIT"
      export MODEL="LocalMapper_paper_checkpoint_${DATASET_NAME}_${RUN_ID}"
      RUN_COUNT=$((RUN_COUNT + 1))
      print_run_line \
        "$DATASET" \
        "paper_checkpoint" \
        "none" \
        "0" \
        "0" \
        "$SEED" \
        "0" \
        "$MODEL"
      if [[ "$PLAN_ONLY" != "1" ]] && should_run_current; then
        "$SCRIPT_DIR/run_pretrained_eval.sh"
        print_progress_banner "$RUN_COUNT" "$TOTAL_RUNS" "$DATASET" "paper_checkpoint" "$SEED" "$MODEL"
      fi
    done
  done
fi

for DATASET_NAME in $DATASETS; do
  for PROFILE in $(budget_profiles_for_dataset "$DATASET_NAME"); do
    IFS=: read -r BUDGET_LABEL_VALUE SAMPLE_LIMIT_VALUE ITERATIONS_VALUE <<< "$PROFILE"
    if [[ -z "${BUDGET_LABEL_VALUE:-}" || -z "${SAMPLE_LIMIT_VALUE:-}" || -z "${ITERATIONS_VALUE:-}" ]]; then
      echo "Invalid budget profile '${PROFILE}'. Expected label:sample_limit:iterations." >&2
      exit 2
    fi
    for MODE in $MODES; do
      for SEED_VALUE in $SEEDS; do
        export DATASET="$DATASET_NAME"
        export INIT_MODE="$MODE"
        export SEED="$SEED_VALUE"
        export SAMPLE_LIMIT="$SAMPLE_LIMIT_VALUE"
        export ITERATIONS="$ITERATIONS_VALUE"
        export BUDGET_LABEL="$BUDGET_LABEL_VALUE"
        export NUM_EPOCHS="$(num_epochs_for_dataset "$DATASET_NAME")"
        export MODEL="LocalMapper_${MODE}_${DATASET_NAME}_${BUDGET_LABEL_VALUE}_${RUN_ID}"
        RUN_COUNT=$((RUN_COUNT + 1))
        ITERATION_COUNT=$((ITERATION_COUNT + ITERATIONS_VALUE))
        print_run_line \
          "$DATASET" \
          "$INIT_MODE" \
          "$BUDGET_LABEL" \
          "$SAMPLE_LIMIT" \
          "$ITERATIONS" \
          "$SEED" \
          "$NUM_EPOCHS" \
          "$MODEL"
        if [[ "$PLAN_ONLY" != "1" ]] && should_run_current; then
          "$SCRIPT_DIR/run_active_learning.sh"
          print_progress_banner "$RUN_COUNT" "$TOTAL_RUNS" "$DATASET" "$INIT_MODE" "$SEED" "$MODEL"
        fi
      done
    done
  done
done

print_totals
run_analysis
