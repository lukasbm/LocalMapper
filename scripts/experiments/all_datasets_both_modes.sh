#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS="${DATASETS:-USPTO_50K Golden NatComm schneider ringreactions metAMDB}"
MODES="${MODES:-scratch pretrained}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${RANDOM}}"
export RUN_ID

default_sample_limit() {
  case "$1" in
    ringreactions) echo "${RINGREACTIONS_SAMPLE_LIMIT:-10}" ;;
    *) echo "${SAMPLE_LIMIT:-200}" ;;
  esac
}

default_iterations() {
  case "$1" in
    ringreactions) echo "${RINGREACTIONS_ITERATIONS:-10}" ;;
    *) echo "${ITERATIONS:-5}" ;;
  esac
}

for DATASET_NAME in $DATASETS; do
  for MODE in $MODES; do
    export DATASET="$DATASET_NAME"
    export INIT_MODE="$MODE"
    export SAMPLE_LIMIT="$(default_sample_limit "$DATASET_NAME")"
    export ITERATIONS="$(default_iterations "$DATASET_NAME")"
    export MODEL="LocalMapper_${MODE}_${DATASET_NAME}_${RUN_ID}"

    echo "### Running DATASET=${DATASET} INIT_MODE=${INIT_MODE} MODEL=${MODEL} ###"
    "$SCRIPT_DIR/run_active_learning.sh"
  done
done
