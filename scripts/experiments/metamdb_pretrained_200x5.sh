#!/usr/bin/env bash
set -euo pipefail

export DATASET="${DATASET:-metAMDB}"
export INIT_MODE="${INIT_MODE:-finetune}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-200}"
export ITERATIONS="${ITERATIONS:-5}"

exec "$(dirname "$0")/run_active_learning.sh"
