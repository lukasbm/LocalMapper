#!/usr/bin/env bash
set -euo pipefail

export DATASET="${DATASET:-ringreactions}"
export INIT_MODE="${INIT_MODE:-finetune}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-10}"
export ITERATIONS="${ITERATIONS:-10}"

exec "$(dirname "$0")/run_active_learning.sh"
