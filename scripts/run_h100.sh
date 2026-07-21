#!/usr/bin/env bash
# End-to-end RYS run for a single H100 box (Lambda).
# Run inside tmux:  bash scripts/run_h100.sh 2>&1 | tee run.log
#
# Stages (each is resumable; re-running the script skips finished work):
#   0. sanity: model loads, scanner forward path works  (~30 min, mostly download)
#   1. layer anatomy (Sapir-Whorf ViT analog)           (~5 min)
#   2. smoke scan on one cheap dataset                  (~1 h)
#   3. full sweep: 10 datasets x 1176 configs + repeats (compute bulk)
#   4. stage-2 confirmation of top configs on fresh images
#   5. heatmaps + report

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-$HOME/rys_data}"
RESULTS="${RESULTS:-results}"
MODEL="${MODEL:-eva18b}"
OUTPUTS="${OUTPUTS:-outputs}"
PY="${PY:-python3}"
# DATASETS: optional space-separated list for the sweep, e.g. "all" to
# include gated ImageNet (needs `huggingface-cli login` + accepted terms).
DATASET_ARGS=()
if [ -n "${DATASETS:-}" ]; then
    read -r -a _ds <<< "$DATASETS"
    DATASET_ARGS=(--datasets "${_ds[@]}")
fi

mkdir -p "$DATA_DIR" "$RESULTS"

echo "=== [0/5] Model loading sanity check ==="
"$PY" scripts/test_model_loading.py --model "$MODEL"

echo "=== [1/5] Layer anatomy ==="
if [ ! -f "$RESULTS/anatomy/anatomy.json" ]; then
    "$PY" scripts/run_anatomy.py --model "$MODEL" --output-dir "$RESULTS/anatomy"
else
    echo "  anatomy.json exists, skipping"
fi

echo "=== [2/5] Smoke scan (eurosat) ==="
if [ ! -f "$RESULTS/.smoke_ok" ]; then
    "$PY" scripts/run_scan.py \
        --data-dir "$DATA_DIR" --output-dir "$RESULTS" --model "$MODEL" \
        --datasets eurosat --resume
    touch "$RESULTS/.smoke_ok"
fi

echo "=== [3/5] Full sweep ==="
# --allow-missing-datasets: one flaky mirror (usually Places365) must not
# abort the sweep; a later --resume run picks up whatever was missing.
"$PY" scripts/run_scan.py \
    --data-dir "$DATA_DIR" --output-dir "$RESULTS" --model "$MODEL" \
    --repeat-scan --resume --allow-missing-datasets "${DATASET_ARGS[@]}"

echo "=== [4/5] Stage-2 confirmation ==="
"$PY" scripts/confirm_top.py \
    --data-dir "$DATA_DIR" --results-dir "$RESULTS" --model "$MODEL" \
    --top-k 20 --n-random-configs 20 --n-test 500 --allow-missing-datasets

echo "=== [5/5] Heatmaps + report ==="
"$PY" visualization/heatmap.py --results-dir "$RESULTS" --output-dir "$OUTPUTS"
"$PY" scripts/report.py --results-dir "$RESULTS" --output-path "$OUTPUTS/report.md"

echo "=== DONE ==="
