#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${1:-repro_outputs}
CONFIG=${2:-examples/configs/default.yaml}

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

python -m smartbuildsim.cli.app data generate "${CONFIG}" --output "${OUTPUT_DIR}/dataset_a.csv"
python -m smartbuildsim.cli.app data generate "${CONFIG}" --output "${OUTPUT_DIR}/dataset_b.csv"

python - <<'PY'
from pathlib import Path
import pandas as pd
import sys

first = Path(sys.argv[1])
second = Path(sys.argv[2])

frame_a = pd.read_csv(first)
frame_b = pd.read_csv(second)

if not frame_a.equals(frame_b):
    raise SystemExit("Reproducibility check failed: generated datasets differ")

print("Reproducibility check passed: datasets are identical")
PY
"${OUTPUT_DIR}/dataset_a.csv" "${OUTPUT_DIR}/dataset_b.csv"
