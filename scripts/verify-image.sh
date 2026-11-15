#!/bin/sh
set -eu

run_dir="$(mktemp -d)"
trap 'rm -rf "$run_dir"' EXIT
python -m mlops_demo prepare "$1" --output "$run_dir/prepared.json"
python -m mlops_demo train "$run_dir/prepared.json" --output "$run_dir/model.json"
python -m mlops_demo evaluate "$run_dir/prepared.json" "$run_dir/model.json" \
  --report "$run_dir/evaluation.json" --max-mae 5
python - "$run_dir" <<'PY'
import sys
from pathlib import Path

from mlops_demo.schemas import DeliveryFeatures
from mlops_demo.workflow import load_approved_model

run = Path(sys.argv[1])
model = load_approved_model(run / "model.json", run / "evaluation.json")
prediction = model.predict(DeliveryFeatures(distance_km=12.5, package_count=3))
assert 30 < prediction < 34, prediction
print(f"Packaged model prediction passed ({prediction:.3f} min).")
PY
