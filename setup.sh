#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

python_command="${PYTHON_COMMAND:-python3}"
if [[ ! -x ".venv/bin/python" ]]; then
  "$python_command" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

for weight in \
  weights/yolo11s-pose.pt \
  weights/lstm/best_checkpoint.pt \
  weights/stgcn/best_checkpoint.pt; do
  if [[ ! -f "$weight" ]]; then
    echo "Missing production weight: $weight" >&2
    exit 1
  fi
done

echo "Setup complete. Start the app with: bash run.sh"

