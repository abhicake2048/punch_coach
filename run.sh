#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "CornerCoach is not set up. Run: bash setup.sh" >&2
  exit 1
fi

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
exec .venv/bin/python -m streamlit run app.py --server.port "${PORT:-8501}"

