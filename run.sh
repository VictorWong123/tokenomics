#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
backend=cpu
if [[ "$(uname -s)" == Darwin ]]; then backend=metal; fi
if [[ "${1:-}" == cpu || "${1:-}" == metal || "${1:-}" == cuda ]]; then backend="$1"; shift; fi
bash scripts/bootstrap.sh "$backend"
exec .venv/bin/python scripts/run_all.py --backend "$backend" "$@"
