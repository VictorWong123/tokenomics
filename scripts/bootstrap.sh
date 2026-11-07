#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
backend="${1:-cpu}"
case "$backend" in
  cpu) metal=OFF; cuda=OFF ;;
  metal) metal=ON; cuda=OFF; [[ "$(uname -s)" == Darwin ]] || { echo 'Metal requires macOS' >&2; exit 2; } ;;
  cuda) metal=OFF; cuda=ON ;;
  *) echo 'usage: bootstrap.sh cpu|metal|cuda' >&2; exit 2 ;;
esac
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
.venv/bin/cmake -S . -B "build-$backend" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_METAL="$metal" -DGGML_CUDA="$cuda"
.venv/bin/cmake --build "build-$backend" -j "${TOKENOMICS_BUILD_JOBS:-4}"
.venv/bin/ctest --test-dir "build-$backend" --output-on-failure
echo "Build and portable checks passed: build-$backend/tokenomics"
