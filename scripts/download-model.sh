#!/usr/bin/env bash
set -euo pipefail

model_id="${1:-breezedeus/pix2text-mfr-1.5}"
output_dir="${2:-models/pix2text-mfr-1.5}"

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI is required. Install huggingface_hub or activate the server venv." >&2
  exit 1
fi

hf download "$model_id" --local-dir "$output_dir"
