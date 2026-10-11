#!/bin/bash
# Fine-tune with the task_mixed_llama70b_completion_only configuration. Run from the repository root.
set -euo pipefail

uv run python src/transfer_learning/train.py \
    --config src/transfer_learning/config/config_task_mixed_llama70b_completion_only.json
