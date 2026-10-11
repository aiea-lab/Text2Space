#!/bin/bash
# Fine-tune with the task_b configuration. Run from the repository root.
set -euo pipefail

uv run python src/transfer_learning/train.py \
    --config src/transfer_learning/config/config_task_b.json \
    --generate_predictions
