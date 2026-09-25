"""Split the full Text2Space dataset into the paper's training file.

The released dataset (``spatial_data.jsonl``, 20000 instances) is partitioned by
instance id into three files:

* ``spatial_data_tested.jsonl``: 1000 evaluation instances (shipped in this repo)
* ``spatial_data_fewshot.jsonl``: 5 few-shot demonstrations (shipped in this repo)
* ``spatial_data_untested.jsonl``: the remaining 18995 training instances

This script writes the training file by removing the ids of the first two from
the full dataset, preserving the original order.

Usage:
    uv run python scripts/split_dataset.py --full datasets/spatial_data.jsonl
"""

import argparse
import json
from pathlib import Path


def read_ids(path: str) -> set[str]:
    with open(path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def main(full: str, held_out: list[str], output: str) -> None:
    excluded: set[str] = set()
    for path in held_out:
        excluded |= read_ids(path)

    kept = 0
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(full) as src, open(output, "w") as dst:
        for line in src:
            if not line.strip():
                continue
            if json.loads(line)["id"] in excluded:
                continue
            dst.write(line if line.endswith("\n") else line + "\n")
            kept += 1
    print(f"Excluded {len(excluded)} ids; wrote {kept} instances to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--full", default="datasets/spatial_data.jsonl")
    parser.add_argument(
        "--held-out",
        nargs="+",
        default=[
            "datasets/spatial_data_tested.jsonl",
            "datasets/spatial_data_fewshot.jsonl",
        ],
    )
    parser.add_argument("--output", default="datasets/spatial_data_untested.jsonl")
    args = parser.parse_args()
    main(args.full, args.held_out, args.output)
