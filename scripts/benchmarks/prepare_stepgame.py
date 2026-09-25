"""Download the StepGame test set used in the paper and convert it to the format
read by ``src/transfer_learning/task_inference/sparp_inference.py``.

Source: the 1000-instance ``small-sparp`` StepGame PS2 test split of SpaRP
(Rizvi et al., 2024), hosted at ``UKPLab/sparp`` on Hugging Face. Records keep
their position as ``id`` and the SpaRP ``targets`` list as ``target``.

Usage:
    uv run python scripts/benchmarks/prepare_stepgame.py
    uv run python scripts/benchmarks/prepare_stepgame.py --output datasets/sparp_stepgame.json
"""

import argparse
import json
import urllib.request
from pathlib import Path

STEPGAME_URL = (
    "https://huggingface.co/datasets/UKPLab/sparp/resolve/main/"
    "small-sparp/SpaRP%20(StepGame)/PS2/test.json"
)
DEFAULT_OUTPUT = "datasets/sparp_stepgame.json"


def main(output: str) -> None:
    print(f"Downloading {STEPGAME_URL} ...")
    with urllib.request.urlopen(STEPGAME_URL) as response:
        source = json.load(response)

    records = [
        {
            "id": index,
            "question": item["question"],
            "target": item["targets"],
            "context": item["context"],
        }
        for index, item in enumerate(source)
    ]

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(records, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    main(parser.parse_args().output)
