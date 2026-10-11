"""Download bAbI Task 19 (path finding) and convert it to the format read by
``src/transfer_learning/task_inference/babi_inference.py``.

Source: the ``en-valid`` variant of bAbI tasks v1.2 (Weston et al., 2015), fetched
from the ParlAI mirror. Each question becomes one record with the story lines
seen so far as ``passage`` and the compass-letter answer (``s,e``) expanded to
words (``south east``). Records are written in split order: train, validation,
test.

Usage:
    uv run python scripts/benchmarks/prepare_babi.py
    uv run python scripts/benchmarks/prepare_babi.py --output datasets/babi_task19_pathfinding.json
"""

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

BABI_URL = "http://parl.ai/downloads/babi/babi.tar.gz"
TASK_FILES = {
    "train": "tasks_1-20_v1-2/en-valid/qa19_train.txt",
    "validation": "tasks_1-20_v1-2/en-valid/qa19_valid.txt",
    "test": "tasks_1-20_v1-2/en-valid/qa19_test.txt",
}
DIRECTION_WORDS = {"n": "north", "s": "south", "e": "east", "w": "west"}
DEFAULT_OUTPUT = "datasets/babi_task19_pathfinding.json"


def parse_task_file(text: str, split: str) -> list[dict[str, str]]:
    """Turn one bAbI task file into passage/question/answer records."""
    records: list[dict[str, str]] = []
    story: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        number, rest = line.split(" ", 1)
        if int(number) == 1:
            story = []
        if "\t" in rest:
            question, answer, _supporting = rest.split("\t")
            records.append(
                {
                    "passage": "\n".join(story),
                    "question": question.strip(),
                    "answer": " ".join(
                        DIRECTION_WORDS[step] for step in answer.strip().split(",")
                    ),
                    "split": split,
                }
            )
        else:
            story.append(rest)
    return records


def main(output: str) -> None:
    print(f"Downloading {BABI_URL} ...")
    with urllib.request.urlopen(BABI_URL) as response:
        archive = io.BytesIO(response.read())

    records: list[dict[str, str]] = []
    with tarfile.open(fileobj=archive, mode="r:gz") as tar:
        for split, member in TASK_FILES.items():
            text = tar.extractfile(member).read().decode("utf-8")
            split_records = parse_task_file(text, split)
            print(f"  {split}: {len(split_records)} questions")
            records.extend(split_records)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(records, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    main(parser.parse_args().output)
