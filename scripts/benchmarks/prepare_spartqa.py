"""Download the SpartQA choose-object (CO) test set and convert it to the format
read by ``src/transfer_learning/task_inference/spartqa_inference.py``.

Two public copies of the same 3594 test questions are combined:

* ``tasksource/spartqa-mchoice`` provides the ``story`` and ``question`` fields.
  Runs of periods (``..``) left by the SpartQA generator are collapsed to one.
* ``RAR-b/spartqa`` (Xiao et al., 2024) provides the query ids
  (``spartqa-q-<n>``) and, through its qrels, the list of accepted answer strings
  used as ``reference_answers``.

Two files are written. ``spartqa.json`` holds all 3594 questions.
``spartqa_unique.json`` keeps the first question of every story (1851 items)
and is the file the paper's SpartQA numbers are computed on.

Usage:
    uv run python scripts/benchmarks/prepare_spartqa.py
"""

import argparse
import csv
import io
import json
import re
import urllib.request
from pathlib import Path

MCHOICE_URL = (
    "https://huggingface.co/datasets/tasksource/spartqa-mchoice/resolve/main/"
    "spartqa_CO_test.jsonl"
)
RARB_BASE = "https://huggingface.co/datasets/RAR-b/spartqa/resolve/main/"
DEFAULT_OUTPUT = "datasets/spartqa.json"
DEFAULT_UNIQUE_OUTPUT = "datasets/spartqa_unique.json"


def fetch_text(url: str) -> str:
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def read_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def clean(text: str) -> str:
    return re.sub(r"\.\.+", ".", text)


def main(output: str, unique_output: str) -> None:
    mchoice = read_jsonl(fetch_text(MCHOICE_URL))
    queries = read_jsonl(fetch_text(RARB_BASE + "queries.jsonl"))
    corpus = {
        doc["_id"]: doc["text"]
        for doc in read_jsonl(fetch_text(RARB_BASE + "corpus.jsonl"))
    }
    qrels_rows = csv.reader(
        io.StringIO(fetch_text(RARB_BASE + "qrels/test.tsv")), delimiter="\t"
    )
    next(qrels_rows)  # header
    relevant: dict[str, list[str]] = {}
    for query_id, doc_id, score in qrels_rows:
        if float(score) > 0:
            relevant.setdefault(query_id, []).append(corpus[doc_id])

    assert len(mchoice) == len(queries), (len(mchoice), len(queries))
    records = []
    for item, query in zip(mchoice, queries):
        story = clean(item["story"])
        question = clean(item["question"])
        assert clean(query["text"]) == f"{story} {question}", query["_id"]
        records.append(
            {
                "id": query["_id"],
                "story": story,
                "question": question,
                "reference_answers": relevant[query["_id"]],
            }
        )

    seen_stories: set[str] = set()
    unique_records = []
    for record in records:
        if record["story"] not in seen_stories:
            seen_stories.add(record["story"])
            unique_records.append(record)

    for path, data in ((output, records), (unique_output, unique_records)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"Wrote {len(data)} records to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--unique-output", default=DEFAULT_UNIQUE_OUTPUT)
    args = parser.parse_args()
    main(args.output, args.unique_output)
