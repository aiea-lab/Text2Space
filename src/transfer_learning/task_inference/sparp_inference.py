"""Inference on SpaRP StepGame spatial reasoning dataset using an OpenAI-compatible API."""

import argparse
import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from tqdm import tqdm

PROMPT_TEMPLATE = """\
You are given a spatial description of point objects in a 2D space viewed from a fixed orientation. Determine the spatial relation of one agent to another by composing the given relations.

Key rules:
- Directions are from a fixed viewpoint: left = west = 9 o'clock, right = east = 3 o'clock, above = north = 12 o'clock, below = south = 6 o'clock.
- Each relation is 1 unit unless stated otherwise. Diagonal relations decompose into 1 unit on each axis (e.g., "upper-left" = 1 above + 1 left).
- Chain relations across multiple objects to find the answer. Units on the same axis add or cancel.

Your answer must include ALL applicable directions from: above, below, left, right.
- If the relation is purely vertical or horizontal, answer with one direction (e.g., "above").
- If the relation is diagonal, answer with exactly two directions separated by a comma (e.g., "above, left").

Provide only the direction(s), nothing else.

Context:
{context}

Question: {question}

Answer:"""

VALID_DIRECTIONS = {"above", "below", "left", "right"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def parse_directions(text: str) -> set[str]:
    """Parse a predicted answer into a set of directions.

    Handles formats like:
    - "above, left"
    - "above and left"
    - "above left"
    - "below, to the right"
    - "above and to the left"
    """
    t = text.strip().lower().rstrip(".").strip()

    # Normalize common phrases
    t = t.replace("to the ", "")
    t = t.replace(" of", "")
    t = t.replace(" and ", ", ")

    # Split on comma or whitespace
    parts = [p.strip() for p in t.replace(",", " ").split()]
    return {p for p in parts if p in VALID_DIRECTIONS}


def score_prediction(prediction: str, ground_truth: list[str]) -> float:
    """Score a prediction against the ground truth direction set.

    Returns 1.0 for exact set match, 0.5 for partial overlap, 0.0 otherwise.
    """
    pred_dirs = parse_directions(prediction)
    gt_dirs = set(ground_truth)

    if not pred_dirs:
        return 0.0
    if pred_dirs == gt_dirs:
        return 1.0
    if pred_dirs & gt_dirs:
        return 0.5
    return 0.0


def run_inference(
    data_path: str,
    output_dir: str,
    base_url: str,
    model: str,
    max_tokens: int = 64,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> None:
    client = OpenAI(
        base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY")
    )

    with open(data_path) as f:
        dataset: list[dict[str, Any]] = json.load(f)
    log.info("Loaded %d instances from %s", len(dataset), data_path)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sparp_stepgame_results.jsonl")

    # Save run config
    config = {
        "dataset": data_path,
        "model": model,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "num_instances": len(dataset),
        "zero_shot": True,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    results: list[dict[str, Any]] = []
    with open(output_path, "w") as out_f:
        for item in tqdm(dataset, desc="SpaRP StepGame"):
            prompt = PROMPT_TEMPLATE.format(
                context=item["context"],
                question=item["question"],
            )
            messages = [{"role": "user", "content": prompt}]

            response_text = ""
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                    )
                    msg = completion.choices[0].message
                    content = msg.content
                    if not content:
                        content = getattr(msg, "reasoning_content", None) or getattr(
                            msg, "reasoning", None
                        )
                    response_text = content.strip() if content else ""
                    break
                except Exception as e:
                    log.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, e)
                    time.sleep(retry_delay)
            else:
                log.error("Instance %d: all retries exhausted", item["id"])

            score = score_prediction(response_text, item["target"])
            result = {
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["target"],
                "prediction": response_text,
                "parsed_prediction": sorted(parse_directions(response_text)),
                "score": score,
            }
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    # Summary
    total = len(results)
    full_correct = sum(1 for r in results if r["score"] == 1.0)
    half_correct = sum(1 for r in results if r["score"] == 0.5)
    total_score = sum(r["score"] for r in results)
    log.info(
        "Done. %d full, %d partial, %d wrong out of %d (avg score %.3f)",
        full_correct,
        half_correct,
        total - full_correct - half_correct,
        total,
        total_score / total if total else 0,
    )

    summary = {
        "total": total,
        "full_correct": full_correct,
        "partial_correct": half_correct,
        "wrong": total - full_correct - half_correct,
        "total_score": total_score,
        "avg_score": total_score / total if total else 0,
        "exact_match_accuracy": full_correct / total if total else 0,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpaRP StepGame inference via vLLM")
    parser.add_argument("--data_path", type=str, default="datasets/sparp_stepgame.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        model_slug = args.model.replace("/", "_")
        output_dir = os.path.join("results", model_slug, "sparp_stepgame")

    run_inference(
        data_path=args.data_path,
        output_dir=output_dir,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
    )
