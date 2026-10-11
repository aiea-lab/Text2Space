"""Inference on bAbI Task 19 (path finding) using an OpenAI-compatible API."""

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
You are given a spatial layout of rooms described by directional relations. Your task is to find the shortest path from one room to another.

The path is always exactly 2 steps. Each step moves to an adjacent room in one of four directions: north, south, east, west. Order matters: the first direction is the first move and the second direction is the second move (e.g., "south east" means go south first, then east).

Provide only the two directions separated by a space, nothing else. Your answer must be one of:
north north, north east, north west, south south, south east, south west, east east, east north, east south, west west, west north, west south

Spatial layout:
{passage}

Question: {question}

Answer:"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def score_prediction(prediction: str, ground_truth: str) -> float:
    """Score a prediction against ground truth.

    Returns 1.0 for full match, 0.5 if only the first step is correct, 0.0 otherwise.
    """
    pred_parts = prediction.strip().lower().split()
    gt_parts = ground_truth.strip().lower().split()
    if len(pred_parts) < 2 or len(gt_parts) < 2:
        return (
            1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0
        )
    if pred_parts[0] == gt_parts[0] and pred_parts[1] == gt_parts[1]:
        return 1.0
    if pred_parts[0] == gt_parts[0]:
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
    output_path = os.path.join(output_dir, "babi_task19_results.jsonl")

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
        for idx, item in enumerate(tqdm(dataset, desc="bAbI Task 19")):
            prompt = PROMPT_TEMPLATE.format(
                passage=item["passage"],
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
                log.error("Instance %d: all retries exhausted", idx)

            score = score_prediction(response_text, item["answer"])
            result = {
                "index": idx,
                "question": item["question"],
                "ground_truth": item["answer"],
                "prediction": response_text,
                "score": score,
                "split": item.get("split", ""),
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
        "Done. %d full, %d half, %d wrong out of %d (avg score %.3f)",
        full_correct,
        half_correct,
        total - full_correct - half_correct,
        total,
        total_score / total if total else 0,
    )

    summary = {
        "total": total,
        "full_correct": full_correct,
        "half_correct": half_correct,
        "total_score": total_score,
        "avg_score": total_score / total if total else 0,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bAbI Task 19 inference via vLLM")
    parser.add_argument(
        "--data_path", type=str, default="datasets/babi_task19_pathfinding.json"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        model_slug = args.model.replace("/", "_")
        output_dir = os.path.join("results", model_slug, "babi_task19")

    run_inference(
        data_path=args.data_path,
        output_dir=output_dir,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
    )
