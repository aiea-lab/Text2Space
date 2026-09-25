"""Inference on SpartQA spatial reasoning dataset using an OpenAI-compatible API."""

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
You are given a spatial description of colored shapes arranged in blocks. Your task is to answer a spatial reasoning question about the layout.

The description uses spatial relations: above, below, to the left of, to the right of, and near. Blocks themselves also have spatial relations to each other (e.g., "Block A is above B"), which affect the positions of the shapes they contain.

Each question asks about the spatial relationship between objects and gives two options. Your answer must be exactly one of:
- The first option
- The second option
- "both of them"

Provide only the answer, nothing else.

Story:
{story}

Question: {question}

Answer:"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def extract_options_from_question(question: str) -> tuple[str, str]:
    """Extract the two options from a SpartQA question.

    Questions follow the pattern: "... <option A> or <option B>?"
    """
    # Strip trailing "?"
    q = question.rstrip("?").strip()
    # Split on last " or " to get the two options
    parts = q.rsplit(" or ", 1)
    if len(parts) == 2:
        option_b = parts[1].strip()
        # option_a is everything after the last "?" or the question stem
        # The stem typically ends with "? " before option_a, but for SpartQA
        # the format is "Which object is X? optA or optB?"
        # We need the part after the question word that introduces options
        option_a = parts[0].strip()
        # option_a still has the question stem; grab text after last "? " or after last "! "
        # Actually the format is: "What is below X? A or B?" - no inner "?"
        # More reliably: the question is already the full text, so option_a
        # is everything in parts[0] that comes after the last comma or question word.
        # Simplest: just return as-is; the matching will normalize.
        return option_a, option_b
    return "", ""


def normalize_answer(text: str) -> str:
    """Normalize an answer for comparison: lowercase, strip, remove leading articles."""
    import re

    t = text.lower().strip().rstrip(".").strip()
    t = re.sub(r"^(a |an |the )", "", t)
    return t


def check_match(prediction: str, reference_answers: list[str], question: str) -> bool:
    """Check if prediction matches any reference answer with fuzzy matching.

    Handles:
    - Exact match (case-insensitive, stripped)
    - Article differences (a/the/an prefix)
    - "The first/second option" resolved to actual option text
    """
    pred_norm = normalize_answer(prediction)
    for ref in reference_answers:
        if normalize_answer(ref) == pred_norm:
            return True

    # Resolve "first option" / "second option" to actual text
    pred_lower = prediction.lower().strip()
    if "first option" in pred_lower or "second option" in pred_lower:
        option_a, option_b = extract_options_from_question(question)
        if "first option" in pred_lower:
            resolved = option_a
        else:
            resolved = option_b
        resolved_norm = normalize_answer(resolved)
        for ref in reference_answers:
            if normalize_answer(ref) == resolved_norm:
                return True

    return False


def run_inference(
    data_path: str,
    output_dir: str,
    base_url: str,
    model: str,
    max_tokens: int = 128,
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
    output_path = os.path.join(output_dir, "spartqa_results.jsonl")

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
        for idx, item in enumerate(tqdm(dataset, desc="SpartQA")):
            prompt = PROMPT_TEMPLATE.format(
                story=item["story"],
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

            result = {
                "id": item.get("id", f"spartqa-{idx}"),
                "question": item["question"],
                "reference_answers": item["reference_answers"],
                "prediction": response_text,
            }
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    # Summary with fuzzy matching
    correct = sum(
        1
        for r in results
        if check_match(r["prediction"], r["reference_answers"], r["question"])
    )
    total = len(results)
    log.info(
        "Done. %d/%d correct (%.1f%%)",
        correct,
        total,
        100 * correct / total if total else 0,
    )

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpartQA inference via vLLM")
    parser.add_argument("--data_path", type=str, default="datasets/spartqa_unique.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        model_slug = args.model.replace("/", "_")
        output_dir = os.path.join("results", model_slug, "spartqa")

    run_inference(
        data_path=args.data_path,
        output_dir=output_dir,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
    )
