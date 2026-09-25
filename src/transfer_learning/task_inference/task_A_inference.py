import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from utils import (
    extract_few_shot_examples,
    load_jsonl,
    load_model,
    process_inference_batch,
)

# Add src directory to path to import llm_helpers
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from src.experiment.inference.llm_helpers import build_messages


def run_task_a_inference(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    few_shot_count: int = 0,
    ascii_format_output: List[str] = None,
    prompt_mode: str = "simple",
    few_shot_mode: str = "system",
    ascii_order: str = "answer_first",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    batch_size: int = 1,
) -> None:
    """Run inference for Task A (Description → ASCII Diagram).

    Args:
        model_path: Path to the saved model directory
        test_data_path: Path to the test data JSONL file
        output_dir: Directory to save the output results
        few_shot_count: Number of few-shot examples to use
        ascii_format_output: List of ASCII views to generate (default: ["grid"])
        prompt_mode: Prompt mode ("simple" or "detailed")
        few_shot_mode: Few-shot mode ("system", "user", or "conversational")
        ascii_order: ASCII order ("answer_first" or "ascii_first")
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        device_map: Device mapping strategy
        torch_dtype: Data type for model weights
        batch_size: Number of samples to process in parallel (default: 1)
    """
    if ascii_format_output is None:
        ascii_format_output = ["grid"]

    print(f"Loading test data from {test_data_path}...")
    all_data = load_jsonl(test_data_path)
    print(f"Loaded {len(all_data)} test cases")

    # Extract few-shot examples (first few_shot_count items)
    few_shot_examples = extract_few_shot_examples(all_data, num_examples=few_shot_count)
    few_shot_ids = [ex.get("id") for ex in few_shot_examples]
    print(f"Using {len(few_shot_examples)} few-shot examples: {few_shot_ids}")

    # Test cases are all items (including few-shot examples in this case)
    test_cases = all_data
    print(f"Running inference on {len(test_cases)} test cases")

    # Load model
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_model(
        model_path, device_map=device_map, torch_dtype=torch_dtype
    )

    # Prepare output file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_jsonl_path = os.path.join(output_dir, "task_a_results.jsonl")

    print(f"Using batch size: {batch_size}")

    # Use inference_mode for better performance
    with torch.inference_mode():
        results = []

        # Process test cases in batches
        for batch_start in tqdm(
            range(0, len(test_cases), batch_size), desc="Running inference"
        ):
            batch_end = min(batch_start + batch_size, len(test_cases))
            batch = test_cases[batch_start:batch_end]

            # Prepare batch data
            batch_ids = []
            batch_messages = []

            for idx, tc in enumerate(batch):
                test_id = tc.get("id", f"unknown-{batch_start + idx}")
                description = tc.get("description", "")

                batch_ids.append(test_id)

                # Build messages using the same approach as task_runner
                messages = build_messages(
                    question="",  # Task A doesn't use query
                    description=description,
                    few_shot_examples=few_shot_examples,
                    ascii_format_input=None,  # Task A doesn't use ASCII input
                    ascii_format_output=ascii_format_output,
                    prompt_mode=prompt_mode,
                    few_shot_mode=few_shot_mode,
                    ascii_order=ascii_order,
                    ascii_input=None,
                    task="A",
                )
                batch_messages.append(messages)

            # Process inference batch using the refactored function
            process_inference_batch(
                batch_messages=batch_messages,
                batch_ids=batch_ids,
                model=model,
                tokenizer=tokenizer,
                output_jsonl_path=output_jsonl_path,
                results=results,
                batch_start=batch_start,
                result_formatter=lambda response: {"ascii": {"grid": response}},
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"Processed {len(results)} test cases")
    print(f"Results saved to: {output_jsonl_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task A inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="datasets/spatial_data_tested.jsonl",
        help="Path to test data JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fine_tuned_model/A/grid/1000",
        help="Directory to save output results",
    )
    parser.add_argument(
        "--few_shot_count", type=int, default=5, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="detailed",
        choices=["simple", "detailed"],
        help="Prompt mode",
    )
    parser.add_argument(
        "--few_shot_mode",
        type=str,
        default="conversational",
        choices=["system", "user", "conversational"],
        help="Few-shot mode",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4 for better GPU utilization)",
    )

    args = parser.parse_args()

    run_task_a_inference(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        few_shot_count=args.few_shot_count,
        ascii_format_output=["grid"],
        prompt_mode=args.prompt_mode,
        few_shot_mode=args.few_shot_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        batch_size=args.batch_size,
    )
