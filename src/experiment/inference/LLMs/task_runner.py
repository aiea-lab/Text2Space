import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from experiment.inference.llm_helpers import (
    TASK_CONFIGS,
    _normalize_ascii_views,
    build_messages,
    build_system_prompt,
    format_few_shot_example,
    json_dumps_compact,
)

DEFAULT_DATA_PATH = "datasets/spatial_data_tested.jsonl"
DEFAULT_FEW_SHOT = 5
DEFAULT_NUM_CASES = 200
DEFAULT_RESULTS_BASE_DIR = "results"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def load_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def format_prompt(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for msg in messages:
        lines.append("-" * 80)
        lines.append(msg.get("role", "").upper())
        lines.append("-" * 80)
        lines.append(msg.get("content", ""))
        lines.append("")
    return "\n".join(lines)


def matches_filter(sample: Dict[str, Any], *, require_ascii: bool = False) -> bool:
    if require_ascii and not sample.get("ascii"):
        return False
    return True


def select_few_shot_examples(
    records: List[Dict[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    few_shot: List[Dict[str, Any]] = []
    seen_labels: set[str] = set()
    for sample in records:
        if not matches_filter(sample):
            continue
        label = sample.get("label")
        if label and label not in seen_labels:
            few_shot.append(sample)
            seen_labels.add(label)
        if len(few_shot) >= limit:
            break
    return few_shot


def select_test_cases(
    records: List[Dict[str, Any]],
    exclude_indices: Iterable[int],
    limit: int,
    require_ascii: bool = True,
) -> List[Dict[str, Any]]:
    exclude = set(exclude_indices)
    test_cases: List[Dict[str, Any]] = []
    for idx, sample in enumerate(records):
        if idx in exclude:
            continue
        if matches_filter(sample, require_ascii=require_ascii):
            test_cases.append(sample)
        if len(test_cases) >= limit:
            break
    return test_cases


def normalize_ascii_views(views_arg: Optional[str]) -> List[str]:
    """
    Normalize ASCII views from command-line argument.
    If views are specified, return them as a list. Otherwise, default to ['grid'].
    """
    if not views_arg:
        return ["grid"]

    valid_views = {"grid", "panel", "simple"}
    views = [v.strip() for v in views_arg.split(",") if v.strip()]
    # Filter to only valid views
    normalized = [v for v in views if v in valid_views]
    return normalized if normalized else ["grid"]


def format_views_for_dir(views: List[str]) -> str:
    """Format ASCII views list as a directory name string."""
    return "-".join(views) if views else "grid"


def call_with_retry(func, max_retries: int, delay: float, *args, **kwargs):
    """Call a function with retries in case of connection/server errors."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            retriable = any(
                token in msg
                for token in (
                    "InternalServerError: upstream connect error",
                    "Connection refused",
                    "remote connection failure",
                    "transport failure",
                    "reset reason",
                )
            )
            if retriable:
                last_exception = e
                logging.warning(
                    "Retry %s/%s after error: %s", attempt + 1, max_retries, msg
                )
                time.sleep(delay)
                continue
            raise
    logging.error("Final failure after retries: %s", last_exception)
    raise last_exception  # noqa: B904


def run_model_case_task_g(
    client: OpenAI,
    model: str,
    tc: Dict[str, Any],
    few_shot_examples: List[Dict[str, Any]],
    *,
    ascii_format_output: Optional[Iterable[str]] = None,
    prompt_mode: str = "simple",
    few_shot_mode: str = "system",
    ascii_order: str = "answer_first",
    task: str = "G",
    max_retries: int = 5,
    delay: float = 2,
    max_tokens: int = 2048,
    timeout: float = 300.0,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Task G: Conversational two-turn task
    Turn 1: User asks to generate ASCII from Description + Query, Assistant responds with ASCII
    Turn 2: User asks to answer the question, Assistant responds with answer (using ASCII from turn 1)

    This tests whether generating ASCII first (turn 1) helps answer better (turn 2).
    """
    # Normalize ASCII format
    output_views = _normalize_ascii_views(ascii_format_output)
    if not output_views:
        output_views = ["grid"]

    # Build system prompt for Task G (two-turn conversational)
    # System prompt should include both Turn 1 and Turn 2 instructions
    simple_mode = prompt_mode == "simple"
    system_prompt = build_system_prompt(
        task_code="G",
        ascii_format_input=None,
        ascii_format_output=output_views,
        include_answer=True,
        include_description_output=False,
        include_query=True,
        simple_mode=simple_mode,
        ascii_order=ascii_order,
    )

    # Build messages array
    messages = [{"role": "system", "content": system_prompt}]

    # Add few-shot examples in conversational format
    task_config = TASK_CONFIGS.get("G", TASK_CONFIGS["C"])
    task_inputs = task_config["inputs"]
    task_outputs = task_config["outputs"]

    if few_shot_examples:
        if few_shot_mode == "system":
            # Add examples to system prompt
            system_examples = "\n\nEXAMPLES:\n"
            for i, ex in enumerate(few_shot_examples, 1):
                system_examples += f"\nExample {i}:\n"
                # Turn 1 format
                desc = ex.get("description", "")
                query = ex.get("query_relation", "")
                ascii_block = ex.get("ascii", {})
                ascii_output = {}
                if ascii_block:
                    for view in output_views:
                        art = ascii_block.get(view)
                        if art:
                            ascii_output[view] = art

                system_examples += f"USER:\nDescription:\n{desc}\n\nQuestion: {query}\n\nPlease generate an ASCII visualization of this scene.\n\n"
                system_examples += (
                    f"ASSISTANT:\n{json_dumps_compact({'ascii': ascii_output})}\n\n"
                )
                system_examples += f"USER:\nNow, based on the ASCII visualization you generated, answer the question: {query}\n\n"
                answer = ex.get("label", "")
                system_examples += (
                    f"ASSISTANT:\n{json_dumps_compact({'answer': answer})}\n"
                )
            messages[0]["content"] += system_examples
        elif few_shot_mode == "conversational":
            # Add examples as conversational turns
            for ex in few_shot_examples:
                desc = ex.get("description", "")
                query = ex.get("query_relation", "")
                ascii_block = ex.get("ascii", {})
                ascii_output = {}
                if ascii_block:
                    for view in output_views:
                        art = ascii_block.get(view)
                        if art:
                            ascii_output[view] = art

                # Turn 1
                messages.append(
                    {
                        "role": "user",
                        "content": f"Description:\n{desc}\n\nQuestion: {query}\n\nPlease generate an ASCII visualization of this scene.",
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": json_dumps_compact({"ascii": ascii_output}),
                    }
                )

                # Turn 2
                answer = ex.get("label", "")
                messages.append(
                    {
                        "role": "user",
                        "content": f"Now, based on the ASCII visualization you generated, answer the question: {query}",
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": json_dumps_compact({"answer": answer}),
                    }
                )

    # Add turn 1 user message for test case
    desc = tc.get("description", "")
    query = tc.get("query_relation", "")
    messages.append(
        {
            "role": "user",
            "content": f"Description:\n{desc}\n\nQuestion: {query}\n\nPlease generate an ASCII visualization of this scene.",
        }
    )

    # Call model for turn 1
    last_exception = None
    generated_ascii = None
    turn1_metadata = {}

    for attempt in range(max_retries):
        try:
            api_params = {
                "model": model,
                "messages": messages,  # Use the conversational messages array
                "max_tokens": max_tokens,
                "timeout": timeout,
            }

            if reasoning_effort:
                api_params["reasoning_effort"] = reasoning_effort

            completion = call_with_retry(
                client.chat.completions.create,
                max_retries,
                delay,
                **api_params,
            )
            message = completion.choices[0].message
            # Use message.content directly for conversation history
            message_content = message.content

            # For parsing, try content first, then fallback to reasoning_content/reasoning
            raw = message_content
            if not raw or not raw.strip():
                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content and reasoning_content.strip():
                    raw = reasoning_content
                else:
                    reasoning = getattr(message, "reasoning", None)
                    if reasoning and reasoning.strip():
                        raw = reasoning

            if not raw or not raw.strip():
                finish_reason = getattr(
                    completion.choices[0], "finish_reason", "unknown"
                )
                error_msg = f"Turn 1: Model returned empty/None content (finish_reason: {finish_reason})"
                logging.warning(
                    "Task G Turn 1: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = ValueError(error_msg)
                time.sleep(delay)
                continue

            # Parse turn 1 output - handle JSON parsing errors explicitly
            try:
                parsed_turn1 = json.loads(raw)
            except json.JSONDecodeError as json_err:
                error_msg = f"Turn 1: Failed to parse JSON response: {str(json_err)[:200]}. Raw content: {str(raw)[:200]}"
                logging.warning(
                    "Task G Turn 1: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = json_err
                time.sleep(delay)
                continue

            # Validate that turn 1 output has the correct structure (must have "ascii" key)
            if not isinstance(parsed_turn1, dict) or "ascii" not in parsed_turn1:
                error_msg = (
                    f"Turn 1 output missing 'ascii' key. Got: {str(parsed_turn1)[:200]}"
                )
                logging.warning(
                    "Task G Turn 1: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = ValueError(error_msg)
                time.sleep(delay)
                continue

            # Validate that the ASCII dict has at least one view
            ascii_dict = parsed_turn1.get("ascii")
            if not ascii_dict or not any(
                view in ascii_dict for view in ["grid", "panel", "simple"]
            ):
                error_msg = (
                    f"Turn 1 ASCII dict is empty or invalid: {str(ascii_dict)[:200]}"
                )
                logging.warning(
                    "Task G Turn 1: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = ValueError(error_msg)
                time.sleep(delay)
                continue

            generated_ascii = parsed_turn1

            # Capture turn 1 metadata
            turn1_metadata = {
                "content": message.content,
                "reasoning": getattr(message, "reasoning", None),
                "reasoning_content": getattr(message, "reasoning_content", None),
                "role": message.role,
                "refusal": message.refusal,
                "finish_reason": getattr(completion.choices[0], "finish_reason", None),
            }

            if hasattr(completion, "usage") and completion.usage:
                turn1_metadata["usage"] = {
                    "completion_tokens": completion.usage.completion_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }

            logging.debug(
                "Task G Turn 1: Successfully generated valid ASCII with structure: %s",
                list(ascii_dict.keys()),
            )

            # Add turn 1 assistant response to messages (for conversational continuity)
            # Extract only the ASCII part (not any answer field that might be present)
            # This ensures Turn 2 only sees the ASCII, not any answer from Turn 1
            ascii_only_response = {"ascii": ascii_dict}
            content_for_conversation = json_dumps_compact(ascii_only_response)

            messages.append({"role": "assistant", "content": content_for_conversation})
            break

        except Exception as e:
            last_exception = e
            logging.warning(
                "Task G Turn 1: Retry %s/%s after error: %s",
                attempt + 1,
                max_retries,
                e,
            )
            time.sleep(delay)
            continue

    if not generated_ascii:
        logging.error(
            "Task G Turn 1: Failed to generate ASCII after retries: %s", last_exception
        )
        return {
            "messages": messages,  # Return messages up to turn 1 user message
            "output": {"error": f"Turn 1 failed: {last_exception}"},
            "turn1_output": None,
            "message_metadata": {
                "turn1": turn1_metadata,
                "turn2": {},
                "error": str(last_exception),
            },
        }

    # Add turn 2 user message (asking to answer the question)
    messages.append(
        {
            "role": "user",
            "content": f"Now, based on the ASCII visualization you generated, answer the question: {query}",
        }
    )

    # Call model for turn 2 (using same conversation, now includes turn 1)
    for attempt in range(max_retries):
        try:
            api_params = {
                "model": model,
                "messages": messages,  # Same conversation array, now includes turn 1
                "max_tokens": max_tokens,
                "timeout": timeout,
            }

            if reasoning_effort:
                api_params["reasoning_effort"] = reasoning_effort

            completion = call_with_retry(
                client.chat.completions.create,
                max_retries,
                delay,
                **api_params,
            )
            message = completion.choices[0].message
            # Use message.content directly
            message_content = message.content

            # For parsing, try content first, then fallback to reasoning_content/reasoning
            raw = message_content
            if not raw or not raw.strip():
                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content and reasoning_content.strip():
                    raw = reasoning_content
                else:
                    reasoning = getattr(message, "reasoning", None)
                    if reasoning and reasoning.strip():
                        raw = reasoning

            if not raw or not raw.strip():
                finish_reason = getattr(
                    completion.choices[0], "finish_reason", "unknown"
                )
                error_msg = f"Turn 2: Model returned empty/None content (finish_reason: {finish_reason})"
                logging.warning(
                    "Task G Turn 2: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = ValueError(error_msg)
                time.sleep(delay)
                continue

            # Parse turn 2 output - handle JSON parsing errors explicitly
            try:
                parsed_turn2 = json.loads(raw)
            except json.JSONDecodeError as json_err:
                error_msg = f"Turn 2: Failed to parse JSON response: {str(json_err)[:200]}. Raw content: {str(raw)[:200]}"
                logging.warning(
                    "Task G Turn 2: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = json_err
                time.sleep(delay)
                continue

            # Capture turn 2 metadata
            turn2_metadata = {
                "content": message.content,
                "reasoning": getattr(message, "reasoning", None),
                "reasoning_content": getattr(message, "reasoning_content", None),
                "role": message.role,
                "refusal": message.refusal,
                "finish_reason": getattr(completion.choices[0], "finish_reason", None),
            }

            if hasattr(completion, "usage") and completion.usage:
                turn2_metadata["usage"] = {
                    "completion_tokens": completion.usage.completion_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }

            logging.debug("Task G Turn 2: Successfully generated answer using ASCII")

            # Add turn 2 assistant response to messages for completeness
            # Use message.content directly (what model actually returned), not the fallback raw
            # If message.content is empty, use the parsed raw content as fallback
            content_for_conversation = (
                message_content if message_content and message_content.strip() else raw
            )
            messages.append({"role": "assistant", "content": content_for_conversation})

            result = {
                "messages": messages,  # Full conversational history
                "output": parsed_turn2,
                "turn1_output": generated_ascii,
                "message_metadata": {
                    "turn1": turn1_metadata,
                    "turn2": turn2_metadata,
                },
            }

            return result

        except Exception as e:
            last_exception = e
            logging.warning(
                "Task G Turn 2: Retry %s/%s after error: %s",
                attempt + 1,
                max_retries,
                e,
            )
            time.sleep(delay)
            continue

    logging.error("Task G Turn 2: Failed after retries: %s", last_exception)
    return {
        "messages": messages,  # Full conversation including turn 1
        "output": {"error": f"Turn 2 failed: {last_exception}"},
        "turn1_output": generated_ascii,
        "message_metadata": {
            "turn1": turn1_metadata,
            "turn2": {},
            "error": str(last_exception),
        },
    }


def run_model_case_task_h(
    client: OpenAI,
    model: str,
    tc: Dict[str, Any],
    few_shot_examples: List[Dict[str, Any]],
    *,
    ascii_format_output: Optional[Iterable[str]] = None,
    prompt_mode: str = "simple",
    few_shot_mode: str = "system",
    ascii_order: str = "answer_first",
    task: str = "H",
    max_retries: int = 5,
    delay: float = 2,
    max_tokens: int = 2048,
    timeout: float = 300.0,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Task H: Two-turn task with ground truth ASCII in Turn 2
    Turn 1: User asks to generate ASCII from Description + Query, Assistant responds with ASCII
    Turn 2: User asks to answer using GROUND TRUTH ASCII (not generated), Assistant responds with answer

    This tests whether using ground truth ASCII helps answer better compared to generated ASCII (Task G).
    """
    logging.info("Task H: Starting run_model_case_task_h for test_id=%s", tc.get("id"))
    # Normalize ASCII format
    output_views = _normalize_ascii_views(ascii_format_output)
    if not output_views:
        output_views = ["grid"]

    # Build system prompt for Task H (two-turn conversational with ground truth ASCII)
    simple_mode = prompt_mode == "simple"
    system_prompt = build_system_prompt(
        task_code="H",
        ascii_format_input=None,
        ascii_format_output=output_views,
        include_answer=True,
        include_description_output=False,
        include_query=True,
        simple_mode=simple_mode,
        ascii_order=ascii_order,
    )

    # Build messages array
    messages = [{"role": "system", "content": system_prompt}]

    # Add few-shot examples in conversational format
    task_config = TASK_CONFIGS.get("H", TASK_CONFIGS["C"])
    task_inputs = task_config["inputs"]
    task_outputs = task_config["outputs"]

    if few_shot_examples:
        if few_shot_mode == "system":
            # Add examples to system prompt
            system_examples = "\n\nEXAMPLES:\n"
            for i, ex in enumerate(few_shot_examples, 1):
                system_examples += f"\nExample {i}:\n"
                desc = ex.get("description", "")
                query = ex.get("query_relation", "")
                ascii_block = ex.get("ascii", {})
                ascii_output = {}
                if ascii_block:
                    for view in output_views:
                        art = ascii_block.get(view)
                        if art:
                            ascii_output[view] = art

                system_examples += f"USER:\nDescription:\n{desc}\n\nQuestion: {query}\n\nPlease generate an ASCII visualization of this scene.\n\n"
                system_examples += (
                    f"ASSISTANT:\n{json_dumps_compact({'ascii': ascii_output})}\n\n"
                )
                system_examples += f"USER:\nNow, based on the ASCII visualization you generated, answer the question: {query}\n\n"
                answer = ex.get("label", "")
                system_examples += (
                    f"ASSISTANT:\n{json_dumps_compact({'answer': answer})}\n"
                )
            messages[0]["content"] += system_examples
        elif few_shot_mode == "conversational":
            # Add examples as conversational turns
            for ex in few_shot_examples:
                desc = ex.get("description", "")
                query = ex.get("query_relation", "")
                ascii_block = ex.get("ascii", {})
                ascii_output = {}
                if ascii_block:
                    for view in output_views:
                        art = ascii_block.get(view)
                        if art:
                            ascii_output[view] = art

                # Turn 1
                messages.append(
                    {
                        "role": "user",
                        "content": f"Description:\n{desc}\n\nQuestion: {query}\n\nPlease generate an ASCII visualization of this scene.",
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": json_dumps_compact({"ascii": ascii_output}),
                    }
                )

                # Turn 2
                answer = ex.get("label", "")
                messages.append(
                    {
                        "role": "user",
                        "content": f"Now, based on the ASCII visualization you generated, answer the question: {query}",
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": json_dumps_compact({"answer": answer}),
                    }
                )

    # Add turn 1 user message for test case
    desc = tc.get("description", "")
    query = tc.get("query_relation", "")
    messages.append(
        {
            "role": "user",
            "content": f"Description:\n{desc}\n\nQuestion: {query}\n\nPlease generate an ASCII visualization of this scene.",
        }
    )

    logging.debug("Task H Turn 1: Calling model for ASCII generation")
    # Call model for turn 1 (same as Task G)
    last_exception = None
    generated_ascii = None
    turn1_metadata = {}

    for attempt in range(max_retries):
        try:
            logging.debug("Task H Turn 1: Attempt %s/%s", attempt + 1, max_retries)
            api_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }

            if reasoning_effort:
                api_params["reasoning_effort"] = reasoning_effort

            logging.debug("Task H Turn 1: Making API call with timeout=%s", timeout)
            completion = call_with_retry(
                client.chat.completions.create,
                max_retries,
                delay,
                **api_params,
            )
            logging.debug("Task H Turn 1: API call completed successfully")
            message = completion.choices[0].message
            message_content = message.content

            raw = message_content
            if not raw or not raw.strip():
                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content and reasoning_content.strip():
                    raw = reasoning_content
                else:
                    reasoning = getattr(message, "reasoning", None)
                    if reasoning and reasoning.strip():
                        raw = reasoning

            if not raw or not raw.strip():
                finish_reason = getattr(
                    completion.choices[0], "finish_reason", "unknown"
                )
                error_msg = f"Turn 1: Model returned empty/None content (finish_reason: {finish_reason})"
                logging.warning(
                    "Task H Turn 1: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = ValueError(error_msg)
                time.sleep(delay)
                continue

            try:
                parsed_turn1 = json.loads(raw)
            except json.JSONDecodeError as json_err:
                error_msg = f"Turn 1: Failed to parse JSON response: {str(json_err)[:200]}. Raw content: {str(raw)[:200]}"
                logging.warning(
                    "Task H Turn 1: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = json_err
                time.sleep(delay)
                continue

            if not isinstance(parsed_turn1, dict) or "ascii" not in parsed_turn1:
                error_msg = (
                    f"Turn 1 output missing 'ascii' key. Got: {str(parsed_turn1)[:200]}"
                )
                logging.warning(
                    "Task H Turn 1: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = ValueError(error_msg)
                time.sleep(delay)
                continue

            ascii_dict = parsed_turn1.get("ascii")
            if not ascii_dict or not any(
                view in ascii_dict for view in ["grid", "panel", "simple"]
            ):
                error_msg = (
                    f"Turn 1 ASCII dict is empty or invalid: {str(ascii_dict)[:200]}"
                )
                logging.warning(
                    "Task H Turn 1: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = ValueError(error_msg)
                time.sleep(delay)
                continue

            generated_ascii = parsed_turn1

            turn1_metadata = {
                "content": message.content,
                "reasoning": getattr(message, "reasoning", None),
                "reasoning_content": getattr(message, "reasoning_content", None),
                "role": message.role,
                "refusal": message.refusal,
                "finish_reason": getattr(completion.choices[0], "finish_reason", None),
            }

            if hasattr(completion, "usage") and completion.usage:
                turn1_metadata["usage"] = {
                    "completion_tokens": completion.usage.completion_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }

            logging.debug(
                "Task H Turn 1: Successfully generated valid ASCII with structure: %s",
                list(ascii_dict.keys()),
            )

            # Add turn 1 assistant response to messages (for conversational continuity)
            ascii_only_response = {"ascii": ascii_dict}
            content_for_conversation = json_dumps_compact(ascii_only_response)
            messages.append({"role": "assistant", "content": content_for_conversation})
            break

        except Exception as e:
            last_exception = e
            logging.warning(
                "Task H Turn 1: Retry %s/%s after error: %s",
                attempt + 1,
                max_retries,
                e,
            )
            time.sleep(delay)
            continue

    if not generated_ascii:
        logging.error(
            "Task H Turn 1: Failed to generate ASCII after retries: %s", last_exception
        )
        return {
            "messages": messages,
            "output": {"error": f"Turn 1 failed: {last_exception}"},
            "turn1_output": None,
            "message_metadata": {
                "turn1": turn1_metadata,
                "turn2": {},
                "error": str(last_exception),
            },
        }

    # Turn 2: Replace generated ASCII with GROUND TRUTH ASCII, but keep conversational format
    # Get ground truth ASCII from test case
    ground_truth_ascii = tc.get("ascii", {})
    if not ground_truth_ascii:
        logging.error("Task H Turn 2: Test case missing ground truth ASCII")
        return {
            "messages": messages,
            "output": {"error": "Test case missing ground truth ASCII"},
            "turn1_output": generated_ascii,
            "message_metadata": {
                "turn1": turn1_metadata,
                "turn2": {},
                "error": "Test case missing ground truth ASCII",
            },
        }

    # Extract ASCII views matching the requested format
    ground_truth_ascii_dict = {}
    if isinstance(ground_truth_ascii, dict):
        for view in output_views:
            art = ground_truth_ascii.get(view)
            if art:
                ground_truth_ascii_dict[view] = art
    else:
        # If ASCII is a string, assume it's the requested view
        if output_views:
            ground_truth_ascii_dict[output_views[0]] = ground_truth_ascii

    if not ground_truth_ascii_dict:
        logging.error("Task H Turn 2: No matching ASCII views found in ground truth")
        return {
            "messages": messages,
            "output": {"error": "No matching ASCII views found in ground truth"},
            "turn1_output": generated_ascii,
            "message_metadata": {
                "turn1": turn1_metadata,
                "turn2": {},
                "error": "No matching ASCII views found in ground truth",
            },
        }

    # Replace Turn 1 assistant response with ground truth ASCII
    # The Turn 1 assistant response is the last message in the array (we just added it)
    # Replace it with ground truth ASCII to maintain conversational format
    ground_truth_ascii_response = {"ascii": ground_truth_ascii_dict}
    messages[-1]["content"] = json_dumps_compact(ground_truth_ascii_response)
    logging.debug("Task H Turn 2: Replaced generated ASCII with ground truth ASCII")

    # Add Turn 2 user message (same conversational format as Task G)
    messages.append(
        {
            "role": "user",
            "content": f"Now, based on the ASCII visualization you generated, answer the question: {query}",
        }
    )

    # Call model for turn 2
    for attempt in range(max_retries):
        try:
            api_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }

            if reasoning_effort:
                api_params["reasoning_effort"] = reasoning_effort

            completion = call_with_retry(
                client.chat.completions.create,
                max_retries,
                delay,
                **api_params,
            )
            message = completion.choices[0].message
            message_content = message.content

            raw = message_content
            if not raw or not raw.strip():
                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content and reasoning_content.strip():
                    raw = reasoning_content
                else:
                    reasoning = getattr(message, "reasoning", None)
                    if reasoning and reasoning.strip():
                        raw = reasoning

            if not raw or not raw.strip():
                finish_reason = getattr(
                    completion.choices[0], "finish_reason", "unknown"
                )
                error_msg = f"Turn 2: Model returned empty/None content (finish_reason: {finish_reason})"
                logging.warning(
                    "Task H Turn 2: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = ValueError(error_msg)
                time.sleep(delay)
                continue

            try:
                parsed_turn2 = json.loads(raw)
            except json.JSONDecodeError as json_err:
                error_msg = f"Turn 2: Failed to parse JSON response: {str(json_err)[:200]}. Raw content: {str(raw)[:200]}"
                logging.warning(
                    "Task H Turn 2: Retry %s/%s - %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = json_err
                time.sleep(delay)
                continue

            turn2_metadata = {
                "content": message.content,
                "reasoning": getattr(message, "reasoning", None),
                "reasoning_content": getattr(message, "reasoning_content", None),
                "role": message.role,
                "refusal": message.refusal,
                "finish_reason": getattr(completion.choices[0], "finish_reason", None),
            }

            if hasattr(completion, "usage") and completion.usage:
                turn2_metadata["usage"] = {
                    "completion_tokens": completion.usage.completion_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }

            logging.debug(
                "Task H Turn 2: Successfully generated answer using ground truth ASCII"
            )

            content_for_conversation = (
                message_content if message_content and message_content.strip() else raw
            )
            messages.append({"role": "assistant", "content": content_for_conversation})

            result = {
                "messages": messages,
                "output": parsed_turn2,
                "turn1_output": generated_ascii,
                "message_metadata": {
                    "turn1": turn1_metadata,
                    "turn2": turn2_metadata,
                },
            }

            return result

        except Exception as e:
            last_exception = e
            logging.warning(
                "Task H Turn 2: Retry %s/%s after error: %s",
                attempt + 1,
                max_retries,
                e,
            )
            time.sleep(delay)
            continue

    logging.error("Task H Turn 2: Failed after retries: %s", last_exception)
    return {
        "messages": messages,
        "output": {"error": f"Turn 2 failed: {last_exception}"},
        "turn1_output": generated_ascii,
        "message_metadata": {
            "turn1": turn1_metadata,
            "turn2": {},
            "error": str(last_exception),
        },
    }


def run_model_case(
    client: OpenAI,
    model: str,
    tc: Dict[str, Any],
    few_shot_examples: List[Dict[str, Any]],
    *,
    ascii_format_input: Optional[Iterable[str]] = None,
    ascii_format_output: Optional[Iterable[str]] = None,
    prompt_mode: str = "simple",
    few_shot_mode: str = "system",
    ascii_order: str = "answer_first",
    ascii_input: Optional[Dict[str, Any]] = None,
    task: str = "C",
    max_retries: int = 5,
    delay: float = 2,
    max_tokens: int = 2048,
    timeout: float = 300.0,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    messages = build_messages(
        question=tc.get("query_relation", ""),
        description=tc.get("description", ""),
        few_shot_examples=few_shot_examples,
        ascii_format_input=ascii_format_input,
        ascii_format_output=ascii_format_output,
        prompt_mode=prompt_mode,
        few_shot_mode=few_shot_mode,
        ascii_order=ascii_order,
        ascii_input=ascii_input,
        task=task,
    )

    last_exception = None
    for attempt in range(max_retries):
        try:
            # Build API call parameters
            api_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }

            # Add reasoning_effort for reasoning models (like gpt-oss)
            if reasoning_effort:
                api_params["reasoning_effort"] = reasoning_effort
                logging.info("Using reasoning_effort=%s", reasoning_effort)

            logging.debug(
                "API params: model=%s, max_tokens=%s, reasoning_effort=%s",
                model,
                max_tokens,
                api_params.get("reasoning_effort", "None"),
            )

            completion = call_with_retry(
                client.chat.completions.create,
                max_retries,
                delay,
                **api_params,
            )
            message = completion.choices[0].message
            raw = message.content

            # Check if content is None or empty - try alternative fields
            if not raw:
                # Some models (like gpt-oss) may use reasoning_content instead
                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content and reasoning_content.strip():
                    raw = reasoning_content
                    logging.info("Using reasoning_content instead of content")
                else:
                    # Try reasoning field
                    reasoning = getattr(message, "reasoning", None)
                    if reasoning and reasoning.strip():
                        raw = reasoning
                        logging.info("Using reasoning instead of content")

            # If still None or empty, log and retry
            if not raw or not raw.strip():
                finish_reason = getattr(
                    completion.choices[0], "finish_reason", "unknown"
                )
                content_val = message.content
                reasoning_val = getattr(message, "reasoning", None)
                reasoning_content_val = getattr(message, "reasoning_content", None)

                error_msg = f"Model returned empty/None content (finish_reason: {finish_reason})"
                logging.warning(
                    "run_model_case: Retry %s/%s - %s. Field values: content='%s', reasoning='%s', reasoning_content='%s'",
                    attempt + 1,
                    max_retries,
                    error_msg,
                    content_val[:50] if content_val else None,
                    reasoning_val[:50] if reasoning_val else None,
                    reasoning_content_val[:50] if reasoning_content_val else None,
                )
                last_exception = ValueError(error_msg)
                time.sleep(delay)
                continue

        except Exception as e:
            last_exception = e
            logging.warning(
                "run_model_case: Retry %s/%s after model error: %s",
                attempt + 1,
                max_retries,
                e,
            )
            time.sleep(delay)
            continue

        try:
            parsed = json.loads(raw)
            result = {"messages": messages, "output": parsed}

            # Capture all message metadata (reasoning, etc.)
            message_metadata = {
                "content": message.content,
                "reasoning": getattr(message, "reasoning", None),
                "reasoning_content": getattr(message, "reasoning_content", None),
                "role": message.role,
                "refusal": message.refusal,
                "finish_reason": getattr(completion.choices[0], "finish_reason", None),
            }

            # Add token usage if available
            if hasattr(completion, "usage") and completion.usage:
                message_metadata["usage"] = {
                    "completion_tokens": completion.usage.completion_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }

            result["message_metadata"] = message_metadata

            if message_metadata["reasoning"]:
                logging.debug(
                    "Captured reasoning content (%d chars)",
                    len(message_metadata["reasoning"]),
                )

            return result
        except Exception as e:
            last_exception = e
            # Log first 500 chars of raw response to help debug
            if raw and raw.strip():
                raw_preview = (raw[:500] + "...") if len(raw) > 500 else raw
            else:
                raw_preview = f"<Empty or None: repr={repr(raw)}>"

            logging.warning(
                "run_model_case: Retry %s/%s after JSON error: %s. Raw response preview: %s",
                attempt + 1,
                max_retries,
                e,
                raw_preview,
            )
            time.sleep(delay)
            continue

    logging.error("Final failure after retries in run_model_case: %s", last_exception)
    return {
        "messages": messages,
        "output": {"error": str(last_exception)},
        "message_metadata": {
            "content": None,
            "reasoning": None,
            "reasoning_content": None,
            "role": None,
            "refusal": None,
            "finish_reason": None,
            "error": str(last_exception),
        },
    }


def save_results(results: List[Dict[str, Any]], result_dir: str, task: str) -> None:
    """Save all results to a JSONL file."""
    os.makedirs(result_dir, exist_ok=True)
    results_path = os.path.join(result_dir, f"task_{task.lower()}_results.jsonl")
    with open(results_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    logging.info("Saved results to %s", results_path)


def append_result(result_entry: Dict[str, Any], result_dir: str, task: str) -> None:
    """Append a single result to the JSONL file and flush immediately."""
    os.makedirs(result_dir, exist_ok=True)
    results_path = os.path.join(result_dir, f"task_{task.lower()}_results.jsonl")
    with open(results_path, "a") as f:
        f.write(json.dumps(result_entry) + "\n")
        f.flush()  # Force write to disk immediately
    logging.debug(
        "Appended result for test_id=%s to %s",
        result_entry.get("test_id"),
        results_path,
    )


def append_reasoning(
    reasoning_entry: Dict[str, Any], result_dir: str, task: str
) -> None:
    """Append reasoning and message metadata to a separate JSONL file."""
    os.makedirs(result_dir, exist_ok=True)
    reasoning_path = os.path.join(result_dir, f"task_{task.lower()}_reasoning.jsonl")
    with open(reasoning_path, "a") as f:
        f.write(json.dumps(reasoning_entry) + "\n")
        f.flush()  # Force write to disk immediately
    logging.debug(
        "Appended reasoning for test_id=%s to %s",
        reasoning_entry.get("test_id"),
        reasoning_path,
    )


def load_existing_test_ids(result_dir: str, task: str) -> set[str]:
    """Load existing test IDs from results file to enable resume functionality."""
    results_path = os.path.join(result_dir, f"task_{task.lower()}_results.jsonl")
    existing_test_ids: set[str] = set()

    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        test_id = data.get("test_id")
                        if test_id:
                            existing_test_ids.add(test_id)
                    except json.JSONDecodeError:
                        continue
            if existing_test_ids:
                logging.info(
                    "Found %s existing results in %s",
                    len(existing_test_ids),
                    results_path,
                )
        except Exception as e:
            logging.warning("Error reading existing results file: %s", e)

    return existing_test_ids


def configure_logging(result_dir: str, task: str) -> None:
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, f"run_task_{task.lower()}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info("Logging initialized. Log file: %s", log_path)


def build_openai_client(token_path: Optional[str], base_url: str) -> OpenAI:
    """Build OpenAI client, handling both remote (with token) and local vllm (no token) setups."""
    # Check if this is a local vllm endpoint
    is_local = "localhost" in base_url or "127.0.0.1" in base_url

    if is_local:
        # Local vllm uses dummy API key
        api_key = "dummy"
        logging.info("OpenAI client initialized with local vllm endpoint: %s", base_url)
    else:
        # Remote endpoint requires a token: --token-path or OPENAI_API_KEY
        if token_path:
            with open(token_path, "r") as f:
                api_key = f.read().strip()
        elif os.environ.get("OPENAI_API_KEY"):
            api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError(
                "Remote endpoint requires --token-path or the OPENAI_API_KEY environment variable"
            )
        logging.info("OpenAI client initialized with remote endpoint: %s", base_url)

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment for a given task")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to spatial dataset JSONL",
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=DEFAULT_FEW_SHOT,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--num-cases", type=int, default=DEFAULT_NUM_CASES, help="Number of test cases"
    )
    parser.add_argument(
        "--ascii-views",
        type=str,
        default="grid",
        help="Comma-separated list of ASCII views to use (e.g., grid,panel).",
    )
    parser.add_argument(
        "--result-dir", type=str, default=None, help="Directory to save results/logs"
    )
    parser.add_argument(
        "--results-base-dir",
        type=str,
        default=DEFAULT_RESULTS_BASE_DIR,
        help=f"Base directory under which to create structured results (default: {DEFAULT_RESULTS_BASE_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3-sdsc",
        help="Model name as expected by the endpoint (e.g. the Hugging Face path for local vLLM)",
    )
    parser.add_argument(
        "--token-path",
        type=str,
        default=None,
        help="Path to a file holding the API key (falls back to OPENAI_API_KEY; not needed for local vLLM)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("A", "B", "C", "D", "E", "F", "G", "H"),
        default="B",
        help="Task type to run: A (description→ASCII), B (ASCII→description), C (description+query→answer), D (ASCII+query→answer), E (description+query→ASCII+answer), F (description+ASCII+query→answer), G (description→ASCII [turn1], then description+generated-ASCII+query→answer [turn2]), H (description→ASCII [turn1], then description+ground-truth-ASCII+query→answer [turn2])",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=("simple", "detailed"),
        default="detailed",
        help="Prompt mode: 'simple' or 'detailed'",
    )
    parser.add_argument(
        "--few-shot-mode",
        type=str,
        choices=("system", "user", "conversational"),
        default="conversational",
        help="Few-shot mode: 'system', 'user', or 'conversational'",
    )
    parser.add_argument(
        "--ascii-order",
        type=str,
        choices=("answer_first", "ascii_first"),
        default="answer_first",
        help="ASCII order in output: 'answer_first' or 'ascii_first'",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for API calls (default: 300.0)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=("low", "medium", "high"),
        default=None,
        help="Reasoning effort for reasoning models like gpt-oss (low/medium/high). Lower = faster, less tokens.",
    )
    args = parser.parse_args()

    ascii_views = normalize_ascii_views(args.ascii_views)
    ascii_views_dir = format_views_for_dir(ascii_views)

    # Sanitize model name for use in file path (replace / and other special chars)
    model_name_safe = args.model.replace("/", "_").replace("\\", "_").replace(" ", "_")

    # For task E (which outputs both ASCII and answer), include ascii_order in the directory structure
    # Task A outputs only ASCII, so it doesn't need ascii_order in the path
    # Task G only outputs answer (ASCII is intermediate), so it doesn't need ascii_order either
    if args.task == "E":
        default_result_dir = os.path.join(
            args.results_base_dir,
            model_name_safe,
            args.task,
            ascii_views_dir,
            args.ascii_order,
            str(args.num_cases),
        )
    else:
        default_result_dir = os.path.join(
            args.results_base_dir,
            model_name_safe,
            args.task,
            ascii_views_dir,
            str(args.num_cases),
        )

    result_dir = args.result_dir if args.result_dir else default_result_dir

    configure_logging(result_dir, args.task)

    logging.info("Running task %s", args.task)
    logging.info("Saving outputs to %s", result_dir)
    logging.info("Loading dataset from %s", args.data_path)
    records = load_records(args.data_path)
    logging.info("Loaded %s samples", len(records))

    random.seed(args.random_seed)
    random.shuffle(records)

    few_shot_examples = select_few_shot_examples(records, args.few_shot)
    few_shot_indices = {records.index(sample) for sample in few_shot_examples}
    logging.info("Selected %s few-shot examples", len(few_shot_examples))

    # Tasks that require ASCII input: B, D, F, H (H needs ground truth ASCII for Turn 2)
    requires_ascii_input = args.task in ("B", "D", "F", "H")
    test_cases = select_test_cases(
        records, few_shot_indices, args.num_cases, require_ascii=requires_ascii_input
    )
    logging.info("Selected %s test cases", len(test_cases))

    # Check for existing results to enable resume functionality
    existing_test_ids = load_existing_test_ids(result_dir, args.task)
    if existing_test_ids:
        # Filter out already processed test cases
        original_count = len(test_cases)
        test_cases = [tc for tc in test_cases if tc.get("id") not in existing_test_ids]
        skipped_count = original_count - len(test_cases)
        if skipped_count > 0:
            logging.info(
                "Resuming: Skipping %s already processed test cases", skipped_count
            )
            logging.info("Remaining test cases to process: %s", len(test_cases))
        if len(test_cases) == 0:
            logging.info("All test cases have already been processed. Exiting.")
            return

    # Build run configuration
    run_config = {
        "data_path": args.data_path,
        "model": args.model,
        "token_path": args.token_path,
        "base_url": args.base_url,
        "task": args.task,
        "prompt_mode": args.prompt_mode,
        "few_shot_mode": args.few_shot_mode,
        "ascii_order": args.ascii_order,
        "random_seed": args.random_seed,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "reasoning_effort": args.reasoning_effort,
        # Processed/computed values
        "ascii_views": ascii_views,
        "ascii_views_dir": ascii_views_dir,
        "result_dir": result_dir,
        "results_base_dir": args.results_base_dir,
        "dataset_total_records": len(records),
        "few_shot_count": len(few_shot_examples),
        "test_case_count": len(test_cases),
    }

    run_config_path = os.path.join(result_dir, "run_config.json")
    with open(run_config_path, "w") as config_file:
        json.dump(run_config, config_file, indent=2)
    logging.info("Saved run configuration to %s", run_config_path)

    client = build_openai_client(args.token_path, args.base_url)

    results: List[Dict[str, Any]] = []

    prompt_dumped = False

    for idx, tc in enumerate(test_cases):
        logging.info(
            "Starting to process case %s/%s | test_id=%s",
            idx + 1,
            len(test_cases),
            tc.get("id"),
        )
        ascii_block = tc.get("ascii")

        # Handle Task G and H separately (multi-turn)
        if args.task == "G":
            # Task G: Generate ASCII first (Turn 1), then use it to answer (Turn 2)
            ascii_output_views = ascii_views

            task_config = dict(
                ascii_format_output=ascii_output_views,
                prompt_mode=args.prompt_mode,
                few_shot_mode=args.few_shot_mode,
                ascii_order=args.ascii_order,
                task=args.task,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
            )

            res_task = run_model_case_task_g(
                client, args.model, tc, few_shot_examples, **task_config
            )

            result_entry = {
                "test_id": tc.get("id"),
                "model_output": res_task.get("output"),
                "turn1_output": res_task.get("turn1_output"),
            }
        elif args.task == "H":
            # Task H: Generate ASCII first (Turn 1), then use GROUND TRUTH ASCII to answer (Turn 2)
            ascii_output_views = ascii_views

            task_config = dict(
                ascii_format_output=ascii_output_views,
                prompt_mode=args.prompt_mode,
                few_shot_mode=args.few_shot_mode,
                ascii_order=args.ascii_order,
                task=args.task,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
            )

            res_task = run_model_case_task_h(
                client, args.model, tc, few_shot_examples, **task_config
            )

            result_entry = {
                "test_id": tc.get("id"),
                "model_output": res_task.get("output"),
                "turn1_output": res_task.get("turn1_output"),
            }

        else:
            # Regular single-turn tasks
            # Set ascii_format_input based on task requirements
            # Tasks B, D, F require ASCII input
            ascii_input_views = ascii_views if args.task in ("B", "D", "F") else None

            # Set ascii_format_output based on task requirements
            # Tasks A and E output ASCII, so we need to specify the views
            ascii_output_views = ascii_views if args.task in ("A", "E") else None

            # Set ascii_input based on task requirements
            # Tasks B, D, F require ASCII input
            ascii_input_data = ascii_block if args.task in ("B", "D", "F") else None

            task_config = dict(
                ascii_format_input=ascii_input_views,
                ascii_format_output=ascii_output_views,
                prompt_mode=args.prompt_mode,
                few_shot_mode=args.few_shot_mode,
                ascii_order=args.ascii_order,
                ascii_input=ascii_input_data,
                task=args.task,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
            )

            res_task = run_model_case(
                client, args.model, tc, few_shot_examples, **task_config
            )

            result_entry = {
                "test_id": tc.get("id"),
                "model_output": res_task.get("output"),
            }

        results.append(result_entry)

        # Save result immediately to prevent data loss
        append_result(result_entry, result_dir, args.task)

        # Extract and save message metadata (reasoning, content, etc.) to separate file
        message_metadata = res_task.get("message_metadata", {})
        if message_metadata:
            # Handle Task G and H (multi-turn) separately
            if args.task in ("G", "H"):
                # Task G and H have nested turn1/turn2 metadata
                reasoning_entry = {
                    "test_id": tc.get("id"),
                    "turn1": message_metadata.get("turn1"),
                    "turn2": message_metadata.get("turn2"),
                    "error": message_metadata.get("error"),
                }
            else:
                # Single-turn tasks have flat metadata structure
                reasoning_entry = {
                    "test_id": tc.get("id"),
                    "content": message_metadata.get("content"),
                    "reasoning": message_metadata.get("reasoning"),
                    "reasoning_content": message_metadata.get("reasoning_content"),
                    "role": message_metadata.get("role"),
                    "refusal": message_metadata.get("refusal"),
                    "finish_reason": message_metadata.get("finish_reason"),
                    "usage": message_metadata.get("usage"),
                    "error": message_metadata.get("error"),
                }
            append_reasoning(reasoning_entry, result_dir, args.task)

        logging.info(
            "Processed case %s/%s | test_id=%s",
            idx + 1,
            len(test_cases),
            tc.get("id"),
        )

        if not prompt_dumped:
            # For Task G and H, save both Turn 1 and Turn 2 prompts
            if args.task in ("G", "H"):
                all_messages = res_task.get("messages", [])

                # Find the test case messages (last occurrences, after few-shot examples)
                # Turn 1: system + few-shot examples + test case user message (waiting for response)
                # Turn 2: system + few-shot examples + turn1 full (user + assistant) + turn2 user message (waiting for response)

                turn1_test_user_idx = -1
                turn1_test_assistant_idx = -1
                turn2_test_user_idx = -1

                # Find test case messages (last occurrences)
                for i in range(len(all_messages) - 1, -1, -1):
                    msg = all_messages[i]
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if (
                            "Please generate an ASCII" in content
                            and turn1_test_user_idx == -1
                        ):
                            turn1_test_user_idx = i
                        elif (
                            "Now, based on the ASCII visualization" in content
                            or "ASCII Reference:" in content
                        ) and turn2_test_user_idx == -1:
                            # Task G uses "Now, based on the ASCII visualization", Task H uses "ASCII Reference:"
                            turn2_test_user_idx = i

                # Find turn 1 assistant response (after turn 1 test user message)
                if turn1_test_user_idx >= 0:
                    for i in range(turn1_test_user_idx + 1, len(all_messages)):
                        if all_messages[i].get("role") == "assistant":
                            turn1_test_assistant_idx = i
                            break

                # Turn 1 prompt: up to and including turn 1 test user message (waiting for response)
                # This includes: system + few-shot examples + test case user message
                if turn1_test_user_idx >= 0:
                    turn1_messages = all_messages[: turn1_test_user_idx + 1]
                else:
                    # Fallback: if we can't find it, use all messages
                    turn1_messages = all_messages

                # Turn 2 prompt: full conversation including turn 1 + turn 2 test user message (waiting for response)
                # This includes: system + few-shot examples + turn1 full (user + assistant) + turn2 user message
                if turn2_test_user_idx >= 0:
                    turn2_messages = all_messages[: turn2_test_user_idx + 1]
                elif turn1_test_assistant_idx >= 0:
                    # Turn 2 user message might not be added yet, include up to turn 1 assistant
                    turn2_messages = all_messages[: turn1_test_assistant_idx + 1]
                else:
                    # Fallback: include all messages
                    turn2_messages = all_messages

                # Save Turn 1 prompt
                prompt_path_turn1 = os.path.join(
                    result_dir, f"task_{args.task.lower()}_prompt_turn1.txt"
                )
                with open(prompt_path_turn1, "w") as prompt_file:
                    prompt_file.write(format_prompt(turn1_messages))
                logging.info("Saved Turn 1 prompt to %s", prompt_path_turn1)

                # Save Turn 2 prompt
                prompt_path_turn2 = os.path.join(
                    result_dir, f"task_{args.task.lower()}_prompt_turn2.txt"
                )
                with open(prompt_path_turn2, "w") as prompt_file:
                    prompt_file.write(format_prompt(turn2_messages))
                logging.info("Saved Turn 2 prompt to %s", prompt_path_turn2)
            else:
                # Regular single-turn tasks
                prompt_path = os.path.join(
                    result_dir, f"task_{args.task.lower()}_prompt.txt"
                )
                with open(prompt_path, "w") as prompt_file:
                    prompt_file.write(format_prompt(res_task.get("messages", [])))
                logging.info("Saved example prompt to %s", prompt_path)

            prompt_dumped = True

    # Final save to ensure all results are written (redundant but safe)
    logging.info(
        "All results have been saved incrementally. Final count: %s", len(results)
    )


if __name__ == "__main__":
    main()
