import json
from typing import Callable, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global device variable - always use CUDA
DEVICE = torch.device("cuda")


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_few_shot_examples(
    data: List[Dict], num_examples: int = 5, exclude_ids: Optional[List[str]] = None
) -> List[Dict]:
    """Extract few-shot examples from the dataset.

    Args:
        data: List of test cases
        num_examples: Number of few-shot examples to extract
        exclude_ids: List of test IDs to exclude from few-shot examples

    Returns:
        List of few-shot example dictionaries
    """
    if exclude_ids is None:
        exclude_ids = []

    # Filter out excluded IDs
    available = [item for item in data if item.get("id") not in exclude_ids]

    # Take the first num_examples
    return available[:num_examples]


def load_model(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a saved model and tokenizer from the given path.

    Args:
        model_path: Path to the saved model directory
        device_map: Device mapping strategy (default: "auto")
        torch_dtype: Data type for model weights (default: "auto")

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set padding side to 'left' for decoder-only models (required for batched inference)
    tokenizer.padding_side = "left"

    # Convert torch_dtype string to actual dtype
    if torch_dtype == "auto":
        dtype = "auto"
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=dtype,
    )

    model.eval()
    print("Model loaded successfully!")

    return model, tokenizer


def chat_completion(
    model: AutoModelForCausalLM,
    messages: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Perform chat completion given a list of messages.

    Args:
        model: The loaded model
        messages: List of message dictionaries with 'role' and 'content' keys
                  Example: [
                      {"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": "Hello!"}
                  ]
        tokenizer: The loaded tokenizer
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling (False for greedy decoding)

    Returns:
        The assistant's response as a string
    """
    # Apply chat template to format the messages
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize the input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
    )

    # Move inputs to the global DEVICE
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    model.eval()
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache for faster generation
        )

    # Decode the generated tokens
    # First decode WITHOUT skipping special tokens to match input_text format
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    while generated_text.startswith("<|endoftext|>"):
        generated_text = generated_text[len("<|endoftext|>") :].strip()

    # Extract only the assistant's response (remove the input prompt)
    if generated_text.startswith(input_text):
        assistant_response = generated_text[len(input_text) :].strip()
    else:
        assistant_response = generated_text.strip()

    # Remove trailing <|im_end|> token if present
    if assistant_response.endswith("<|im_end|>"):
        assistant_response = assistant_response[: -len("<|im_end|>")].strip()

    return assistant_response


def batch_chat_completion(
    model,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> List[str]:
    """Perform batched chat completion for multiple message sequences.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        messages_list: List of message sequences to process
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling

    Returns:
        List of assistant responses
    """
    # Apply chat template to all messages
    input_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_list
    ]

    # Tokenize all inputs with padding
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Move inputs to the global DEVICE
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate responses
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,  # Enable KV cache for faster generation
    )

    # Decode all generated tokens
    responses = []
    for i, output in enumerate(outputs):
        # First decode WITHOUT skipping special tokens to match input_text format
        generated_text = tokenizer.decode(output, skip_special_tokens=False)

        while generated_text.startswith("<|endoftext|>"):
            generated_text = generated_text[len("<|endoftext|>") :].strip()

        # Extract only the assistant's response (remove the input prompt)
        input_text = input_texts[i]
        if generated_text.startswith(input_text):
            assistant_response = generated_text[len(input_text) :].strip()
        else:
            assistant_response = generated_text.strip()

        # Remove trailing <|im_end|> token if present
        while assistant_response.endswith("<|im_end|>") or assistant_response.endswith(
            "<|endoftext|>"
        ):
            if assistant_response.endswith("<|im_end|>"):
                assistant_response = assistant_response[: -len("<|im_end|>")].strip()
            if assistant_response.endswith("<|endoftext|>"):
                assistant_response = assistant_response[: -len("<|endoftext|>")].strip()

        responses.append(assistant_response)

    return responses


def process_inference_batch(
    batch_messages: List[List[Dict[str, str]]],
    batch_ids: List[str],
    model,
    tokenizer,
    output_jsonl_path: str,
    results: List[Dict],
    batch_start: int,
    result_formatter: Callable[[str], Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> None:
    """Process a batch of inference requests and save results.

    This function encapsulates the common pattern of:
    1. Running batch or single inference based on batch size
    2. Processing responses with a custom formatter
    3. Saving results incrementally to JSONL
    4. Handling errors gracefully

    Args:
        batch_messages: List of message sequences for inference
        batch_ids: List of test IDs corresponding to each message sequence
        model: The loaded model
        tokenizer: The loaded tokenizer
        output_jsonl_path: Path to the output JSONL file
        results: List to append results to
        batch_start: Starting index of this batch (for error reporting)
        result_formatter: Callback function that takes a response string and returns
                         the model_output dict (e.g., {"answer": response} or {"ascii": {"grid": response}})
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
    """
    try:
        # Run batched inference
        if len(batch_messages) > 1:
            responses = batch_chat_completion(
                model=model,
                tokenizer=tokenizer,
                messages_list=batch_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        else:
            # Single sample - use simple_chat
            response = chat_completion(
                model=model,
                tokenizer=tokenizer,
                messages=batch_messages[0],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            responses = [response]

        # Save results for this batch
        for test_id, response in zip(batch_ids, responses):
            result_entry = {
                "test_id": test_id,
                "model_output": result_formatter(response),
            }
            results.append(result_entry)

            # Save incrementally - APPEND mode to avoid O(N²) file I/O
            with open(output_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Error processing batch starting at {batch_start}: {e}")
        # Save error entries for all items in the batch
        for test_id in batch_ids:
            result_entry = {
                "test_id": test_id,
                "model_output": {"error": str(e)},
            }
            results.append(result_entry)
            with open(output_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
