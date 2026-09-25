import argparse
import os
from typing import List, Optional, Union

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install with: pip install vllm")


def merge_lora_weights(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: Union[torch.dtype, str] = torch.bfloat16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> None:
    """Merge LoRA adapter weights with base model for vLLM deployment.

    Args:
        base_model_name: Name or path of the base model (e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507")
        adapter_path: Path to the saved LoRA adapter weights
        output_path: Path where the merged model will be saved
        torch_dtype: Data type for model weights. Can be torch.dtype (e.g., torch.bfloat16)
                     or string "auto" to auto-detect from model config (default: torch.bfloat16)
        device_map: Device mapping strategy (default: "auto")
        trust_remote_code: Whether to trust remote code (default: True)

    Raises:
        ImportError: If PEFT is not installed
        FileNotFoundError: If adapter_path doesn't exist
    """

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    print("=" * 80)
    print("Merging LoRA Weights for vLLM Deployment")
    print("=" * 80)
    print(f"Base model: {base_model_name}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output path: {output_path}")
    print(f"Torch dtype: {torch_dtype}")
    print("=" * 80)

    # Load base model
    print("\n[1/4] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    print(f"✓ Base model loaded: {base_model.__class__.__name__}")

    # Load LoRA adapter
    print("\n[2/4] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print(f"✓ LoRA adapter loaded from: {adapter_path}")

    # Merge weights
    print("\n[3/4] Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    print("✓ LoRA weights merged successfully")

    # Save merged model
    print(f"\n[4/4] Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    print("✓ Merged model saved")

    # Save tokenizer
    print("\nSaving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(output_path)
    print("✓ Tokenizer saved")

    print("\n" + "=" * 80)
    print("✓ Merge complete! Model ready for vLLM deployment.")
    print(f"✓ Merged model location: {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter weights with base model for vLLM deployment"
    )

    # Required arguments
    parser.add_argument(
        "--base-model-name",
        type=str,
        required=True,
        help="Name or path of the base model (e.g., 'Qwen/Qwen3-30B-A3B-Instruct-2507')",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the saved LoRA adapter weights",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path where the merged model will be saved",
    )

    # Optional arguments
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help="Data type for model weights (default: 'auto')",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping strategy (default: 'auto')",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Whether to trust remote code (default: True)",
    )

    args = parser.parse_args()

    merge_lora_weights(
        base_model_name=args.base_model_name,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
