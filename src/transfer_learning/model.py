"""Model loading and PEFT configuration for Task A fine-tuning.

This module handles loading the Qwen model with optional PEFT/LoRA
for parameter-efficient fine-tuning.
"""

import os
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. Install with: pip install peft")


def load_tokenizer(
    model_name: str,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    """Load tokenizer for the model.

    Args:
        model_name: Name or path of the model
        cache_dir: Directory to cache the tokenizer
        trust_remote_code: Whether to trust remote code

    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    # Ensure tokenizer has necessary tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_base_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    use_flash_attention: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> AutoModelForCausalLM:
    """Load base model without PEFT.

    Args:
        model_name: Name or path of the model
        cache_dir: Directory to cache the model
        torch_dtype: Data type for model weights ("auto", "float16", "bfloat16", "float32")
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        use_flash_attention: Whether to use flash attention
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision

    Returns:
        Loaded model
    """
    # Convert torch_dtype string to actual dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, "auto")

    # Configure quantization if requested
    quantization_config = None
    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if load_in_4bit else None,
            bnb_4bit_use_double_quant=True if load_in_4bit else None,
            bnb_4bit_quant_type="nf4" if load_in_4bit else None,
        )

    # Model loading arguments
    model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "cache_dir": cache_dir,
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    return model


def setup_peft_model(
    model: AutoModelForCausalLM,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None,
    lora_bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> AutoModelForCausalLM:
    """Setup PEFT/LoRA for the model.

    Args:
        model: Base model to apply PEFT to
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        lora_target_modules: List of module names to apply LoRA to
        lora_bias: Bias handling ("none", "all", or "lora_only")
        task_type: Task type for PEFT

    Returns:
        Model with PEFT applied
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is required but not installed. Install with: pip install peft"
        )

    # Default target modules for Qwen models
    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    # Prepare model for k-bit training if quantized
    if hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
        model = prepare_model_for_kbit_training(model)
    elif hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias=lora_bias,
        task_type=TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else task_type,
    )

    # Apply PEFT
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def load_model_and_tokenizer(
    model_name: str,
    cache_dir: Optional[str] = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    use_flash_attention: bool = True,
    use_peft: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None,
    lora_bias: str = "none",
    task_type: str = "CAUSAL_LM",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with optional PEFT configuration.

    Args:
        model_name: Name or path of the model
        cache_dir: Directory to cache the model
        torch_dtype: Data type for model weights
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        use_flash_attention: Whether to use flash attention
        use_peft: Whether to use PEFT/LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        lora_target_modules: List of module names to apply LoRA to
        lora_bias: Bias handling
        task_type: Task type for PEFT
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = load_tokenizer(
        model_name=model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    print(f"Loading base model from {model_name}...")
    model = load_base_model(
        model_name=model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        use_flash_attention=use_flash_attention,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )

    if use_peft:
        print("Setting up PEFT/LoRA...")
        model = setup_peft_model(
            model=model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_bias=lora_bias,
            task_type=task_type,
        )

    return model, tokenizer


def load_model_from_config(config) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from configuration object.

    Args:
        config: Configuration object with model settings

    Returns:
        Tuple of (model, tokenizer)
    """
    return load_model_and_tokenizer(
        model_name=config.model.model_name,
        cache_dir=config.model.model_cache_dir,
        torch_dtype=config.model.torch_dtype,
        device_map=config.model.device_map,
        trust_remote_code=config.model.trust_remote_code,
        use_flash_attention=config.model.use_flash_attention,
        use_peft=config.model.use_peft,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        lora_target_modules=config.model.lora_target_modules,
        lora_bias=config.model.lora_bias,
        task_type=config.model.task_type,
        load_in_8bit=config.model.load_in_8bit,
        load_in_4bit=config.model.load_in_4bit,
    )


def save_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    save_full_model: bool = False,
):
    """Save model and tokenizer.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save to
        save_full_model: Whether to save full model or just adapter weights
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Save model
    if hasattr(model, "save_pretrained"):
        if save_full_model and hasattr(model, "merge_and_unload"):
            # Merge LoRA weights and save full model
            print("Merging LoRA weights and saving full model...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_dir)
        else:
            # Save adapter weights only (for PEFT models)
            print("Saving adapter weights...")
            model.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")
