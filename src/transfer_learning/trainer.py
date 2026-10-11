"""Training loop and evaluation for Task A fine-tuning.

This module implements the training loop using HuggingFace Trainer
with custom callbacks and evaluation logic.
"""

import os
from typing import Any, Dict, Optional

import torch
from transformers import (
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from .logger import TrainingLogger
from .utils import cleanup_checkpoints


class LoggingCallback(TrainerCallback):
    """Callback for custom logging during training."""

    def __init__(self, logger: TrainingLogger):
        """Initialize callback.

        Args:
            logger: Custom logger instance
        """
        self.logger = logger

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.logger.log("Training started")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        self.logger.log("Training completed")

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        epoch = int(state.epoch) if state.epoch is not None else 0
        self.logger.log(f"Epoch {epoch} started")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        epoch = int(state.epoch) if state.epoch is not None else 0
        self.logger.log(f"Epoch {epoch} completed")


class CheckpointCleanupCallback(TrainerCallback):
    """Callback to cleanup old checkpoints."""

    def __init__(self, keep_last_n: int = 3):
        """Initialize callback.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        self.keep_last_n = keep_last_n

    def on_save(self, args, state, control, **kwargs):
        """Called after saving a checkpoint."""
        cleanup_checkpoints(
            output_dir=args.output_dir,
            keep_last_n=self.keep_last_n,
            checkpoint_prefix="checkpoint",
        )


def create_training_arguments(config) -> SFTConfig:
    """Create TrainingArguments from config.

    Args:
        config: Configuration object

    Returns:
        TrainingArguments instance
    """
    # Prepare save strategy arguments
    save_strategy = config.training.save_strategy
    save_kwargs = {}

    if save_strategy == "steps" and config.training.save_steps is not None:
        save_kwargs["save_steps"] = config.training.save_steps

    # When load_best_model_at_end is True, eval_strategy must match save_strategy
    # If save_strategy is "epoch", we need to use eval_strategy="epoch" as well
    # Otherwise, use "steps" for both
    if config.training.load_best_model_at_end:
        eval_strategy = save_strategy
    else:
        # If not loading best model, we can keep eval_strategy as "steps" for more frequent evaluation
        eval_strategy = "steps"

    sft_kwargs = {}
    if config.data.max_seq_length is not None:
        sft_kwargs["max_length"] = config.data.max_seq_length
    if config.training.gradient_checkpointing:
        sft_kwargs["gradient_checkpointing"] = True
        sft_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return SFTConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_grad_norm=config.training.max_grad_norm,
        # Optimizer and scheduler
        optim=config.training.optimizer,
        lr_scheduler_type=config.training.lr_scheduler_type,
        # Evaluation - must match save_strategy when load_best_model_at_end is True
        eval_strategy=eval_strategy,
        eval_steps=config.training.eval_steps if eval_strategy == "steps" else None,
        save_strategy=save_strategy,
        logging_steps=config.training.logging_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        # Mixed precision
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        # Other settings
        dataloader_num_workers=config.training.dataloader_num_workers,
        remove_unused_columns=config.training.remove_unused_columns,
        report_to=["tensorboard"] if config.logging.use_tensorboard else [],
        logging_dir=(
            config.logging.tensorboard_dir if config.logging.use_tensorboard else None
        ),
        # Disable default logging to avoid conflicts
        disable_tqdm=False,
        logging_first_step=True,
        completion_only_loss=config.training.completion_only_loss,
        packing=config.training.packing,
        **sft_kwargs,
        **save_kwargs,
    )


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config,
    custom_logger: Optional[TrainingLogger] = None,
):
    """Create trainer instance.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Configuration object
        custom_logger: Custom logger instance

    Returns:
        Trainer instance (SFTTrainer or TaskATrainer)
    """
    # Create training arguments
    training_args = create_training_arguments(config)

    # Create callbacks
    callbacks = []
    if custom_logger is not None:
        callbacks.append(LoggingCallback(custom_logger))
    callbacks.append(
        CheckpointCleanupCallback(keep_last_n=config.training.save_total_limit)
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Add custom logger to SFTTrainer if provided
    if custom_logger is not None:
        trainer.custom_logger = custom_logger

    return trainer


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config,
    custom_logger: Optional[TrainingLogger] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Train the model.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Configuration object
        custom_logger: Custom logger instance
        resume_from_checkpoint: Path to checkpoint to resume from

    Returns:
        Training metrics dictionary
    """
    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        custom_logger=custom_logger,
    )

    # Log training info
    if custom_logger is not None:
        custom_logger.log(f"Training dataset size: {len(train_dataset)}")
        custom_logger.log(f"Evaluation dataset size: {len(eval_dataset)}")
        custom_logger.log(f"Number of epochs: {config.training.num_epochs}")
        custom_logger.log(f"Batch size: {config.training.batch_size}")
        custom_logger.log(
            f"Gradient accumulation steps: {config.training.gradient_accumulation_steps}"
        )
        custom_logger.log(
            f"Effective batch size: {config.training.batch_size * config.training.gradient_accumulation_steps}"
        )
        custom_logger.log(f"Learning rate: {config.training.learning_rate}")

    # Train
    if custom_logger is not None:
        custom_logger.log("Starting training...")

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    if custom_logger is not None:
        custom_logger.log("Saving final model...")

    trainer.save_model(config.training.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    if custom_logger is not None:
        custom_logger.log("Training completed successfully")
        custom_logger.log_metrics(metrics, prefix="final_train_")

    return metrics


def evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    config,
    custom_logger: Optional[TrainingLogger] = None,
) -> Dict[str, Any]:
    """Evaluate the model.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        config: Configuration object
        custom_logger: Custom logger instance

    Returns:
        Evaluation metrics dictionary
    """
    # Create trainer for evaluation
    training_args = create_training_arguments(config)

    # Use SFTTrainer for evaluation
    if custom_logger is not None:
        custom_logger.log("Using SFTTrainer for evaluation (chat format)")

    # SFTTrainer requires a train_dataset even for evaluation-only
    # We pass eval_dataset as train_dataset to satisfy the requirement
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Evaluate
    if custom_logger is not None:
        custom_logger.log("Starting evaluation...")

    metrics = trainer.evaluate()

    # Save evaluation metrics
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    if custom_logger is not None:
        custom_logger.log("Evaluation completed")
        custom_logger.log_metrics(metrics, prefix="eval_")

    return metrics


def generate_predictions(
    model,
    tokenizer,
    dataset,
    config,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    custom_logger: Optional[TrainingLogger] = None,
) -> list:
    """Generate predictions on a dataset.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer
        dataset: Dataset to generate predictions for
        config: Configuration object
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        custom_logger: Custom logger instance

    Returns:
        List of predictions
    """
    if custom_logger is not None:
        custom_logger.log(f"Generating predictions for {len(dataset)} samples...")

    model.eval()
    predictions = []

    device = next(model.parameters()).device

    for i, sample in enumerate(dataset):
        # Chat format: apply chat template to get the prompt (without assistant response)
        # Extract only system and user messages (exclude assistant response)
        messages = sample["messages"]
        prompt_messages = [msg for msg in messages if msg["role"] != "assistant"]

        # Apply chat template to create the input prompt
        input_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Get reference from the assistant message
        reference_text = next(
            (msg["content"] for msg in messages if msg["role"] == "assistant"),
            sample.get("ascii", ""),
        )

        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=config.data.max_description_length,
            truncation=True,
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract only the generated part (remove input)
        if generated_text.startswith(input_text):
            generated_text = generated_text[len(input_text) :].strip()
        # Remove trailing <|im_end|> token if present
        if generated_text.endswith("<|im_end|>"):
            generated_text = generated_text[: -len("<|im_end|>")].strip()

        predictions.append(
            {
                "record_id": sample.get("record_id", f"sample_{i}"),
                "input": input_text,
                "prediction": generated_text,
                "reference": reference_text,
            }
        )

        if (i + 1) % 10 == 0 and custom_logger is not None:
            custom_logger.log(f"Generated {i + 1}/{len(dataset)} predictions")

    if custom_logger is not None:
        custom_logger.log("Prediction generation completed")

    return predictions
