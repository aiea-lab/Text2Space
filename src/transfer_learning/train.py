"""Main training script for Task A fine-tuning.

This script orchestrates the entire fine-tuning pipeline:
- Load configuration
- Setup logging
- Load model and tokenizer
- Prepare datasets
- Train model
- Evaluate and save results

Usage:
    python train.py --config config.json
    python train.py --data_path datasets/spatial_data.jsonl --output_dir outputs/my_run
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.transfer_learning.config import Config, get_default_config
from src.transfer_learning.dataset import prepare_datasets
from src.transfer_learning.logger import setup_logger_from_config
from src.transfer_learning.model import load_model_from_config, save_model
from src.transfer_learning.trainer import (
    evaluate_model,
    generate_predictions,
    train_model,
)
from src.transfer_learning.utils import save_predictions, set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model")

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file",
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to JSONL data file",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name or path",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate",
    )

    # Other arguments
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation, no training",
    )
    parser.add_argument(
        "--test_prediction_only",
        action="store_true",
        help="Only run test set predictions, no evaluation",
    )
    parser.add_argument(
        "--generate_predictions",
        action="store_true",
        help="Generate predictions on test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Disable tokenizers parallelism to avoid fork warnings
    # This prevents warnings when using tokenizers before multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Parse arguments
    args = parse_args()

    # Load or create configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = Config.load(args.config)
    else:
        print("Using default configuration")
        config = get_default_config()

    # Override config with command-line arguments
    if args.data_path:
        config.data.data_path = args.data_path
    if args.max_samples:
        config.data.max_samples = args.max_samples
    if args.model_name:
        config.model.model_name = args.model_name
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.seed:
        config.data.seed = args.seed

    # Set random seed
    set_seed(config.data.seed)
    print(f"Random seed set to {config.data.seed}")

    # Setup logging
    logger = setup_logger_from_config(config)
    logger.log("=" * 80)
    logger.log(f"Description: {config.description}")
    logger.log("=" * 80)

    # Save configuration
    config_path = os.path.join(config.training.output_dir, "config.json")
    config.save(config_path)
    logger.log(f"Configuration saved to {config_path}")

    # Log configuration
    logger.log_hyperparameters(config.to_dict())

    # Load model and tokenizer
    logger.log("Loading model and tokenizer...")
    model, tokenizer = load_model_from_config(config)
    logger.log_model_info(model)

    # Prepare datasets
    logger.log("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        data_path=config.data.data_path,
        task=config.data.task,
        ascii_view=config.data.ascii_view,
        max_description_length=config.data.max_description_length,
        require_ascii=config.data.require_ascii,
        min_num_components=config.data.min_num_components,
        max_num_components=config.data.max_num_components,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        max_samples=config.data.max_samples,
        seed=config.data.seed,
        prompt_completion_format=config.data.prompt_completion_format,
    )

    logger.log(f"Train dataset: {len(train_dataset)} samples")
    logger.log(f"Validation dataset: {len(val_dataset)} samples")
    logger.log(f"Test dataset: {len(test_dataset)} samples")

    # Training
    if not args.eval_only:
        logger.log("=" * 80)
        logger.log("Starting training")
        logger.log("=" * 80)

        train_metrics = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            config=config,
            custom_logger=logger,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

        logger.log("=" * 80)
        logger.log("Training completed")
        logger.log("=" * 80)

        # Save final model
        final_model_dir = os.path.join(config.training.output_dir, "final_model")
        logger.log(f"Saving final model to {final_model_dir}")
        save_model(model, tokenizer, final_model_dir)

    if not args.test_prediction_only:
        # Evaluation on test set
        logger.log("=" * 80)
        logger.log("Evaluating on test set")
        logger.log("=" * 80)

        test_metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=test_dataset,
            config=config,
            custom_logger=logger,
        )

        logger.log("Test set evaluation completed")
        logger.log_metrics(test_metrics, prefix="test_")

    # Generate predictions if requested
    if args.generate_predictions:
        logger.log("=" * 80)
        logger.log("Generating predictions on test set")
        logger.log("=" * 80)

        predictions = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            config=config,
            custom_logger=logger,
        )

        # Save predictions
        predictions_path = os.path.join(
            config.training.output_dir, "test_predictions.jsonl"
        )
        save_predictions(predictions, predictions_path, format="jsonl")
        logger.log(f"Predictions saved to {predictions_path}")

    # Finish logging
    logger.log("=" * 80)
    logger.log("All tasks completed successfully!")
    logger.log("=" * 80)
    logger.finish()

    print(f"\nTraining completed! Results saved to: {config.training.output_dir}")


if __name__ == "__main__":
    main()
