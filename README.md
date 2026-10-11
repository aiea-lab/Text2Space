
# Text2Space: A Benchmark Pairing Spatial Descriptions, ASCII Layouts, and Spatial QA

Code and data for our COLM 2026 paper - Learning to Draw ASCII Improves Spatial Reasoning in Language Models

**Shiyuan Huang\*, Li Liu\*, Jincheng He, Leilani H. Gilpin** — University of California, Santa Cruz
*\*Equal contribution*

[Paper](https://arxiv.org/abs/2604.14641) · [Dataset](https://huggingface.co/datasets/ShiyuanHuang/Text2Space)

## About

LLMs can read spatial layouts but struggle to construct them. We introduce **Text2Space**, a dataset pairing natural-language spatial descriptions with ground-truth ASCII grid layouts and spatial QA pairs, which lets us separate failures in building a spatial representation from failures in reasoning over one. Training models on layout construction (Desc→ASCII) improves spatial reasoning from text alone, even when no ASCII is produced at inference, and the gains transfer to StepGame, bAbI Task 19, and SpartQA.

## Dataset

The full dataset (20,000 instances) is on Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("ShiyuanHuang/Text2Space")
```

or as a single file:

```bash
curl -L -o datasets/spatial_data.jsonl \
  https://huggingface.co/datasets/ShiyuanHuang/Text2Space/resolve/main/spatial_data.jsonl
```

Every field is documented in [`docs/SPATIAL_DATA_SCHEMA.md`](docs/SPATIAL_DATA_SCHEMA.md); the layout-uniqueness annotation is specified in [`docs/UNIQUENESS_ALGORITHM.md`](docs/UNIQUENESS_ALGORITHM.md).

The paper partitions the 20,000 instances by id into three files. The two small ones ship with this repository; the training file is derived from the full dataset.

| File | Instances | Role |
|------|-----------|------|
| `datasets/spatial_data_tested.jsonl` | 1,000 | Evaluation set behind every number in the paper |
| `datasets/spatial_data_fewshot.jsonl` | 5 | Few-shot demonstrations used in all prompts |
| `datasets/spatial_data_untested.jsonl` | 18,995 | Fine-tuning data; `scripts/split_dataset.py` writes it from the full file |

```bash
uv run python scripts/split_dataset.py --full datasets/spatial_data.jsonl
```

The evaluation and few-shot sets are exactly what `task_runner.py` selects from the full file with `--random-seed 42 --few-shot 5 --num-cases 1000` (step 2 below), so inference on the full file reproduces the paper's instances.

`datasets/sample_instance.json` is one instance for reference; `spatial_data_sample_10.jsonl` and `spatial_data_sample_50.jsonl` are small subsets for quick tests.

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) (Python 3.13 or newer):

```bash
uv sync
```

Run every command below from the repository root with `uv run python ...`. Fine-tuning and local vLLM serving need a Linux GPU machine; dataset generation, evaluation, and API-based inference run anywhere.

## Repository structure

```
Text2Space/
├── src/
│   ├── data/                    # Dataset generation
│   │   ├── spatial_map.py               # 2D spatial-relation engine, ASCII rendering, uniqueness
│   │   ├── spatial_data_generator.py    # Synthetic instance generator (balanced sampling)
│   │   ├── spatial_utils.py, spatial_constants.py
│   │   └── visualize_spatial_data_stats.py
│   ├── experiment/inference/    # Zero/few-shot inference through an OpenAI-compatible API
│   │   ├── LLMs/task_runner.py          # Driver for tasks A-H
│   │   └── llm_helpers.py               # Prompt construction
│   ├── evaluation/              # ascii_evaluator.py + per-task evaluators/
│   ├── benchmark.py             # Evaluate + analyze every task in a run directory
│   ├── analysis/                # Per-factor accuracy breakdowns
│   ├── dashboard/               # Aggregate metrics across runs into one JSON
│   ├── plot/plot_results.py     # Base vs fine-tuned comparison figure
│   ├── transfer_learning/       # LoRA/QLoRA fine-tuning
│   │   ├── train.py, config/            # Trainer + the paper's training configs
│   │   ├── vllm_model_transfer.py       # Merge an adapter for vLLM serving
│   │   └── task_inference/              # Fine-tuned-model inference + external benchmarks
│   └── test/                    # Unit tests for the spatial engine and evaluator
├── scripts/
│   ├── split_dataset.py         # Full dataset -> fine-tuning split
│   ├── benchmarks/              # Download + convert bAbI, StepGame, SpartQA
│   ├── train/                   # One launcher per fine-tuning run in the paper
│   └── backfill_reasoning_steps.py
├── datasets/                    # Evaluation split, few-shot set, samples
└── docs/                        # Data schema, uniqueness algorithm
```

## Reproduction pipeline

Stages connect through fixed file layouts: inference writes `results/<run>/<task>/<view>/<n>/task_<x>_results.jsonl`, `benchmark.py` reads those and writes `results/<run>/outputs/evaluation/`, and the aggregation and plotting steps read the evaluation JSON.

### 1. Generate the dataset

The released dataset was produced with:

```bash
uv run python src/data/spatial_data_generator.py \
  --num-instances 20000 --min-components 2 --max-components 8 \
  --min-hops 1 --max-hops 12 --seed 42 \
  --output datasets/spatial_data.jsonl
```

This writes the JSONL file, one JPG per instance under `datasets/images/` (skip with `--no-generate-images`), `sample_instance.json`, and a `*_stats.txt` summary. Balancing over component count, query type, relation count, and terminology is on by default; `--target-unique-ratio`, `--target-direct-ratio`, `--stratify-unique-by-components`, and the `--no-balance-*` flags control it. The current generator also emits a `reasoning_steps` field that the released file does not have; `scripts/backfill_reasoning_steps.py` adds it to an existing file.

Plot the distribution of any JSONL file:

```bash
uv run python src/data/visualize_spatial_data_stats.py \
  --input datasets/spatial_data.jsonl --output datasets/spatial_analysis \
  --stats --format png pdf
```

### 2. Run inference (tasks A-H)

`task_runner.py` talks to any OpenAI-compatible endpoint: the OpenAI API, or a local vLLM server for open-weight models. The eight tasks are:

| Task | Input | Output |
|------|-------|--------|
| A | description | ASCII layout |
| B | ASCII layout | description |
| C | description + query | direction |
| D | ASCII layout + query | direction |
| E | description + query | ASCII layout + direction |
| F | description + ASCII layout + query | direction |
| G | description (turn 1), then description + generated ASCII + query (turn 2) | direction |
| H | description (turn 1), then description + ground-truth ASCII + query (turn 2) | direction |

The paper's setting for every model and task:

```bash
uv run python src/experiment/inference/LLMs/task_runner.py \
  --task C --model <model-name> \
  --data-path datasets/spatial_data.jsonl \
  --base-url http://localhost:8000/v1 \
  --num-cases 1000 --few-shot 5 --ascii-views grid \
  --prompt-mode detailed --few-shot-mode conversational \
  --random-seed 42 --max-tokens 512 --results-base-dir results
```

Task E additionally takes `--ascii-order answer_first|ascii_first`; both settings appear in the paper. The generation budget differed by model: 512 tokens for the Llama-3 and Qwen3-235B runs, 512 or 1024 for Qwen3-30B, 2048 for the fine-tuned Qwen3-30B models, gpt-oss-20b, and Qwen2.5-7B, and 4096 for GPT-4.1 and GPT-5-mini. Outputs land in `results/<model>/<task>/grid/1000/` as `task_<x>_results.jsonl`, a `run_config.json` with every argument, the reasoning trace, and the rendered prompts. Rerunning the same command resumes from the ids already in the results file.

For a hosted API, pass `--base-url https://api.openai.com/v1` and either `--token-path <file containing the key>` or set `OPENAI_API_KEY`. A `localhost` endpoint needs no key. Reasoning models accept `--reasoning-effort`.

### 3. Evaluate

```bash
# One run
uv run python src/benchmark.py --results-dir results/<model> --dataset datasets/spatial_data.jsonl

# Every run under a root
uv run python src/benchmark.py --results-root results --dataset datasets/spatial_data.jsonl
```

`benchmark.py` finds every `task_<x>_results.jsonl` below the run directory, evaluates it, and writes `outputs/evaluation/task_<x>_evaluation.json` plus `evaluation_summary.json`; it then breaks accuracy down by dataset factor into `outputs/analysis/factor_analysis.json`. `--evaluate-only`, `--analyze-only`, `--task c --task e`, `--no-save`, and `--with-factor-plots` narrow or extend the run.

How each task is scored:

| Task | Metric | Method |
|------|--------|--------|
| A | `accuracy`, `average_instance_accuracy` | Generated ASCII must realize every relation stated in the description; the second metric gives partial credit per relation |
| B | `accuracy`, `average_instance_accuracy` | Generated description must cover the relations in the ground-truth layout |
| C, D, F, H | `accuracy` | Predicted direction matches the label; synonyms, cardinal, and clock phrasings are accepted |
| E, G | `accuracy`, `avg_ascii_accuracy`, `consistency_rate` | Answer correctness, relation accuracy of the generated ASCII, and whether the ASCII implies the model's own answer |

The three ASCII evaluation modes (all pairwise relations, stated relations only, single queried relation) live in `src/evaluation/ascii_evaluator.py`, which also works as a standalone command (`--help`). Factor analysis reports accuracy by `num_components`, `num_relations`, `query_type`, `terminology_used`, `is_directly_stated`, `has_unique_layout`, and `ambiguous_stages`.

### 4. Aggregate and plot

```bash
# Factor breakdown for one run (also produced by benchmark.py)
uv run python src/analysis/run_analyzer.py --run-dir results/<model>

# Every evaluated run under results/ into one JSON
uv run python -m src.dashboard.cli --results-dir results --output outputs/dashboard_data.json

# Base vs fine-tuned comparison figure (tasks C, E, G, H)
uv run python src/plot/plot_results.py \
  --base-dir results/<base-run> --finetune-dir results/<fine-tuned-run> --output-dir outputs/plots
```

### 5. Fine-tune

Training is config-driven. The configs in `src/transfer_learning/config/` are the runs in the paper; `scripts/train/` holds a one-line launcher for each.

| Config | Base model | Training task |
|--------|-----------|---------------|
| `config_task_a_completion_only.json` | Qwen3-30B-A3B-Instruct-2507 | A: description → ASCII |
| `config_task_b_completion_only.json` | Qwen3-30B-A3B-Instruct-2507 | B: ASCII → description |
| `config_task_mixed_completion_only.json` | Qwen3-30B-A3B-Instruct-2507 | 50/50 mix of A and B |
| `config_task_a_random_format.json` | Qwen3-30B-A3B-Instruct-2507 | A with a random ASCII style per sample |
| `config_task_a.json`, `config_task_b.json`, `config_task_mixed.json` | Qwen3-30B-A3B-Instruct-2507 | Same tasks with loss on the full sequence |
| `config_task_{a,b,mixed}_llama70b_completion_only.json` | Meta-Llama-3-70B-Instruct (8-bit QLoRA) | A, B, mixed |

All configs read `datasets/spatial_data_untested.jsonl`, use 5,000 samples, LoRA rank 16, one epoch, and write to `saved_models/<run>/`. The Llama-3-8B runs use the `llama70b` configs with `model_name` set to `meta-llama/Meta-Llama-3-8B-Instruct` and `load_in_8bit` removed; everything else (attention-projection adapters, learning rate 1e-4, effective batch size 16) is unchanged.

```bash
bash scripts/train/train_task_a_completion_only.sh
# equivalent to
uv run python src/transfer_learning/train.py \
  --config src/transfer_learning/config/config_task_a_completion_only.json
```

`train.py` also accepts `--eval_only`, `--generate_predictions`, `--resume_from_checkpoint`, and overrides such as `--model_name`, `--data_path`, `--num_epochs`, `--batch_size`, `--learning_rate`, `--max_samples`, `--seed`.

To evaluate a fine-tuned model on tasks A-H, merge the adapter, serve it with vLLM, and run step 2 against the local endpoint:

```bash
uv run python src/transfer_learning/vllm_model_transfer.py \
  --base-model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --adapter-path saved_models/task_a_completion_only_fine_tune/final_model \
  --output-path saved_models/task_a_merged
uv run vllm serve saved_models/task_a_merged --port 8000
```

`task_inference/task_A_inference.py` and `task_C_inference.py` run tasks A and C directly with `transformers` instead of a server (`--model_path`, `--test_data_path`, `--output_dir`, `--few_shot_count`, `--prompt_mode`, `--batch_size`); their outputs are evaluated with `benchmark.py` exactly as in step 3.

### 6. External benchmarks

The paper tests transfer on bAbI Task 19, StepGame, and SpartQA. The benchmark files are not redistributed here; the scripts below download the public sources and write the exact files used in the paper: bAbI `en-valid` task 19 from the ParlAI mirror; the 1,000-instance SpaRP StepGame PS2 test set from `UKPLab/sparp`; the 3,594 SpartQA choose-object test questions from `tasksource/spartqa-mchoice` with answer sets from `RAR-b/spartqa`, reduced to the first question of each story.

```bash
uv run python scripts/benchmarks/prepare_babi.py       # datasets/babi_task19_pathfinding.json
uv run python scripts/benchmarks/prepare_stepgame.py   # datasets/sparp_stepgame.json
uv run python scripts/benchmarks/prepare_spartqa.py    # datasets/spartqa.json, spartqa_unique.json
```

Then run inference through any OpenAI-compatible endpoint; each script writes results, a `run_config.json`, and a `summary.json` with accuracy:

```bash
uv run python src/transfer_learning/task_inference/babi_inference.py \
  --base_url http://localhost:8000/v1 --model <model> --output_dir results/<model>/babi_task19
uv run python src/transfer_learning/task_inference/sparp_inference.py \
  --base_url http://localhost:8000/v1 --model <model> --output_dir results/<model>/sparp_stepgame
uv run python src/transfer_learning/task_inference/spartqa_inference.py \
  --base_url http://localhost:8000/v1 --model <model> --output_dir results/<model>/spartqa
```

Sources: bAbI (Weston et al., 2015), StepGame (Shi et al., 2022) as packaged in SpaRP (Rizvi et al., 2024), SpartQA (Mirzaee et al., 2021) as packaged by tasksource and RAR-b (Xiao et al., 2024).

## Tests

```bash
uv run python src/test/test_ascii_evaluator.py                 # evaluator suite
uv run python src/test/test_comprehensive_uniqueness.py        # uniqueness algorithm
uv run python src/test/test_vertical_horizontal_uniqueness.py  # component-wise uniqueness
uv run python src/test/test_reasoning_steps.py                 # chain-of-thought steps
```

## Citation

```bibtex
@article{huang2026learning,
  title   = {Learning to Draw {ASCII} Improves Spatial Reasoning in Language Models},
  author  = {Huang, Shiyuan and Liu, Li and He, Jincheng and Gilpin, Leilani H.},
  journal = {arXiv preprint arXiv:2604.14641},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.14641}
}
```
Accepted at COLM 2026. We will update this entry with the proceedings citation once available.
