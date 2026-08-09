
# Text2Space: A Benchmark Pairing Spatial Descriptions, ASCII Layouts, and Spatial QA

Code and data for our COLM 2026 paper - Learning to Draw ASCII Improves Spatial Reasoning in Language Models

**Shiyuan Huang\*, Li Liu\*, Jincheng He, Leilani H. Gilpin** — University of California, Santa Cruz
*\*Equal contribution*

[Paper](https://arxiv.org/abs/2604.14641) · [Dataset](https://huggingface.co/datasets/ShiyuanHuang/Text2Space)

## About

LLMs can read spatial layouts but struggle to construct them. We introduce **Text2Space**, a dataset pairing natural-language spatial descriptions with ground-truth ASCII grid layouts and spatial QA pairs, which lets us separate failures in building a spatial representation from failures in reasoning over one. Training models on layout construction (Desc→ASCII) improves spatial reasoning from text alone, even when no ASCII is produced at inference, and the gains transfer to StepGame, bAbI Task 19, and SpartQA.

## Dataset

Available now on Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("ShiyuanHuang/Text2Space")
```

## Code

**Coming soon.** We are preparing the data generation, evaluation, and fine-tuning scripts for release. Please watch this repo for updates. Thanks!


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
