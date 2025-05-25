[![arXiv](https://img.shields.io/badge/arXiv-2505.01855-b31b1b.svg)](https://arxiv.org/abs/2505.01855)

# Intra-Layer Recurrence in Transformers for Language Modeling

The code for the paper: [Intra-Layer Recurrence in Transformers for Language Modeling](https://arxiv.org/abs/2505.01855)
Accepted at CanadianAI 2025.

## Abstract

Transformer models have established new benchmarks in natural language processing; however, their increasing depth results in substantial growth in parameter counts. While existing recurrent transformer methods address this issue by reprocessing layers multiple times, they often apply recurrence indiscriminately across entire blocks of layers. In this work, we investigate Intra-Layer Recurrence (ILR), a more targeted approach that applies recurrence selectively to individual layers within a single forward pass. Our experiments show that allocating more iterations to earlier layers yields optimal results. These findings suggest that ILR offers a promising direction for optimizing recurrent structures in transformer architectures.

![Figure 1](https://raw.githubusercontent.com/ant-8/Layer-Recurrent-Transformers/refs/heads/main/figure1.JPG)

## Dataset Setup
TODO

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

We use [Weights & Biases](https://wandb.ai/) for logging. Before running any experiments, make sure to log in:

```bash
wandb login
```

### Experiment Configs

Experiment configuration files are located in:

```
./experiment_setups/
```

To run an experiment:

```bash
python run_experiment.py --config ./experiment_setups/[CONFIG].json
```

You can also specify a device with the `--device` flag:

```bash
python run_experiment.py --config ./experiment_setups/example.json --device cuda:0
```

---

## Model Overview

### `modeling_llama.py`

Contains a modified version of the LLaMa architecture that supports **Intra-Layer Recurrence (ILR)**.

### Initializing an ILR-supported LLaMa Model

```python
from looped_llama_configuration import LoopedLlamaConfig
from modeling_llama import LoopedLlamaForCausalLM

config = LoopedLlamaConfig(
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    loop_map=[3, 2, 2, 1],
    vocab_size=32000,
    max_position_embeddings=1024,
    tie_word_embeddings=True,
    _attn_implementation="eager",
    positional_encoding="nope",
    use_cache=False
)

model = LoopedLlamaForCausalLM(config)
```

- `loop_map` corresponds to the **reuse map** defined in the paper.

---

## Notes

- We conducted preliminary experiments on **GPT-2-based architectures** and alternative positional encodings (**CoPE**, **FIRE**), but these were incomplete due to time and resource constraints and were not included in the paper.

- Due to architectural changes, the default `generate()` method from HF Transformers is not supported and has been overwritten with a naive implementation:
  - No KV-caching
  - Sampling only supports temperature-based decoding

## 📄 Citation

(citation coming soon after official proceedings publication).
