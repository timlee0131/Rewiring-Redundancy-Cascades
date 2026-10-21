# On the Runway Cascade of Transformers for Language Modeling

This is the official demo and 

The repository is deliberately small. The method is one attention class
([`model.py`](model.py)), and everything else is a conventional language-model
training and evaluation loop.

| File | Contents |
| --- | --- |
| `model.py` | RoPE, ALiBi, SwiGLU, the baseline `MultiHeadAttention`, and `RedundancyCascadeAttention` |
| `data.py` | Corpus tokenisation (in-memory or memory-mapped) and sliding-window datasets |
| `train.py` | Single-device training loop |
| `evaluate.py` | Perplexity for a saved checkpoint, plus attention-map dumps |
| `demo.py` | Small side-by-side comparison of both variants |

This is a demonstration of the mechanism, not an analysis suite. If you want to
study the method at scale, lift `RedundancyCascadeAttention` out of
[`model.py`](model.py) and drop it into your own transformer: it is a
self-contained `nn.Module` with the same `(output, attn, beta)` interface as the
baseline attention next to it.

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

Train both variants at a small scale and compare them. This streams 30M tokens
of TinyStories and takes 10-15 minutes on one GPU:

```bash
python demo.py
```

The tokenised corpus is cached under `./data`, so only the first run pays for
downloading and tokenising it.

It reports validation perplexity for the standard baseline and for the
redundancy cascade, each a ~16M-parameter model trained from the same seed on
the same batches. With the default dot-product scorer the cascade adds no
parameters, so the two models are parameter-matched by construction and differ
only in the attention mechanism.

TinyStories rather than wikitext-2 because of the parameter-to-token ratio: of
those 16M parameters, 12.9M are the tied GPT-2 embedding table and only 3.2M are
transformer weights, which is far more capacity than wikitext-2's 2.4M tokens can
train. A model that has only learned token frequencies tells you nothing about
its attention mechanism. For a fast smoke test rather than a result:

```bash
python demo.py --dataset wikitext --max_steps 200
```

## Training

```bash
# Baseline
python train.py --variant std --dataset wikitext --seq_len 1024

# Redundancy cascade
python train.py --variant redwire --scorer dot --dataset wikitext --seq_len 1024
```

Checkpoints go to `./checkpoints/best_model.pt` and carry their own
configuration, so evaluation does not need the architecture respecified.

For corpora too large to tokenise into memory, add `--use_memmap`, which streams
the corpus to a flat `uint16` file under `--data_dir` once and memory-maps it
thereafter.

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

Perplexity uses overlapping windows with overlap masking: with
`--eval_stride` smaller than `--eval_seq_len`, every token is scored exactly
once and with the maximum available left context.

```bash
# Evaluate at a different sequence length than the model was trained on
python evaluate.py --checkpoint ... --eval_seq_len 2048 --eval_stride 1024

# Save attention weights and gates for one sequence, for plotting
python evaluate.py --checkpoint ... --save_attention analysis/attn.pt
```

## Reproducing the paper

Published runs use C4 (English) and the following two sweeps. Both are run once
per variant, `--variant std` and `--variant redwire`.

Shared settings: `--dataset c4 --use_memmap --max_tokens 20000000000
--learning_rate 6e-4 --weight_decay 0.1 --dropout 0.1 --grad_clip 1.0
--max_steps 37500 --eval_interval 3750 --max_seq_len 4096`.

**Sequence length** (at `d_model 1536`, `n_layers 24`, `n_heads 24`):

| `--seq_len` | `--stride` | `--eval_seq_len` | `--eval_stride` | `--batch_size` |
| --- | --- | --- | --- | --- |
| 128 | 64 | 128 | 64 | 128 |
| 256 | 128 | 256 | 128 | 128 |
| 512 | 256 | 512 | 256 | 32 |
| 1024 | 512 | 1024 | 512 | 32 |
| 2048 | 1024 | 2048 | 1024 | 8 |

**Depth and width** (at `--seq_len 1024 --stride 512 --batch_size 32`):

| `--n_layers` | `--n_heads` | `--d_model` |
| --- | --- | --- |
| 8 | 8 | 512 |
| 12 | 12 | 768 |
| 16 | 16 | 1024 |
| 20 | 20 | 1280 |
| 24 | 24 | 1536 |

The `--batch_size` values above are **per GPU**. Published runs were
distributed with `--accumulation_steps 8`, giving an effective batch of
`batch_size * 8 * <number of GPUs>` sequences per optimiser step. The released
code is single-device, so to match the effective batch on one GPU set
`--accumulation_steps` to `8 * <number of GPUs>`.


## Citation

```bibtex
@inproceedings{
anonymous2026on,
title={On the Runway Cascade of Transformers for Language Modeling},
author={Lee, Hunjae and Clark, Corey},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=BG7L2BEqkO}
}
```