"""Minimal side-by-side comparison of standard and Redundancy-Cascade attention.

Trains both variants from the same seed on the same batches of TinyStories at a
small scale, then reports validation perplexity for each. Takes 10-15 minutes on
one GPU:

    python demo.py


This is the same code path as the full-scale runs in train.py, only smaller.
See README.md for the configurations used in the paper.
"""

import argparse
import math
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from data import DATASETS, build_loaders
from model import Transformer
from train import train

DEMO_CONFIG = dict(
    d_model=384,
    n_layers=6,
    n_heads=6,
    seq_len=1024,
    batch_size=64,
    dropout=0.1,
    learning_rate=6e-4,
    max_steps=1500,
    eval_interval=500,
)


def build_demo_model(variant, config, vocab_size, seed):
    # Same seed for both variants. With the default dot-product scorer the
    # cascade adds no parameters, so both start from identical weights and
    # differ only in the attention mechanism.
    torch.manual_seed(seed)
    np.random.seed(seed)

    return Transformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        max_seq_len=config.seq_len,
        dropout=config.dropout,
        tie_weights=True,
        variant=variant,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="tinystories", choices=DATASETS)
    parser.add_argument("--max_tokens", type=int, default=100_000_000)
    parser.add_argument("--max_eval_tokens", type=int, default=1_000_000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    for key, value in DEMO_CONFIG.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    config = parser.parse_args()

    device = torch.device(config.device)
    if device.type == "cpu":
        print(
            "No GPU detected. This will be slow; consider "
            "--max_steps 100 --max_tokens 1000000 for a smoke test.\n"
        )

    # Both variants see identical data in an identical order.
    torch.manual_seed(config.seed)
    train_loader, eval_loader = build_loaders(
        dataset_name=config.dataset,
        seq_len=config.seq_len,
        stride=config.seq_len,
        eval_seq_len=config.seq_len,
        # Overlapping eval windows, so the average scored token is predicted
        # with ~3/4 of a window of left context instead of ~1/2.
        eval_stride=config.seq_len // 4,
        batch_size=config.batch_size,
        max_tokens=config.max_tokens,
        max_eval_tokens=config.max_eval_tokens,
        num_workers=config.num_workers,
        # Cache the tokenised corpus under ./data, so only the first run pays
        # for downloading and tokenising it.
        use_memmap=True,
    )

    vocab_size = 50257  # GPT-2

    # --max_steps is the binding constraint; epochs are whatever it takes to
    # reach it. Over 30M tokens of TinyStories the default does not complete
    # even one pass, so nothing is seen twice.
    num_epochs = max(1, math.ceil(config.max_steps / len(train_loader)))
    tokens_per_step = config.batch_size * config.seq_len
    print(
        f"\n{config.max_steps} steps x {tokens_per_step:,} tokens/step = "
        f"{config.max_steps * tokens_per_step:,} tokens "
        f"({config.max_steps / len(train_loader):.2f} passes over the corpus)"
    )

    results = {}

    for variant in ("std", "redwire"):
        label = "standard attention" if variant == "std" else "redundancy cascade"
        print(f"\n{'=' * 64}\n{label} ({variant})\n{'=' * 64}")

        model = build_demo_model(variant, config, vocab_size, config.seed)
        print(f"{model.count_parameters():,} parameters\n")

        # Reseed so dropout and data shuffling are identical across variants.
        torch.manual_seed(config.seed)
        nll, ppl = train(
            model,
            train_loader,
            eval_loader,
            device,
            num_epochs=num_epochs,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            eval_interval=config.eval_interval,
            log_interval=config.eval_interval,
            save_dir=None,
        )
        results[variant] = (nll, ppl)

    print(f"\n{'=' * 64}\nResults after {config.max_steps} steps on {config.dataset}\n{'=' * 64}")
    print(f"{'variant':<24}{'val loss':<12}{'val ppl':<12}")
    for variant, (nll, ppl) in results.items():
        label = "standard" if variant == "std" else "redundancy cascade"
        print(f"{label:<24}{nll:<12.4f}{ppl:<12.2f}")

    delta = results["std"][1] - results["redwire"][1]
    direction = "lower" if delta > 0 else "higher"
    print(
        f"\nRedundancy cascade perplexity is {abs(delta):.2f} {direction} "
        f"than the standard baseline at this scale."
    )
    print(
        "Note: this is a small-scale demonstration, not the paper's result. "
        "See README.md for the full configurations."
    )


if __name__ == "__main__":
    main()
