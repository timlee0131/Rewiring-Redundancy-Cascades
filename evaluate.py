"""Evaluate a trained checkpoint.

    python evaluate.py --checkpoint checkpoints/best_model.pt
    python evaluate.py --checkpoint ... --eval_seq_len 2048 --eval_stride 1024

The architecture is read from the checkpoint; any flag you pass explicitly
overrides it, which is how you evaluate at a sequence length the model was not
trained on.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import EvalSlidingWindowDataset, load_tokens, open_tokens, prepare_memmap
from train import build_model, build_parser, evaluate

# Architecture flags that must match the checkpoint for the weights to load.
ARCH_KEYS = (
    "variant",
    "scorer",
    "alpha",
    "beta_min",
    "global_summary",
    "placebo_mode",
    "no_rope",
    "vocab_size",
    "d_model",
    "n_layers",
    "n_heads",
    "max_seq_len",
)


def load_checkpoint(path, config, device, explicit_flags):
    """Rebuild the model from the checkpoint, letting explicit CLI flags win."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved = checkpoint.get("config") or {}

    for key in ARCH_KEYS:
        if key in saved and key not in explicit_flags:
            setattr(config, key, saved[key])

    if not saved:
        print(
            "Checkpoint carries no config; using command-line architecture flags. "
            "They must match how the model was trained."
        )

    model = build_model(config)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Tolerate checkpoints saved from a DistributedDataParallel wrapper.
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model.to(device).eval()

    print(f"Loaded {path}")
    print(
        f"  variant: {config.variant}"
        + (f" (scorer {config.scorer})" if config.variant == "redwire" else "")
    )
    print(f"  parameters: {model.count_parameters():,}")
    if "step" in checkpoint:
        print(f"  trained for {checkpoint['step']:,} steps")
    if "val_ppl" in checkpoint:
        print(f"  checkpoint val ppl: {checkpoint['val_ppl']:.2f}")

    return model


def explicitly_passed(argv, parser):
    """Names of options actually present on the command line.

    Used so that a flag given on the command line overrides the architecture
    recorded in the checkpoint, while everything else is inherited from it.
    """
    passed = set()
    for action in parser._actions:
        if action.dest == "help":
            continue
        for opt in action.option_strings:
            if opt in argv or any(arg.startswith(f"{opt}=") for arg in argv):
                passed.add(action.dest)
                break
    return passed


def main():
    parser = build_parser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--save_attention",
        type=str,
        default=None,
        help="path to save attention weights and gates for one sequence",
    )

    config = parser.parse_args()
    explicit = explicitly_passed(sys.argv[1:], parser)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    model = load_checkpoint(config.checkpoint, config, device, explicit)

    eval_dataset_name = config.eval_dataset or config.dataset
    eval_seq_len = config.eval_seq_len or config.seq_len
    eval_stride = config.eval_stride or eval_seq_len

    if config.use_memmap:
        tokens = open_tokens(
            prepare_memmap(
                eval_dataset_name, "validation", config.max_eval_tokens, config.data_dir
            )
        )
    else:
        tokens, _ = load_tokens(
            eval_dataset_name, "validation", config.max_eval_tokens, config.data_dir
        )

    dataset = EvalSlidingWindowDataset(tokens, eval_seq_len, eval_stride)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"\nEvaluating on {eval_dataset_name} [validation]: {len(dataset):,} windows "
        f"of {eval_seq_len} tokens (stride {eval_stride})"
    )

    nll, ppl = evaluate(model, loader, device)
    print(f"\nValidation loss {nll:.4f} | perplexity {ppl:.2f}")

    if config.save_attention:
        # One sequence only: the maps are (n_layers, n_heads, L, L).
        input_ids, _ = next(iter(loader))
        input_ids = input_ids[:1].to(device)
        with torch.no_grad():
            _, attns = model(input_ids, is_causal=True, return_attn=True)

        save_path = Path(config.save_attention)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(
            {
                "variant": config.variant,
                "scorer": config.scorer if config.variant == "redwire" else None,
                "input_ids": input_ids.cpu(),
                "attn": [a.cpu() for a, _ in attns],
                "beta": [b.cpu() if b is not None else None for _, b in attns],
            },
            save_path,
        )
        print(f"\nSaved attention maps for {len(attns)} layers to {save_path}")


if __name__ == "__main__":
    main()
