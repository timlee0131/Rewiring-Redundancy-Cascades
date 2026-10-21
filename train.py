"""Train a standard or Redundancy-Cascade transformer for language modelling.

Single-device training loop. Examples:

    python train.py --variant std
    python train.py --variant redwire --scorer dot

See README.md for the configurations used in the paper.
"""

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from data import DATASETS, build_loaders
from model import ATTENTION_VARIANTS, RedundancyCascadeAttention, Transformer

# Perplexity of a uniform distribution over GPT-2's vocabulary is ~50k; any
# loss above this is a diverged run and exp() would overflow the report.
MAX_REPORTED_NLL = 20.0


def perplexity(nll):
    return math.exp(min(nll, MAX_REPORTED_NLL))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Token-averaged negative log-likelihood and perplexity on masked targets."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (input_ids, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Reported perplexity is computed in fp32, not in the bfloat16 used for
        # training, so that the metric does not depend on the training dtype.
        logits = model(input_ids, is_causal=True)

        # Context-only positions are -100 in the dataset and ignored here.
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += (targets != -100).sum().item()

    if was_training:
        model.train()

    if total_tokens == 0:
        return float("inf"), float("inf")

    avg_nll = total_loss / total_tokens
    return avg_nll, perplexity(avg_nll)


def param_groups(model, weight_decay):
    """Split parameters into decayed and non-decayed groups.

    Matrices (embeddings, projections, the LM head) are decayed; LayerNorm
    gains and biases are not. Decaying a LayerNorm gain pulls it toward zero,
    which fights the normalisation rather than regularising anything.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train(
    model,
    train_loader,
    eval_loader,
    device,
    num_epochs=1,
    max_steps=None,
    learning_rate=6e-4,
    weight_decay=0.1,
    grad_clip=1.0,
    accumulation_steps=1,
    eval_interval=500,
    eval_batches=None,
    log_interval=50,
    save_dir=None,
    config=None,
    quiet=False,
):
    """Train ``model`` in place. Returns the best (nll, ppl) seen on the eval set.

    ``config`` is stored alongside the weights so that ``evaluate.py`` can
    rebuild the architecture without being told about it again.
    """
    model.to(device)
    model.train()

    optimizer = AdamW(
        param_groups(model, weight_decay),
        lr=learning_rate,
        betas=(0.9, 0.95),  # GPT-3 style
    )

    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    total_steps = max(1, total_steps)  # the scheduler divides by this

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.01),
        num_training_steps=total_steps,
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    best_nll = float("inf")
    best_ppl = float("inf")
    step = 0
    done = False

    def log(msg):
        if not quiet:
            print(msg)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        if done:
            break

        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids, is_causal=True)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )

            (loss / accumulation_steps).backward()

            last_in_epoch = (batch_idx + 1) == len(train_loader)
            if (batch_idx + 1) % accumulation_steps != 0 and not last_in_epoch:
                continue

            # bfloat16 has the dynamic range of fp32, so no gradient scaler.
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % log_interval == 0:
                batch_loss = loss.item()
                log(
                    f"epoch {epoch + 1} | step {step}/{total_steps} | "
                    f"loss {batch_loss:.4f} | ppl {perplexity(batch_loss):8.2f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e}"
                )

            if step % eval_interval == 0 or step >= total_steps:
                nll, ppl = evaluate(model, eval_loader, device, max_batches=eval_batches)
                marker = ""

                if nll < best_nll:
                    best_nll, best_ppl = nll, ppl
                    marker = "  *best*"
                    if save_dir is not None:
                        torch.save(
                            {
                                "step": step,
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "val_nll": nll,
                                "val_ppl": ppl,
                                "config": dict(config) if config else None,
                            },
                            save_dir / "best_model.pt",
                        )

                log(f"  eval @ step {step}: nll {nll:.4f} | ppl {ppl:.2f}{marker}")

            if step >= total_steps:
                log(f"Reached {total_steps} optimisation steps.")
                done = True
                break

    return best_nll, best_ppl


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    # Model
    parser.add_argument("--variant", type=str, default="std", choices=ATTENTION_VARIANTS)
    parser.add_argument(
        "--scorer",
        type=str,
        default="dot",
        choices=list(RedundancyCascadeAttention.SCORERS),
        help="redundancy scorer (redwire only)",
    )
    parser.add_argument("--alpha", type=float, default=5.0, help="gate sharpness (redwire only)")
    parser.add_argument("--beta_min", type=float, default=0.05, help="gate floor (redwire only)")
    parser.add_argument("--no_rope", action="store_true", help="use ALiBi instead of RoPE")

    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--d_model", type=int, default=1536)
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--n_heads", type=int, default=24)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Data
    parser.add_argument("--dataset", type=str, default="tinystories", choices=DATASETS)
    parser.add_argument("--eval_dataset", type=str, default=None, choices=DATASETS)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--eval_seq_len", type=int, default=None)
    parser.add_argument("--eval_stride", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=10_000_000)
    parser.add_argument("--max_eval_tokens", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--use_memmap",
        action="store_true",
        help="stream the corpus to disk and memory-map it (needed above ~1B tokens)",
    )

    # Optimisation
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accumulation_steps", type=int, default=1)

    # Bookkeeping
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    return parser


def build_model(config):
    attn_kwargs = {}
    if config.variant == "redwire":
        attn_kwargs = dict(
            scorer=config.scorer,
            alpha=config.alpha,
            beta_min=config.beta_min
        )

    return Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        tie_weights=True,
        variant=config.variant,
        use_rope=not config.no_rope,
        **attn_kwargs,
    )


def main():
    config = build_parser().parse_args()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device)
    print("Config:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")

    train_loader, eval_loader = build_loaders(
        dataset_name=config.dataset,
        eval_dataset_name=config.eval_dataset,
        seq_len=config.seq_len,
        stride=config.stride,
        eval_seq_len=config.eval_seq_len,
        eval_stride=config.eval_stride,
        batch_size=config.batch_size,
        max_tokens=config.max_tokens,
        max_eval_tokens=config.max_eval_tokens,
        num_workers=config.num_workers,
        cache_dir=config.data_dir,
        use_memmap=config.use_memmap,
    )

    model = build_model(config)
    print(f"\n{config.variant} model: {model.count_parameters():,} parameters\n")

    best_nll, best_ppl = train(
        model,
        train_loader,
        eval_loader,
        device,
        num_epochs=config.num_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        accumulation_steps=config.accumulation_steps,
        eval_interval=config.eval_interval,
        log_interval=config.log_interval,
        save_dir=config.save_dir,
        config=vars(config),
    )

    print(f"\nBest validation loss {best_nll:.4f} | perplexity {best_ppl:.2f}")
    print(f"Checkpoint: {Path(config.save_dir) / 'best_model.pt'}")


if __name__ == "__main__":
    main()
