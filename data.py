"""Token loading and sliding-window datasets.

Small corpora (wikitext-2, TinyStories, Shakespeare) are tokenised straight
into memory. For the large C4 runs, ``prepare_memmap`` streams the corpus to a
flat ``uint16`` file on disk once, and ``open_tokens`` memory-maps it so that
training never holds the corpus in RAM. Both paths hand the datasets a plain
numpy array, so the datasets themselves do not care which one you used.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast

DATASETS = ("wikitext", "tinystories", "shakespeare", "c4")

# GPT-2's vocabulary is 50257 tokens, which fits in uint16.
TOKEN_DTYPE = np.uint16

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def get_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # We tokenise whole corpora, not model-length inputs; silence the warning.
    tokenizer.model_max_length = int(1e30)
    return tokenizer


# ---------------------------------------------------------------------------
# Corpus iterators
# ---------------------------------------------------------------------------


def _iter_texts(dataset_name, split, cache_dir="./data"):
    """Yield raw documents for a corpus and split."""
    from datasets import load_dataset as hf_load_dataset

    if dataset_name == "wikitext":
        for row in hf_load_dataset("wikitext", "wikitext-2-raw-v1", split=split):
            yield row["text"]

    elif dataset_name == "tinystories":
        # Streamed: the full corpus is ~2GB on disk and callers typically want
        # a few tens of millions of tokens from the front of it.
        for row in hf_load_dataset(
            "roneneldan/TinyStories", split=split, streaming=True
        ):
            yield row["text"]

    elif dataset_name == "shakespeare":
        import urllib.request

        cache_path = Path(cache_dir) / "shakespeare.txt"
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        if not cache_path.exists():
            print(f"Downloading Shakespeare to {cache_path}...")
            urllib.request.urlretrieve(SHAKESPEARE_URL, cache_path)

        text = cache_path.read_text(encoding="utf-8")
        # Single file with no official splits: hold out the last 10%.
        cut = int(len(text) * 0.9)
        yield text[:cut] if split == "train" else text[cut:]

    elif dataset_name == "c4":
        for row in hf_load_dataset(
            "allenai/c4", "en", split=split, streaming=True, trust_remote_code=True
        ):
            yield row["text"]

    else:
        raise ValueError(f"unknown dataset: {dataset_name} (expected one of {DATASETS})")


def _iter_token_chunks(texts, tokenizer, max_tokens, docs_per_chunk=1000):
    """Tokenise documents in batches, yielding uint16 arrays up to ``max_tokens``."""
    collected = 0
    buffer = []

    def flush(buffer):
        nonlocal collected
        tokens = tokenizer.encode("\n".join(buffer), add_special_tokens=False)
        if collected + len(tokens) > max_tokens:
            tokens = tokens[: max_tokens - collected]
        collected += len(tokens)
        return np.asarray(tokens, dtype=TOKEN_DTYPE)

    for text in texts:
        if collected >= max_tokens:
            break
        buffer.append(text)
        if len(buffer) >= docs_per_chunk:
            yield flush(buffer)
            buffer = []

    if buffer and collected < max_tokens:
        yield flush(buffer)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_tokens(dataset_name, split="train", max_tokens=10_000_000, cache_dir="./data"):
    """Tokenise a corpus into an in-memory uint16 array."""
    tokenizer = get_tokenizer()
    print(f"Loading {dataset_name} [{split}] (up to {max_tokens:,} tokens)...")

    chunks = list(
        _iter_token_chunks(
            _iter_texts(dataset_name, split, cache_dir), tokenizer, max_tokens
        )
    )
    tokens = (
        np.concatenate(chunks) if chunks else np.empty(0, dtype=TOKEN_DTYPE)
    )

    print(f"  {len(tokens):,} tokens")
    return tokens, tokenizer


def prepare_memmap(
    dataset_name, split="train", max_tokens=10_000_000, cache_dir="./data"
):
    """Stream a corpus to a flat uint16 file on disk, once. Returns its path.

    Two uses: corpora too large to tokenise into memory (the 20B-token C4
    runs), and avoiding repeated tokenisation across runs. The file is reused
    on subsequent calls, so only the first run pays the cost.
    """
    cache_path = Path(cache_dir) / "tokens"
    cache_path.mkdir(exist_ok=True, parents=True)
    filepath = cache_path / f"{dataset_name}-{split}-{max_tokens}.bin"

    if filepath.exists():
        print(f"Found cached tokens: {filepath} ({filepath.stat().st_size // 2:,} tokens)")
        return filepath

    tokenizer = get_tokenizer()
    print(f"Tokenising {dataset_name} [{split}] to {filepath} (one-time)...")

    tmp_path = filepath.with_suffix(".bin.partial")
    written = 0
    next_report = 10_000_000
    with open(tmp_path, "wb") as f:
        for chunk in _iter_token_chunks(
            _iter_texts(dataset_name, split, cache_dir), tokenizer, max_tokens
        ):
            f.write(chunk.tobytes())
            written += len(chunk)
            if written >= next_report:
                print(f"  {written:,} / {max_tokens:,} tokens")
                next_report += 10_000_000

    # Only publish the final name once the write completed, so an interrupted
    # run cannot leave a truncated file that looks like a valid cache.
    tmp_path.rename(filepath)
    print(f"  wrote {written:,} tokens ({filepath.stat().st_size / 1e6:.1f} MB)")
    return filepath


def open_tokens(path):
    """Memory-map a token file written by ``prepare_memmap``."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class SlidingWindowDataset(Dataset):
    """Training windows: ``(input_ids, target_ids)`` shifted by one token."""

    def __init__(self, tokens, seq_len, stride):
        if not 1 <= stride <= seq_len:
            raise ValueError("stride must satisfy 1 <= stride <= seq_len")

        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        return max(0, (len(self.tokens) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


class EvalSlidingWindowDataset(Dataset):
    """Evaluation windows with overlap masking.

    With ``stride < seq_len`` each window carries extra left context, but a
    token must only be scored once or the perplexity is wrong. Positions that
    serve as context only are set to ``-100`` so that ``cross_entropy`` ignores
    them: window 0 scores everything, and every later window scores only the
    ``stride`` tokens it is the first to reach.
    """

    def __init__(self, tokens, seq_len, stride):
        if not 1 <= stride <= seq_len:
            raise ValueError("stride must satisfy 1 <= stride <= seq_len")

        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        return max(0, (len(self.tokens) - self.seq_len) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = np.asarray(self.tokens[start : start + self.seq_len], dtype=np.int64)

        input_ids = torch.from_numpy(chunk[:-1])
        target_ids = torch.from_numpy(chunk[1:])

        score_start = 0 if idx == 0 else max(0, len(target_ids) - self.stride)
        masked_targets = torch.full_like(target_ids, -100)
        masked_targets[score_start:] = target_ids[score_start:]

        return input_ids, masked_targets


def build_loaders(
    dataset_name="tinystories",
    eval_dataset_name=None,
    seq_len=256,
    stride=256,
    eval_seq_len=None,
    eval_stride=None,
    batch_size=16,
    max_tokens=10_000_000,
    max_eval_tokens=1_000_000,
    num_workers=2,
    cache_dir="./data",
    use_memmap=False,
):
    """Build train and eval dataloaders. Returns ``(train_loader, eval_loader)``."""
    from torch.utils.data import DataLoader

    eval_dataset_name = eval_dataset_name or dataset_name
    eval_seq_len = eval_seq_len or seq_len
    eval_stride = eval_stride or eval_seq_len

    if use_memmap:
        train_tokens = open_tokens(
            prepare_memmap(dataset_name, "train", max_tokens, cache_dir)
        )
        eval_tokens = open_tokens(
            prepare_memmap(eval_dataset_name, "validation", max_eval_tokens, cache_dir)
        )
    else:
        train_tokens, _ = load_tokens(dataset_name, "train", max_tokens, cache_dir)
        eval_tokens, _ = load_tokens(
            eval_dataset_name, "validation", max_eval_tokens, cache_dir
        )

    train_dataset = SlidingWindowDataset(train_tokens, seq_len, stride)
    eval_dataset = EvalSlidingWindowDataset(eval_tokens, eval_seq_len, eval_stride)

    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError(
            "not enough tokens for the requested sequence length; "
            f"train={len(train_tokens):,} tokens, eval={len(eval_tokens):,} tokens, "
            f"seq_len={seq_len}, eval_seq_len={eval_seq_len}"
        )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    eval_loader = DataLoader(eval_dataset, shuffle=False, **loader_kwargs)

    print(
        f"train: {len(train_dataset):,} windows | eval: {len(eval_dataset):,} windows"
    )
    return train_loader, eval_loader
