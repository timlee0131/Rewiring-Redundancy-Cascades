"""Transformer language model with standard and Redundancy-Cascade attention.

Two attention variants are provided and are interchangeable within the same
transformer stack:

    MultiHeadAttention          -- the baseline ("std")
    RedundancyCascadeAttention  -- the method proposed in the paper ("redwire")

Both return (output, attn, beta) so that a single training loop can drive
either one. ``beta`` is ``None`` for the baseline.
"""

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (https://arxiv.org/abs/2104.09864)."""

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cache_key = None
        self._cos_cached = None
        self._sin_cached = None

    def _compute_cos_sin(self, seq_len, device):
        key = (seq_len, device)
        if key != self._cache_key:
            self._cache_key = key

            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
            emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        """Rotate queries and keys of shape (batch, n_heads, seq_len, head_dim)."""
        cos, sin = self._compute_cos_sin(q.shape[2], q.device)
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot


class ALiBi(nn.Module):
    """Additive linear bias (https://arxiv.org/abs/2108.12409)."""

    def __init__(self, n_heads, max_seq_len=2048):
        super().__init__()
        slopes = self._get_slopes(n_heads)
        self.register_buffer(
            "slopes", torch.tensor(slopes, dtype=torch.float32), persistent=False
        )
        self._seq_len_cached = 0
        self.register_buffer("bias", torch.empty(0), persistent=False)

    @staticmethod
    def _get_slopes(n_heads):
        """Geometric slope schedule from the ALiBi paper."""

        def _power_of_2_slopes(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start**i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return _power_of_2_slopes(n_heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes_a = _power_of_2_slopes(closest_power_of_2)
        slopes_b = _power_of_2_slopes(2 * closest_power_of_2)
        slopes_b = slopes_b[0::2][: n_heads - closest_power_of_2]
        return slopes_a + slopes_b

    def forward(self, seq_len):
        """Return bias of shape (1, n_heads, seq_len, seq_len) for causal positions."""
        if seq_len > self._seq_len_cached:
            positions = torch.arange(seq_len, device=self.slopes.device)
            distance = positions.unsqueeze(1) - positions.unsqueeze(0)
            distance = distance.clamp(min=0).float()
            bias = -self.slopes.view(-1, 1, 1) * distance.unsqueeze(0)
            self.bias = bias.unsqueeze(0)
            self._seq_len_cached = seq_len

        return self.bias[:, :, :seq_len, :seq_len]


@lru_cache(maxsize=8)
def _causal_mask(seq_len, device):
    """Additive (-inf above the diagonal) causal mask of shape (seq_len, seq_len).

    Cached: every layer of every forward pass needs the same mask.
    """
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_seq_len=2048, use_rope=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.use_rope = use_rope
        if use_rope:
            self.pos_enc = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        else:
            self.pos_enc = ALiBi(n_heads, max_seq_len=max_seq_len)

    def forward(self, x, is_causal=True):
        bsz, seq_len, _ = x.size()

        qkv = rearrange(
            self.qkv(x),
            "b s (three h d) -> three b h s d",
            three=3,
            h=self.n_heads,
            d=self.head_dim,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (bsz, n_heads, seq_len, head_dim)

        if self.use_rope:
            q, k = self.pos_enc(q, k)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if not self.use_rope:
            logits = logits + self.pos_enc(seq_len)

        if is_causal:
            logits = logits + _causal_mask(seq_len, x.device)

        attn = F.softmax(logits, dim=-1)

        out = torch.matmul(self.attn_dropout(attn), v)
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.resid_dropout(self.out_proj(out))

        return out, attn, None


class RedundancyCascadeAttention(nn.Module):
    """Causal attention with redundancy-gated edge rescaling.

    Args:
        beta_min: floor on the gate, so that logits cannot vanish entirely.
        alpha: sharpness of the redundancy penalty.
        scorer: how ``R`` is computed, either ``dot`` (scaled dot product) or
            ``bilinear`` (through a learned matrix shared across heads).
        global_summary: with ``scorer='dot'``, score against the running mean of
            all previous values instead of only the previous position.
        placebo_mode: replace the redundancy probe with Gaussian noise. Control
            for "does the gate carry signal, or does any multiplicative mask
            help?".
    """

    SCORERS = ("dot", "bilinear")

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        max_seq_len=2048,
        use_rope=True,
        beta_min=0.05,
        scorer="dot",
        alpha=5.0,
        eps=1e-12,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        if scorer not in self.SCORERS:
            raise ValueError(f"unknown scorer: {scorer} (expected one of {self.SCORERS})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.alpha = alpha
        self.eps = eps
        self.beta_min = beta_min
        self.scorer = scorer

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.use_rope = use_rope
        if use_rope:
            self.pos_enc = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        else:
            self.pos_enc = ALiBi(n_heads, max_seq_len=max_seq_len)

        if scorer == "bilinear":
            # Shared across heads.
            self.W = nn.Parameter(
                torch.randn(self.head_dim, self.head_dim) / math.sqrt(self.head_dim)
            )

    def _redundancy(self, v_gate, seq_len, device):
        """Redundancy scores of shape (bsz, 1, seq_len, seq_len), pre-sigmoid."""

        # Position i is scored against v_{i-1}.
        prev_idx = torch.clamp(torch.arange(seq_len, device=device) - 1, min=0)
        v_probe = v_gate[:, :, prev_idx, :]

        if self.scorer == "bilinear":
            return torch.matmul(v_probe, torch.matmul(v_gate, self.W).transpose(-2, -1))

        # 'dot'
        return torch.matmul(v_probe, v_gate.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

    def forward(self, x, is_causal=True):
        bsz, seq_len, _ = x.size()

        qkv = rearrange(
            self.qkv(x),
            "b s (three h d) -> three b h s d",
            three=3,
            h=self.n_heads,
            d=self.head_dim,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (bsz, n_heads, seq_len, head_dim)

        if self.use_rope:
            q, k = self.pos_enc(q, k)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if not self.use_rope:
            logits = logits + self.pos_enc(seq_len)

        i = torch.arange(seq_len, device=x.device).view(1, 1, seq_len, 1)
        j = torch.arange(seq_len, device=x.device).view(1, 1, 1, seq_len)
        causal = j <= i
        pinned = (j == i) | (j == i - 1) | (j == 0)  # never gated
        eligible = j <= i - 2  # content-scaled candidates

        # The gate is computed from a single head's values (the last one) and
        # broadcast across all heads.
        v_gate = v[:, -1:, :, :]
        R = torch.sigmoid(self._redundancy(v_gate, seq_len, x.device))
        beta = torch.exp(-self.alpha * R).clamp(min=self.beta_min)

        beta = torch.where(pinned, torch.ones_like(beta), beta)
        beta = torch.where(eligible, beta, torch.ones_like(beta))
        beta = beta.expand(-1, self.n_heads, -1, -1)

        attn = torch.softmax(logits.masked_fill(~causal, float("-inf")), dim=-1)

        attn_rewired = attn * beta
        attn_rewired = attn_rewired / attn_rewired.sum(dim=-1, keepdim=True).clamp(
            min=self.eps
        )

        out = torch.matmul(self.attn_dropout(attn_rewired), v)
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.resid_dropout(self.out_proj(out))

        return out, attn, beta


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or (d_model * 8) // 3

        self.w1 = nn.Linear(d_model, self.d_ff, bias=False)
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


ATTENTION_VARIANTS = ("std", "redwire")


def build_attention(variant, d_model, n_heads, dropout, max_seq_len, use_rope, **kwargs):
    """Construct an attention module. Unused kwargs raise, rather than being
    silently dropped -- passing ``--scorer`` to the baseline should be an error,
    not a no-op."""
    if variant == "std":
        return MultiHeadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            **kwargs,
        )
    if variant == "redwire":
        return RedundancyCascadeAttention(
            d_model,
            n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            **kwargs,
        )
    raise ValueError(
        f"unknown attention variant: {variant} (expected one of {ATTENTION_VARIANTS})"
    )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        max_seq_len=2048,
        variant="std",
        **attn_kwargs,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = build_attention(
            variant, d_model, n_heads, dropout, max_seq_len, **attn_kwargs
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, is_causal=True):
        attn_out, attn, beta = self.attn(self.ln1(x), is_causal=is_causal)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x, attn, beta


class Transformer(nn.Module):
    """Decoder-only transformer. ``variant`` selects the attention mechanism."""

    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        max_seq_len=2048,
        dropout=0.1,
        tie_weights=True,
        variant="std",
        use_rope=True,
        **attn_kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.variant = variant

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    variant=variant,
                    use_rope=use_rope,
                    **attn_kwargs,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """GPT-2 / GPT-3 style initialisation."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, is_causal=True, return_attn=False):
        """Map (batch, seq_len) token ids to (batch, seq_len, vocab_size) logits.

        With ``return_attn=True`` also returns a list of ``(attn, beta)`` pairs,
        one per layer. Keep it ``False`` on the training path: the attention
        matrices are (batch, n_heads, seq_len, seq_len) and holding all layers
        at once is expensive.
        """
        seq_len = input_ids.size(1)
        assert (
            seq_len <= self.max_seq_len
        ), f"sequence length {seq_len} exceeds max {self.max_seq_len}"

        x = self.emb_dropout(self.token_emb(input_ids))

        attns = []
        for block in self.blocks:
            x, attn, beta = block(x, is_causal=is_causal)
            if return_attn:
                attns.append((attn, beta))

        logits = self.lm_head(self.ln_f(x))

        return (logits, attns) if return_attn else logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Sample ``max_new_tokens`` continuations of ``input_ids``."""
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            cond = input_ids[:, -self.max_seq_len :]
            logits = self(cond, is_causal=True)[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        if was_training:
            self.train()
        return input_ids

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
