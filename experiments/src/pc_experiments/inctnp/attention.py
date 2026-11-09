# Adapted from the incTNP codebase, https://github.com/philipmortimer/incTNP-code (commit e9e8e8d):
# `tnp/networks/attention.py`, `tnp/networks/attention_layers.py` and `tnp/networks/kv_cache.py`.
# Philip Mortimer, Cristiana Diaconu, Tommy Rochussen, Bruno Mlodozeniec and Richard E. Turner,
# "Incremental Transformer Neural Processes", ICML 2026, https://arxiv.org/abs/2602.18955.
# That codebase builds on https://github.com/cambridge-mlg/tnp by Matthew Ashman and Cristiana
# Diaconu.
#
# Kept: multi-head self- and cross-attention, their pre-/post-norm layers, and the key-value
# cache. Removed: the `check_shapes` decorators, linear and kernel-regression attention, the
# fixed-size cache, the flash-attention and TNP-A code paths, cached cross-attention keys, and
# the options nothing here sets (dropout rate, activation, feedforward width, a mask on
# self-attention). Module and parameter names are unchanged, so state dicts are interchangeable.
"""Multi-head attention with a key-value cache for causal self-attention."""

from collections.abc import Callable
from functools import partial

import einops
import torch
from torch import Tensor, nn

# Maps a layer tag to that layer's cached tensors. Entries are replaced, never modified in place,
# so a shallow copy of the dictionary is an independent cache.
Cache = dict[str, Tensor | tuple[Tensor, Tensor]]


def update_kv_cache(k_new: Tensor, v_new: Tensor, cache: Cache, tag: str) -> tuple[Tensor, Tensor]:
    """Append keys and values, shape `(m, h, n_new, d)`, to `cache[tag]` and return all of them."""
    if tag in cache:
        k, v = cache[tag]
        k_new = torch.cat((k, k_new), dim=2)
        v_new = torch.cat((v, v_new), dim=2)
    cache[tag] = (k_new, v_new)
    return k_new, v_new


def update_ctx_cache(zc_new: Tensor, cache: Cache, tag: str) -> None:
    """Append context representations, shape `(m, n_new, dz)`, to `cache[tag]`."""
    cached = cache.get(tag)
    if isinstance(cached, Tensor):
        zc_new = torch.cat((cached, zc_new), dim=1)
    cache[tag] = zc_new


class BaseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        qk_dim: int,
        v_dim: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == v_dim)

        self.to_q = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(v_dim, inner_dim, bias=False)
        # The dropout layers here and below do nothing at rate zero. They stay because they fix
        # the positions of their neighbours in the state dict.
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, v_dim), nn.Dropout(0.0))
            if project_out
            else nn.Identity()
        )

    def propagate(
        self,
        xq: Tensor,
        xk: Tensor,
        xv: Tensor,
        mask: Tensor | None = None,
        kv_cache: Cache | None = None,
        kv_tag: str | None = None,
        use_causal: bool = False,
    ) -> Tensor:
        """Attend from `xq` to `xk` and `xv`.

        Args:
            xq: queries, shape `(m, nq, dqk)`.
            xk: keys, shape `(m, nkv, dqk)`.
            xv: values, shape `(m, nkv, dv)`.
            mask: boolean, shape `(nq, nkv)`, true where attention is allowed.
            kv_cache: if given with `kv_tag`, the new keys and values are appended to
                `kv_cache[kv_tag]`, and the queries attend to every cached token and causally
                to each other. `mask` and `use_causal` are then ignored.
            kv_tag: the layer's entry in `kv_cache`.
            use_causal: causal attention between queries and keys of equal length.

        Returns:
            Shape `(m, nq, dv)`.
        """
        q = einops.rearrange(self.to_q(xq), "m n (h d) -> m h n d", h=self.num_heads)
        k = einops.rearrange(self.to_k(xk), "m n (h d) -> m h n d", h=self.num_heads)
        v = einops.rearrange(self.to_v(xv), "m n (h d) -> m h n d", h=self.num_heads)

        if kv_cache is not None and kv_tag is not None:
            k, v = update_kv_cache(k, v, kv_cache, kv_tag)
            # A causal mask offset by the number of cached tokens, which `is_causal` cannot express.
            q_len, k_len = q.shape[2], k.shape[2]
            mask = torch.tril(torch.ones(k_len, k_len, dtype=torch.bool, device=k.device))[-q_len:]
            use_causal = False

        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=self.scale, is_causal=use_causal
        )
        out = einops.rearrange(out, "m h n d -> m n (h d)")
        return self.to_out(out)


class MultiHeadSelfAttention(BaseMultiHeadAttention):
    def __init__(self, *, embed_dim: int, **kwargs) -> None:
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    def forward(
        self,
        x: Tensor,
        kv_cache: Cache | None = None,
        kv_tag: str | None = None,
        use_causal: bool = False,
    ) -> Tensor:
        return super().propagate(x, x, x, kv_cache=kv_cache, kv_tag=kv_tag, use_causal=use_causal)


class MultiHeadCrossAttention(BaseMultiHeadAttention):
    def __init__(self, *, embed_dim: int, **kwargs) -> None:
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    def forward(self, xq: Tensor, xkv: Tensor, mask: Tensor | None = None) -> Tensor:
        return super().propagate(xq, xkv, xkv, mask)


class BaseMultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attention: Callable[..., BaseMultiHeadAttention],
        norm_first: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.attn = attention(**kwargs)

        self.ff_block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.0),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_first = norm_first

        self.attn_dropout = nn.Dropout(0.0)


class MultiHeadSelfAttentionLayer(BaseMultiHeadAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs) -> None:
        attention = partial(MultiHeadSelfAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    def attn_block(
        self,
        x: Tensor,
        kv_cache: Cache | None = None,
        kv_tag: str | None = None,
        use_causal: bool = False,
    ) -> Tensor:
        x = self.attn(x, kv_cache=kv_cache, kv_tag=kv_tag, use_causal=use_causal)
        return self.attn_dropout(x)

    def forward(
        self,
        x: Tensor,
        kv_cache: Cache | None = None,
        kv_tag: str | None = None,
        use_causal: bool = False,
    ) -> Tensor:
        """Self-attention block on `x`, shape `(m, n, d)`; see `propagate` for the cache."""
        if self.norm_first:
            x = x + self.attn_block(self.norm1(x), kv_cache, kv_tag, use_causal)
            x = x + self.ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self.attn_block(x, kv_cache, kv_tag, use_causal))
            x = self.norm2(x + self.ff_block(x))

        return x


class MultiHeadCrossAttentionLayer(BaseMultiHeadAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs) -> None:
        attention = partial(MultiHeadCrossAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    def attn_block(self, xq: Tensor, xkv: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.attn(xq, xkv, mask=mask)
        return self.attn_dropout(x)

    def forward(self, xq: Tensor, xkv: Tensor, mask: Tensor | None = None) -> Tensor:
        """Cross-attention block from `xq`, shape `(m, nq, d)`, to `xkv`, shape `(m, nkv, d)`."""
        if self.norm_first:
            xq = xq + self.attn_block(self.norm1(xq), self.norm1(xkv), mask)
            xq = xq + self.ff_block(self.norm2(xq))
        else:
            xq = self.norm1(xq + self.attn_block(xq, xkv, mask))
            xq = self.norm2(xq + self.ff_block(xq))

        return xq
