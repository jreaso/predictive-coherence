# Adapted from the incTNP codebase, https://github.com/philipmortimer/incTNP-code (commit e9e8e8d):
# `TNPTransformerFullyMaskedEncoder` in `tnp/networks/transformer.py`.
# Philip Mortimer, Cristiana Diaconu, Tommy Rochussen, Bruno Mlodozeniec and Richard E. Turner,
# "Incremental Transformer Neural Processes", ICML 2026, https://arxiv.org/abs/2602.18955.
# That codebase builds on https://github.com/cambridge-mlg/tnp by Matthew Ashman and Cristiana
# Diaconu.
#
# Kept: the forward pass and the incremental `encode_context` and `query`. Removed: the
# `check_shapes` decorators, the other encoders, the fixed-size cache, the flash-attention and
# cached cross-attention options, and the mask on self-attention, which is always causal here.
# Module and parameter names are unchanged.
"""Transformer encoder whose context can be extended one block at a time."""

import copy

from torch import Tensor, nn

from pc_experiments.inctnp.attention import (
    Cache,
    MultiHeadCrossAttentionLayer,
    MultiHeadSelfAttentionLayer,
    update_ctx_cache,
)


class TNPTransformerFullyMaskedEncoder(nn.Module):
    """Context tokens self-attend, and target tokens cross-attend to the context at every layer.

    With causal self-attention a context token never sees later tokens, so appending tokens
    leaves earlier representations unchanged. `encode_context` and `query` use this to reproduce
    `forward` from a cache at a cost linear in the context length.
    """

    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mhsa_layer: MultiHeadSelfAttentionLayer,
    ) -> None:
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)

    def forward(
        self,
        xc: Tensor,
        xt: Tensor,
        mask_ca: Tensor | None = None,
        use_causal: bool = False,
    ) -> Tensor:
        """Target representations after attending to the context.

        Args:
            xc: context tokens, shape `(m, nc, d)`.
            xt: target tokens, shape `(m, nt, d)`.
            mask_ca: boolean cross-attention mask, shape `(nt, nc)`.
            use_causal: causal self-attention over the context.

        Returns:
            Shape `(m, nt, d)`.
        """
        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers, strict=True):
            xc = mhsa_layer(xc, use_causal=use_causal)
            xt = mhca_layer(xt, xc, mask=mask_ca)

        return xt

    def encode_context(self, zc_new: Tensor, kv_cache: Cache) -> None:
        """Append context tokens, shape `(m, nc_new, d)`, to `kv_cache`, which is updated in place.

        Caches each self-attention layer's keys and values, and each layer's context
        representations for the targets to attend to.
        """
        for i, mhsa_layer in enumerate(self.mhsa_layers):
            zc_new = mhsa_layer(zc_new, kv_cache=kv_cache, kv_tag=f"layer_{i}_sa")
            update_ctx_cache(zc_new, kv_cache, f"context_layer_{i}")

    def query(self, zt: Tensor, kv_cache: Cache) -> Tensor:
        """Target representations, shape `(m, nt, d)`, from the context held in `kv_cache`."""
        for i, mhca_layer in enumerate(self.mhca_layers):
            zt = mhca_layer(zt, kv_cache[f"context_layer_{i}"])
        return zt


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
