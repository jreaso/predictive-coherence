# Adapted from the incTNP codebase, https://github.com/philipmortimer/incTNP-code (commit e9e8e8d):
# `tnp/models/incTNPBatched.py` (incTNP-Seq), `TNPDecoder` in `tnp/models/tnp.py`,
# `tnp/networks/mlp.py`, `tnp/likelihoods/gaussian.py`, the loss for `BatchedCausalTNP` in
# `tnp/utils/np_functions.py`, and the architecture in `experiments/configs/models/inctnpSeq.yml`.
# Philip Mortimer, Cristiana Diaconu, Tommy Rochussen, Bruno Mlodozeniec and Richard E. Turner,
# "Incremental Transformer Neural Processes", ICML 2026, https://arxiv.org/abs/2602.18955.
# That codebase builds on https://github.com/cambridge-mlg/tnp by Matthew Ashman and Cristiana
# Diaconu.
#
# Changed: the key-value cache is an argument rather than state on the module, so that one
# network can serve many contexts at once; training and prediction are separate methods rather
# than a switch on `self.training`; the identity `x_encoder` and `y_encoder`, which hold no
# parameters, the likelihood's noise floor, which was zero, and the `check_shapes` decorators are
# removed. Module and parameter names are unchanged, so state dicts are interchangeable.
"""The incTNP-Seq network: a transformer neural process with a causal, cacheable context."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from pc_experiments.inctnp.attention import (
    Cache,
    MultiHeadCrossAttentionLayer,
    MultiHeadSelfAttentionLayer,
)
from pc_experiments.inctnp.transformer import TNPTransformerFullyMaskedEncoder


class MLP(nn.Module):
    """`num_layers` hidden layers of `width` units with ReLU activations."""

    def __init__(self, in_dim: int, out_dim: int, num_layers: int, width: int) -> None:
        super().__init__()
        layers = (width,) * num_layers
        net: list[nn.Module] = [nn.Linear(in_dim, layers[0])]
        for i in range(1, len(layers)):
            net.append(nn.ReLU())
            net.append(nn.Linear(layers[i - 1], layers[i]))
        net.append(nn.ReLU())
        net.append(nn.Linear(layers[-1], out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HeteroscedasticNormalLikelihood(nn.Module):
    def forward(self, x: Tensor) -> Normal:
        """Split the last dimension into a mean and a softplus-transformed variance."""
        loc, log_var = x.chunk(2, dim=-1)
        return Normal(loc, nn.functional.softplus(log_var) ** 0.5)


class TNPDecoder(nn.Module):
    def __init__(self, z_decoder: MLP) -> None:
        super().__init__()
        self.z_decoder = z_decoder

    def forward(self, z: Tensor) -> Tensor:
        return self.z_decoder(z)


class IncTNPBatchedEncoder(nn.Module):
    """Embeds `(x, y, flag)` tokens and runs the transformer. The flag marks targets, whose `y`
    is zeroed."""

    def __init__(
        self, transformer_encoder: TNPTransformerFullyMaskedEncoder, xy_encoder: MLP
    ) -> None:
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder

    def preprocess_context(self, xc: Tensor, yc: Tensor) -> Tensor:
        flag = torch.zeros_like(yc[..., :1])
        return self.xy_encoder(torch.cat((xc, yc, flag), dim=-1))

    def preprocess_targets(self, xt: Tensor, dim_y: int) -> Tensor:
        yt = xt.new_zeros(*xt.shape[:-1], dim_y)
        flag = torch.ones_like(yt[..., :1])
        return self.xy_encoder(torch.cat((xt, yt, flag), dim=-1))

    def predict_encoder(self, xc: Tensor, yc: Tensor, xt: Tensor) -> Tensor:
        zc = self.preprocess_context(xc, yc)
        zt = self.preprocess_targets(xt, yc.shape[-1])
        return self.transformer_encoder(zc, zt, use_causal=True)

    def train_encoder(self, x: Tensor, y: Tensor) -> Tensor:
        n = x.shape[1]
        zc = self.preprocess_context(x, y)
        # The first point is never a target: the network is not trained on an empty context.
        zt = self.preprocess_targets(x[:, 1:], y.shape[-1])
        # Target `j` is point `j + 1` of the sequence, and attends to points `0, ..., j`.
        mask_ca = torch.tril(torch.ones(n - 1, n, dtype=torch.bool, device=x.device))
        return self.transformer_encoder(zc, zt, mask_ca=mask_ca, use_causal=True)


class IncTNPSeq(nn.Module):
    """incTNP-Seq: every point of a sequence is predicted from the points before it.

    Context tokens attend causally to one another and target tokens attend to the context, so a
    context can be extended without recomputing what is already encoded. The predictive depends
    on the order of the context. It is not defined for an empty context.

    Inputs have shape `(m, n, dim_x)` and outputs `(m, n, dim_y)`, for a batch of `m` tasks.
    """

    def __init__(
        self, encoder: IncTNPBatchedEncoder, decoder: TNPDecoder, likelihood: nn.Module, dim_y: int
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood
        self.dim_y = dim_y

    def forward(self, xc: Tensor, yc: Tensor, xt: Tensor) -> Normal:
        """Marginal predictives at `xt` given the context, with `batch_shape` `(m, nt, dim_y)`."""
        return self.likelihood(self.decoder(self.encoder.predict_encoder(xc, yc, xt)))

    def forward_sequence(self, x: Tensor, y: Tensor) -> Normal:
        """Predictive of each `y[:, i]`, `i >= 1`, given `(x, y)[:, :i]`, in one pass.

        Returns:
            `batch_shape` `(m, n - 1, dim_y)`.
        """
        return self.likelihood(self.decoder(self.encoder.train_encoder(x, y)))

    def encode_context(self, xc: Tensor, yc: Tensor, cache: Cache) -> None:
        """Append `(xc, yc)` to the context held in `cache`, which is updated in place.

        An empty dictionary is an empty context. Entries are replaced rather than modified, so a
        shallow copy taken beforehand still holds the shorter context.
        """
        zc = self.encoder.preprocess_context(xc, yc)
        self.encoder.transformer_encoder.encode_context(zc, cache)

    def query(self, xt: Tensor, cache: Cache) -> Normal:
        """As `forward`, from the context held in `cache`."""
        zt = self.encoder.preprocess_targets(xt, self.dim_y)
        return self.likelihood(self.decoder(self.encoder.transformer_encoder.query(zt, cache)))


def sequence_nll(network: IncTNPSeq, x: Tensor, y: Tensor) -> Tensor:
    """Mean negative log likelihood of every point after the first, given the points before it."""
    return -network.forward_sequence(x, y).log_prob(y[:, 1:]).mean()


@dataclass(frozen=True)
class Architecture:
    dim_x: int = 1
    dim_y: int = 1
    embed_dim: int = 128
    num_heads: int = 8
    head_dim: int = 16
    num_layers: int = 5
    norm_first: bool = True


def build_network(
    architecture: Architecture, head: nn.Module | None = None, per_output: int = 2
) -> IncTNPSeq:
    """The network of the authors' one-dimensional Gaussian process configuration.

    Tokens are `(x, y, flag)`, and the decoder returns `per_output` numbers for each of the
    `dim_y` outputs, which `head` makes a predictive of: by default a mean and a variance for a
    Gaussian.
    """
    embed_dim, dim_y = architecture.embed_dim, architecture.dim_y
    layer = dict(
        embed_dim=embed_dim,
        num_heads=architecture.num_heads,
        head_dim=architecture.head_dim,
        norm_first=architecture.norm_first,
    )
    transformer = TNPTransformerFullyMaskedEncoder(
        num_layers=architecture.num_layers,
        mhca_layer=MultiHeadCrossAttentionLayer(**layer),
        mhsa_layer=MultiHeadSelfAttentionLayer(**layer),
    )
    xy_encoder = MLP(architecture.dim_x + dim_y + 1, embed_dim, num_layers=2, width=embed_dim)
    z_decoder = MLP(embed_dim, per_output * dim_y, num_layers=2, width=embed_dim)
    return IncTNPSeq(
        IncTNPBatchedEncoder(transformer, xy_encoder),
        TNPDecoder(z_decoder),
        HeteroscedasticNormalLikelihood() if head is None else head,
        dim_y,
    )
