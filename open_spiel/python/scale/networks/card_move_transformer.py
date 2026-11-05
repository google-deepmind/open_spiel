# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The card move transformer: one token per card, moves as pointer logits.

Importing this module registers the torso and head kinds and the NAMED
networks in the catalog. actor_critic.LAZY_TORSOS / LAZY_POLICY_HEADS and
catalog.LAZY name this module, so a network config or catalog name that uses
them imports it (no import by the caller).

A network for trick-taking card games whose actions are cards plus a few
special moves (a bid, a pass, a trump call), read from card planes
f32[N, C, n_card] (one C-vector per card; e.g. encoders.hearts_card_planes).

  CardMoveTransformerTorso  (torso kind "card_move_transformer")
    tokens: value, decision, one token per special action, a no-op token,
    then the n_card card tokens. Each card token is Linear(C, D) of its
    card's C plane values ("embedder"); the special tokens start at zero.
    + a learned positional encoding [T, D]; `depth` pre-LayerNorm blocks
    (LayerNorm, multi-head self-attention with biases, no mask, residual;
    LayerNorm, Linear(D, ff_factor D), ReLU, Linear(ff_factor D, D),
    residual); a final LayerNorm ("norm_out"). D = 8 *
    embed_dim_per_head_over8 * n_head; LayerNorm eps 1e-5. The no-op token
    is attended to like any other and has no logit: a seat in these games is
    never asked to act when it has no move, so it is a fixed extra token of
    the sequence. Returns {"value": [N, D] token 0, "decision": [N, D]
    token 1, "specials": [N, n_special_action, D], "cards": [N, n_card, D]}.

  CardMovePointerHead  (policy head kind "card_move_pointer")
    q = q_proj(decision); k = k_proj([cards, specials]);
    logits = k . q / sqrt(D): actions 0..n_card-1 are the cards, then the
    special actions in token order (num_actions = n_card + n_special_action).

The value is ValueHead(kind="scalar", source="value") on token 0. Its init is
the ValueHead default (flax lecun_normal kernel, zero bias) unless
value_init="torch_uniform": kernel and bias U(+-1/sqrt(D)), the uniform
fan-in init of a PyTorch nn.Linear(D, 1).

Masks: the network takes no legal mask; decide and every loss mask illegal
logits.

Dtypes: params f32; matmuls on `dtype` operands ("float32" or "bfloat16")
with f32 accumulation, each Linear returning `dtype` (bias added in
`dtype`); residual stream, LayerNorms, softmax and logits f32.

Init: every torso and pointer-head Linear kernel and bias U(+-1/sqrt(fan_in)),
LayerNorm 1 / 0,
positional encoding a normal of std pos_emb_std truncated at +-2.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Mapping, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from open_spiel.python.scale.networks import actor_critic
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.networks import gin_tfv2
from open_spiel.python.scale.networks import layers

F32 = jnp.float32
TOK_VALUE = 0
TOK_DECISION = 1
#: value, decision and no-op tokens besides the special-action tokens.
N_FIXED_TOKENS = 3
LN_EPS = 1e-5


def uniform_kernel_init(rng, shape, dtype=F32):
  """U(+-1/sqrt(fan_in)) over a [fan_in, fan_out] kernel.

  Args:
    rng: the PRNG key.
    shape: the kernel shape.
    dtype: its dtype.

  Returns:
    The kernel.
  """
  bound = 1.0 / math.sqrt(shape[0])
  return jax.random.uniform(rng, shape, dtype, -bound, bound)


def uniform_bias_init(fan_in: int):
  """The bias init U(+-1/sqrt(fan_in)).

  Args:
    fan_in: the layer's input width.

  Returns:
    The initialiser.
  """
  bound = 1.0 / math.sqrt(fan_in)
  return lambda rng, shape, dtype=F32: jax.random.uniform(
      rng, shape, dtype, -bound, bound
  )


def _mixed_linear_fwd(x, kernel, bias, dtype):
  """The forward of mixed_linear and its residuals.

  Args:
    x: the input.
    kernel: f32[in, out].
    bias: f32[out].
    dtype: the operand dtype.

  Returns:
    (the output in `dtype`, the residuals).
  """
  xc, wc = x.astype(dtype), kernel.astype(dtype)
  y = gin_tfv2.mm(xc, wc, dtype, dtype) + bias.astype(dtype)
  return y, (xc, wc, jnp.zeros((0,), x.dtype))


def _mixed_linear_bwd(dtype, res, g):
  """The backward of mixed_linear: f32 parameter gradients.

  Args:
    dtype: the operand dtype.
    res: the forward's residuals.
    g: the output gradient.

  Returns:
    (dx, dkernel, dbias).
  """
  xc, wc, x_proto = res
  g = g.astype(dtype)
  dx = jax.lax.dot_general(
      g, wc, (((g.ndim - 1,), (1,)), ((), ())), preferred_element_type=dtype
  ).astype(x_proto.dtype)
  rows = (-1, xc.shape[-1])
  dw = jax.lax.dot_general(
      xc.reshape(rows),
      g.reshape(-1, g.shape[-1]),
      (((0,), (0,)), ((), ())),
      preferred_element_type=F32,
  )
  db = jnp.sum(g.reshape(-1, g.shape[-1]), axis=0, dtype=F32)
  return dx, dw, db


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def mixed_linear(x, kernel, bias, dtype):
  """Computes x @ kernel + bias on `dtype` operands, accumulated in f32.

  The output is in `dtype`.

  Its gradient: x's in `dtype` (cast to x's dtype); the kernel's and bias's
  accumulated in f32 and returned as f32, so a parameter gradient summed over
  every row is not rounded to `dtype`.

  Args:
    x: the input.
    kernel: f32[in, out].
    bias: f32[out].
    dtype: the operand dtype.

  Returns:
    The output in `dtype`.
  """
  return _mixed_linear_fwd(x, kernel, bias, dtype)[0]


mixed_linear.defvjp(_mixed_linear_fwd, _mixed_linear_bwd)


def _mixed_linear_f32_fwd(x, kernel, bias, dtype):
  """The forward of mixed_linear_f32 and its residuals.

  Args:
    x: the input.
    kernel: f32[in, out].
    bias: f32[out].
    dtype: the operand dtype.

  Returns:
    (the f32 output, the residuals).
  """
  xc, wc = x.astype(dtype), kernel.astype(dtype)
  y = gin_tfv2.mm(xc, wc, dtype, F32) + bias.astype(F32)
  return y, (xc, wc, jnp.zeros((0,), x.dtype))


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def mixed_linear_f32(x, kernel, bias, dtype):
  """mixed_linear with an f32 output: the f32 accumulation plus the f32 bias.

  The gradient is mixed_linear's (the output gradient cast to `dtype`, f32
  parameter gradients).

  Args:
    x: the input.
    kernel: f32[in, out].
    bias: f32[out].
    dtype: the operand dtype.

  Returns:
    The output in f32.
  """
  return _mixed_linear_f32_fwd(x, kernel, bias, dtype)[0]


mixed_linear_f32.defvjp(_mixed_linear_f32_fwd, _mixed_linear_bwd)


class Linear(nn.Module):
  """A dense layer: f32 params, `dtype` operands and output (mixed_linear).

  Attributes:
    features: the output width.
    dtype: the compute dtype name.
    f32_out: return f32 (the `dtype` product accumulated in f32, plus the f32
      bias) instead of rounding the output to `dtype`; same params.
  """

  features: int
  dtype: str = "float32"
  f32_out: bool = False

  @nn.compact
  def __call__(self, x):
    """Applies the layer.

    Args:
      x: [..., in].

    Returns:
      [..., features].
    """
    fan_in = x.shape[-1]
    kernel = self.param(
        "kernel", uniform_kernel_init, (fan_in, self.features), F32
    )
    bias = self.param("bias", uniform_bias_init(fan_in), (self.features,), F32)
    cdt = layers.dtype_of(self.dtype)
    if cdt == F32:
      return gin_tfv2.mm(x, kernel, F32, F32) + bias
    if self.f32_out:
      return mixed_linear_f32(x, kernel, bias, cdt)
    return mixed_linear(x, kernel, bias, cdt)


def _layer_norm(name: str) -> nn.LayerNorm:
  """An f32 LayerNorm of eps LN_EPS.

  Args:
    name: the module name.

  Returns:
    The LayerNorm.
  """
  return nn.LayerNorm(
      epsilon=LN_EPS,
      dtype=F32,
      param_dtype=F32,
      use_fast_variance=False,
      name=name,
  )


class SelfAttentionBlock(nn.Module):
  """Pre-LN multi-head self-attention, then a pre-LN ReLU FFN; f32 residual.

  Attributes:
    d_model: the width.
    n_head: the heads.
    ff_factor: the FFN width over d_model.
    dtype: the compute dtype name.
    attention: gin_tfv2.ATTENTION_IMPLS.
    center_keys: subtract each head's mean key over the tokens before
      attending (opt-in). Exact-arithmetic invariant (a per-query
      constant logit shift), so a fork can turn it on; the keys are computed
      and centred in f32 and only then cast to `dtype`, so a large common-mode
      key no longer sets the bf16 rounding of the token-dependent part.
  """

  d_model: int
  n_head: int
  ff_factor: int
  dtype: str
  attention: str = "xla"
  center_keys: bool = False

  @nn.compact
  def __call__(self, x):
    """Applies the block.

    Args:
      x: the tokens [N, T, d_model].

    Returns:
      The tokens after the block, [N, T, d_model].
    """
    d, nh = self.d_model, self.n_head
    hd = d // nh
    cdt = layers.dtype_of(self.dtype)
    h = _layer_norm("layer_norm1")(x)
    q, k, v = (
        Linear(
            d,
            self.dtype,
            f32_out=self.center_keys and n == "k",
            name=f"{n}_proj",
        )(h).reshape(h.shape[:-1] + (nh, hd))
        for n in ("q", "k", "v")
    )
    if self.center_keys:  # k [N, T, H, hd], f32: the token mean per head
      k = k - jnp.mean(k, axis=-3, keepdims=True)
    o = gin_tfv2.attend(
        q.astype(cdt), k.astype(cdt), v.astype(cdt), None, self.attention
    )
    o = Linear(d, self.dtype, name="out_proj")(o.reshape(o.shape[:-2] + (d,)))
    x = x + o
    h = _layer_norm("layer_norm2")(x)
    h = Linear(self.ff_factor * d, self.dtype, name="linear1")(h)
    h = Linear(d, self.dtype, name="linear2")(nn.relu(h))
    return x + h


class CardMoveTransformerTorso(nn.Module):
  """Planes [N, in_channels, n_card] -> the value, decision and move tokens.

  Its output is {"value", "decision", "specials", "cards"}.

  Attributes:
    depth: the blocks.
    embed_dim_per_head_over8: D / (8 n_head).
    n_head: the heads.
    ff_factor: the FFN width over D.
    in_channels: the planes per card.
    n_card: the card tokens.
    n_special_action: the special-action tokens.
    pos_emb_std: the positional encoding's init std.
    dtype: the compute dtype name.
    attention: gin_tfv2.ATTENTION_IMPLS.
    center_keys: SelfAttentionBlock.center_keys (opt-in).
  """

  depth: int = 8
  embed_dim_per_head_over8: int = 6
  n_head: int = 8
  ff_factor: int = 4
  in_channels: int = 56
  n_card: int = 52
  n_special_action: int = 0
  pos_emb_std: float = 0.1
  dtype: str = "float32"
  attention: str = "xla"
  center_keys: bool = False

  @property
  def embed_dim(self) -> int:
    """The model width D."""
    return 8 * self.embed_dim_per_head_over8 * self.n_head

  @property
  def n_special(self) -> int:
    """The fixed tokens plus the special-action tokens."""
    return N_FIXED_TOKENS + self.n_special_action

  def _check(self):
    """Raises ValueError on an invalid config."""
    for name in (
        "depth",
        "n_head",
        "embed_dim_per_head_over8",
        "ff_factor",
        "in_channels",
        "n_card",
    ):
      if int(getattr(self, name)) < 1:
        raise ValueError(
            f"CardMoveTransformerTorso {name} must be >= 1, got "
            f"{getattr(self, name)}"
        )
    if int(self.n_special_action) < 0:
      raise ValueError(
          f"n_special_action must be >= 0, got {self.n_special_action}"
      )
    if self.attention not in gin_tfv2.ATTENTION_IMPLS:
      raise ValueError(
          f"attention must be one of {gin_tfv2.ATTENTION_IMPLS}, got "
          f"{self.attention!r}"
      )

  @nn.compact
  def __call__(self, planes):
    """Applies the torso.

    Args:
      planes: [N, in_channels, n_card].

    Returns:
      {"value", "decision", "specials", "cards"} token features.
    """
    self._check()
    if planes.ndim != 3 or planes.shape[1:] != (self.in_channels, self.n_card):
      raise ValueError(
          f"CardMoveTransformerTorso wants planes [N, {self.in_channels}, "
          f"{self.n_card}], got {planes.shape}"
      )
    d, ns = self.embed_dim, self.n_special
    x = jnp.swapaxes(
        planes.astype(F32), 1, 2
    )  # [N, n_card, C]: one token per card
    x = Linear(d, self.dtype, name="embedder")(x)
    x = jnp.pad(x, ((0, 0), (ns, 0), (0, 0)))
    std = float(self.pos_emb_std)
    pos = self.param(
        "positional_encoding",
        lambda rng, shape, dtype=F32: std
        * jax.random.truncated_normal(rng, -2.0 / std, 2.0 / std, shape, dtype),
        (ns + self.n_card, d),
        F32,
    )
    x = x + pos[None]
    for i in range(self.depth):
      x = SelfAttentionBlock(
          d_model=d,
          n_head=self.n_head,
          ff_factor=self.ff_factor,
          dtype=self.dtype,
          attention=self.attention,
          center_keys=bool(self.center_keys),
          name=f"layer_{i}",
      )(x)
    out = _layer_norm("norm_out")(x)
    first = TOK_DECISION + 1
    return {
        "value": out[:, TOK_VALUE],
        "decision": out[:, TOK_DECISION],
        "specials": out[:, first : first + self.n_special_action],
        "cards": out[:, ns:],
    }


class CardMovePointerHead(nn.Module):
  """Pointer logits over the card and special-action tokens (module doc).

  Attributes:
    num_actions: n_card + n_special_action.
    dtype: the compute dtype name.
  """

  num_actions: int
  dtype: str = "float32"

  @nn.compact
  def __call__(self, feats):
    """The pointer logits.

    Args:
      feats: the torso's output.

    Returns:
      Logits [N, num_actions].
    """
    cdt = layers.dtype_of(self.dtype)
    dec, cards, specials = feats["decision"], feats["cards"], feats["specials"]
    n = cards.shape[1] + specials.shape[1]
    if self.num_actions != n:
      raise ValueError(
          f"CardMovePointerHead has {n} move tokens, got num_actions="
          f"{self.num_actions}"
      )
    d = dec.shape[-1]
    q = Linear(d, self.dtype, name="q_proj")(dec)  # [N, D]
    k = Linear(d, self.dtype, name="k_proj")(
        jnp.concatenate([cards, specials], axis=1)
    )
    logits = jnp.einsum(
        "nkd,nd->nk", k.astype(cdt), q.astype(cdt), preferred_element_type=F32
    ) / F32(math.sqrt(d))
    return logits.astype(F32)


def card_move_transformer(
    *,
    depth: int,
    embed_dim_per_head_over8: int,
    n_head: int,
    ff_factor: int,
    in_channels: int,
    n_card: int,
    n_special_action: int,
    compute_dtype: str,
    pos_emb_std: float = 0.1,
    attention: str = "xla",
    chunk_size: Optional[int] = None,
    value_init: str = "lecun",
    center_keys: bool = False,
) -> dict[str, Any]:
  """The network config of a card move transformer (module doc).

  Args:
    depth: the blocks.
    embed_dim_per_head_over8: D / (8 n_head).
    n_head: the heads.
    ff_factor: the FFN width over D.
    in_channels: the planes per card.
    n_card: the card tokens.
    n_special_action: the special-action tokens.
    compute_dtype: "float32" or "bfloat16".
    pos_emb_std: the positional encoding's init std.
    attention: gin_tfv2.ATTENTION_IMPLS (execution only).
    chunk_size: rows per forward chunk (execution only).
    value_init: the value head's init, "lecun" (the default; the config
      carries no init_scheme, so its hash is unchanged) or "torch_uniform"
      (the uniform fan-in init of a PyTorch nn.Linear).
    center_keys: SelfAttentionBlock.center_keys; the torso config carries the
      key only when True, so the default config and its hash are unchanged.

  Returns:
    The config dict (actor_critic.build_network's input).

  Raises:
    ValueError: an unknown value_init.
  """
  assert compute_dtype in ("float32", "bfloat16"), compute_dtype
  return with_options(
      {
          "compute_dtype": compute_dtype,
          "torso": {
              "kind": "card_move_transformer",
              "depth": int(depth),
              "embed_dim_per_head_over8": int(embed_dim_per_head_over8),
              "n_head": int(n_head),
              "ff_factor": int(ff_factor),
              "in_channels": int(in_channels),
              "n_card": int(n_card),
              "n_special_action": int(n_special_action),
              "pos_emb_std": float(pos_emb_std),
              "attention": attention,
          },
          "policy_head": {"kind": "card_move_pointer"},
          "value_head": {"kind": "scalar", "source": "value"},
          "chunk_size": chunk_size,
      },
      value_init=value_init,
      center_keys=center_keys,
  )


def with_options(
    network: Mapping[str, Any],
    *,
    value_init: str = "lecun",
    center_keys: bool = False,
) -> dict[str, Any]:
  """A card move transformer config with its opt-in options set.

  card_move_transformer's rule holds: an option at its default leaves no key,
  so a config built with the defaults equals `network` without the options
  (and has its hash).

  Args:
    network: a card_move_transformer config (e.g. a NAMED network).
    value_init: card_move_transformer's value_init.
    center_keys: card_move_transformer's center_keys.

  Returns:
    The config, a copy.

  Raises:
    ValueError: `network` is not a card move transformer, or an unknown
      value_init.
  """
  if network.get("torso", {}).get("kind") != "card_move_transformer":
    raise ValueError(
        f"not a card move transformer config: {network.get('torso')}"
    )
  if value_init not in ("lecun", layers.TORCH_UNIFORM):
    raise ValueError(
        f"value_init must be 'lecun' or {layers.TORCH_UNIFORM!r}, got "
        f"{value_init!r}"
    )
  out = dict(network)
  torso = {k: v for k, v in network["torso"].items() if k != "center_keys"}
  if center_keys:
    torso["center_keys"] = True
  value_head = {
      k: v for k, v in network["value_head"].items() if k != "init_scheme"
  }
  if value_init != "lecun":
    value_head["init_scheme"] = value_init
  out["torso"], out["value_head"] = torso, value_head
  return out


#: The model block of the card move transformer recipes: depth 8, 8 heads,
#: D = 384 (head dim 48), FFN 4D.
BASE_MODEL = dict(depth=8, embed_dim_per_head_over8=6, n_head=8, ff_factor=4)

#: Named networks (catalog.register at import), bf16 compute, f32 params,
#: cuDNN attention on a GPU ("auto"): the card move transformer (BASE_MODEL)
#:   hearts: over the hearts card planes [56, 52] (encoders.
#:     hearts_card_planes), 52 card moves; 14,535,553 params.
#:   euchre: over the euchre card planes [64, 24] (encoders.
#:     euchre_card_planes), 24 card moves then 7 special moves (pass, the four
#:     suits, alone, with partner: envs.euchre's action ids 24..30);
#:     14,530,561 params.
NAMED = {
    "hearts_card_move_transformer_bf16": card_move_transformer(
        **BASE_MODEL,
        in_channels=56,
        n_card=52,
        n_special_action=0,
        compute_dtype="bfloat16",
        attention="auto",
    ),
    "euchre_card_move_transformer_bf16": card_move_transformer(
        **BASE_MODEL,
        in_channels=64,
        n_card=24,
        n_special_action=7,
        compute_dtype="bfloat16",
        attention="auto",
    ),
}

actor_critic.register_torso("card_move_transformer", CardMoveTransformerTorso)
actor_critic.register_policy_head("card_move_pointer", CardMovePointerHead)
for _name, _cfg in NAMED.items():
  catalog.register(_name, _cfg)
