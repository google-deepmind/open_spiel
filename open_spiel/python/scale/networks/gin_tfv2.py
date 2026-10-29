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

"""The gin card-token transformer (gin_tfv2): a torso and a pointer head.

CardTokenTorso reads a gin card-token vector (`encoders.gin_card_tokens`,
laid out as `encoders.card_tokens_layout` says): 56 tokens, the 52 cards plus
[ACT] [VAL] [CTX] [OPP], through pre-RMSNorm attention blocks with a learned
meld-relation bias. PointerPolicyHead scores discards, the upcard and melds
by pointer attention over the card tokens. The op sequence (fused QKV, the
row MLP applied to the stacked embedding tables, the `_sum_rows` custom_vjp,
the float64 pointer scores) is fixed, so a checkpoint gives the same outputs
bit for bit on the same hardware.

Importing this module registers the torso kind "card_tokens" and the policy
head kind "pointer" with `networks.actor_critic`; `card_tokens(...)` builds a
network config:

  from open_spiel.python.scale.networks import gin_tfv2
  cfg = gin_tfv2.card_tokens(d_model=192, num_layers=3, num_heads=6,
                             ffn_ratio=2, compute_dtype="bfloat16",
                             value_head="scalar", card_fields=4,
                             global_fields=24)
  net = actor_critic.build_network(cfg, num_actions=241)
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.encoders import card_tokens_layout as ct
from open_spiel.python.scale.networks import actor_critic
from open_spiel.python.scale.networks import layers

F32 = jnp.float32

NUM_GLOBAL_TOKENS = 4
TOK_ACT, TOK_VAL, TOK_CTX, TOK_OPP = 0, 1, 2, 3
NUM_TOKENS = NUM_GLOBAL_TOKENS + ct.NUM_CARDS  # 56
NUM_RELATIONS = 15
#: auto: cuDNN on CUDA for bf16/fp16, else XLA.
ATTENTION_IMPLS = ("xla", "cudnn", "auto")
_EMBED_BWD_CHUNK = 131072
#: cuDNN's fused attention runs on sequences padded to a multiple of this.
CUDNN_SEQ_MULTIPLE = 64
#: Compute dtypes for which attention "auto" takes cuDNN on a CUDA host. An
#: f32 network stays on the XLA path: cuDNN pads the sequence and sums in a
#: different order, so "auto" would change an f32 network's bits on a GPU.
CUDNN_AUTO_DTYPES = (jnp.bfloat16, jnp.float16)


def card_relation_ids() -> np.ndarray:
  """The meld relation of every card pair.

  Returns:
    i32[52, 52]: 0 the same card; 1..12 the same suit at that rank gap; 13
    another suit and rank; 14 the same rank.
  """
  s = np.arange(ct.NUM_CARDS) // 13
  r = np.arange(ct.NUM_CARDS) % 13
  same_s = s[:, None] == s[None, :]
  same_r = r[:, None] == r[None, :]
  gap = np.abs(r[:, None] - r[None, :])
  rel = np.where(same_s, gap, 13 + same_r.astype(np.int64))
  assert rel.min() == 0 and rel.max() == NUM_RELATIONS - 1
  return rel.astype(np.int32)


def card_deadwood_value() -> np.ndarray:
  """The deadwood value index of every card (rank, capped at 9).

  Returns:
    i32[52].
  """
  r = np.arange(ct.NUM_CARDS) % 13
  return np.minimum(r, 9).astype(np.int32)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _sum_rows(table, idx, bwd_dtype):
  """Sums table rows: exact gathers forward, a multi-hot matmul backward.

  Args:
    table: f32[n, d].
    idx: i32[..., k] row indices.
    bwd_dtype: the backward matmul's dtype.

  Returns:
    sum_j table[idx[..., j]], f32[..., d].
  """
  return _sum_rows_fwd(table, idx, bwd_dtype)[0]


def _sum_rows_fwd(table, idx, bwd_dtype):
  """The forward of `_sum_rows` and its residuals.

  Args:
    table: f32[n, d].
    idx: i32[..., k].
    bwd_dtype: unused.

  Returns:
    (the sum, (idx, table.shape)).
  """
  del bwd_dtype
  out = jnp.take(table, idx[..., 0], axis=0)
  for j in range(1, idx.shape[-1]):
    out = out + jnp.take(table, idx[..., j], axis=0)
  return out, (idx, table.shape)


def _bwd_chunk_rows(rows):
  """Rows per backward chunk and the padding rows.

  Args:
    rows: the number of rows.

  Returns:
    (chunk rows, pad rows).
  """
  if rows <= _EMBED_BWD_CHUNK:
    return rows, 0
  for c in range(_EMBED_BWD_CHUNK, _EMBED_BWD_CHUNK // 8, -1):
    if rows % c == 0:
      return c, 0
  return _EMBED_BWD_CHUNK, (-rows) % _EMBED_BWD_CHUNK


def _sum_rows_bwd(bwd_dtype, res, g):
  """The backward of `_sum_rows`: a chunked multi-hot matmul.

  Args:
    bwd_dtype: the matmul dtype.
    res: the forward's residuals.
    g: the output cotangent.

  Returns:
    (the table cotangent, None).
  """
  idx, (n, d) = res
  k = idx.shape[-1]
  fi = idx.reshape(-1, k)
  fg = g.reshape(-1, d).astype(bwd_dtype)
  chunk, pad = _bwd_chunk_rows(fi.shape[0])
  if pad:
    fi = jnp.pad(fi, ((0, pad), (0, 0)), constant_values=n)
    fg = jnp.pad(fg, ((0, pad), (0, 0)))
  fi = fi.reshape(-1, chunk, k)
  fg = fg.reshape(-1, chunk, d)
  cols = jnp.arange(n, dtype=fi.dtype)
  precision = jax.lax.Precision.HIGHEST if fg.dtype == jnp.float32 else None

  def body(acc, x):
    """Adds one chunk's multi-hot product to the accumulator.

    Args:
      acc: f32[n, d].
      x: (indices, cotangents) of one chunk.

    Returns:
      (the new accumulator, None).
    """
    i, gg = x
    hot = i[:, 0, None] == cols
    for j in range(1, k):
      hot = hot | (i[:, j, None] == cols)
    return (
        acc
        + jnp.einsum(
            "cn,cd->nd",
            hot.astype(gg.dtype),
            gg,
            preferred_element_type=jnp.float32,
            precision=precision,
        ),
        None,
    )

  acc, _ = jax.lax.scan(body, jnp.zeros((n, d), jnp.float32), (fi, fg))
  return acc.astype(g.dtype), None


_sum_rows.defvjp(_sum_rows_fwd, _sum_rows_bwd)


class Table(nn.Module):
  """An nn.Embed table as a param (path <name>/embedding), nothing applied.

  Attributes:
    num_embeddings: the rows.
    features: the columns.
  """

  num_embeddings: int
  features: int

  @nn.compact
  def __call__(self):
    """The table.

    Returns:
      f32[num_embeddings, features].
    """
    return self.param(
        "embedding",
        nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
        (self.num_embeddings, self.features),
        F32,
    )


def _dense_general_kernel_init(rng, shape, dtype=F32):
  """lecun_normal over the flattened output axes (DenseGeneral's init).

  Args:
    rng: the key.
    shape: the kernel shape.
    dtype: the dtype.

  Returns:
    The kernel.
  """
  flat = (int(shape[0]), int(np.prod(shape[1:])))
  return jnp.reshape(nn.initializers.lecun_normal()(rng, flat, dtype), shape)


class Kernel(nn.Module):
  """A Dense/DenseGeneral kernel (+ bias) as params, nothing applied.

  Attributes:
    shape: the kernel shape.
    use_bias: whether there is a bias (shape[1:]).
  """

  shape: tuple[int, ...]
  use_bias: bool = False

  @nn.compact
  def __call__(self):
    """The kernel, or (kernel, bias).

    Returns:
      The kernel, or (kernel, bias) with use_bias.
    """
    kernel = self.param("kernel", _dense_general_kernel_init, self.shape, F32)
    if not self.use_bias:
      return kernel
    return kernel, self.param(
        "bias", nn.initializers.zeros, self.shape[1:], F32
    )


def mm(x, w, dtype, out_dtype=None):
  """The product x @ w over x's last axis.

  Args:
    x: the left operand.
    w: the right operand.
    dtype: the operands' dtype.
    out_dtype: the result's dtype (default `dtype`).

  Returns:
    The product.
  """
  return jax.lax.dot_general(
      x.astype(dtype),
      w.astype(dtype),
      (((x.ndim - 1,), (0,)), ((), ())),
      preferred_element_type=out_dtype or dtype,
  )


def attend_padded(
    q, k, v, bias, implementation: str, multiple: int = CUDNN_SEQ_MULTIPLE
):
  """Self-attention over a sequence padded to a multiple of `multiple` tokens.

  jax.nn.dot_product_attention over q, k, v [N, T, H, D] padded to the next
  multiple of `multiple` tokens, with the real length T passed as the query
  and key lengths (the pad keys are masked out of every softmax, the pad
  queries' rows dropped): the unpadded attention's function and gradients.
  bias [..., T, T] is zero-padded. The lengths are built from q, so a
  vmapped caller batches them with q.

  Args:
    q: [N, T, H, D].
    k: [N, T, H, D].
    v: [N, T, H, D].
    bias: [..., T, T] or None.
    implementation: jax.nn.dot_product_attention's implementation.
    multiple: the padded length's multiple.

  Returns:
    [N, T, H, D].

  Raises:
    ValueError: q and k have different lengths.
  """
  t = q.shape[1]
  if k.shape[1] != t:
    raise ValueError(
        f"attend_padded: self-attention only, q has {t} tokens, k {k.shape[1]}"
    )
  tp = -(-t // multiple) * multiple
  if tp == t:
    return jax.nn.dot_product_attention(
        q, k, v, bias=bias, implementation=implementation
    )

  def pad(x):
    """Pads the sequence axis of `x` to `tp`.

    Args:
      x: [B, T, H, D].

    Returns:
      [B, tp, H, D].
    """
    return jnp.pad(x, ((0, 0), (0, tp - t), (0, 0), (0, 0)))

  if bias is not None:
    bias = jnp.pad(
        bias, [(0, 0)] * (bias.ndim - 2) + [(0, tp - t), (0, tp - t)]
    )
  lens = (q[:, 0, 0, 0] * 0).astype(jnp.int32) + jnp.int32(t)
  o = jax.nn.dot_product_attention(
      pad(q),
      pad(k),
      pad(v),
      bias=bias,
      query_seq_lengths=lens,
      key_value_seq_lengths=lens,
      implementation=implementation,
  )
  return o[:, :t]


@functools.cache
def cuda_backend() -> bool:
  """Whether this process has a CUDA (GPU) backend, so cuDNN can trace.

  Returns:
    bool.
  """
  try:
    return bool(jax.devices("gpu"))
  except RuntimeError:
    return False


def auto_uses_cudnn(dtype) -> bool:
  """Whether attention "auto" runs cuDNN's fused attention for `dtype`.

  True in a process with a CUDA backend for a bf16 or fp16 compute dtype
  (CUDNN_AUTO_DTYPES); everything else is the XLA path.

  Args:
    dtype: the network's compute dtype.

  Returns:
    bool.
  """
  dt = jnp.dtype(dtype)
  return dt in tuple(jnp.dtype(d) for d in CUDNN_AUTO_DTYPES) and cuda_backend()


def attend(q, k, v, bias, implementation: str):
  """Self-attention with `implementation` "xla", "cudnn" or "auto".

  "auto" is cuDNN's fused attention when lowered for CUDA and the compute
  dtype (q's) is bf16 or fp16 (auto_uses_cudnn), the XLA path on any other
  platform (lax.platform_dependent) and for an f32 network on any platform,
  so one bf16 config trains with cuDNN on GPU and runs anywhere, and an f32
  network under "auto" is the "xla" network bit for bit everywhere. The two
  paths compute the same function up to floating-point summation order. The
  cuDNN path pads the sequence to a multiple of CUDNN_SEQ_MULTIPLE
  (attend_padded): its backward has no kernel for some lengths. An explicit
  "cudnn" runs cuDNN whatever the dtype.

  "auto" in a process with no CUDA backend (a TPU or CPU host, including a
  CPU host exporting for ("cuda", "tpu")) takes the XLA path in every
  branch: platform_dependent traces every branch up front, and jax's cuDNN
  attention checks for cuDNN in Python while it traces.

  Args:
    q: [N, T, H, D].
    k: [N, T, H, D].
    v: [N, T, H, D].
    bias: [..., T, T] or None.
    implementation: "xla", "cudnn" or "auto".

  Returns:
    [N, T, H, D].
  """
  if implementation == "cudnn":
    return attend_padded(q, k, v, bias, "cudnn")
  if implementation != "auto":
    return jax.nn.dot_product_attention(
        q, k, v, bias=bias, implementation=implementation
    )

  def xla(q_, k_, v_, b_):
    """XLA attention.

    Args:
      q_: queries.
      k_: keys.
      v_: values.
      b_: the bias, or None.

    Returns:
      The attention output.
    """
    return jax.nn.dot_product_attention(
        q_, k_, v_, bias=b_, implementation="xla"
    )

  if not auto_uses_cudnn(q.dtype):
    return xla(q, k, v, bias)

  def cudnn(q_, k_, v_, b_):
    """Attention through cuDNN over a padded sequence (attend_padded).

    Args:
      q_: queries.
      k_: keys.
      v_: values.
      b_: the bias, or None.

    Returns:
      The attention output.
    """
    return attend_padded(q_, k_, v_, b_, "cudnn")

  return jax.lax.platform_dependent(q, k, v, bias, cuda=cudnn, default=xla)


class _Block(nn.Module):
  """Pre-RMSNorm attention (QK-norm, no QKV bias, zero-init out) + GELU FFN.

  Attributes:
    d_model: the token width.
    num_heads: the attention heads.
    ffn_ratio: the FFN width over d_model.
    dtype: the compute dtype.
    attention: the attention implementation (ATTENTION_IMPLS).
  """

  d_model: int
  num_heads: int
  ffn_ratio: int
  dtype: Any
  attention: str

  @nn.compact
  def __call__(self, x, bias):
    """One block.

    Args:
      x: [N, T, d_model] tokens.
      bias: [H, T, T] attention bias.

    Returns:
      The new tokens, x's dtype.
    """
    nh = self.num_heads
    hd = self.d_model // nh
    d = self.d_model
    h = nn.RMSNorm(dtype=self.dtype, param_dtype=F32, name="attn_norm")(x)
    w = jnp.concatenate(
        [
            Kernel((d, nh, hd), name=n)().reshape(d, nh * hd)
            for n in ("q", "k", "v")
        ],
        axis=-1,
    )
    qkv = mm(h, w, self.dtype).reshape(h.shape[:-1] + (3, nh, hd))
    q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
    q = nn.RMSNorm(dtype=self.dtype, param_dtype=F32, name="q_norm")(q).astype(
        self.dtype
    )
    k = nn.RMSNorm(dtype=self.dtype, param_dtype=F32, name="k_norm")(k).astype(
        self.dtype
    )
    o = attend(q, k, v, bias[None].astype(self.dtype), self.attention)
    o = nn.DenseGeneral(
        self.d_model,
        axis=(-2, -1),
        use_bias=False,
        dtype=self.dtype,
        param_dtype=F32,
        kernel_init=nn.initializers.zeros,
        name="attn_out",
    )(o)
    x = x + o.astype(x.dtype)
    h = nn.RMSNorm(dtype=self.dtype, param_dtype=F32, name="ffn_norm")(x)
    h = nn.Dense(
        self.ffn_ratio * self.d_model,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=F32,
        name="ffn_in",
    )(h)
    h = nn.Dense(
        self.d_model,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=F32,
        kernel_init=nn.initializers.zeros,
        name="ffn_out",
    )(nn.gelu(h))
    return x + h.astype(x.dtype)


class CardTokenTorso(nn.Module):
  """The gin card-token transformer torso.

  56 tokens: 52 cards (suit/rank/deadwood-value embeds + card_proj of the
  card fields + the folded touch-row history) and [ACT] [VAL] [CTX] [OPP];
  num_layers pre-RMSNorm blocks with a learned per-layer/head meld-relation
  bias on the card block; a final f32 RMSNorm.

  Attributes:
    d_model: the token width.
    num_layers: the attention blocks.
    num_heads: the attention heads.
    ffn_ratio: the FFN width over d_model.
    dtype: the compute dtype, "float32" or "bfloat16".
    residual_dtype: the residual stream's dtype, "float32" or "bfloat16".
    row_mlp: fold the touch rows through a GELU row MLP (else summed).
    attention: the attention implementation (ATTENTION_IMPLS).
    card_fields: the fields per card in the input.
    global_fields: the global fields in the input.
  """

  d_model: int = 192
  num_layers: int = 4
  num_heads: int = 6
  ffn_ratio: int = 2
  dtype: str = "bfloat16"
  residual_dtype: str = "float32"
  row_mlp: bool = True
  attention: str = "xla"
  card_fields: int = 4
  global_fields: int = 24

  def _check(self):
    """Checks the fields.

    Raises:
      ValueError: a field out of range.
    """
    if self.d_model % self.num_heads:
      raise ValueError(
          f"d_model={self.d_model} not divisible by num_heads={self.num_heads}"
      )
    if self.attention not in ATTENTION_IMPLS:
      raise ValueError(
          f"attention must be one of {ATTENTION_IMPLS}, got {self.attention!r}"
      )
    if self.ffn_ratio < 1 or self.num_layers < 1:
      raise ValueError("ffn_ratio and num_layers must be >= 1")
    if self.dtype not in ("float32", "bfloat16") or self.residual_dtype not in (
        "float32",
        "bfloat16",
    ):
      raise ValueError(
          "dtype/residual_dtype must be float32|bfloat16: "
          f"{self.dtype}, {self.residual_dtype}"
      )

  @nn.compact
  def __call__(self, obs):
    """The torso features of a batch of card-token vectors.

    Args:
      obs: f32[N, D] card-token vectors.

    Returns:
      {"act": f32[N, d], "value": f32[N, d], "cards": f32[N, 52, d],
      "upcard": f32[N, 52] (the is-upcard card field)}.

    Raises:
      ValueError: obs is not [N, D], or a field is out of range.
    """
    self._check()
    if obs.ndim != 2:
      raise ValueError(f"CardTokenTorso wants [N, D] obs, got {obs.shape}")
    cdt = layers.dtype_of(self.dtype)
    d = self.d_model
    n = obs.shape[0]
    cards, glob, touch = ct.split(
        obs.astype(F32), self.card_fields, self.global_fields
    )

    cid = jnp.arange(ct.NUM_CARDS, dtype=jnp.int32)
    ident = (
        nn.Embed(4, d, param_dtype=F32, name="suit_embed")(cid // 13)
        + nn.Embed(13, d, param_dtype=F32, name="rank_embed")(cid % 13)
        + nn.Embed(10, d, param_dtype=F32, name="value_embed")(
            jnp.asarray(card_deadwood_value())
        )
    )
    x_cards = ident[None] + layers.dense(d, cdt, "card_proj")(
        cards.astype(cdt)
    ).astype(F32)

    card1, kind, t1 = touch[..., 0], touch[..., 1], touch[..., 2]
    now = jnp.round(glob[:, -1]).astype(jnp.int32)
    valid = kind != ct.KIND_PAD
    age = jnp.clip(now[:, None] - t1, 0, ct.EVENTS_MAX)
    tables = [
        Table(ct.NUM_KINDS, d, name="kind_embed")(),
        Table(ct.NUM_TIME_CODES, d, name="time_embed")(),
        Table(ct.EVENTS_MAX + 1, d, name="age_embed")(),
    ]
    off_t = ct.NUM_KINDS
    off_a = ct.NUM_KINDS + ct.NUM_TIME_CODES
    idx = [kind, t1 + off_t, age + off_a]
    table = jnp.concatenate(tables, axis=0)
    onehot = jax.nn.one_hot(card1, ct.NUM_TOUCH_CARD_CODES, dtype=cdt)
    if self.row_mlp:
      w_in, b_in = Kernel((d, d), use_bias=True, name="row_mlp_in")()
      table = jnp.dot(table, w_in, precision=jax.lax.Precision.HIGHEST)
      idx.append(jnp.full_like(kind, table.shape[0]))
      table = jnp.concatenate([table, b_in[None]], axis=0)
      row = _sum_rows(table, jnp.stack(idx, axis=-1), cdt).astype(cdt)
      row = jnp.where(valid[..., None], nn.gelu(row), 0).astype(cdt)
      w_out, b_out = Kernel((d, d), use_bias=True, name="row_mlp_out")()
      summed = jnp.einsum(
          "nrc,nrd->ncd", onehot, row, preferred_element_type=F32
      )
      count = jnp.einsum(
          "nrc,nr->nc", onehot, valid.astype(cdt), preferred_element_type=F32
      )
      folded = mm(summed, w_out, cdt, F32) + count[..., None] * b_out.astype(
          cdt
      ).astype(F32)
    else:
      row = _sum_rows(table, jnp.stack(idx, axis=-1), cdt)
      row = jnp.where(valid[..., None], row, 0).astype(cdt)
      folded = jnp.einsum(
          "nrc,nrd->ncd", onehot, row, preferred_element_type=F32
      )
    x_cards = x_cards + folded[:, 1:]

    g = self.param(
        "global_tokens",
        nn.initializers.normal(0.02),
        (NUM_GLOBAL_TOKENS, d),
        F32,
    )
    g = jnp.broadcast_to(g[None], (n, NUM_GLOBAL_TOKENS, d))
    glob_feats = jnp.concatenate(
        [glob[:, :-1], (now.astype(F32) / ct.EVENTS_MAX)[:, None]], axis=-1
    )
    ctx = layers.dense(d, cdt, "ctx_proj")(glob_feats.astype(cdt))
    ctx = ctx.astype(F32) + folded[:, 0]
    opp_counts = (
        jnp.stack(
            [jnp.sum((kind == k) & valid, axis=-1) for k in ct.OPP_KINDS],
            axis=-1,
        ).astype(F32)
        / 16.0
    )
    opp_feats = jnp.concatenate(
        [cards[..., ct.CARD_FIELD_OPP_KNOWN], opp_counts], axis=-1
    )
    opp = layers.dense(d, cdt, "opp_proj")(opp_feats.astype(cdt))
    g = g.at[:, TOK_CTX].add(ctx).at[:, TOK_OPP].add(opp.astype(F32))
    x = jnp.concatenate([g, x_cards], axis=1).astype(
        layers.dtype_of(self.residual_dtype)
    )

    rel = jnp.asarray(card_relation_ids())
    for i in range(self.num_layers):
      tbl = self.param(
          f"rel_bias_{i}",
          nn.initializers.zeros,
          (NUM_RELATIONS, self.num_heads),
          F32,
      )
      card_bias = jnp.transpose(tbl[rel], (2, 0, 1))
      bias = jnp.zeros((self.num_heads, NUM_TOKENS, NUM_TOKENS), F32)
      bias = bias.at[:, NUM_GLOBAL_TOKENS:, NUM_GLOBAL_TOKENS:].set(card_bias)
      x = _Block(
          d_model=d,
          num_heads=self.num_heads,
          ffn_ratio=self.ffn_ratio,
          dtype=cdt,
          attention=self.attention,
          name=f"block_{i}",
      )(x, bias)
    x = nn.RMSNorm(dtype=F32, param_dtype=F32, name="final_norm")(x.astype(F32))
    return {
        "act": x[:, TOK_ACT],
        "value": x[:, TOK_VAL],
        "cards": x[:, NUM_GLOBAL_TOKENS:],
        "upcard": cards[..., ct.CARD_FIELD_UPCARD],
    }


def _cards_and_melds():
  """The JAX gin game's card and meld tables (imported on first use).

  Lazy: the networks package imports without the games package; only a
  pointer head needs it.

  Returns:
    The open_spiel.python.jax.games.gin_rummy.cards_and_melds module.
  """
  # pylint: disable-next=g-import-not-at-top,import-outside-toplevel
  from open_spiel.python.jax.games.gin_rummy import cards_and_melds  # noqa: PLC0415

  return cards_and_melds


def meld_incidence() -> np.ndarray:
  """Which cards each gin meld holds, normalised by its size.

  Returns:
    f32[185, 52]: row m is meld m's cards / its size (the JAX game's meld
    order).
  """
  cm = _cards_and_melds()
  m = np.asarray(cm.get_meld_cards_bool(), dtype=np.float32)
  assert m.shape == (cm.NUM_MELDS, cm.NUM_CARDS), m.shape
  return m / m.sum(axis=1, keepdims=True)


class PointerPolicyHead(nn.Module):
  """Gin pointer policy over CardTokenTorso features.

  pointer(p)[n, c] = <W_q^p h_act, W_k^p h_c> / sqrt(d), computed as a
  per-row query u = W_k^T W_q h_act dotted with the card tokens, for p in
  (discard, upcard, meld). Logits: 0-51 discard pointer; 52 draw upcard =
  special[0] + the upcard pointer at the card flagged upcard; 53-55 =
  special[1:4] (stock, pass, knock) from an MLP on [ACT]; 56-240 melds = the
  meld pointer at the mean of the meld's card scores (the 185 x 52
  incidence).

  Attributes:
    num_actions: the gin actions (241).
    dtype: the compute dtype.
    score_dtype: the dtype of the scaled pointer scores and the logit
      assembly. "float64" (the default) keeps the scaled pointer scores in
      float64 explicitly rather than through dtype promotion, so trained
      checkpoints reproduce bit for bit.
  """

  num_actions: int = 241
  dtype: str = "bfloat16"
  score_dtype: str = "float64"

  @nn.compact
  def __call__(self, feats):
    """The gin logits.

    Args:
      feats: CardTokenTorso's features.

    Returns:
      f32[N, 241].

    Raises:
      ValueError: num_actions is not the gin action count.
    """
    cm = _cards_and_melds()
    assert (
        cm.ACTION_DRAW_UPCARD,
        cm.ACTION_DRAW_STOCK,
        cm.ACTION_PASS,
        cm.ACTION_KNOCK,
    ) == (52, 53, 54, 55)
    if self.num_actions != cm.NUM_ACTIONS:
      raise ValueError(
          f"PointerPolicyHead emits the {cm.NUM_ACTIONS} gin actions, "
          f"got num_actions={self.num_actions}"
      )
    cdt = layers.dtype_of(self.dtype)
    sdt = layers.dtype_of(self.score_dtype)
    h_act, h_cards, is_up = feats["act"], feats["cards"], feats["upcard"]
    d = h_act.shape[-1]
    scale = (1.0 / np.sqrt(d)).astype(np.float64)

    us = []
    for name in ("discard", "upcard", "meld"):
      wq = Kernel((d, d), name=f"{name}_q")()
      wk = Kernel((d, d), name=f"{name}_k")()
      q = mm(h_act, wq, cdt)
      us.append(mm(q, wk.T, cdt, F32))
    u = jnp.stack(us, axis=1)  # [N, 3, d]
    ptr = jnp.einsum(
        "npd,nkd->npk",
        u.astype(cdt),
        h_cards.astype(cdt),
        preferred_element_type=F32,
    )
    ptr = ptr.astype(sdt) * jnp.asarray(scale, sdt)  # [N, 3, 52]
    discard = ptr[:, 0]
    upcard_ptr = jnp.sum(ptr[:, 1] * is_up.astype(sdt), axis=-1)
    special = layers.dense(
        4, F32, "special_out", kernel_init=nn.initializers.zeros
    )(nn.gelu(layers.dense(d, cdt, "special_hidden")(h_act.astype(cdt))))
    melds = jnp.einsum(
        "mc,nc->nm", jnp.asarray(meld_incidence()).astype(sdt), ptr[:, 2]
    )
    return jnp.concatenate(
        [
            discard,
            (special[:, 0].astype(sdt) + upcard_ptr)[:, None],
            special[:, 1:4].astype(sdt),
            melds,
        ],
        axis=-1,
    ).astype(F32)


def card_tokens(
    *,
    d_model: int,
    num_layers: int,
    num_heads: int,
    ffn_ratio: int,
    compute_dtype: str,
    value_head: str,
    card_fields: int,
    global_fields: int,
    hl_gauss: Optional[dict[str, Any]] = None,
    residual_dtype: str = "float32",
    row_mlp: bool = True,
    attention: str = "xla",
    chunk_size: Optional[int] = 16384,
) -> dict[str, Any]:
  """The network config of a gin card-token transformer.

  Args:
    d_model: the token width.
    num_layers: the attention blocks.
    num_heads: the attention heads.
    ffn_ratio: the FFN width over d_model.
    compute_dtype: "float32" or "bfloat16".
    value_head: "scalar" or "hl_gauss".
    card_fields: the fields per card in the input.
    global_fields: the global fields in the input.
    hl_gauss: {num_bins, v_min, v_max, sigma_ratio} when value_head is
      "hl_gauss", else None.
    residual_dtype: the residual stream's dtype.
    row_mlp: fold the touch rows through the row MLP.
    attention: execution only, the attention implementation.
    chunk_size: execution only, rows per forward chunk.

  Returns:
    The network config (`actor_critic.build_network`).
  """
  vh: dict[str, Any] = {
      "kind": value_head,
      "hidden": int(d_model),
      "hidden_activation": "gelu",
      "dtype": compute_dtype,
      "out_dtype": "float32",
  }
  if value_head == "hl_gauss":
    assert hl_gauss is not None, "an hl_gauss value head needs its support"
    vh.update(
        num_bins=int(hl_gauss["num_bins"]),
        v_min=float(hl_gauss["v_min"]),
        v_max=float(hl_gauss["v_max"]),
        sigma_ratio=float(hl_gauss["sigma_ratio"]),
    )
  else:
    assert hl_gauss is None, hl_gauss
  return {
      "compute_dtype": compute_dtype,
      "torso": {
          "kind": "card_tokens",
          "d_model": int(d_model),
          "num_layers": int(num_layers),
          "num_heads": int(num_heads),
          "ffn_ratio": int(ffn_ratio),
          "residual_dtype": residual_dtype,
          "row_mlp": bool(row_mlp),
          "attention": attention,
          "card_fields": int(card_fields),
          "global_fields": int(global_fields),
      },
      "policy_head": {"kind": "pointer", "score_dtype": "float64"},
      "value_head": vh,
      # An opponent-hand auxiliary head: no loss reads it here; it is part of
      # the param tree so trained checkpoints load unchanged.
      "aux_heads": [{
          "key": "opp_hand",
          "features": 1,
          "source": "cards",
          "dtype": "float32",
          "squeeze": True,
      }],
      "chunk_size": chunk_size,
  }


actor_critic.register_torso("card_tokens", CardTokenTorso)
actor_critic.register_policy_head("pointer", PointerPolicyHead)
