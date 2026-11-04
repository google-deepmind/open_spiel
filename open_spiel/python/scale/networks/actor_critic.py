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

"""ActorCritic: a torso, policy, value and aux heads, from a JSON config.

A network is data: its config is recorded in the policy's arch_kwargs, so an
artifact rebuilds the exact module without code. `build_network` is the one
config -> module builder.

  net = actor_critic.build_network(catalog.get("mlp_256x2"), num_actions=7)
  params = net.init(key, jnp.zeros((1, 126), jnp.float32))
  out = net.apply(params, x)       # {"logits", "value", "value_logits", "aux"}

Network config (JSON; every key is a module field, unknown keys raise):
  {
    "compute_dtype": "float32" | "bfloat16",  # default `dtype` of every part
    "torso": {"kind": "mlp" | "cnn" | "recurrent" | a registered kind, ...},
             # recurrent: {"kind": "recurrent", "hidden_size", "inner": {torso}}
    "policy_head": {"kind": "dense" | a registered kind, ...},
    "value_head": {"kind": "scalar" | "hl_gauss", ...},
    "aux_heads": [{"key": "my_aux", "source": "value", ...}, ...],
    "chunk_size": null | int,   # execution only: rows per forward chunk
    "remat": true | false,      # execution only (default true): recompute
                                # each chunk's forward in the backward pass
                                # (nn.remat)
  }

Outputs (all float32): logits [N, A], value [N], value_logits [N, bins] or
None, aux {key: ...}. Param tree: torso/..., policy_head/..., value_head/...,
aux_<key>/.... register_torso / register_policy_head add a module kind, so a
plugin's network family builds from the same config format.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from open_spiel.python.scale.networks import heads
from open_spiel.python.scale.networks import torsos

F32 = jnp.float32

TORSOS = {"mlp": torsos.MLPTorso, "cnn": torsos.CNNTorso}
POLICY_HEADS = {"dense": heads.DensePolicyHead}


def register_torso(kind: str, module: Any) -> None:
  """Registers a torso kind (a flax module class taking the config's keys).

  Args:
    kind: the config's torso "kind".
    module: the flax module class.

  Raises:
    ValueError: `kind` is "recurrent" or registered to another class.
  """
  if TORSOS.get(kind, module) is not module or kind == "recurrent":
    raise ValueError(f"torso kind {kind!r} is already registered")
  TORSOS[kind] = module


def register_policy_head(kind: str, module: Any) -> None:
  """Registers a policy head kind (a flax module class taking config keys).

  Args:
    kind: the config's policy head "kind".
    module: the flax module class.

  Raises:
    ValueError: `kind` is registered to another class.
  """
  if POLICY_HEADS.get(kind, module) is not module:
    raise ValueError(f"policy head kind {kind!r} is already registered")
  POLICY_HEADS[kind] = module


_TUPLE_FIELDS = ("widths", "relative_stack_sizes")


#: aux keys a network's own outputs fill (NetworkPolicy.evaluate writes the
#: value head's logits under "value_logits"), so no aux head may use them.
RESERVED_AUX_KEYS = ("value_logits",)


@dataclasses.dataclass(frozen=True)
class AuxSpec:
  """One aux head of a network config (heads.AuxHead fields plus its key).

  Attributes:
    key: the aux output's name.
    features: the head's output width.
    source: the torso feature it reads.
    dtype: the compute dtype.
    init_scheme: "lecun" or "orthogonal".
    squeeze: drop the trailing size-1 axis (needs features == 1).
  """

  key: str
  features: int = 1
  source: str = "value"
  dtype: str = "float32"
  init_scheme: str = "lecun"
  squeeze: bool = True

  def __post_init__(self):
    if self.squeeze and self.features != 1:
      raise ValueError(
          f"aux head {self.key!r}: squeeze needs features == 1, got "
          f"{self.features}; pass squeeze=False"
      )


def _f32(x):
  """`x` as float32, None kept.

  Args:
    x: An array or None.

  Returns:
    The array or None.
  """
  return None if x is None else x.astype(F32)


def chunk_plan(n: int, chunk_size: int) -> tuple[int, int]:
  """(chunks, rows per chunk) of a chunked forward over n rows.

  There are ceil(n / chunk_size) chunks of equal size ceil(n / chunks),
  rounded up to a multiple of 8 rows but never above chunk_size; the last
  chunk is padded. Equal chunks keep the padding under 8 rows per chunk, where
  chunk_size-row chunks could pad up to chunk_size - 1 rows.

  Args:
    n: the number of rows.
    chunk_size: the most rows per chunk.

  Returns:
    (number of chunks, rows per chunk).

  Raises:
    ValueError: `n` or `chunk_size` is below 1.
  """
  n, c = int(n), int(chunk_size)
  if n < 1 or c < 1:
    raise ValueError(
        f"chunk_plan needs n >= 1 rows and chunk_size >= 1, got {n}, {c}"
    )
  if n <= c:
    return 1, n
  num = _ceil_div(n, c)
  size = min(c, _ceil_div(_ceil_div(n, num), 8) * 8)
  num = _ceil_div(n, size)
  return num, size


def _ceil_div(a: int, b: int) -> int:
  """The ceiling of a / b.

  Args:
    a: The dividend.
    b: The divisor.

  Returns:
    The quotient.
  """
  return -(-a // b)


class ActorCritic(nn.Module):
  """A torso, a policy head, a value head and aux heads (module doc).

  Attributes:
    torso: the torso module.
    policy_head: the policy head module.
    value_head: the value head module.
    aux_specs: the aux heads.
    recurrent: whether the torso is recurrent (observe / evaluate_carry).
    chunk_size: rows per forward chunk, or None for one forward.
    remat: recompute each chunk's forward in the backward pass.
    aux: the aux head modules by key (built in setup).
  """

  torso: nn.Module
  policy_head: nn.Module
  value_head: nn.Module
  aux_specs: tuple[AuxSpec, ...] = ()
  recurrent: bool = False
  chunk_size: Optional[int] = None
  # chunked forward: nn.remat per chunk (memory for recompute)
  remat: bool = True

  def setup(self):
    """Builds the aux heads; their keys must be distinct.

    Raises:
      ValueError: two aux heads share a key.
    """
    keys = [s.key for s in self.aux_specs]
    if len(set(keys)) != len(keys):
      raise ValueError(f"duplicate aux head keys {keys}")
    self.aux = {
        s.key: heads.AuxHead(**{
            f.name: getattr(s, f.name)
            for f in dataclasses.fields(s)
            if f.name != "key"
        })
        for s in self.aux_specs
    }

  def _heads(self, feats) -> dict[str, Any]:
    """Every head's output on the torso features.

    Args:
      feats: The torso features.

    Returns:
      {logits, value, value_logits, aux}.
    """
    value, value_logits = self.value_head(feats)
    return {
        "logits": _f32(self.policy_head(feats)),
        "value": _f32(value),
        "value_logits": _f32(value_logits),
        "aux": {k: _f32(m(feats)) for k, m in self.aux.items()},
    }

  def _forward(self, x):
    """The heads on the torso of `x`.

    Args:
      x: The encoded input.

    Returns:
      The heads' outputs.
    """
    return self._heads(self.torso(x))

  def __call__(self, x):
    """Stateless forward over x [N, ...].

    Args:
      x: the encoded features, [N, ...].

    Returns:
      {"logits", "value", "value_logits", "aux"}, all float32.

    Raises:
      ValueError: the network is recurrent.
    """
    if self.recurrent:
      raise ValueError(
          "a recurrent network runs through observe / evaluate_carry"
      )
    n = x.shape[0]
    if (
        self.chunk_size is None
        or n <= int(self.chunk_size)
        or self.is_initializing()
    ):
      return self._forward(x)
    num, c = chunk_plan(n, int(self.chunk_size))
    pad = num * c - n
    if pad:  # pad rows repeat row 0 (a valid input) and are dropped below
      x = jnp.concatenate([x, jnp.broadcast_to(x[:1], (pad,) + x.shape[1:])])
    chunks = x.reshape((num, c) + x.shape[1:])

    def body(mdl, carry, xc):
      """The heads' outputs of one chunk (scan step).

      Args:
        mdl: the module.
        carry: the unused scan carry.
        xc: one chunk of rows.

      Returns:
        (carry, the chunk's outputs).
      """
      # Uses private _forward of a module in this package.
      return carry, mdl._forward(xc)  # pylint: disable=protected-access

    # remat (the default) keeps one chunk's activations live in the backward
    # pass and recomputes its forward; remat=False stores every chunk's
    # activations instead (no recompute, more memory). Same math either way.
    step = nn.remat(body, prevent_cse=False) if self.remat else body
    scanned = nn.scan(
        step,
        variable_broadcast="params",
        split_rngs={"params": False},
        in_axes=0,
        out_axes=0,
    )
    _, out = scanned(self, None, chunks)
    return jax.tree.map(lambda y: y.reshape((num * c,) + y.shape[2:])[:n], out)

  # Recurrent API: the carry h changes only in observe.
  def observe(self, x, h, extra=None):
    """The recurrent carry after observing x.

    Args:
      x: the encoded features, [N, ...].
      h: the carry f32[N, hidden_size].
      extra: extra GRU inputs (the last-action one-hot), or None.

    Returns:
      The new carry.
    """
    return self.torso.step(x, h, extra)

  def evaluate_carry(self, h):
    """The heads' outputs on a recurrent carry.

    Args:
      h: the carry f32[N, hidden_size].

    Returns:
      {"logits", "value", "value_logits", "aux"}, all float32.
    """
    return self._heads(self.torso.features(h))

  def init_recurrent(self, x, h, extra=None):
    """Observe then evaluate: touches every recurrent param (for init).

    Args:
      x: the encoded features, [N, ...].
      h: the carry f32[N, hidden_size].
      extra: extra GRU inputs, or None.

    Returns:
      The heads' outputs on the new carry.
    """
    return self.evaluate_carry(self.observe(x, h, extra))


def _fields(
    cfg: Mapping[str, Any], compute_dtype: str, default_dtype: bool = True
) -> dict[str, Any]:
  """A module's fields from its config.

  Args:
    cfg: The config.
    compute_dtype: The compute dtype.
    default_dtype: Whether to default dtype to it.

  Returns:
    The fields.
  """
  out = {
      k: tuple(v) if k in _TUPLE_FIELDS else v
      for k, v in cfg.items()
      if k != "kind"
  }
  if default_dtype:
    out.setdefault("dtype", compute_dtype)
  return out


def _torso(cfg: Mapping[str, Any], compute_dtype: str) -> nn.Module:
  """The torso module of a config.

  Args:
    cfg: The config.
    compute_dtype: The compute dtype.

  Returns:
    The torso.

  Raises:
    ValueError: The torso kind is unknown.
  """
  kind = cfg["kind"]
  if kind == "recurrent":
    rest = {k: v for k, v in cfg.items() if k not in ("kind", "inner")}
    inner = _torso(cfg["inner"], compute_dtype)
    rest.setdefault("dtype", compute_dtype)
    return torsos.RecurrentTorso(inner=inner, **rest)
  if kind not in TORSOS:
    raise ValueError(
        f"unknown torso kind {kind!r}; known {sorted(TORSOS) + ['recurrent']}"
    )
  return TORSOS[kind](**_fields(cfg, compute_dtype))


def build_network(cfg: Mapping[str, Any], num_actions: int) -> ActorCritic:
  """The ActorCritic for a network config (see the module doc).

  Args:
    cfg: the network config.
    num_actions: the policy head's number of actions.

  Returns:
    The ActorCritic module.

  Raises:
    ValueError: an unknown key or kind, a num_actions mismatch, or repeated or
      reserved aux head keys.
  """
  known = {
      "compute_dtype",
      "torso",
      "policy_head",
      "value_head",
      "aux_heads",
      "chunk_size",
      "remat",
  }
  extra = set(cfg) - known
  if extra:
    raise ValueError(f"unknown network config keys {sorted(extra)}")
  cdt = cfg.get("compute_dtype", "float32")
  torso = _torso(cfg["torso"], cdt)
  ph = dict(cfg["policy_head"])
  kind = ph.pop("kind")
  if kind not in POLICY_HEADS:
    raise ValueError(
        f"unknown policy head kind {kind!r}; known {sorted(POLICY_HEADS)}"
    )
  ph.setdefault("num_actions", num_actions)
  if ph["num_actions"] != num_actions:
    raise ValueError(
        f"policy head num_actions {ph['num_actions']} != {num_actions}"
    )
  policy_head = POLICY_HEADS[kind](**_fields(ph, cdt))
  vh = dict(cfg["value_head"])
  value_head = heads.ValueHead(
      **{**_fields(vh, cdt), "kind": vh.get("kind", "scalar")}
  )
  aux = tuple(AuxSpec(**{"dtype": cdt, **a}) for a in cfg.get("aux_heads", ()))
  keys = [a.key for a in aux]
  if len(set(keys)) != len(keys):
    raise ValueError(f"aux head keys repeat: {keys}")
  taken = sorted(set(keys) & set(RESERVED_AUX_KEYS))
  if taken:
    raise ValueError(
        f"aux head keys {taken} are reserved: the policy writes them itself"
    )
  return ActorCritic(
      torso=torso,
      policy_head=policy_head,
      value_head=value_head,
      aux_specs=aux,
      recurrent=cfg["torso"]["kind"] == "recurrent",
      chunk_size=cfg.get("chunk_size"),
      remat=_remat(cfg.get("remat", True)),
  )


def _remat(v) -> bool:
  """Checks the remat flag.

  Args:
    v: The flag value.

  Returns:
    The flag.

  Raises:
    ValueError: It is not a bool.
  """
  if not isinstance(v, bool):
    raise ValueError(f"network remat must be true or false, got {v!r}")
  return v
