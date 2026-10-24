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

"""The network catalog: named network configs and the builders that spell them.

A network config is the JSON that actor_critic.build_network turns into a
module, and it is recorded in every artifact. Spelling each architecture
family through one builder gives it one spelling, so equal networks have
equal configs and equal config hashes.

  cfg = catalog.mlp([256, 256], "relu", "separate")
  cfg = catalog.get("mlp_256x4")          # a fresh copy of a named config
  catalog.register("my_net", cfg)         # a plugin's named network

Builders:
  mlp(...)  MLPTorso (shared or separate trunks) + dense policy head + scalar
            value head (optional hidden layer, optional tanh) and optional
            aux heads.
  cnn(...)  CNNTorso + dense policy head + scalar value head.
A key equal to the module default ("lecun" init, value hidden 0, no tanh) is
left out.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

LECUN = "lecun"


def _init(scheme: str) -> dict[str, Any]:
  """The init kwargs of a scheme.

  Args:
    scheme: The init scheme.

  Returns:
    {} for LeCun, else {init_scheme: scheme}.
  """
  return {} if scheme == LECUN else {"init_scheme": scheme}


def mlp(
    widths: Sequence[int],
    activation: str,
    layout: str,
    *,
    init_scheme: str = LECUN,
    compute_dtype: str = "float32",
    value_hidden: int = 0,
    value_hidden_activation: str = "relu",
    value_tanh: bool = False,
    aux_heads: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
  """An MLP network config (module doc).

  Args:
    widths: hidden widths of the trunk(s).
    activation: a layers.ACTIVATIONS name.
    layout: "shared" (one trunk) or "separate" (actor and critic trunks).
    init_scheme: "lecun" or "orthogonal".
    compute_dtype: "float32" or "bfloat16" (params are float32 either way).
    value_hidden: width of a value-head hidden layer (0: none).
    value_hidden_activation: its activation.
    value_tanh: squash the value with tanh.
    aux_heads: aux heads (heads.AuxHead), each {"key": the aux output's
      name, "source": the torso feature it reads, optionally "features"
      (default 1) and other AuxHead fields}; the init scheme is added.

  Returns:
    The network config dict.

  Raises:
    ValueError: an unknown layout or compute dtype, or an aux head without a
      key and a source.
  """
  if layout not in ("shared", "separate"):
    raise ValueError(f"layout must be 'shared' or 'separate', got {layout!r}")
  if compute_dtype not in ("float32", "bfloat16"):
    raise ValueError(
        f"compute_dtype must be float32 or bfloat16, got {compute_dtype!r}"
    )
  head = _init(init_scheme)
  value = {"kind": "scalar", **head}
  if int(value_hidden):
    value.update(
        hidden=int(value_hidden), hidden_activation=value_hidden_activation
    )
  if value_tanh:
    value["tanh"] = True
  net = {
      "compute_dtype": compute_dtype,
      "torso": {
          "kind": "mlp",
          "widths": [int(w) for w in widths],
          "layout": layout,
          "activation": activation,
          **head,
      },
      "policy_head": {"kind": "dense", **head},
      "value_head": value,
  }
  if aux_heads:
    for h in aux_heads:
      if "key" not in h or "source" not in h:
        raise ValueError(
            f"an aux head needs a key and a source, got {dict(h)!r}"
        )
    net["aux_heads"] = [{**dict(h), **head} for h in aux_heads]
  return net


def cnn(
    channels: int,
    relative_stack_sizes: Sequence[float],
    num_blocks: int,
    dense_hidden: int,
    *,
    value_tanh: bool = False,
    compute_dtype: str = "float32",
) -> dict[str, Any]:
  """An IMPALA-CNN network config (torsos.CNNTorso).

  Stack i has relative_stack_sizes[i] * channels channels and num_blocks
  residual blocks, then a dense_hidden layer.

  Args:
    channels: the base channel count.
    relative_stack_sizes: each stack's channels as a fraction of `channels`.
    num_blocks: residual blocks per stack.
    dense_hidden: the width of the dense layer after the stacks.
    value_tanh: squash the value with tanh.
    compute_dtype: "float32" or "bfloat16".

  Returns:
    The network config dict.
  """
  value = {"kind": "scalar"}
  if value_tanh:
    value["tanh"] = True
  return {
      "compute_dtype": compute_dtype,
      "torso": {
          "kind": "cnn",
          "channels": int(channels),
          "relative_stack_sizes": [float(r) for r in relative_stack_sizes],
          "num_blocks": int(num_blocks),
          "dense_hidden": int(dense_hidden),
      },
      "policy_head": {"kind": "dense"},
      "value_head": value,
  }


NAMED: dict[str, dict[str, Any]] = {
    # Shared 4 x 256 ReLU trunk, a 256-unit value layer and a tanh value: the
    # AlphaZero example network.
    "mlp_256x4": mlp(
        [256] * 4, "relu", "shared", value_hidden=256, value_tanh=True
    ),
    # Shared 2 x 256 ReLU trunk, a 256-unit value layer and a tanh value: the
    # behaviour-cloning example network.
    "mlp_256x2": mlp(
        [256, 256], "relu", "shared", value_hidden=256, value_tanh=True
    ),
}


def register(name: str, config: dict[str, Any]) -> None:
  """Adds a named network config; a name holds one config.

  Args:
    name: the config's name.
    config: the network config.

  Raises:
    TypeError: `name` is not a non-empty string or `config` is not a dict.
    ValueError: the name holds a different config.
  """
  if not isinstance(name, str) or not name:
    raise TypeError(f"a network name is a non-empty string, got {name!r}")
  if not isinstance(config, dict):
    raise TypeError(f"a network config is a dict, got {type(config).__name__}")
  if name in NAMED and NAMED[name] != config:
    raise ValueError(f"network {name!r} is already in the catalog")
  NAMED[name] = copy.deepcopy(config)


def get(name: str) -> dict[str, Any]:
  """A fresh copy of the named network config.

  Args:
    name: the config's name.

  Returns:
    A deep copy of the config.

  Raises:
    KeyError: an unknown name.
  """
  if name not in NAMED:
    raise KeyError(f"unknown network {name!r} (catalog: {sorted(NAMED)})")
  return copy.deepcopy(NAMED[name])
