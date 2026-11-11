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

"""Test helpers for networks: encoders, steps and network configs."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn


class ObsInfo(NamedTuple):
  """An InfoState that is a plain feature array.

  Attributes:
    obs: the features [B, ...].
  """

  obs: jax.Array


def obs_field(info):
  """Encoder for ObsInfo: the features are the info's obs array.

  Args:
    info: an ObsInfo, leaves [B, ...].

  Returns:
    info.obs.
  """
  return info.obs


def kuhn_onehot(info):
  """Kuhn InfoState -> one_hot(info_index).

  Args:
    info: the InfoState, leaves [B, ...].

  Returns:
    f32[B, 12].
  """
  return jax.nn.one_hot(
      kuhn.info_index(info), kuhn.NUM_INFO_STATES, dtype=jnp.float32
  )


def kuhn_planes(info):
  """Kuhn InfoState -> planes (for a CNN torso).

  Args:
    info: the InfoState, leaves [B, ...].

  Returns:
    f32[B, 2, 6, 1].
  """
  return kuhn_onehot(info).reshape(-1, 2, 6, 1)


KUHN_ONEHOT = f"{__name__}:kuhn_onehot"
KUHN_PLANES = f"{__name__}:kuhn_planes"
OBS_FIELD = f"{__name__}:obs_field"

ENCODED_ROWS: list[int] = []  # the row counts recording_obs was traced at


def recording_obs(info):
  """obs_field that records the batch size it encodes (in ENCODED_ROWS).

  Args:
    info: an ObsInfo, leaves [B, ...].

  Returns:
    info.obs.
  """
  ENCODED_ROWS.append(int(info.obs.shape[0]))
  return info.obs


RECORDING_OBS = f"{__name__}:recording_obs"


def obs_steps(
    x, seat=0, legal=None, num_actions=None, new_hand=True, new_match=True
):
  """A SeatStep [B] whose info is ObsInfo(x).

  Args:
    x: the features [B, ...].
    seat: the seat of every row.
    legal: bool[B, A] legal mask; None is all legal.
    num_actions: A, needed when `legal` is None.
    new_hand: the new_hand flag of every row.
    new_match: the new_match flag of every row.

  Returns:
    The SeatStep.
  """
  b = x.shape[0]
  if legal is None:
    legal = jnp.ones((b, num_actions), jnp.bool_)
  return types.SeatStep(
      info=ObsInfo(obs=jnp.asarray(x)),
      legal_mask=jnp.asarray(legal, jnp.bool_),
      seat=jnp.full((b,), seat, jnp.int32),
      to_move=jnp.full((b,), seat, jnp.int32),
      new_hand=jnp.full((b,), new_hand),
      new_match=jnp.full((b,), new_match),
      last_action=jnp.full((b,), types.NO_ACTION, jnp.int32),
      last_actor=jnp.full((b,), types.NO_ACTOR, jnp.int32),
  )


def mlp_config(
    dtype="float32", layout="shared", widths=(16, 16), value="scalar", aux=True
):
  """A small MLP network config.

  Args:
    dtype: the compute dtype.
    layout: "shared" or "separate".
    widths: the trunk widths.
    value: the value head kind.
    aux: add a "deadwood" aux head.

  Returns:
    The network config.
  """
  vh = {"kind": value}
  if value == "hl_gauss":
    vh.update(num_bins=11, v_min=-2.0, v_max=2.0)
  cfg = {
      "compute_dtype": dtype,
      "torso": {"kind": "mlp", "widths": list(widths), "layout": layout},
      "policy_head": {"kind": "dense"},
      "value_head": vh,
  }
  if aux:
    cfg["aux_heads"] = [{"key": "deadwood", "source": "value"}]
  return cfg


def cnn_config(dtype="float32"):
  """A small CNN network config.

  Args:
    dtype: the compute dtype.

  Returns:
    The network config.
  """
  return {
      "compute_dtype": dtype,
      "torso": {
          "kind": "cnn",
          "channels": 8,
          "relative_stack_sizes": [0.5, 1.0],
          "num_blocks": 1,
          "dense_hidden": 16,
      },
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar", "tanh": True},
  }


def rnn_config(dtype="float32", use_last_action=True):
  """A small recurrent network config.

  Args:
    dtype: the compute dtype.
    use_last_action: feed the last-action one-hot to the GRU.

  Returns:
    The network config.
  """
  return {
      "compute_dtype": dtype,
      "torso": {
          "kind": "recurrent",
          "hidden_size": 8,
          "use_last_action": use_last_action,
          "inner": {"kind": "mlp", "widths": [16], "layout": "shared"},
      },
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }
