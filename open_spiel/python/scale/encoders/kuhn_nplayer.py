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

"""Encoders of envs.kuhn_nplayer (KuhnNInfo), one family per encoding.

Sized by the game's `players` option:

  kuhn_nplayer_info_state_tensor   f32[6N-1]: OpenSpiel kuhn_poker(players=N)
      information_state_tensor: one_hot(seat, N) ++ one_hot(own card, N+1)
      ++ betting [2N-1, 2] (row k: one_hot of action k once played), flat.
  kuhn_nplayer_observation_tensor  f32[3N+1]: OpenSpiel's
      observation_tensor: one_hot(seat, N) ++ one_hot(own card, N+1) ++ each
      seat's chips in (ante 1 plus its bets so far). Not perfect recall for
      N > 2.

Both equal pyspiel's tensors on every node of random games (the tests).
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import kuhn_nplayer

F32 = jnp.float32
GAME = "kuhn_nplayer"
INFO_STATE = "kuhn_nplayer_info_state_tensor"
OBSERVATION = "kuhn_nplayer_observation_tensor"


def _check(info: kuhn_nplayer.KuhnNInfo, n: int) -> None:
  if info.bets.shape != (kuhn_nplayer.max_actions(n),):
    raise ValueError(
        f"kuhn_nplayer({n}) encoder got bets of shape {info.bets.shape}"
    )


def _seat_and_card(info, n: int) -> jax.Array:
  """The seat and card one-hots of an infoset.

  Args:
    info: The KuhnNPlayer infoset.
    n: The player count.

  Returns:
    f32[2n + 1].
  """
  card = jnp.asarray(info.card, jnp.int32)
  card_oh = jnp.where(
      card >= 0,
      jax.nn.one_hot(jnp.maximum(card, 0), n + 1, dtype=F32),
      jnp.zeros((n + 1,), F32),
  )
  return jnp.concatenate([jax.nn.one_hot(info.seat, n, dtype=F32), card_oh])


def info_state_tensor(info: kuhn_nplayer.KuhnNInfo, n: int) -> jax.Array:
  """OpenSpiel's information-state tensor f32[6N-1] (module doc).

  Args:
    info: the KuhnNInfo.
    n: N, the number of seats.

  Returns:
    f32[6N-1].
  """
  _check(info, n)
  m = kuhn_nplayer.max_actions(n)
  played = (jnp.arange(m) < info.num_bets)[:, None]
  betting = jnp.where(
      played, jax.nn.one_hot(jnp.maximum(info.bets, 0), 2, dtype=F32), 0.0
  )
  return jnp.concatenate(
      [_seat_and_card(info, n), betting.reshape(-1).astype(F32)]
  )


def observation_tensor(info: kuhn_nplayer.KuhnNInfo, n: int) -> jax.Array:
  """OpenSpiel's observation tensor f32[3N+1] (module doc).

  Args:
    info: the KuhnNInfo.
    n: N, the number of seats.

  Returns:
    f32[3N+1].
  """
  _check(info, n)
  m = kuhn_nplayer.max_actions(n)
  k = jnp.arange(m)
  is_bet = (k < info.num_bets) & (info.bets == kuhn_nplayer.BET)
  by_seat = (k % n)[None, :] == jnp.arange(n)[:, None]  # [N, 2N-1]
  chips = 1 + jnp.sum(by_seat & is_bet[None, :], axis=1)
  return jnp.concatenate([_seat_and_card(info, n), chips.astype(F32)])


def _players(env_spec) -> int:
  """The player count of an env spec.

  Args:
    env_spec: The EnvSpec.

  Returns:
    Its players option.
  """
  return int(env_spec.kwargs["players"])


def _info_member(env_spec) -> registry.Encoder:
  """The info-state encoder of an env spec.

  Args:
    env_spec: The EnvSpec.

  Returns:
    The Encoder.
  """
  n = _players(env_spec)
  return registry.Encoder(
      name=registry.family_name(INFO_STATE, env_spec),
      game=GAME,
      shape=(6 * n - 1,),
      dtype=F32,
      fn=lambda info: info_state_tensor(info, n),
      reference=f"OpenSpiel kuhn_poker(players={n}) information_state_tensor",
  )


def _obs_member(env_spec) -> registry.Encoder:
  """The observation encoder of an env spec.

  Args:
    env_spec: The EnvSpec.

  Returns:
    The Encoder.
  """
  n = _players(env_spec)
  return registry.Encoder(
      name=registry.family_name(OBSERVATION, env_spec),
      game=GAME,
      shape=(3 * n + 1,),
      dtype=F32,
      fn=lambda info: observation_tensor(info, n),
      reference=f"OpenSpiel kuhn_poker(players={n}) observation_tensor",
  )


registry.register_family(INFO_STATE, GAME, _info_member, options=("players",))
registry.register_family(OBSERVATION, GAME, _obs_member, options=("players",))
