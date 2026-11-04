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

"""JaxGameEnv: the base class of every Env over a JAX game.

JAX games are pure-JAX OpenSpiel games: a state is a pytree and every
method is jittable. This class supplies the dynamics and chance layers of the
Env contract from the Game; a per-game subclass adds what is game specific:
`info_state`, and where needed `labels`, `world_from_info` and
`visible_action`, plus the class attributes `name` and `max_len`.

  class MyGameEnv(JaxGameEnv):
    name = "my_game"
    max_len = 20

    def __init__(self):
      super().__init__(MyGame())

    def info_state(self, s, seat):
      return MyInfo(seat=jnp.asarray(seat, jnp.int32), board=s.board)

The player to move is read from the Game's `current_player(state)` method
when it has one, else from the state's `current_player` field. actor() is -1
at terminal states and at chance nodes.
"""

from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.scale.core import chance


class JaxGameEnv:
  """Env over one JAX Game (module doc).

  Subclasses set name, max_len, zero_sum.

  Attributes:
    name: The env's registry name (set by the subclass).
    max_len: The longest game in plies (set by the subclass).
    zero_sum: Whether the returns sum to zero (set by the subclass).
    game: The wrapped JAX Game.
    num_players: The game's number of seats.
    num_actions: The game's number of actions.
    max_chance_outcomes: The game's largest number of chance outcomes.
  """

  name: str = ""
  max_len: int = 0
  zero_sum: bool = True

  def __init__(self, game: game_lib.Game):
    if not isinstance(game, game_lib.Game):
      raise TypeError(
          f"a JaxGameEnv wraps a JAX Game, got {type(game).__name__}"
      )
    if not self.name:
      raise TypeError(f"{type(self).__name__}: subclass must set name")
    if self.max_len <= 0:
      raise TypeError(f"{type(self).__name__}: subclass must set max_len")
    self.game = game
    self.num_players = int(game.num_players)
    self.num_actions = int(game.num_actions)
    self.max_chance_outcomes = int(game.max_chance_outcomes)
    method = getattr(game, "current_player", None)
    if callable(method):
      self._current_player = method
    else:
      s0 = game.new_initial_state()
      assert hasattr(s0, "current_player"), (
          f"{type(game).__name__}: neither a current_player() method nor a "
          "state.current_player field"
      )
      self._current_player = lambda s: s.current_player

  # -- raw layer
  # ----------------------------------------------------------------
  def root(self) -> Any:
    """The game's initial state, chance unresolved.

    Returns:
      The root state (it may be a chance node).
    """
    return self.game.new_initial_state()

  def apply(self, s, a) -> Any:
    """Applies player action `a`; chance stays unresolved.

    Args:
      s: The state.
      a: The player action id.

    Returns:
      The successor state.
    """
    return self.game.apply_action(s, jnp.asarray(a, jnp.int32))

  def is_chance(self, s) -> jax.Array:
    """Whether `s` is a chance node.

    Args:
      s: The state.

    Returns:
      bool: True at a chance node (False at terminal).
    """
    cp = jnp.asarray(self._current_player(s), jnp.int32)
    return (cp == game_lib.CHANCE_PLAYER_ID) & ~self.terminal(s)

  def apply_chance(self, s, outcome) -> Any:
    """Applies a known chance outcome at a chance node.

    Args:
      s: The state.
      outcome: The chance outcome id.

    Returns:
      The successor state.
    """
    return self.game.apply_action(s, jnp.asarray(outcome, jnp.int32))

  # -- resolved layer
  # -----------------------------------------------------------
  def init(self, deal_key) -> Any:
    """The root with every leading chance node resolved.

    Args:
      deal_key: PRNG key that resolves the leading chance nodes.

    Returns:
      The first decision state (or terminal).
    """
    return chance.resolve(self, self.root(), deal_key)[0]

  def step(self, s, a, chance_key) -> Any:
    """Applies `a`, then resolves every chance node that follows.

    Args:
      s: The state.
      a: The player action id.
      chance_key: PRNG key that resolves the chance nodes after the action.

    Returns:
      The successor state.
    """
    return chance.resolve(self, self.apply(s, a), chance_key)[0]

  # -- queries
  # ------------------------------------------------------------------
  def chance_probs(self, s) -> jax.Array:
    """The outcome probabilities at a chance node.

    Args:
      s: The state.

    Returns:
      f32[max_chance_outcomes] outcome probabilities.
    """
    return jnp.asarray(self.game.chance_outcomes(s), jnp.float32)

  def terminal(self, s) -> jax.Array:
    """Whether the game is over at `s`.

    Args:
      s: The state.

    Returns:
      bool: True at a terminal state.
    """
    return jnp.asarray(s.is_terminal, jnp.bool_)

  def actor(self, s) -> jax.Array:
    """The seat to move.

    Args:
      s: The state.

    Returns:
      i32 seat in [0, P); -1 at a terminal state.
    """
    cp = jnp.asarray(self._current_player(s), jnp.int32)
    return jnp.where(self.terminal(s), jnp.int32(-1), cp)

  def legal_mask(self, s) -> jax.Array:
    """The actor's legal actions.

    Args:
      s: The state.

    Returns:
      bool[A] mask; all False at terminal.
    """
    return jnp.asarray(self.game.legal_actions_mask(s), jnp.bool_)

  def returns(self, s) -> jax.Array:
    """Per-seat returns.

    Args:
      s: The state.

    Returns:
      f32[P] returns; meaningful at terminal.
    """
    return jnp.asarray(self.game.returns(s), jnp.float32)

  def info_state(self, s, seat) -> Any:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).
    """
    raise NotImplementedError(f"{type(self).__name__} must define info_state")

  def labels(self, s, seat) -> Mapping[str, jax.Array]:
    """Privileged targets for learners (e.g. the opponent's hand).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      A mapping of label name to array.
    """
    del s, seat
    return {}

  def visible_action(self, s, a, seat) -> jax.Array:
    """Default: every player action is public.

    Args:
      s: The state.
      a: The player action id.
      seat: The seat.

    Returns:
      i32 action.
    """
    del s, seat
    return jnp.asarray(a, jnp.int32)

  def world_from_info(self, info) -> Optional[Any]:
    """The one state consistent with `info`, when it determines it.

    Args:
      info: One seat's InfoState.
    """
    del info
    return None
