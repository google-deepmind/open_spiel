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

"""Go, OpenSpiel's `go`, with Tromp-Taylor scoring.

State: a `GameState` whose board holds a chain id per cell (0 empty, > 0 a
black chain, < 0 a white chain), with a cached legal-action mask and the
ko point. Black (player 0) moves first. Actions: the point
row * board_size + col, and `board_size**2` to pass, as in OpenSpiel.
Observation: f32[board_size, board_size, 3], planes (the observer's stones,
the opponent's stones, a constant plane equal to the observer's id).
Returns: after two consecutive passes, or `max_game_length` moves (default
`2 * board_size**2`), the Tromp-Taylor area score with `komi` decides: +1 /
-1, or 0 for a tie.

Ko and superko, as in OpenSpiel: simple ko makes the immediate recapture
illegal; a move (not a pass) that recreates an earlier board position is
legal but ends the game at once as a draw (positional superko, OpenSpiel's
`superko_`). Positions are compared by 64-bit Zobrist hashes of the board, as
OpenSpiel compares them; the state keeps one hash per move played. Suicide
is illegal. Handicap stones are not supported (`handicap=0`).
"""

import typing
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

if typing.TYPE_CHECKING:
  import pyspiel


class GameState(NamedTuple):
  """State of a Go game.

  Attributes:
    board: i32[board_size**2]: 0 empty, > 0 a black chain id, < 0 a white
      chain id.
    current_player: 0 (black) or 1 (white).
    is_terminal: Whether the game has ended.
    winner: -1 (no winner/draw), 0 (black wins), 1 (white wins).
    legal_actions_mask: bool[num_actions].
    move_count: Number of moves played so far.
    last_action: Last action played (-1 if none).
    ko_point: Point where recapture is forbidden (-1 if no ko).
    consecutive_passes: Number of consecutive passes.
    position_hashes: u32[max_game_length + 1, 2]: the 64-bit Zobrist hash of
      the board after each move (row 0: the empty board); rows after
      `move_count` are unused.
    superko: Whether the last move recreated an earlier position (the game
      is then over, a draw).
  """

  board: jax.Array
  current_player: jax.Array
  is_terminal: jax.Array
  winner: jax.Array
  legal_actions_mask: jax.Array
  move_count: jax.Array
  last_action: jax.Array
  ko_point: jax.Array
  consecutive_passes: jax.Array
  position_hashes: jax.Array
  superko: jax.Array


# Seed of the Zobrist keys: fixed, so a hash names a position across runs.
_ZOBRIST_SEED = 2765481


class Go(game_lib.Game, interop.PyspielInterop):
  """Go on a `board_size` square board with `komi` points to white.

  Attributes:
    board_size: The board is board_size x board_size.
    komi: Points added to white's Tromp-Taylor score.
    max_game_length: Moves after which the game ends and is scored.
    neighbours: i32[num_cells, 4] the neighbours of each cell, -1 off the board.
    num_cells: The number of cells, board_size * board_size.
    num_neighbours: i32[num_cells] the number of neighbours of each cell.
    pass_action: The action id of a pass.
  """

  def __init__(
      self,
      board_size: int = 19,
      komi: float = 7.5,
      handicap: int = 0,
      max_game_length: int | None = None,
  ):
    """OpenSpiel's `go` parameters and defaults.

    Args:
      board_size: The board is board_size x board_size.
      komi: Points added to white's Tromp-Taylor score.
      handicap: Must be 0: handicap stones are not supported.
      max_game_length: Moves after which the game ends and is scored;
        default (OpenSpiel's) 2 * board_size**2.
    """
    if handicap != 0:
      raise ValueError(f"go supports only handicap=0, got {handicap!r}")
    if max_game_length is None:
      max_game_length = 2 * board_size * board_size
    if (
        isinstance(max_game_length, bool)
        or not isinstance(max_game_length, int)
        or max_game_length < 1
    ):
      raise ValueError(
          f"go max_game_length must be a positive int, got {max_game_length!r}"
      )
    self.board_size = board_size
    self.komi = komi
    self.num_cells = board_size * board_size
    self.pass_action = self.num_cells
    self.max_game_length = max_game_length
    # Zobrist keys: u32[num_cells, 2 (black, white), 2 (64-bit hash words)].
    self._zobrist = np.random.default_rng(_ZOBRIST_SEED).integers(
        0, 2**32, size=(self.num_cells, 2, 2), dtype=np.uint32
    )

    # Precompute neighbour indices for each cell.
    self.neighbours = jnp.array(self._precompute_neighbours())
    self.num_neighbours = np.array(
        [np.sum(self.neighbours[i] >= 0) for i in range(self.num_cells)],
        dtype=np.int32,
    )

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return self.num_cells + 1

  @property
  def num_players(self) -> int:
    """The number of players."""
    return 2

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (self.board_size, self.board_size, 3)

  def _precompute_neighbours(self):
    """Precomputes neighbour indices for each cell.

    Returns:
      int32[num_cells, 4], neighbour indices, -1 past an edge.
    """
    neighbours = np.full((self.num_cells, 4), -1, dtype=np.int32)
    for i in range(self.num_cells):
      r, c = divmod(i, self.board_size)
      n = 0
      for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
          neighbours[i, n] = nr * self.board_size + nc
          n += 1
    return neighbours

  def new_initial_state(self) -> GameState:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    legal = jnp.ones((self.num_actions,), dtype=jnp.bool_)
    return GameState(
        board=jnp.zeros((self.num_cells,), dtype=jnp.int32),
        current_player=jnp.array(0, dtype=jnp.int32),
        is_terminal=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(-1, dtype=jnp.int32),
        legal_actions_mask=legal,
        move_count=jnp.array(0, dtype=jnp.int32),
        last_action=jnp.array(-1, dtype=jnp.int32),
        ko_point=jnp.array(-1, dtype=jnp.int32),
        consecutive_passes=jnp.array(0, dtype=jnp.int32),
        position_hashes=self.position_history(
            jnp.zeros((self.num_cells,), dtype=jnp.int32), jnp.int32(0)
        ),
        superko=jnp.array(False, dtype=jnp.bool_),
    )

  def position_hash(self, board: jax.Array) -> jax.Array:
    """Returns the 64-bit Zobrist hash of `board` (stones only).

    Args:
      board: The board.

    Returns:
      u32[2].
    """
    keys = jnp.asarray(self._zobrist)
    stones = jnp.stack([board > 0, board < 0], axis=-1)  # [cells, color]
    # Sums wrap modulo 2**32: a hash of the set of (cell, color) stones.
    return jnp.sum(
        jnp.where(stones[..., None], keys, jnp.uint32(0)),
        axis=(0, 1),
        dtype=jnp.uint32,
    )

  def position_history(
      self, board: jax.Array, move_count: jax.Array
  ) -> jax.Array:
    """Returns a position history holding only `board`, up to `move_count`.

    For rebuilding a state from its position alone: the earlier positions
    are not known, so a later repetition of one of them is not detected; a
    repetition of `board` is.

    Args:
      board: The board.
      move_count: The number of moves played.

    Returns:
      u32[max_game_length + 1, 2]: `board`'s hash at every row up to
      `move_count`.
    """
    rows = jnp.arange(self.max_game_length + 1) <= move_count
    return jnp.where(
        rows[:, None], self.position_hash(board)[None, :], jnp.uint32(0)
    )

  def _expand_mask(self, mask: jax.Array) -> jax.Array:
    """Expands a boolean mask to include neighbours of set cells.

    Args:
      mask: bool[num_cells], the mask.

    Returns:
      bool[num_cells], the mask and its neighbours.
    """
    expanded = jnp.zeros(self.num_cells, dtype=jnp.int32)
    for n_idx in range(4):
      n = self.neighbours[:, n_idx]
      valid = n >= 0
      contrib = (mask & valid).astype(jnp.int32)
      expanded = expanded.at[jnp.clip(n, 0, self.num_cells - 1)].add(contrib)
    return expanded > 0

  def legal_actions_for(
      self,
      board: jax.Array,
      my_sign: jax.Array,
      ko_point: jax.Array,
  ) -> jax.Array:
    """Returns the legal moves on `board` of the side with stone sign `my_sign`.

    Args:
      board: The board.
      my_sign: +1 black, -1 white.
      ko_point: The cell ko forbids, or -1.

    Returns:
      bool[num_actions], pass included.
    """
    opp_sign = -my_sign

    is_empty = board == 0

    num_pseudo, idx_sum, idx_squared_sum = self._count_liberties(board)

    chain_ix = jnp.abs(board) - 1
    safe_chain_ix = jnp.where(is_empty, 0, chain_ix)

    in_atari = (idx_sum[safe_chain_ix] ** 2) == idx_squared_sum[
        safe_chain_ix
    ] * num_pseudo[safe_chain_ix]

    has_liberty = (board * my_sign > 0) & (~in_atari)
    can_kill = (board * opp_sign > 0) & in_atari

    def is_adj_ok(xy):
      """Returns whether a cell has an empty, capturable or living neighbour.

      Args:
        xy: The cell.

      Returns:
        bool[].
      """
      n = self.neighbours[xy]
      valid = n >= 0
      n_safe = jnp.clip(n, 0, self.num_cells - 1)

      n_is_empty = is_empty[n_safe] & valid
      n_can_kill = can_kill[n_safe] & valid
      n_has_liberty = has_liberty[n_safe] & valid

      return (n_is_empty | n_can_kill | n_has_liberty).any()

    adj_ok = jax.vmap(is_adj_ok)(jnp.arange(self.num_cells))

    not_ko = jnp.arange(self.num_cells) != ko_point

    board_legal = is_empty & not_ko & adj_ok
    return jnp.append(board_legal, jnp.array(True))

  def tromp_taylor_score(self, board: jax.Array) -> jax.Array:
    """Computes the Tromp-Taylor score.

    Args:
      board: The board.

    Returns:
      The score; positive when black wins.
    """
    black_stones = jnp.sum(board > 0)
    white_stones = jnp.sum(board < 0)

    is_empty = board == 0
    indices = jnp.arange(self.num_cells, dtype=jnp.int32)
    labels = jnp.where(is_empty, indices, self.num_cells)

    def propagate_cond(val):
      """Returns whether the region labels are still changing.

      Args:
        val: (labels, old labels, count).

      Returns:
        bool[].
      """
      labels, old_labels, count = val
      return jnp.any(labels != old_labels) & (count < self.num_cells)

    def propagate_body(val):
      """Propagates the minimum label one step through empty cells.

      Args:
        val: (labels, old labels, count).

      Returns:
        (new labels, labels, count + 1).
      """
      labels, _, count = val
      new_labels = labels.copy()
      for n_idx in range(4):
        n = self.neighbours[:, n_idx]
        valid = (n >= 0) & is_empty
        n_safe = jnp.clip(n, 0, self.num_cells - 1)
        n_labels = labels[n_safe]
        candidate = jnp.where(valid & is_empty, n_labels, self.num_cells)
        new_labels = jnp.minimum(new_labels, candidate)
      return new_labels, labels, count + 1

    labels, _, _ = jax.lax.while_loop(
        propagate_cond,
        propagate_body,
        (
            labels,
            jnp.full(self.num_cells, self.num_cells + 1, dtype=jnp.int32),
            jnp.array(0),
        ),
    )

    all_black_touch = []
    all_white_touch = []
    for n_idx in range(4):
      n = self.neighbours[:, n_idx]
      valid = (n >= 0) & is_empty
      n_safe = jnp.clip(n, 0, self.num_cells - 1)
      n_color = board[n_safe]
      all_black_touch.append((valid & (n_color > 0)).astype(jnp.int32))
      all_white_touch.append((valid & (n_color < 0)).astype(jnp.int32))

    any_black_touch = jnp.any(jnp.stack(all_black_touch), axis=0).astype(
        jnp.int32
    )
    any_white_touch = jnp.any(jnp.stack(all_white_touch), axis=0).astype(
        jnp.int32
    )

    borders_black = (
        jax.ops.segment_max(
            any_black_touch, labels, num_segments=self.num_cells + 1
        )
        > 0
    )
    borders_white = (
        jax.ops.segment_max(
            any_white_touch, labels, num_segments=self.num_cells + 1
        )
        > 0
    )

    cell_borders_black = borders_black[labels] & is_empty
    cell_borders_white = borders_white[labels] & is_empty

    black_territory = jnp.sum(
        (cell_borders_black & (~cell_borders_white)).astype(jnp.int32)
    )
    white_territory = jnp.sum(
        (cell_borders_white & (~cell_borders_black)).astype(jnp.int32)
    )

    return (
        black_stones.astype(jnp.float32)
        + black_territory.astype(jnp.float32)
        - white_stones.astype(jnp.float32)
        - white_territory.astype(jnp.float32)
        - self.komi
    )

  def _count_liberties(self, board: jax.Array):
    """The liberties of every cell's group, deduplicated by sums.

    Args:
      board: int[num_cells], the board.

    Returns:
      int[num_cells], the liberty count of each cell's group.
    """
    is_empty = board == 0

    def count_neighbor(xy):
      """Counts a cell's empty neighbours, with sums for deduplication.

      Args:
        xy: The cell.

      Returns:
        (count, sum of index + 1, sum of (index + 1)**2).
      """
      n = self.neighbours[xy]
      valid = n >= 0
      n_safe = jnp.clip(n, 0, self.num_cells - 1)
      n_is_empty = is_empty[n_safe] & valid
      return (
          n_is_empty.sum(),
          jnp.where(n_is_empty, n_safe + 1, 0).sum(),
          jnp.where(n_is_empty, (n_safe + 1) ** 2, 0).sum(),
      )

    num_pseudo, idx_sum, idx_squared_sum = jax.vmap(count_neighbor)(
        jnp.arange(self.num_cells)
    )

    segment_ids = jnp.abs(board)
    num_segments = self.num_cells + 1

    sum_num_pseudo = jax.ops.segment_sum(
        num_pseudo, segment_ids, num_segments=num_segments
    )
    sum_idx_sum = jax.ops.segment_sum(
        idx_sum, segment_ids, num_segments=num_segments
    )
    sum_idx_squared_sum = jax.ops.segment_sum(
        idx_squared_sum, segment_ids, num_segments=num_segments
    )

    return sum_num_pseudo[1:], sum_idx_sum[1:], sum_idx_squared_sum[1:]

  def apply_action(self, state: GameState, action: jax.Array) -> GameState:
    """Returns the state after `action`.

    Args:
      state: The current state.
      action: i32[], a legal player action, or a chance action id with
        nonzero probability at a chance node.

    Returns:
      The next state. Applying an illegal action gives an unspecified state;
      it does not raise, because the method is traced.
    """
    # Cell ids fit int32; an int64 action (x64) would leak into the board.
    action = jnp.asarray(action, jnp.int32)
    is_pass = action == self.pass_action
    my_sign = 1 - 2 * state.current_player
    opp_sign = -my_sign

    num_pseudo, idx_sum, idx_squared_sum = self._count_liberties(state.board)

    action_safe = jnp.clip(action, 0, self.num_cells - 1)
    adj_ixs = self.neighbours[action_safe]
    valid_adj = adj_ixs >= 0
    adj_ixs_safe = jnp.clip(adj_ixs, 0, self.num_cells - 1)
    adj_ids = state.board[adj_ixs_safe]

    chain_ix = jnp.abs(adj_ids) - 1
    in_atari = (idx_sum[chain_ix] ** 2) == idx_squared_sum[
        chain_ix
    ] * num_pseudo[chain_ix]

    safe_idx_sum = jnp.where(idx_sum[chain_ix] > 0, idx_sum[chain_ix], 1)
    single_liberty = (idx_squared_sum[chain_ix] // safe_idx_sum) - 1

    is_killed = (
        valid_adj
        & (adj_ids * opp_sign > 0)
        & in_atari
        & (single_liberty == action)
    )

    surrounded_stones = (state.board[:, None] == adj_ids) & (is_killed[None, :])
    # dtype pin: jnp.sum promotes its integer accumulator to int64
    # under x64 (numpy sum semantics), so pin dtype rather than only the
    # summand — otherwise num_captured leaks int64 into the lax.select below.
    num_captured = jnp.sum(surrounded_stones.any(axis=-1), dtype=jnp.int32)
    board_after_capture = jnp.where(
        surrounded_stones.any(axis=-1), 0, state.board
    )

    new_id = (action + 1) * my_sign
    board_after_place = board_after_capture.at[action].set(new_id)

    adj_ids_after_place = board_after_place[adj_ixs_safe]
    is_my_chain = adj_ids_after_place * my_sign > 0
    should_merge = valid_adj & is_my_chain

    tgt_ids = adj_ids_after_place
    smallest_id = jnp.min(jnp.where(should_merge, jnp.abs(tgt_ids), 9999))
    smallest_id = jnp.minimum(jnp.abs(new_id), smallest_id) * my_sign

    merge_mask = (board_after_place == new_id) | (
        should_merge[None, :] & (board_after_place[:, None] == tgt_ids[None, :])
    ).any(axis=-1)
    board_after_merge = jnp.where(merge_mask, smallest_id, board_after_place)

    new_board = jax.lax.select(is_pass, state.board, board_after_merge)
    # dtype pin: a bare `0` literal is int64 under x64 while
    # num_captured is int32 — lax.select requires exact dtype match.
    num_captured = jax.lax.select(is_pass, jnp.int32(0), num_captured)

    captured_mask = (
        (board_after_capture == 0) & (state.board * opp_sign > 0) & (~is_pass)
    )
    # dtype pin: jnp.arange is strongly-typed int64 under x64, so
    # single_captured_pos would collide with the int32 branch of the new_ko
    # lax.select below. Cell indices fit int32; pinning keeps new_ko int32,
    # matching the initial state's ko_point.
    captured_positions = jnp.where(
        captured_mask, jnp.arange(self.num_cells, dtype=jnp.int32), -1
    )
    single_captured_pos = jnp.max(captured_positions)

    played_in_enemy_eye = jnp.array(True)
    for n_idx in range(4):
      n = adj_ixs[n_idx]
      n_valid = n >= 0
      n_s = jnp.clip(n, 0, self.num_cells - 1)
      n_id = state.board[n_s]
      is_ok = (~n_valid) | (n_id * opp_sign > 0)
      played_in_enemy_eye = played_in_enemy_eye & is_ok

    new_ko = jax.lax.select(
        (~is_pass) & (num_captured == 1) & played_in_enemy_eye,
        single_captured_pos,
        jnp.array(-1, dtype=jnp.int32),
    )

    new_consecutive = jax.lax.select(
        is_pass, state.consecutive_passes + 1, jnp.array(0, dtype=jnp.int32)
    )
    new_move_count = state.move_count + 1
    # Positional superko (OpenSpiel's): a move, not a pass, that recreates a
    # position seen before ends the game as a draw.
    new_hash = self.position_hash(new_board)
    rows = jnp.arange(self.max_game_length + 1) <= state.move_count
    seen = jnp.any(
        jnp.all(state.position_hashes == new_hash[None, :], axis=-1) & rows
    )
    superko = seen & ~is_pass
    new_position_hashes = state.position_hashes.at[new_move_count].set(
        new_hash, mode="drop"
    )
    is_terminal = (
        (new_consecutive >= 2)
        | (new_move_count >= self.max_game_length)
        | superko
    )

    def compute_winner(board):
      """Returns the winner by Tromp-Taylor score.

      Args:
        board: The final board.

      Returns:
        i32[]: 0 black, 1 white, -1 a draw.
      """
      score = self.tromp_taylor_score(board)
      return jax.lax.select(
          score > 0,
          jnp.array(0, dtype=jnp.int32),
          jax.lax.select(
              score < 0,
              jnp.array(1, dtype=jnp.int32),
              jnp.array(-1, dtype=jnp.int32),
          ),
      )

    winner = jax.lax.cond(
        is_terminal & ~superko,
        compute_winner,
        lambda _: jnp.array(-1, dtype=jnp.int32),
        new_board,
    )

    next_player = 1 - state.current_player
    next_sign = 1 - 2 * next_player
    legal = jax.lax.select(
        is_terminal,
        jnp.zeros((self.num_actions,), dtype=jnp.bool_),
        self.legal_actions_for(new_board, next_sign, new_ko),
    )

    return GameState(
        board=new_board,
        current_player=next_player,
        is_terminal=is_terminal,
        winner=winner,
        legal_actions_mask=legal,
        move_count=new_move_count,
        last_action=action,
        ko_point=new_ko,
        consecutive_passes=new_consecutive,
        position_hashes=new_position_hashes,
        superko=superko,
    )

  def current_player(self, state: GameState) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node. The
      value at a terminal state is game-specific.
    """
    return state.current_player

  def is_terminal(self, state: GameState) -> jax.Array:
    """Returns whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the game is over.
    """
    return state.is_terminal

  def is_chance_node(self, state: GameState) -> jax.Array:
    """Returns whether the next action is a chance outcome.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the next action is a chance outcome.
    """
    del state  # Unused.
    return jnp.bool_(False)

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """Returns the legal player actions.

    Args:
      state: The current state.

    Returns:
      bool[num_actions]: the legal player actions. All False at a terminal state
      and, for most games, at a chance node.
    """
    return state.legal_actions_mask

  def observation_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns what `observing_player` observes now.

    Args:
      state: The current state.
      observing_player: i32[], the observing player.

    Returns:
      f32[*obs_shape]: the observation tensor.
    """
    board_2d = state.board.reshape(self.board_size, self.board_size)

    my_sign = jnp.where(observing_player == 0, 1, -1)
    opp_sign = -my_sign

    my_pieces = (board_2d * my_sign > 0).astype(jnp.float32)
    opp_pieces = (board_2d * opp_sign > 0).astype(jnp.float32)
    player_plane = jnp.full(
        (self.board_size, self.board_size),
        jnp.array(observing_player).astype(jnp.float32),
    )

    return jnp.stack([my_pieces, opp_pieces, player_plane], axis=-1).astype(
        jnp.float32
    )

  def information_state_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns the observation: the board determines the information state.

    Args:
      state: The state.
      observing_player: i32[], the observing player.

    Returns:
      The observation tensor.
    """
    return self.observation_tensor(state, observing_player)

  def returns(self, state: GameState) -> jax.Array:
    """Returns each player's return.

    Args:
      state: The current state.

    Returns:
      f32[num_players]: each player's return; zero before the end.
    """
    is_win0 = (state.winner == 0) & state.is_terminal
    is_win1 = (state.winner == 1) & state.is_terminal

    p0 = jnp.where(is_win0, 1.0, jnp.where(is_win1, -1.0, 0.0))
    p1 = jnp.where(is_win1, 1.0, jnp.where(is_win0, -1.0, 0.0))

    return jnp.stack([p0, p1])

  def observation_from_os(
      self, os_state: "pyspiel.State", player: int
  ) -> jax.Array:
    """Returns the JAX game's observation tensor of `player` at `os_state`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      The observation tensor the JAX game gives `player` in the same position.

    Raises:
      NotImplementedError: if the game has no pyspiel conversion.
    """
    os_obs = np.asarray(os_state.observation_tensor(player)).reshape(
        4, self.board_size, self.board_size
    )
    black_pieces = os_obs[0]
    white_pieces = os_obs[1]

    is_player_0 = player == 0

    my_pieces = jnp.where(is_player_0, black_pieces, white_pieces)
    opp_pieces = jnp.where(is_player_0, white_pieces, black_pieces)

    player_plane = jnp.full((self.board_size, self.board_size), float(player))

    return jnp.stack([my_pieces, opp_pieces, player_plane], axis=-1).astype(
        jnp.float32
    )
