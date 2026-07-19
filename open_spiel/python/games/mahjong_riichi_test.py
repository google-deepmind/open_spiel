# Copyright 2019 DeepMind Technologies Limited
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

"""Tests for mahjong_riichi.py."""

import random

from absl.testing import absltest

from open_spiel.python.games import mahjong_riichi as mr
import pyspiel


def _deal_rigged(state, seat_hands, wait_tile_owner=None):
  """Applies DEAL chance actions so each seat's initial hand matches
  `seat_hands` (a dict: seat -> list of tile types), with everyone else
  filled in from the remaining tile pool. Returns nothing; mutates `state`.
  """
  next_copy = {}
  used_ids = []

  def take(t):
    c = next_copy.get(t, 0)
    tid = t * 4 + c
    next_copy[t] = c + 1
    used_ids.append(tid)
    return tid

  ordered_ids = {}
  for seat, types in seat_hands.items():
    ordered_ids[seat] = [take(t) for t in types]

  remaining_pool = [i for i in range(mr.NUM_TILES) if i not in used_ids]
  full_order = []
  pool_idx = 0
  sizes = {0: 14, 1: 13, 2: 13, 3: 13}
  for seat in range(4):
    ids = ordered_ids.get(seat, [])
    needed = sizes[seat] - len(ids)
    ids = ids + remaining_pool[pool_idx:pool_idx + needed]
    pool_idx += needed
    full_order.extend(ids)
  full_order.extend(remaining_pool[pool_idx:])
  del wait_tile_owner
  for tid in full_order:
    state.apply_action(tid)


class MahjongRiichiTest(absltest.TestCase):

  def test_load_default(self):
    game = pyspiel.load_game("python_mahjong_riichi")
    self.assertEqual(game.num_players(), 4)

  def test_deal_gives_dealer_fourteen_tiles(self):
    game = pyspiel.load_game("python_mahjong_riichi")
    state = game.new_initial_state()
    while state.current_player() == pyspiel.PlayerId.CHANCE:
      outcomes = state.chance_outcomes()
      state.apply_action(outcomes[0][0])
    self.assertLen(state._hands[0], 14)
    for p in (1, 2, 3):
      self.assertLen(state._hands[p], 13)
    self.assertEqual(state.current_player(), 0)

  def test_action_encoding_is_disjoint(self):
    ranges = [
        range(mr._DISCARD_BASE, mr._RIICHI_DISCARD_BASE),
        range(mr._RIICHI_DISCARD_BASE, mr._CLOSED_KAN_BASE),
        range(mr._CLOSED_KAN_BASE, mr._ADDED_KAN_BASE),
        range(mr._ADDED_KAN_BASE, mr._TSUMO_ACTION),
    ]
    singles = [mr._TSUMO_ACTION, mr._KYUUSHU_ACTION, mr._PASS_ACTION,
               mr._CHI_LOW_ACTION, mr._CHI_MID_ACTION, mr._CHI_HIGH_ACTION,
               mr._PON_ACTION, mr._OPEN_KAN_ACTION, mr._RON_ACTION]
    seen = set()
    for r in ranges:
      for a in r:
        self.assertNotIn(a, seen)
        seen.add(a)
    for a in singles:
      self.assertNotIn(a, seen)
      seen.add(a)
    self.assertLen(seen, mr._NUM_DISTINCT_ACTIONS)

  def test_tenhou_tsumo(self):
    """Dealer dealt a complete winning hand: yakuman via Tenhou."""
    game = pyspiel.load_game("python_mahjong_riichi")
    state = game.new_initial_state()
    winning_hand = [1, 2, 3, 5, 6, 7, 12, 13, 14, 16, 16, 19, 20, 21]
    _deal_rigged(state, {0: winning_hand})
    self.assertIn(mr._TSUMO_ACTION, state.legal_actions())
    state.apply_action(mr._TSUMO_ACTION)
    self.assertTrue(state.is_terminal())
    returns = state.returns()
    self.assertAlmostEqual(sum(returns), 0.0)
    self.assertGreater(returns[0], 0)
    self.assertAlmostEqual(returns[0], -3 * returns[1])
    self.assertAlmostEqual(returns[1], returns[2])
    self.assertAlmostEqual(returns[2], returns[3])

  def test_ron_and_furiten(self):
    game = pyspiel.load_game("python_mahjong_riichi")
    state = game.new_initial_state()
    dealer_hand = [1, 2, 3, 5, 6, 7, 12, 13, 14, 16, 16, 19, 20, 27]
    _deal_rigged(state, {
        0: dealer_hand,
        1: [21],  # 4s: one of the dealer's two waits, once tenpai.
        2: [18],  # 1s: dealer's other wait.
    })
    self.assertEqual(state._waits(0), set())  # Not tenpai yet (14 tiles).
    state.apply_action(mr._discard_action(27))  # Dealer reaches tenpai.
    self.assertEqual(state._waits(0), {18, 21})

    state.apply_action(mr._discard_action(21))  # p1 discards a dealer wait.
    self.assertEqual(state._phase, mr.Phase.REACT_RON)
    self.assertEqual(state.current_player(), 0)
    state.apply_action(mr._PASS_ACTION)  # Dealer declines a legal ron.
    self.assertTrue(state._temporary_furiten[0])

    # Furiten blocks ron on the *other* wait too, not just the declined tile.
    state.apply_action(mr._discard_action(18))  # p2 discards the other wait.
    self.assertEqual(state._phase, mr.Phase.TURN)  # No ron offered.

  def test_ron_settlement_is_zero_sum(self):
    game = pyspiel.load_game("python_mahjong_riichi")
    state = game.new_initial_state()
    dealer_hand = [1, 2, 3, 5, 6, 7, 12, 13, 14, 16, 16, 19, 20, 27]
    _deal_rigged(state, {0: dealer_hand, 1: [21]})
    state.apply_action(mr._discard_action(27))
    state.apply_action(mr._discard_action(21))
    self.assertEqual(state.current_player(), 0)
    state.apply_action(mr._RON_ACTION)
    self.assertTrue(state.is_terminal())
    self.assertAlmostEqual(sum(state.returns()), 0.0)
    self.assertGreater(state.returns()[0], 0)
    self.assertLess(state.returns()[1], 0)
    self.assertEqual(state.returns()[2], 0)
    self.assertEqual(state.returns()[3], 0)

  def test_riichi_locks_discard_to_drawn_tile(self):
    game = pyspiel.load_game("python_mahjong_riichi")
    state = game.new_initial_state()
    # Tenpai-after-discard hand for the dealer (ryanmen on 1s/4s once 27 goes).
    dealer_hand = [1, 2, 3, 5, 6, 7, 12, 13, 14, 16, 16, 19, 20, 27]
    _deal_rigged(state, {0: dealer_hand})
    legal = state.legal_actions()
    self.assertIn(mr._riichi_discard_action(27), legal)
    state.apply_action(mr._riichi_discard_action(27))
    self.assertTrue(state._riichi[0])
    self.assertEqual(state._scores[0], 25000 - 1000)

  @staticmethod
  def _run_random_games(num_sims, seed_offset=0):
    game = pyspiel.load_game("python_mahjong_riichi")
    for trial in range(num_sims):
      rng = random.Random(trial + seed_offset)
      state = game.new_initial_state()
      steps = 0
      while not state.is_terminal() and steps < 3000:
        if state.current_player() == pyspiel.PlayerId.CHANCE:
          outcomes = state.chance_outcomes()
          actions, probs = zip(*outcomes)
          action = rng.choices(actions, weights=probs)[0]
        else:
          action = rng.choice(state.legal_actions())
        state.apply_action(action)
        steps += 1
      if not state.is_terminal():
        raise RuntimeError(f"Trial {trial} did not terminate in time")
      if abs(sum(state.returns())) > 1e-6:
        raise RuntimeError(
            f"Trial {trial} was not zero-sum: {state.returns()}")

  def test_random_games_are_zero_sum_and_terminate(self):
    self._run_random_games(30, seed_offset=9000)

  def test_random_sim(self):
    """Runs OpenSpiel's standard API-consistency checks, with serialization."""
    game = pyspiel.load_game("python_mahjong_riichi")
    pyspiel.random_sim_test(game, num_sims=5, serialize=True, verbose=False)


if __name__ == "__main__":
  absltest.main()
