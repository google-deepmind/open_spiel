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

"""Tests for LP solvers."""

import numpy as np
import pyspiel
from absl.testing import absltest, parameterized

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability, sequence_form_lp
from open_spiel.python.utils import file_utils


class SFLPTest(parameterized.TestCase):
  def test_rock_paper_scissors(self):
    game = pyspiel.load_game_as_turn_based("matrix_rps")
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    self.assertAlmostEqual(val1, 0)
    self.assertAlmostEqual(val2, 0)

  def test_kuhn_poker(self):
    game = pyspiel.load_game("kuhn_poker")
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    # value from Kuhn 1950 or https://en.wikipedia.org/wiki/Kuhn_poker
    self.assertAlmostEqual(val1, -1 / 18)
    self.assertAlmostEqual(val2, +1 / 18)

  def test_kuhn_poker_efg(self):
    game = pyspiel.load_efg_game(pyspiel.get_kuhn_poker_efg_data())
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    # value from Kuhn 1950 or https://en.wikipedia.org/wiki/Kuhn_poker
    self.assertAlmostEqual(val1, -1 / 18)
    self.assertAlmostEqual(val2, +1 / 18)

  def test_leduc_poker(self):
    game = pyspiel.load_game("leduc_poker")
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    # values obtained from Appendix E.2 of Lanctot et al. 2017, A Unified
    # Game-Theoretic Approach to Multiagent Reinforcement Learning.
    # https://arxiv.org/abs/1711.00832
    self.assertAlmostEqual(val1, -0.085606424078, places=6)
    self.assertAlmostEqual(val2, 0.085606424078, places=6)

  def test_iigoofspiel4(self):
    game = pyspiel.load_game_as_turn_based(
      "goofspiel",
      {
        "imp_info": True,
        "num_cards": 4,
        "points_order": "descending",
      },
    )
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    # symmetric game, should be 0
    self.assertAlmostEqual(val1, 0)
    self.assertAlmostEqual(val2, 0)

  def test_exploitablity(self):
    # exploitability test for a player's / joint policies
    # loading the game from Kuhn 1950 or
    # https://en.wikipedia.org/wiki/Kuhn_poker
    game = pyspiel.load_game("kuhn_poker")
    # solving the game as the linear programme
    (_, _, pi1, pi2) = sequence_form_lp.solve_zero_sum_game(game)

    # the way to do it is to merge the policies to get the joint policy
    # of the game
    merged_policy = policy.merge_tabular_policies([pi1, pi2], game)
    expl_pi = exploitability.exploitability(game, merged_policy)
    self.assertAlmostEqual(0.0, expl_pi)

  @parameterized.parameters(
    "guess_the_ace", "kuhn_poker_with_raise", "kuhn_poker"
  )
  def test_sequential_equlirium_runs(self, name):
    # for the subgame perfect equilibrium test
    filename = file_utils.find_file(
      f"open_spiel/games/efg_game/games/{name}.efg", 2
    )
    game = pyspiel.load_game("efg_game(filename=" + filename + ")")
    val_classic, _, pi1, pi2 = sequence_form_lp.solve_zero_sum_game(game)

    merged_policy = policy.merge_tabular_policies([pi1, pi2], game)
    expl_classic = exploitability.exploitability(game, merged_policy)

    val_noisy, _, pi1sp, pi2sp = sequence_form_lp.solve_perturbed_zero_sum_game(
      game, eps=1e-6 if name == "guess_the_ace" else 1e-4
    )
    merged_policy_sp = policy.merge_tabular_policies([pi1sp, pi2sp], game)
    expl_pi_noisy = exploitability.exploitability(game, merged_policy_sp)

    self.assertAlmostEqual(0.0, expl_classic)
    self.assertAlmostEqual(expl_classic, expl_pi_noisy, 3)
    self.assertAlmostEqual(val_classic, val_noisy, 3)

  def test_sequential_equlirium_guess_the_ace(self):
    # "Guess the ace" is an exemplar game from
    # Peter Bro Miltersen and Troels
    # Bjerre Sørensen, "Computing sequential equilibria for two-player games",
    # https://www.itu.dk/~trbj/papers/seqeqsoda.pdf.
    # The game tree could be written as follows:
    #
    #     Nature (dealer shuffles deck)
    # ├── AS on top          (prob 1/52)
    # │   ├── P1: Stop ──────► Payoff (0, 0)
    # │   └── P1: Play ──────► P2 infoset (cannot see card)
    # │                       ├── Guess AS ────► (-1000, +1000) <- st. eq.
    # │                       └── Guess Other ──► (0, 0)
    # └── Other on top       (prob 51/52)
    #     ├── P1: Stop ──────► Payoff (0, 0)
    #     └── P1: Play ──────► (same P2 infoset)
    #                         ├── Guess AS ────► (0, 0)
    #                         └── Guess Other ──► (-1000, +1000) <- seq. eq.

    filename = file_utils.find_file(
      "open_spiel/games/efg_game/games/guess_the_ace.efg", 2
    )
    game = pyspiel.load_game("efg_game(filename=" + filename + ")")

    _, _, pi1_std, pi2_std = sequence_form_lp.solve_zero_sum_game(game)

    _, _, pi1_pert, pi2_pert = sequence_form_lp.solve_perturbed_zero_sum_game(
      game, eps=1e-6
    )
    p2_state_other = "1-1-1-"
    std_guess_other = pi2_std.policy_for_key(p2_state_other)

    # Baseline solution outputs non-sequential equilibrium [1, 0]
    self.assertTrue(np.allclose(std_guess_other, np.array([1, 0]), atol=1e-1))
    # Perturbed solution outputs sequential equilibrium [0, 1] for the 2nd pl.
    player1_as = pi1_std.policy_for_key("0-0-2-")
    player1_other = pi1_std.policy_for_key("0-0-1-")
    self.assertTrue(np.allclose(player1_as, np.array([1, 0]), atol=1e-2))
    self.assertTrue(np.allclose(player1_other, np.array([1, 0]), atol=1e-2))

    perturb_guess_other = pi2_pert.policy_for_key(p2_state_other)
    self.assertTrue(
        np.allclose(perturb_guess_other, np.array([0, 1]), atol=1e-2)
    )
    # Perturbed solution outputs sequential equilibrium [1, 0] for the 1st pl.
    player1_as = pi1_pert.policy_for_key("0-0-2-")
    player1_other = pi1_pert.policy_for_key("0-0-1-")
    self.assertTrue(np.allclose(player1_as, np.array([1, 0]), atol=1e-2))
    self.assertTrue(np.allclose(player1_other, np.array([1, 0]), atol=1e-2))

  def test_sequential_equlirium_kuhn_poker(self):
    # "Poker with raise" is an exemplar game from
    # Peter Bro Miltersen and Troels
    # Bjerre Sørensen, "Computing sequential equilibria for two-player games",
    # https://www.itu.dk/~trbj/papers/seqeqsoda.pdf.
    # The game tree could be written as Kuhn poker with additional "raise"
    # action.
    #
    # In this variant, players ante $1. Nature deals one card to each player
    # from a 3-card deck (Ace, King, Queen). The betting proceeds like
    # standard Kuhn Poker, but with the added option
    # to Raise exactly once per hand.
    filename = file_utils.find_file(
      "open_spiel/games/efg_game/games/kuhn_poker_with_raise.efg", 2
    )
    game = pyspiel.load_game("efg_game(filename=" + filename + ")")

    _, _, pi1_std, _ = sequence_form_lp.solve_zero_sum_game(game)

    _, _, pi1_pert, _ = sequence_form_lp.solve_perturbed_zero_sum_game(
        game, eps=1e-2
    )

    # Player 1 holds the 2 (King), (B)ets, and suddenly faces a (R)aise from Player 2.
    # A rational Player 2 will only raise if they hold the King. 
    # Because Player 1 is holding the King, it is impossible for Player 2 to have it. 
    # Therefore, a rational Player 2 will never, ever raise.
    # The probability of reaching this node is strictly 0.0.
    not_on_the_pass = "0-0-9-2br"

    has_to_bet_std = pi1_std.policy_for_key(not_on_the_pass)
    has_to_bet_pert = pi1_pert.policy_for_key(not_on_the_pass)
    self.assertTrue(has_to_bet_std[0] > 0.1 and has_to_bet_std[1] < 0.9)
    # Called -> outplayed
    self.assertFalse(has_to_bet_pert[0] > 0.1 and has_to_bet_pert[1] < 0.9)
    

  @absltest.skip("Takes too long. Might not pass.")
  def test_tictactoe(self):
    game = pyspiel.load_game("tic_tac_toe")
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    self.assertAlmostEqual(val1, 0)
    self.assertAlmostEqual(val2, 0)

  # This test takes too long for non-glpk solvers, and glpk solver is not
  # supported within google's internal cvxopt import. When solving via glpk,
  # (locally, outside of google's testing framework), the test takes >300
  # seconds, so it is disabled by default, but still left here for reference.
  # Note, value is taken from an independent implementation but also found in
  # Neller & Lanctot 2013, An Introduction to Counterfactual Regret Minimization
  # http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
  #
  @absltest.skip("Might've taken too long.")
  def test_liars_dice(self):
    game = pyspiel.load_game("liars_dice")
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game, solver="ecos")
    self.assertAlmostEqual(val1, -0.027131782945736)
    self.assertAlmostEqual(val2, 0.027131782945736)


if __name__ == "__main__":
  absltest.main()
