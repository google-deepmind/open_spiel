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

"""Tests that the Legal list names each move as OpenSpiel does.

A legal action's label, and a chance outcome's, is the game's own
`action_to_string(current_player, action)` at the position, nothing added:
the client's Legal list shows the id, that label and, for a chance outcome,
its probability, in id order. These pin the labels at every position of
games played through a session, against a state rebuilt here from the
history.
"""

import random

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import session as session_lib

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)

# Plies played per game: through the deal, the bidding and well into play.
_PLIES = 120


def _expected(state: pyspiel.State) -> list[tuple[int, str]]:
  """The game's own label of every legal action or chance outcome."""
  player = state.current_player()
  if state.is_chance_node():
    actions = [a for a, p in state.chance_outcomes() if p > 0]
  else:
    actions = state.legal_actions()
  return [
      (int(action), str(state.action_to_string(player, action)))
      for action in sorted(actions)
  ]


class LegalLabelsTest(absltest.TestCase):

  def test_every_label_is_the_games_own(self) -> None:
    for game in ("chess", "hearts", "euchre", "gin_rummy"):
      with self.subTest(game):
        self.check_labels(game)

  def test_every_tic_tac_toe_label_is_the_games_own(self) -> None:
    # A game ends in at most 9 plies, and in at least 5.
    for seed in range(3):
      with self.subTest(seed=seed):
        self.check_labels("tic_tac_toe", minimum=5, seed=seed)

  def check_labels(self, game: str, minimum: int = 21, seed: int = 0) -> None:
    """Plays `game` at random, checking the labels at every position."""
    num = pyspiel.load_game(game).num_players()
    session = session_lib.Session.create(
        game,
        {},
        seed=2,
        chance_policy=contract.ChancePolicy.EXPOSE,
        controllers=[controllers.HUMAN_SPEC] * num,
    )
    rng = random.Random(f"{game}/{seed}" if seed else game)
    checked = 0
    for _ in range(_PLIES):
      frame = session.frame(_ORACLE)
      if frame.status == contract.SessionStatus.TERMINAL:
        break
      state = pyspiel.load_game(game).new_initial_state()
      for action in session.history:
        state.apply_action(action)
      chance = frame.chance_outcomes or ()
      shown = [(c.id, c.label) for c in chance] or [
          (a.id, a.label) for a in frame.legal_actions or ()
      ]
      self.assertEqual(sorted(shown), _expected(state), f"ply {frame.ply}")
      checked += 1
      choice = rng.choice(shown)[0]
      if chance:
        session.apply_chance(_ORACLE, choice)
      else:
        session.apply_action(_ORACLE, choice)
    self.assertGreaterEqual(checked, minimum)

  def test_a_chance_outcome_carries_its_probability(self) -> None:
    session = session_lib.Session.create(
        "hearts",
        {},
        seed=2,
        chance_policy=contract.ChancePolicy.EXPOSE,
        controllers=[controllers.HUMAN_SPEC] * 4,
    )
    frame = session.frame(_ORACLE)
    state = pyspiel.load_game("hearts").new_initial_state()
    self.assertEqual(
        [(c.id, c.label, c.probability) for c in frame.chance_outcomes or ()],
        [
            (int(a), state.action_to_string(state.current_player(), a), p)
            for a, p in state.chance_outcomes()
            if p > 0
        ],
    )


if __name__ == "__main__":
  absltest.main()
