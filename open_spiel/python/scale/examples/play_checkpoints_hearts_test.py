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

"""play_checkpoints on hearts: one artifact against three seats, end to end.

A freshly initialised hearts network on disk (the public loader reads it as
it reads a trained artifact) plays one seat against uniform random play in
the other three, every deal four times with the artifact in each seat. The
saved games replay in pyspiel move for move (every move legal, the same
actors) with the same returns, and the summary's payoffs are the records'.
"""

import json
import os
import sys

import numpy as np
import pytest

from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.examples import play_checkpoints
from open_spiel.python.scale.serve.tests import fresh_artifact
import pyspiel

GAME = "hearts"
NUM_PLAYERS = 4
GAMES = 8  # two deals, the artifact in each of the four seats


@pytest.fixture(name="artifact_dir", scope="module")
def _artifact_dir(tmp_path_factory):
  """A freshly initialised hearts network, artifact path@1.

  Args:
    tmp_path_factory: pytest's.

  Returns:
    The artifact folder.
  """
  return fresh_artifact.write(
      tmp_path_factory.mktemp("hearts_art"),
      registry.make(GAME),
      "hearts_info_state",
      step=1,
  )


def test_hearts_artifact_against_three_uniform_seats(
    artifact_dir, tmp_path, capsys
):
  """Hearts: a against three uniform seats, saved, replayed in pyspiel."""
  out_dir = str(tmp_path / "vs_uniform")
  summary = play_checkpoints.run(
      f"{artifact_dir}@1", "uniform", GAMES, 3, out_dir, probs=True
  )
  printed = capsys.readouterr().out
  assert "a in seat 3:" in printed, printed
  assert (
      f"replayed {GAMES} saved games in pyspiel: every move legal; "
      f"{GAMES} with the same returns"
      in printed
  ), printed
  with open(
      os.path.join(out_dir, play_checkpoints.SUMMARY), encoding="utf-8"
  ) as f:
    assert json.load(f) == json.loads(json.dumps(summary))
  assert summary["game"] == GAME and summary["num_players"] == NUM_PLAYERS
  params = summary["pyspiel_params"]
  assert isinstance(params, dict) and not params, params
  assert summary["num_records"] == GAMES
  assert sorted(k for k in summary["result"] if k.startswith("a_seat")) == [
      f"a_seat{r}" for r in range(NUM_PLAYERS)
  ]
  _, records = play_checkpoints.read_trajectories(out_dir)
  records = list(records)
  assert len(records) == GAMES
  a_pay, b_pay = [], []
  for i, rec in enumerate(records):
    assert (rec["index"], rec["deal"]) == (i, i // NUM_PLAYERS)
    assert rec["seats"] == [
        "a" if q == i % NUM_PLAYERS else "b" for q in range(NUM_PLAYERS)
    ]
    assert rec["game_string"] == GAME
    # hearts' histories are pyspiel's: no map, no translation
    assert rec["history"] == rec["env_history"]
    assert sum(rec["returns"]) in (78.0, 26.0)  # 26 after a moon
    state = play_checkpoints.replay_in_pyspiel(rec)
    assert state.history() == rec["history"]
    assert state.returns() == pytest.approx(rec["returns"], abs=0)
    a = rec["seats"].index("a")
    a_pay.append(rec["returns"][a])
    b_pay.append(
        np.mean([rec["returns"][q] for q in range(NUM_PLAYERS) if q != a])
    )
    # every decision's distribution, the acting side's (a's or b's)
    assert len(rec["policy"]) == len(rec["history"])
    for act, who, dist in zip(rec["history"], rec["actors"], rec["policy"]):
      if who < 0:
        assert dist is None
        continue
      assert act in [x for x, _ in dist]
      assert sum(p for _, p in dist) == pytest.approx(1.0, abs=1e-6)
  # The four games of a deal share its chance outcomes: the same deal.
  for d in range(GAMES // NUM_PLAYERS):
    deals = {
        tuple(records[NUM_PLAYERS * d + r]["history"][:53])
        for r in range(NUM_PLAYERS)
    }
    assert len(deals) == 1
  assert summary["result"]["a"]["mean"] == pytest.approx(np.mean(a_pay))
  assert summary["result"]["b_mean"] == pytest.approx(np.mean(b_pay))
  for r in range(NUM_PLAYERS):
    assert summary["result"][f"a_seat{r}"]["mean"] == pytest.approx(
        np.mean(a_pay[r::NUM_PLAYERS])
    )
  # the pyspiel game the records name is the one the artifact plays
  assert pyspiel.load_game(records[0]["game_string"]).num_players() == 4


def test_games_must_be_a_multiple_of_the_players(artifact_dir, tmp_path):
  """--games is a multiple of the four seats, at least two deals."""
  for games in (6, 4):
    with pytest.raises(Exception, match="multiple of 4"):
      play_checkpoints.run(
          f"{artifact_dir}@1", "uniform", games, 0, str(tmp_path / f"g{games}")
      )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
