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

"""Play a hearts or euchre card move transformer checkpoint against uniform.

For each game, a small card move transformer checkpoint (one block, freshly
initialised; the public loader reads it as it reads a trained one) is played
by play_checkpoints in one seat against uniform random play in the other
three, every deal with the agent in each seat, as the package docs' "Play
against it" command does. Hearts' games replay in pyspiel move for move;
euchre's replay needs a pyspiel with the corrected euchre (959-entry
tensor), so on a released pyspiel the games are played but not replayed (the
docs' --noverify_pyspiel).
"""

import sys

import pytest

from open_spiel.python.scale import deps
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.examples import play_checkpoints
from open_spiel.python.scale.networks import card_move_transformer as cmt
from open_spiel.python.scale.serve.tests import fresh_artifact
import pyspiel

pytest.importorskip(deps.GAMES_PACKAGE)

GAMES = 8  # two deals, the agent in each of the four seats
#: game -> (encoder, special-move tokens)
CHECKPOINTS = {
    "hearts": ("hearts_card_planes", 0),
    "euchre": ("euchre_card_planes", 7),
}


def _replays_in_pyspiel(game: str) -> bool:
  """Whether the installed pyspiel replays `game`'s records.

  Args:
    game: the game name.

  Returns:
    False for euchre on a released pyspiel, else True.
  """
  if game != "euchre":
    return True
  shape = pyspiel.load_game(game).information_state_tensor_shape()
  return list(shape) == [registry.EUCHRE_CORRECTED_TENSOR]


@pytest.mark.parametrize("game", sorted(CHECKPOINTS))
def test_checkpoint_plays_against_three_uniform_seats(game, tmp_path):
  encoder, special = CHECKPOINTS[game]
  enc = enc_registry.get(encoder)
  net = cmt.card_move_transformer(
      depth=1,
      embed_dim_per_head_over8=1,
      n_head=1,
      ff_factor=1,
      in_channels=enc.shape[0],
      n_card=enc.shape[1],
      n_special_action=special,
      compute_dtype="float32",
  )
  ckpt = fresh_artifact.write(
      tmp_path / f"{game}_ckpt",
      registry.make(game),
      encoder,
      step=1,
      network=net,
  )
  out_dir = str(tmp_path / "vs_uniform")
  summary = play_checkpoints.run(
      f"{ckpt}@1",
      "uniform",
      GAMES,
      0,
      out_dir,
      verify=_replays_in_pyspiel(game),
  )
  assert summary["game"] == game and summary["num_players"] == 4
  assert summary["num_records"] == GAMES
  _, records = play_checkpoints.read_trajectories(out_dir)
  assert len(list(records)) == GAMES


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
