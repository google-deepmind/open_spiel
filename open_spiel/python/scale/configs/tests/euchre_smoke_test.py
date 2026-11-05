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

"""configs.euchre_smoke: its shape, and a run end to end on a CPU."""

import sys

import numpy as np
import pytest

from open_spiel.python.scale import deps

# The euchre config imports the optional JAX games package: skip before the
# imports below need it.
pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.scale import train
from open_spiel.python.scale.configs import euchre_smoke as es
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import payoff_views

# pylint: enable=g-import-not-at-top,wrong-import-position


def test_config_shape():
  """Config shape."""
  cfg = common.load_config("euchre_smoke")
  assert cfg.env == "euchre" and cfg.controller == "scheduled_promotion"
  assert cfg.league.lineup is None and cfg.league.self_play_seats == "one"
  assert cfg.league.payoff == "evaluator"
  env = common.build_env(cfg)
  assert sides_lib.sides_of(env) == sides_lib.Sides.partnerships(
      4, 2, constant_sum=True
  )


def test_euchre_league_runs_end_to_end_on_cpu(tmp_path):
  """Euchre league runs end to end on cpu."""
  cfg = es.get_config()
  res = train.run(cfg, checkpoint_dir=str(tmp_path / "ckpt"))
  lg = res.league
  ls = lg.learners["main"]
  assert int(ls.updates) == es.NUM_UPDATES
  live = [m for m in lg.members.values() if m.lifecycle != "evicted"]
  assert len(live) >= 2
  assert len(ls.mix.support()) >= 2
  # Seating: the learner holds both seats of one partnership, alternating
  # North-South and East-West by row; the other partnership is one pool
  # member on both of its seats (league.sampler).
  # Reaches private _draw of the code under test: the seating it draws.
  # pylint: disable-next=protected-access
  table, _, counts, _ = lg._draw(ls, np.int64(0))
  b = lg.config.num_envs
  persp = np.asarray(table.perspective_seat)
  np.testing.assert_array_equal(persp, [[0, 2], [1, 3]] * (b // 2))
  slot = np.asarray(table.seat_slot)
  rows = np.arange(b)
  assert np.all(slot[rows, persp[:, 0]] == 0)
  assert np.all(slot[rows, persp[:, 1]] == 0)
  theirs = (persp + 1) % 4
  assert np.all(slot[rows, theirs[:, 0]] >= 1)
  np.testing.assert_array_equal(
      slot[rows, theirs[:, 0]], slot[rows, theirs[:, 1]]
  )
  assert sum(counts.values()) >= b
  pm = lg.payoff
  assert pm is not None
  assert np.all(np.isfinite(pm.mean)) and np.all(np.abs(pm.mean) <= 4)
  print(payoff_views.league_table(pm, lg.view_kind))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider", "-s"]
      )
  )
