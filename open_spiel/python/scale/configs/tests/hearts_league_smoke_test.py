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

"""configs.hearts_league_smoke: its shape, and a run end to end on a CPU."""

import sys

import numpy as np
import pytest

from open_spiel.python.scale import deps

# The hearts config imports the optional JAX games package: skip before the
# imports below need it.
pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.scale import train
from open_spiel.python.scale.configs import hearts_league_smoke as hls
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import payoff_views

# pylint: enable=g-import-not-at-top,wrong-import-position


def test_config_shape():
  """Config shape."""
  cfg = common.load_config("hearts_league_smoke")
  assert cfg.env == "hearts" and cfg.controller == "scheduled_promotion"
  assert cfg.league.lineup is None and cfg.league.self_play_seats == "one"
  assert cfg.league.payoff == "evaluator"
  env = common.build_env(cfg)
  assert sides_lib.sides_of(env) == sides_lib.Sides.individual(
      4, constant_sum=False
  )


def test_hearts_league_runs_end_to_end_on_cpu(tmp_path):
  """Hearts league runs end to end on cpu."""
  cfg = hls.get_config()
  res = train.run(cfg, checkpoint_dir=str(tmp_path / "ckpt"))
  lg = res.league
  ls = lg.learners["main"]
  assert int(ls.updates) == hls.NUM_UPDATES
  live = [m for m in lg.members.values() if m.lifecycle != "evicted"]
  assert len(live) >= 2
  assert len(ls.mix.support()) >= 2
  # Seating: the learner on one seat per row (cycling over the four), and the
  # other three seats pool members drawn independently (league.sampler).
  # Reaches private _draw of the code under test: the seating it draws.
  table, _, counts, _ = lg._draw(ls, np.int64(0))  # pylint: disable=protected-access
  b = lg.config.num_envs
  persp = np.asarray(table.perspective_seat)
  np.testing.assert_array_equal(persp, np.arange(b) % 4)
  slot = np.asarray(table.seat_slot)
  rows = np.arange(b)
  assert np.all(slot[rows, persp] == 0)
  others = np.stack(
      [slot[r, [q for q in range(4) if q != persp[r]]] for r in rows]
  )
  assert np.all(others >= 1)  # never the learner's slot
  assert sum(counts.values()) == 3 * b  # three pool seats per row
  pm = lg.payoff
  assert pm is not None and lg.view_kind == "field"
  assert np.all(np.isfinite(pm.mean)) and np.all(
      (pm.mean >= 0) & (pm.mean <= 36)
  )
  print(payoff_views.league_table(pm, lg.view_kind))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider", "-s"]
      )
  )
