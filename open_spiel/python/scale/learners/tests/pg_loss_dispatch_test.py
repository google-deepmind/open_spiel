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

"""Contract tests for PGLearner's loss dispatch.

Contract tests for PGLearner's loss dispatch: each loss reaches exactly its
function (losses.pg_loss for ppo / mmd, a registered LossSpec's own loss),
the magnet follows the loss, the registry refuses a clash, and an unknown loss
raises at every dispatch point instead of running pg_loss.
"""

import dataclasses
import sys

import jax
import pytest

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners.tests import pg_fixtures as fx
from open_spiel.python.scale.learners.tests import toy_loss


def _cfg(loss, **kw):
  """A PGConfig for `loss`, KL only for MMD.

  Args:
    loss: The loss.
    **kw: Overrides.

  Returns:
    The PGConfig.
  """
  kl = schedules.Schedule("constant", 0.2 if loss == "mmd" else 0.0)
  base = dict(
      loss=loss,
      lr=1e-3,
      anneal_lr=False,
      num_updates=10,
      update_epochs=1,
      num_minibatches=2,
      kl_coef=kl,
  )
  base.update(kw)
  return pg.PGConfig(**base)


def _setup(loss, **kw):
  """A learner for `loss`, its state and a self-play batch.

  Args:
    loss: The loss.
    **kw: PGConfig overrides.

  Returns:
    (learner, state, batch).
  """
  pol = fx.make_policy("kuhn_poker", None)
  learner = pg.PGLearner(pol, _cfg(loss, **kw))
  params = pol.init_params(jax.random.key(0))
  batch = fx.self_play_batch(pol, params, game="kuhn_poker", b=8, t=8, seed=1)
  return learner, learner.init_from_params(params), batch


def test_loss_families_and_the_registry():
  """Loss families and the registry."""
  assert pg.PG_LOSSES == ("ppo", "mmd")
  assert toy_loss.NAME in pg.losses() and set(pg.PG_LOSSES) <= set(pg.losses())
  assert (
      pg.extension("mmd") is None
      and pg.extension(toy_loss.NAME) is toy_loss.SPEC
  )
  assert pg.register_loss(toy_loss.SPEC) is toy_loss.SPEC  # idempotent
  with pytest.raises(ValueError, match="built-in"):
    pg.register_loss(dataclasses.replace(toy_loss.SPEC, name="mmd"))
  with pytest.raises(ValueError, match="already registered"):
    pg.register_loss(dataclasses.replace(toy_loss.SPEC, check=None))
  with pytest.raises(AssertionError, match="loss_config"):
    _cfg("mmd", loss_config=toy_loss.ToyLossConfig())
  with pytest.raises(AssertionError, match="ToyLossConfig"):
    _cfg(toy_loss.NAME, loss_config=loss_lib.PGLossConfig())
  with pytest.raises(AssertionError, match="magnet_every"):
    _cfg("mmd", magnet_every=2)


@pytest.mark.parametrize(
    "loss,kw,want",
    [
        ("ppo", {}, False),
        ("mmd", {}, True),
        (toy_loss.NAME, {}, False),
        (toy_loss.NAME, {"magnet_every": 3}, True),
    ],
)
def test_magnet_known_answer(loss, kw, want):
  """Magnet known answer."""
  learner, s, _ = _setup(loss, **kw)
  # Reaches private _has_magnet of the code under test.
  assert learner._has_magnet is want  # pylint: disable=protected-access
  assert (s.magnet is not None) is want


@pytest.mark.parametrize(
    "loss,called",
    [("ppo", "pg_loss"), ("mmd", "pg_loss"), (toy_loss.NAME, "toy")],
)
def test_each_loss_reaches_its_function(monkeypatch, loss, called):
  """Each loss reaches its function."""
  # Reaches private _EXTENSION_LOSSES of the code under test.
  # pylint: disable=protected-access
  calls = []
  real = loss_lib.pg_loss

  def _spy(*a, **k):
    """Records the call, then runs the real pg_loss.

    Args:
      *a: pg_loss's positional arguments.
      **k: pg_loss's keyword arguments.

    Returns:
      pg_loss's output.
    """
    calls.append("pg_loss")
    return real(*a, **k)

  def _toy(*a, **k):
    """Records the call, then runs the toy loss.

    Args:
      *a: The loss's positional arguments.
      **k: The loss's keyword arguments.

    Returns:
      The toy loss's output.
    """
    calls.append("toy")
    return toy_loss.loss(*a, **k)

  monkeypatch.setattr(loss_lib, "pg_loss", _spy)
  monkeypatch.setitem(
      pg._EXTENSION_LOSSES,
      # Reaches private members of the code under test.
      toy_loss.NAME,  # pylint: disable=protected-access
      dataclasses.replace(toy_loss.SPEC, loss=_toy),
  )
  learner, s, batch = _setup(loss)
  _, m = learner.update(s, batch, jax.random.key(2))
  assert calls and set(calls) == {called}, calls
  assert ("kl_penalty" in m) is (loss == "mmd")


def _break(learner, loss="a2c"):
  object.__setattr__(learner.cfg, "loss", loss)  # past PGConfig's own check


def test_config_refuses_an_unknown_loss():
  """Config refuses an unknown loss."""
  with pytest.raises(AssertionError, match="loss must be one of"):
    _cfg("a2c")
  with pytest.raises(AssertionError, match="loss must be one of"):
    dataclasses.replace(_cfg("mmd"), loss="a2c")


@pytest.mark.parametrize("start", ["ppo", "mmd", toy_loss.NAME])
def test_unknown_loss_raises_at_every_dispatch_point(start):
  """Unknown loss raises at every dispatch point."""
  learner, s, batch = _setup(start)
  _break(learner)
  with pytest.raises(ValueError, match="unknown PG loss 'a2c'"):
    # Reaches private _has_magnet of the code under test.
    _ = learner._has_magnet  # pylint: disable=protected-access
  with pytest.raises(ValueError, match="unknown PG loss 'a2c'"):
    learner.prepare(s, batch)
  with pytest.raises(ValueError, match="unknown PG loss 'a2c'"):
    learner.update(s, batch, jax.random.key(0))


@pytest.mark.parametrize("start", ["ppo", "mmd"])
def test_unknown_loss_in_rows_and_loss_raises(start):
  """Unknown loss in rows and loss raises."""
  learner, s, batch = _setup(start)
  prep = learner.prepare(s, batch)
  _break(learner)
  with pytest.raises(ValueError, match="unknown PG loss 'a2c'"):
    # Reaches private _rows of the code under test.
    learner._rows(batch, prep)  # pylint: disable=protected-access
  with pytest.raises(ValueError, match="unknown PG loss 'a2c'"):
    learner.update_prepared(s, batch, prep, jax.random.key(0))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
