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

"""core.hooks: declared hooks raise when misused, and are the only way in.

Every hook the framework reads goes through a declared Protocol.

Known answers: an undeclared read, a missing required hook, a hook of the
wrong kind and a failing property each raise; the league refuses, at a field
view, a controller that declares neither supports_sides nor pairwise; a run
hook that defines none of its hooks is refused; a canonical deal form with a
malformed every_ply is refused at registration.
"""

# pylint: disable=protected-access

import sys
import types as pytypes

import pytest

from open_spiel.python.scale.core import hooks
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.league import controller as controller_lib
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league import sampler
from open_spiel.python.scale.sources import deals


def _proto(name, **kw):
  """A throwaway Protocol (removed from PROTOCOLS by the caller's fixture).

  Args:
    name: its name.
    **kw: Protocol keyword arguments.

  Returns:
    The Protocol.
  """
  return hooks.Protocol(
      name,
      [
          hooks.method("act", "act() -> int"),
          hooks.value("level", "an int >= 0", check=lambda v: v >= 0),
      ],
      **kw,
  )


@pytest.fixture(name="scratch")
def _scratch(monkeypatch):
  """PROTOCOLS restored after the test."""
  monkeypatch.setattr(hooks, "PROTOCOLS", dict(hooks.PROTOCOLS))
  yield


def test_get_require_and_check_raise_when_used_wrongly(scratch):
  """Get, require and check raise when used wrongly."""
  del scratch
  p = _proto("toy")

  class Full:
    level = 2

    def act(self):
      """The toy hook.

      Returns:
        1.
      """
      return 1

  class Empty:
    pass

  assert p.get(Full(), "act")() == 1 and p.get(Full(), "level") == 2
  assert p.get(Empty(), "act") is None
  assert p.defined(Full()) == ("act", "level")
  none = p.defined(Empty())
  assert isinstance(none, tuple) and not none, none
  with pytest.raises(hooks.UndeclaredHookError, match="'acts' is not a toy"):
    p.get(Full(), "acts")
  with pytest.raises(hooks.MissingHookError, match="defines no act hook"):
    p.require(Empty(), "act")

  class NotCallable:
    act = 3

  with pytest.raises(hooks.MalformedHookError, match="'act' is a method"):
    p.get(NotCallable(), "act")
  with pytest.raises(hooks.MalformedHookError, match="malformed"):
    p.check(pytypes.SimpleNamespace(level=-1))
  with pytest.raises(hooks.MalformedHookError, match="malformed"):
    p.check(pytypes.SimpleNamespace(level="x"))  # the check raised TypeError

  class Broken:

    @property
    def level(self):
      raise AttributeError("a bug inside the property")

  with pytest.raises(AttributeError, match="a bug inside"):
    p.get(Broken(), "level")

  # A value of None is the documented "not defined".
  assert p.get(pytypes.SimpleNamespace(act=None), "act") is None
  # A function's attributes are hooks too.

  def fn():
    return None

  fn.level = 0
  assert p.get(fn, "level") == 0


def test_at_least_one_and_declaration_errors(scratch):
  """At least one and declaration errors."""
  del scratch
  p = _proto("toy", at_least_one=True)
  with pytest.raises(hooks.MissingHookError, match="defines none"):
    p.check(object())
  x = pytypes.SimpleNamespace(act=lambda: 0)
  assert p.check(x) is x
  with pytest.raises(ValueError, match="already declared"):
    _proto("toy")
  with pytest.raises(ValueError, match="twice"):
    hooks.Protocol("dup", [hooks.method("a", ""), hooks.value("a", "")])
  with pytest.raises(ValueError, match="no hooks"):
    hooks.Protocol("none", [])
  with pytest.raises(TypeError, match="not a Hook"):
    hooks.Protocol("bad", ["a"])


def test_every_framework_protocol_is_declared():
  """Every framework protocol is declared and documented."""
  want = {
      "league controller": {
          "supports_sides",
          "pairwise",
          "reserved_pool_positions",
          "training_row_filter",
          "draw_guards",
          "gate_request",
      },
      "promotion gate": {"training_row_filter", "draw_guards"},
      "run hook": {"initial_eval", "after_update"},
      "evaluator": {"sides", "gate_estimate"},
      "match engine": {"table_keys"},
      "env": {"sides", "pair_scorer"},
      "canonical deal form": {"hash_tag", "every_ply"},
  }
  for name, names in want.items():
    assert set(hooks.PROTOCOLS[name].hooks) == names, name
    assert all(h.doc for h in hooks.PROTOCOLS[name].hooks.values()), name
  text = hooks.describe()
  assert all(f"{name}:" in text for name in want)
  assert controller_lib.CONTROLLER_HOOKS is hooks.PROTOCOLS["league controller"]
  assert runner.RUN_HOOKS is hooks.PROTOCOLS["run hook"]
  assert match.EVALUATOR_HOOKS is hooks.PROTOCOLS["evaluator"]
  assert sides_lib.ENV_HOOKS is hooks.PROTOCOLS["env"]
  assert deals.CANON_HOOKS is hooks.PROTOCOLS["canonical deal form"]


def test_pairwise_declarations_of_the_core_controllers():
  """The core controllers declare pairwise (PAIRWISE_CONTROLLERS retired)."""
  assert not hasattr(loop, "PAIRWISE_CONTROLLERS")
  pairwise = {
      controller_lib.GatedPool("main"): True,
      controller_lib.ContinualExploiter("main"): True,
      controller_lib.BROracle("main"): True,
      controller_lib.PBT(["a", "b"]): True,
      controller_lib.PSRO(meta_solver="nash"): True,
      controller_lib.PSRO(meta_solver="uniform"): False,
      controller_lib.FixedMix("x", sampler.Mix(1.0, {})): False,
  }
  proto = controller_lib.CONTROLLER_HOOKS
  for c, want in pairwise.items():
    assert proto.require(c, "pairwise") is want, type(c).__name__


def _check_sides(controller, sides):
  """League._check_sides on a stand-in League holding only `sides`.

  Args:
    controller: the controller.
    sides: the core.sides.Sides.

  Returns:
    The stand-in's view kind.
  """
  me = pytypes.SimpleNamespace(sides=sides)
  cfg = pytypes.SimpleNamespace(lineup=None, eval_design="auto")
  env = pytypes.SimpleNamespace(name="toy3")
  loop.League._check_sides(me, env, controller, cfg)
  return me.view_kind


class _Bare:
  """A controller that declares no sides hook."""

  def step(self, view):
    """A controller step that does nothing.

    Args:
      view: the league view (unused).

    Returns:
      No actions.
    """
    del view
    return []


def test_a_controller_without_a_sides_hook_is_refused_at_a_field_view():
  """A controller without a sides hook is refused at a field view."""
  three = sides_lib.Sides.individual(3, True)
  two = sides_lib.Sides.individual(2, True)
  # At a pair view nothing is asked of it (unchanged).
  assert _check_sides(_Bare(), two) == "pair"
  with pytest.raises(hooks.MissingHookError, match="defines no pairwise"):
    _check_sides(_Bare(), three)
  with pytest.raises(hooks.MissingHookError, match="defines no pairwise"):
    _check_sides(
        controller_lib.Composite(
            [controller_lib.FixedMix("x", sampler.Mix(1.0, {})), _Bare()]
        ),
        three,
    )
  # Declared: refused (pairwise) or allowed, as before.
  with pytest.raises(ValueError, match="two-player game"):
    _check_sides(controller_lib.PBT(["a", "b"]), three)
  assert _check_sides(controller_lib.PSRO(meta_solver="uniform"), three) == (
      "field"
  )
  assert (
      _check_sides(controller_lib.FixedMix("x", sampler.Mix(1.0, {})), three)
      == "field"
  )

  class Says(_Bare):
    pairwise = "no"

  with pytest.raises(hooks.MalformedHookError, match="pairwise"):
    _check_sides(Says(), two)

  class Flag(_Bare):
    supports_sides = True

  with pytest.raises(hooks.MalformedHookError, match="supports_sides"):
    _check_sides(Flag(), two)


def test_a_run_hook_that_hooks_nothing_is_refused(monkeypatch):
  """A run hook that hooks nothing is refused."""
  monkeypatch.setattr(runner, "_RUN_HOOKS", [])

  class Nothing:

    def after_updates(self, *a):  # misspelled: not a hook
      del a

  runner.register_run_hook(lambda cfg: Nothing())
  with pytest.raises(hooks.MissingHookError, match="defines none"):
    runner.run_hooks(None)
  monkeypatch.setattr(runner, "_RUN_HOOKS", [])

  class NotCallable:
    initial_eval = False

  runner.register_run_hook(lambda cfg: NotCallable())
  with pytest.raises(hooks.MalformedHookError, match="initial_eval"):
    runner.run_hooks(None)
  monkeypatch.setattr(runner, "_RUN_HOOKS", [])

  class Fine:

    def initial_eval(self):
      """The run hook: no initial eval.

      Returns:
        False.
      """
      return False

  runner.register_run_hook(lambda cfg: None)
  runner.register_run_hook(lambda cfg: Fine())
  got = runner.run_hooks(None)
  assert len(got) == 1 and isinstance(got[0], Fine)


def test_env_and_canon_hooks_are_checked(monkeypatch):
  """Env and canonical-form hooks are checked."""
  env = pytypes.SimpleNamespace(num_players=2, zero_sum=True, pair_scorer=3)
  with pytest.raises(TypeError, match="pair_scorer"):
    sides_lib.pair_scorer_of(env)
  env.pair_scorer = None
  assert sides_lib.pair_scorer_of(env) is sides_lib.mean_pair_score
  monkeypatch.setattr(deals, "_CANON", {})

  def canon(info, labels):
    del info, labels

  canon.every_ply = "yes"
  with pytest.raises(TypeError, match="every_ply"):
    deals.register_canon("toy_every", canon)
  canon.every_ply = True
  deals.register_canon("toy_every", canon)
  every = deals.every_ply("toy_every")
  assert isinstance(every, bool) and every
  with pytest.raises(TypeError, match="not callable"):
    deals.register_canon("toy_none", 3)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
