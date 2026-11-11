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

"""Tables through the Arena.

Exact N-player scores, the two-player view, self-play zeros.

* table_keys under swap2's plan are core.arena_engine.deal_keys bit for bit,
  so a swap2 table replays the two-player Evaluator's games.
* TableResult.as_head_to_head of a swap2 table equals the two-player
  head_to_head (and mixture_head_to_head) within 1e-12 on kuhn and leduc;
  head_to_head itself is pinned bit for bit by testing/goldens.
* nplayer_matrix: pure strategies give exact per-position, per-side and
  placement results under every design (expected values by
  nplayer_matrix.exact_returns); mixed strategies agree with the exact
  expectation within 5 SE, vs a field and in the substitute gate.
* Self against self: 0 deal by deal under ARGMAX (kuhn3 cyclic, bridge
  swap2 in points and IMPs); substitute(S, S) is 0 deal by deal even when
  sampling (shared keys); sampled self-play tables have mean 0 within 5 SE.
* Per-seat tables (mixed sides) on team_pennies_2v2 and the uneven
  team_pennies_2v1: every seat's and side's score exact under pure
  strategies, within 5 SE of the exact expectation when sampling.
* The constant-sum invariant holds where declared and is not asserted for
  general-sum matching_pennies_3p.
* vs_field and substitute at a subject position other than 0, on the uneven
  odd_side_3v1v1 (three sides: a team of three, two lone seats): the
  subject rotates over every side of its position's class, scores exactly
  under pure strategies (hand-checked values), within 5 SE of the exact
  expectation against a sampled field; a design that does not rotate the
  position over its class is refused.
"""

import itertools
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena_engine
from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import nplayer_matrix as nm
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.eval import designs
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import payoff
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.policies import registry as pol_registry
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform

SEAT_INDEX = f"{nm.__name__}:seat_index"


# -- keys ----------------------------------------------------------------------
@pytest.mark.parametrize("duplicate", [True, False])
def test_swap2_table_keys_are_deal_keys(duplicate):
  """Swap2 table keys are deal keys."""
  plan = designs.key_plan(
      designs.swap2(sides_lib.Sides.individual(2, True)), duplicate
  )
  for seed, n in ((0, 9), (11, 3)):
    d0, p0 = arena_engine.deal_keys(jax.random.key(seed), n, duplicate)
    d1, p1 = arena_engine.table_keys(jax.random.key(seed), n, plan, 2)
    np.testing.assert_array_equal(
        jax.random.key_data(d0), jax.random.key_data(d1)
    )
    np.testing.assert_array_equal(
        jax.random.key_data(p0), jax.random.key_data(p1)
    )


def test_table_keys_follow_the_plan():
  """Table keys follow the plan."""
  d = designs.substitute(sides_lib.Sides.individual(3, True))
  dk, pk = arena_engine.table_keys(
      jax.random.key(3), 4, designs.key_plan(d, True), 3
  )
  dk = np.asarray(jax.random.key_data(dk)).reshape(4, 6, -1)
  pk = np.asarray(jax.random.key_data(pk)).reshape(4, 6, 3, -1)
  assert np.all(dk == dk[:, :1]) and not np.array_equal(dk[0], dk[1])
  np.testing.assert_array_equal(pk[:, :3], pk[:, 3:])
  seen = set()
  for r in range(3):
    for q in range(3):
      seen.add(pk[0, r, q].tobytes())
  assert len(seen) == 9
  # A deal's keys do not depend on how many deals are drawn.
  dk2, _ = arena_engine.table_keys(
      jax.random.key(3), 2, designs.key_plan(d, True), 3
  )
  np.testing.assert_array_equal(
      np.asarray(jax.random.key_data(dk2)).reshape(2, 6, -1), dk[:2]
  )


# -- the two-player view -------------------------------------------------------
def _two_player(game):
  """(env, network entrants, uniform) of a two-player game.

  Args:
    game: "kuhn" or "leduc".

  Returns:
    (env, [network entrants], the uniform entrant).
  """
  if game == "kuhn":
    env = env_registry.make("kuhn_poker")
    kw = common.network_policy_kwargs(
        "kuhn_poker",
        "kuhn_poker_info_index",
        [12],
        2,
        2,
        catalog.mlp([32], "relu", "separate"),
    )
    pol = pol_registry.build("network", kw)
  else:
    cfg = common.load_config("mmd_leduc_poker")
    env, pol = common.build_env(cfg), pol_registry.build(
        cfg.policy_arch, cfg.policy_kwargs
    )
  nets = [
      match.Entrant(
          f"net{i}", pol, pol.init_params(jax.random.key(i)), types.SAMPLE
      )
      for i in range(3)
  ]
  uni = match.Entrant(
      "uniform",
      uniform.UniformPolicy(env.name, env.num_actions, env.num_players),
      {},
      types.SAMPLE,
  )
  return env, nets, uni


def _assert_match_close(got: payoff.MatchResult, want: payoff.MatchResult):
  """Every field equal, the numbers within 1e-12.

  Args:
    got: The MatchResult computed.
    want: The MatchResult expected.
  """
  for f in payoff.MatchResult._fields:
    g, w = getattr(got, f), getattr(want, f)
    if f == "seat":
      for gs, ws in zip(g, w):
        np.testing.assert_allclose(
            np.array(gs), np.array(ws), rtol=0, atol=1e-12, err_msg=f
        )
    elif f in (
        "mean",
        "diff",
        "a_payoff",
        "b_payoff",
        "se_unpaired",
        "variance_reduction",
    ):
      np.testing.assert_allclose(
          np.array(g, np.float64),
          np.array(w, np.float64),
          rtol=0,
          atol=1e-12,
          err_msg=f,
      )
    else:
      assert g == w, (f, g, w)


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
@pytest.mark.parametrize("rule", [None, types.ARGMAX])
def test_as_head_to_head_matches_head_to_head(game, rule):
  """As head to head matches head to head."""
  env, nets, uni = _two_player(game)
  ev = match.Evaluator(env, match.EvaluatorConfig(max_rows=64, rule=rule))
  for a, b, n, seed in (
      (nets[0], uni, 48, 1),
      (nets[1], nets[2], 48, 2),
      (nets[0], nets[0], 16, 3),
  ):
    old = ev.head_to_head(a, b, n, jax.random.key(seed))
    tab = ev.table([a, b], n, jax.random.key(seed), "swap2")
    _assert_match_close(tab.as_head_to_head(), old)
    assert tab.check_sum() == 0.0


def test_as_head_to_head_matches_mixture_head_to_head():
  """As head to head matches mixture head to head."""
  env, nets, uni = _two_player("kuhn")
  ev = match.Evaluator(env, match.EvaluatorConfig(max_rows=64))
  key, n = jax.random.key(6), 40
  old = ev.mixture_head_to_head(
      nets[:2],
      [0.25, 0.75],
      [nets[2], uni],
      [0.6, 0.4],
      n,
      key,
      a_name="A",
      b_name="B",
  )
  # The same draws (match.Evaluator.mixture_head_to_head), played as a swap2
  # table.
  k_play, k_draw = rng.split(key)
  ka, kb = rng.split(k_draw)
  ia = np.asarray(rng.choice(ka, 2, (n,), p=np.array([0.25, 0.75])), np.int32)
  ib = np.asarray(rng.choice(kb, 2, (n,), p=np.array([0.6, 0.4])), np.int32)
  np.testing.assert_array_equal(ia, old.a_members)
  lineup = np.stack([ia, ib + 2], 1)[:, None, :]
  tab = ev.play_table(
      nets[:2] + [nets[2], uni], lineup, designs.swap2(ev.sides), n, k_play
  )
  _assert_match_close(tab.as_head_to_head("A", "B"), old.result)


# -- nplayer_matrix: exact -----------------------------------------------------
def _matrix_entrants(env, strategies, rule):
  """One Entrant per strategy.

  A distribution over actions, every seat alike.

  Args:
    env: the NPlayerMatrixEnv.
    strategies: one action distribution per entrant.
    rule: the entrants' DecisionRule.

  Returns:
    A list of Entrant.
  """
  pol = tabular.TabularPolicy(
      "nplayer_matrix",
      env.num_players,
      env.num_actions,
      env.num_players,
      SEAT_INDEX,
  )
  out = []
  for i, s in enumerate(strategies):
    logits = jnp.log(jnp.clip(jnp.asarray(s, jnp.float32), 1e-30, 1.0))
    logits = jnp.broadcast_to(logits, (env.num_players, env.num_actions))
    out.append(
        match.Entrant(f"e{i}", pol, pol.params_from_logits(logits), rule)
    )
  return out


def _exact_rows(variant, design, lineup, strategies):
  """f64[R, P]: exact expected seat returns of each row of one deal's lineup.

  Args:
    variant: The nplayer_matrix variant.
    design: The Design.
    lineup: The deal's lineup.
    strategies: Each entrant's strategy.

  Returns:
    f64[R, P].
  """
  rows = designs.table_rows(design, np.asarray(lineup)[None])[0]
  return np.stack(
      [nm.exact_returns(variant, [strategies[e] for e in row]) for row in rows]
  )


def _onehot(a, n=2):
  """The pure strategy of action `a`.

  Args:
    a: The action.
    n: The actions.

  Returns:
    f64[n].
  """
  return np.eye(n)[a]


PURE = {  # variant -> (design names, per-entrant pure actions)
    "odd_one_out_3p": (("cyclic", "cyclic+reversed", "substitute"), (0, 0, 1)),
    "odd_one_out_3p_c3": (("cyclic", "cyclic+reversed"), (1, 0, 1)),
    "matching_pennies_3p": (("cyclic", "cyclic+reversed"), (0, 1, 1)),
    "odd_one_out_4p": (("cyclic", "cyclic+reversed"), (0, 1, 1, 1)),
    "team_pennies_2v2": (("swap2", "swap2+partner"), (0, 1, 1, 0)),
    "team_pennies_2v1": (("cyclic+classes", "substitute"), (0, 1, 1)),
}


@pytest.mark.parametrize("variant", sorted(PURE))
def test_pure_strategies_score_exactly_under_every_design(variant):
  """Pure strategies score exactly under every design."""
  env = nm.NPlayerMatrixEnv(variant)
  sides = sides_lib.sides_of(env)
  names, acts = PURE[variant]
  strategies = [_onehot(a, env.num_actions) for a in acts]
  ev = match.Evaluator(env, match.EvaluatorConfig(name="exact", max_rows=8))
  ents = _matrix_entrants(env, strategies, types.ARGMAX)
  for name in names:
    d = designs.make(name, sides)
    t = sides.num_sides
    if d.num_blocks == 2:
      # S = e0, I = e1; field e2 then e0
      lineup = np.array([[0, 2, 1], [1, 2, 0]])
      seat_lineup = lineup
    elif variant == "team_pennies_2v1":
      # a mixed pair: e0 at seat 0, e1 at seat 2
      seat_lineup = np.array([[0, 2, 1]])
    elif name == "swap2+partner":
      # different partners: e0 + e2 against e1 + e3
      seat_lineup = np.array([[0, 1, 2, 3]])
    else:
      seat_lineup = designs.seat_lineup(sides, np.arange(t)[None])
    n = 3
    res = ev.play_table(
        ents,
        np.broadcast_to(seat_lineup, (n,) + seat_lineup.shape),
        d,
        n,
        jax.random.key(0),
    )
    want = _exact_rows(variant, d, seat_lineup, strategies)
    np.testing.assert_array_equal(
        res.returns, np.broadcast_to(want, res.returns.shape)
    )
    side_want = sides.side_returns(want)
    for b in range(d.num_blocks):
      rows = d.rows_of(b)
      for k in range(t):
        exp = side_want[rows, d.side_perms[rows, k]].mean()
        est = res.position_score(k, b)
        assert est.mean == pytest.approx(exp, abs=1e-12) and est.se == 0.0, (
            name,
            b,
            k,
        )
        for s, e in enumerate(res.by_side(k, b)):
          sel = rows[d.side_perms[rows, k] == s]
          if sel.size:
            assert e.mean == pytest.approx(side_want[sel, s].mean(), abs=1e-12)
          else:
            assert e is None
      pl = res.placement(0, b)
      assert pl.sum() == pytest.approx(1.0, abs=1e-12)
    total = res.check_sum()
    if variant == "matching_pennies_3p":
      assert total is None
    else:
      assert total == pytest.approx(
          {"odd_one_out_3p_c3": 3.0}.get(variant, 0.0), abs=1e-12
      )


@pytest.mark.parametrize("variant", ["team_pennies_2v2", "team_pennies_2v1"])
def test_per_seat_table_scores_every_seat_exactly(variant):
  """Evaluator.table with a mixed side scores every seat exactly.

  Different entrants at the partner seats, pure strategies: every seat's
  return, the side's and each seat's score are exact, and the result records
  the entrant of every seat.

  Args:
    variant: the nplayer_matrix variant.
  """
  env = nm.NPlayerMatrixEnv(variant)
  sides = sides_lib.sides_of(env)
  acts = (0, 1, 1, 0)
  strategies = [_onehot(a) for a in acts]
  ents = _matrix_entrants(env, strategies, types.ARGMAX)
  ev = match.Evaluator(env, match.EvaluatorConfig(name="exact", max_rows=8))
  fills = [
      (ents[0], ents[1]),
      ents[2] if sides.side_sizes[1] == 1 else (ents[2], ents[3]),
  ]
  res = ev.table(fills, 3, jax.random.key(0))
  d = res.design
  assert d.name == (
      "swap2" if variant == "team_pennies_2v2" else "cyclic+classes"
  )
  seat_ent = np.zeros(env.num_players, np.int32)
  for k, f in enumerate(fills):
    for q, e in zip(
        sides.seats(k),
        f if isinstance(f, tuple) else (f,) * len(sides.seats(k)),
    ):
      seat_ent[q] = ents.index(e)
  np.testing.assert_array_equal(
      res.lineup[:, 0], np.broadcast_to(seat_ent, (3, env.num_players))
  )
  assert res.entrants == tuple(
      ents[i].name for i in sorted(set(seat_ent.tolist()))
  )
  want = _exact_rows(variant, d, seat_ent[None], strategies)
  np.testing.assert_array_equal(
      res.returns, np.broadcast_to(want, res.returns.shape)
  )
  for q in range(env.num_players):
    exp = np.mean([want[r, d.seat_perms[r][q]] for r in range(d.num_rows)])
    assert res.seat_score(q).mean == pytest.approx(exp, abs=1e-12)
  side_want = sides.side_returns(want)
  for k in range(sides.num_sides):
    exp = np.mean([side_want[r, d.side_perms[r, k]] for r in range(d.num_rows)])
    assert res.position_score(k).mean == pytest.approx(exp, abs=1e-12)
  assert res.check_sum() == 0.0
  with pytest.raises(ValueError, match="its fill names"):
    ev.table([(ents[0],), ents[2]], 3, jax.random.key(0))


def test_two_v_one_mixed_strategies_agree_with_the_exact_expectation():
  """Two v one mixed strategies agree with the exact expectation."""
  env = nm.NPlayerMatrixEnv("team_pennies_2v1")
  s = [np.array([0.8, 0.2]), np.array([0.3, 0.7]), np.array([0.9, 0.1])]
  ents = _matrix_entrants(env, s, types.SAMPLE)
  res = match.Evaluator(env, match.EvaluatorConfig(max_rows=4096)).table(
      [(ents[0], ents[1]), ents[2]], 4000, jax.random.key(8)
  )
  exact = nm.exact_returns("team_pennies_2v1", [s[0], s[2], s[1]])
  for q in range(3):
    est = res.seat_score(q)
    assert abs(est.mean - exact[q]) < 5 * est.se + 1e-12, (q, est, exact)
  assert res.check_sum() == 0.0


def test_odd_one_out_placements_are_exact():
  """Odd one out placements are exact."""
  env = nm.NPlayerMatrixEnv("odd_one_out_3p")
  ents = _matrix_entrants(
      env, [_onehot(0), _onehot(0), _onehot(1)], types.ARGMAX
  )
  res = match.Evaluator(env).table(
      ents, 4, jax.random.key(1), "cyclic+reversed"
  )
  np.testing.assert_allclose(res.placement(2), [1.0, 0.0, 0.0], atol=1e-15)
  # The two equal seats tie for 2nd-3rd.
  np.testing.assert_allclose(res.placement(0), [0.0, 0.5, 0.5], atol=1e-15)
  assert res.diff(2, 0).mean == 3.0


# -- nplayer_matrix: sampled against the exact expectation ---------------------
def test_vs_field_agrees_with_the_exact_expectation():
  """Vs field agrees with the exact expectation."""
  env = nm.NPlayerMatrixEnv("matching_pennies_3p")
  subj = np.array([0.8, 0.2])
  field_s = [np.array([0.1, 0.9]), np.array([0.6, 0.4])]
  w = np.array([0.3, 0.7])
  ents = _matrix_entrants(env, [subj] + field_s, types.SAMPLE)
  res = match.Evaluator(env, match.EvaluatorConfig(max_rows=4096)).vs_field(
      ents[0], ents[1:], w, 4000, jax.random.key(2)
  )
  mix = w[0] * field_s[0] + w[1] * field_s[1]
  exact = np.mean([
      nm.exact_returns(
          "matching_pennies_3p", [subj if q == r else mix for q in range(3)]
      )[r]
      for r in range(3)
  ])
  est = res.position_score(0)
  assert abs(est.mean - exact) < 5 * est.se, (est, exact)
  assert res.check_sum() is None  # general-sum: logged, not asserted


def test_substitute_estimates_the_exact_replacement_value():
  """Substitute estimates the exact replacement value."""
  env = nm.NPlayerMatrixEnv("odd_one_out_3p")
  s_s, i_s, f_s = (
      np.array([0.9, 0.1]),
      np.array([0.5, 0.5]),
      np.array([0.2, 0.8]),
  )
  ents = _matrix_entrants(env, [s_s, i_s, f_s], types.SAMPLE)
  ev = match.Evaluator(env, match.EvaluatorConfig(max_rows=4096))
  res = ev.substitute(
      ents[0], ents[1], [ents[2]], [1.0], 4000, jax.random.key(3)
  )

  def val(x):
    """The exact value of x against the field, averaged over the positions.

    Args:
      x: a strategy.

    Returns:
      The value.
    """
    return np.mean([
        nm.exact_returns(
            "odd_one_out_3p", [x if q == r else f_s for q in range(3)]
        )[r]
        for r in range(3)
    ])

  est = res.block_diff(0)
  assert abs(est.mean - (val(s_s) - val(i_s))) < 5 * est.se, (
      est,
      val(s_s) - val(i_s),
  )
  # Shared keys make the paired difference far less noisy than unpaired blocks.
  unpaired = np.sqrt(
      res.position_score(0, 0).se ** 2 + res.position_score(0, 1).se ** 2
  )
  assert est.se < unpaired
  # Self against self: exactly 0 on every deal, although every seat samples.
  same = ev.substitute(
      ents[1], ents[1], [ents[2]], [1.0], 64, jax.random.key(4)
  )
  np.testing.assert_array_equal(same.deal_scores(0, 0), same.deal_scores(0, 1))
  # play did vary
  assert np.any(same.deal_scores(0, 0) != same.deal_scores(0, 0)[0])


# -- subject positions at uneven sides (3v1v1) --------------------------------
V311 = "odd_side_3v1v1"


@pytest.mark.parametrize(
    "position, want",
    # Field: every seat plays 0; subject: every seat plays 1.
    # 0: the team (x = 1 ^ 1 ^ 1 = 1) is the odd side on both rows: 2.
    # 1: a lone side, rotated over sides 1 and 2: at side 1 it is odd and
    #    collects the bonus (2 + 1), at side 2 it is odd and pays nothing
    #    (seat 1 plays 0): (3 + 2) / 2. Fixed at side 1 it would be 3.
    # 2: the same two rows in the other order.
    [(0, 2.0), (1, 2.5), (2, 2.5)],
)
def test_vs_field_and_substitute_rotate_the_subject_over_its_class(
    position, want
):
  """3v1v1: the subject at `position` holds every side of its class.

  Exact under pure strategies (se 0), by the hand-checked value and by the
  design's rows; substitute's gate statistic is S's score minus I's, with the
  incumbent (every seat 0) scoring exactly 0 against the all-0 field.

  Args:
    position: the subject's side position.
    want: the subject's exact score.
  """
  env = nm.NPlayerMatrixEnv(V311)
  sides = sides_lib.sides_of(env)
  assert sides.side_sizes == (3, 1, 1)
  assert sides.rotation_classes == ((0,), (1, 2))
  ents = _matrix_entrants(
      env, [_onehot(1), _onehot(0), _onehot(0)], types.ARGMAX
  )
  ev = match.Evaluator(env, match.EvaluatorConfig(name="exact", max_rows=8))
  res = ev.vs_field(
      ents[0], [ents[2]], [1.0], 4, jax.random.key(0), position=position
  )
  cls = sides.rotation_classes[0 if position == 0 else 1]
  assert sorted(set(res.design.side_perms[:, position].tolist())) == list(cls)
  est = res.position_score(position)
  assert est.mean == want and est.se == 0.0, (position, est)
  lineup = res.lineup[0]
  rows = _exact_rows(V311, res.design, lineup, [_onehot(1), _onehot(0)])
  side_want = sides.side_returns(rows)
  d = res.design
  by_rows = [side_want[r, d.side_perms[r, position]] for r in range(d.num_rows)]
  assert est.mean == pytest.approx(np.mean(by_rows), abs=1e-12)
  assert res.check_sum() == 0.0

  sub = ev.substitute(
      ents[0],
      ents[1],
      [ents[2]],
      [1.0],
      4,
      jax.random.key(1),
      position=position,
  )
  assert sorted(
      set(sub.design.side_perms[sub.design.rows_of(0), position].tolist())
  ) == list(cls)
  assert sub.position_score(position, 0).mean == want
  assert sub.position_score(position, 1).mean == 0.0
  gate = sub.block_diff(position)
  assert gate.mean == want and gate.se == 0.0
  assert sub.check_sum() == 0.0


def test_vs_field_at_a_lone_position_agrees_with_the_exact_expectation():
  """3v1v1, subject on a lone side against a sampled two-member field.

  Each other side draws one member per deal (the team's three seats play
  that one member); the exact value enumerates the members of the two other
  sides and averages the subject's side return over the lone sides it holds.
  """
  env = nm.NPlayerMatrixEnv(V311)
  sides = sides_lib.sides_of(env)
  subj = np.array([0.3, 0.7])
  members = [np.array([0.9, 0.1]), np.array([0.2, 0.8])]
  w = np.array([0.4, 0.6])
  ents = _matrix_entrants(env, [subj] + members, types.SAMPLE)
  ev = match.Evaluator(env, match.EvaluatorConfig(max_rows=4096))
  res = ev.vs_field(ents[0], ents[1:], w, 4000, jax.random.key(5), position=1)

  def value(at_side):
    """The exact side return of the subject at `at_side` against the field.

    Args:
      at_side: the side the subject holds.

    Returns:
      The expectation over the two other sides' member draws.
    """
    total = 0.0
    others = [k for k in range(3) if k != at_side]
    for picks in itertools.product(range(2), repeat=2):
      who = {at_side: subj}
      who.update({k: members[j] for k, j in zip(others, picks)})
      strat = [who[sides.side(q)] for q in range(env.num_players)]
      sr = sides.side_returns(nm.exact_returns(V311, strat))
      total += w[picks[0]] * w[picks[1]] * sr[at_side]
    return total

  exact = (value(1) + value(2)) / 2
  assert value(1) != value(2)  # the two lone sides differ
  est = res.position_score(1)
  assert abs(est.mean - exact) < 5 * est.se, (est, exact)
  by = res.by_side(1)
  assert by[0] is None and by[1] is not None and by[2] is not None


def test_a_design_that_does_not_rotate_the_position_is_refused():
  """A design that leaves the subject's position on one side of its class."""
  env = nm.NPlayerMatrixEnv(V311)
  sides = sides_lib.sides_of(env)
  ents = _matrix_entrants(env, [_onehot(1), _onehot(0)], types.ARGMAX)
  ev = match.Evaluator(env, match.EvaluatorConfig(name="exact", max_rows=8))
  fixed = designs.Design(
      "cyclic+classes", sides, (tuple(range(5)),), (0,), (0,)
  )
  with pytest.raises(ValueError, match="rotation class"):
    ev.vs_field(
        ents[0], [ents[1]], [1.0], 4, jax.random.key(0), fixed, position=1
    )
  # The team's class is itself: the identity design rotates it fully.
  res = ev.vs_field(
      ents[0], [ents[1]], [1.0], 4, jax.random.key(0), fixed, position=0
  )
  assert res.position_score(0).mean == 2.0
  with pytest.raises(ValueError, match="position 3"):
    ev.vs_field(ents[0], [ents[1]], [1.0], 4, jax.random.key(0), position=3)
  with pytest.raises(ValueError, match="position -1"):
    ev.substitute(
        ents[0], ents[1], [ents[1]], [1.0], 4, jax.random.key(0), position=-1
    )


# -- self against self ---------------------------------------------------------
def _net(env, encoder_name, width=32):
  """A small network policy of `env` over encoder `encoder_name`.

  Args:
    env: The env.
    encoder_name: The encoder.
    width: The hidden width.

  Returns:
    The policy.
  """
  enc = enc_registry.get(encoder_name)
  kw = common.network_policy_kwargs(
      env.name,
      enc.name,
      list(enc.shape),
      env.num_actions,
      env.num_players,
      catalog.mlp([width], "relu", "separate"),
  )
  return pol_registry.build("network", kw)


def test_kuhn3_self_play_is_zero_per_deal_under_argmax():
  """Kuhn3 self play is zero per deal under argmax."""
  env = env_registry.make("kuhn_nplayer", players=3)
  pol = _net(env, "kuhn_nplayer_info_state_tensor")
  e = match.Entrant(
      "net", pol, pol.init_params(jax.random.key(0)), types.ARGMAX
  )
  ev = match.Evaluator(env, match.EvaluatorConfig(max_rows=256))
  res = ev.table([e, e, e], 64, jax.random.key(5))
  assert res.design.name == "cyclic"
  for k in range(3):
    np.testing.assert_array_equal(res.deal_scores(k), 0.0)
  assert np.any(res.returns != 0.0)
  assert res.check_sum() == 0.0
  # Two other nets in the field: a real table, still constant-sum.
  others = [
      match.Entrant(
          f"o{i}", pol, pol.init_params(jax.random.key(i + 1)), types.ARGMAX
      )
      for i in range(2)
  ]
  sub = ev.substitute(e, e, others, [0.5, 0.5], 32, jax.random.key(6))
  np.testing.assert_array_equal(sub.block_diff(0).mean, 0.0)


def test_kuhn3_sampled_self_play_has_mean_zero():
  """Kuhn3 sampled self play has mean zero."""
  env = env_registry.make("kuhn_nplayer", players=3)
  u = match.Entrant(
      "u", uniform.UniformPolicy(env.name, env.num_actions, 3), {}, types.SAMPLE
  )
  res = match.Evaluator(env, match.EvaluatorConfig(max_rows=4096)).table(
      [u, u, u], 3000, jax.random.key(7), "cyclic+reversed"
  )
  est = res.position_score(0)
  assert est.se > 0.0 and abs(est.mean) < 5 * est.se, est


@pytest.fixture(name="bridge_env", scope="module")
def _bridge_env_fixture():
  """The default bridge env.

  Returns:
    The BridgeEnv.
  """
  # Lazy: the bridge env needs the optional JAX games package.
  from open_spiel.python.scale.envs import bridge  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return bridge.BridgeEnv()


def test_bridge_self_play_is_zero_per_deal_under_argmax(bridge_env):
  """Bridge self play is zero per deal under argmax."""
  env = bridge_env
  assert sides_lib.sides_of(env) == sides_lib.Sides.partnerships(4, 2, True)
  pol = _net(env, "bridge_info_state", width=16)
  e = match.Entrant(
      "net", pol, pol.init_params(jax.random.key(0)), types.ARGMAX
  )
  ev = match.Evaluator(env, match.EvaluatorConfig(max_rows=32))
  r = ev.head_to_head(e, e, 8, jax.random.key(8))  # IMPs (bridge's pair scorer)
  np.testing.assert_array_equal(r.a_payoff[:, 0], -r.a_payoff[:, 1])
  assert r.mean.mean == 0.0 and r.mean.se == 0.0 and np.isnan(r.se_unpaired)
  tab = ev.table([e, e], 8, jax.random.key(8))
  assert np.any(tab.returns != 0.0)  # contracts were bid and played
  np.testing.assert_array_equal(tab.deal_scores(0), 0.0)
  np.testing.assert_array_equal(tab.deal_scores(0, scorer=env.pair_scorer), 0.0)
  # partners agree
  np.testing.assert_array_equal(tab.returns, tab.returns[:, :, [2, 3, 0, 1]])
  assert tab.check_sum() == 0.0


def test_bridge_sampled_self_play_has_mean_zero(bridge_env):
  """Bridge sampled self play has mean zero."""
  env = bridge_env
  u = match.Entrant(
      "u", uniform.UniformPolicy(env.name, env.num_actions, 4), {}, types.SAMPLE
  )
  ev = match.Evaluator(env, match.EvaluatorConfig(max_rows=128))
  tab = ev.table([u, u], 128, jax.random.key(9))
  for scorer in (None, env.pair_scorer):
    est = tab.position_score(0, scorer=scorer)
    assert est.se > 0.0 and abs(est.mean) < 5 * est.se, (scorer, est)


def test_arena_validate_accepts_every_design_row():
  """Each row's perspective seat sits in the subject's slot.

  core.arena validate.
  """
  for sides in (
      sides_lib.Sides.individual(3, True),
      sides_lib.Sides.partnerships(4, 2, True),
      sides_lib.Sides((0, 1, 0, 2, 0), True),
  ):
    for name in designs.NAMES:
      try:
        d = designs.make(name, sides)
      except ValueError:
        continue
      lineup = np.zeros((1, d.num_blocks, sides.num_players), np.int32)
      lineup[0, :, list(sides.seats(0))] = 1
      rows = designs.table_rows(d, lineup)[0]
      for perm, row in zip(d.seat_perms, rows):
        assert row[perm[0]] == 1


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
