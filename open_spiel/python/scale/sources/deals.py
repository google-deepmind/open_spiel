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

"""Canonical deal hashes: training data never holds an evaluation deal.

An evaluation on fixed deals is only unbiased if no training row started from
one of them. Each game with curated evaluation deals defines a canonical form
of a deal (registered with `register_canon`), computed from what a training
row carries at the first ply of a hand (`steps.new_hand`): the seat's
InfoState and the privileged labels. The hash is
sha256(tag + game + rules_sha + canonical bytes), so two rule sets never
share a hash (tag: HASH_TAG).

    index = deals.DealIndex.from_eval_sets(env, [("eval", seed, num_pairs)])
    index.save(path)
    shards.ShardWriter(out_dir, env, rules_sha=rs, info_schema_version=1,
                       deal_exclusion=index)  # drops matching rows

A game whose labels carry the whole deal on every ply, not only at a hand
start, marks its canonical form with `every_ply = True` (an attribute of the
function). batch_hashes then hashes every valid ply of its rows, so a row
cut from the middle of an evaluation deal is caught too.

A canonical form is whatever identifies a deal regardless of how play goes
on from it: for a card game, typically every seat's initial hand in seat
order plus any dealt public card. A game's GameSpec.canon names its function
(for example envs.gin.deal_canon).

DealIndex holds the hashes of every evaluation set of a game.
`from_eval_sets(env, [(name, seed, num_pairs), ...])` recomputes the deals
the Evaluator plays for jax.random.key(seed) (duplicate, seat-swapped, hand
0). It is saved as JSON (sorted hashes and their sources) and identified by
its sha256.

Enforcement (`sources.shards`): a ShardWriter given a DealIndex drops every
row with a hand that starts from an indexed deal and records the drops in the
manifest; a row with a decision before its first hand start cannot be
checked and is refused (shards.Error). A ShardReader given a DealIndex refuses
a shard written without the same index and recomputes every row; one hit
refuses the shard.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Optional, Sequence

from etils import epath
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import arena_engine
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry

#: The domain-separation tag hashed first into every deal hash.
HASH_TAG = b"deal_hash\0"
#: The "format" field of a saved DealIndex (part of the index's sha256).
INDEX_FORMAT = "deal_index"


Canon = Callable[[Any, Mapping[str, Any]], np.ndarray]
_CANON: dict[str, Canon] = {}


def register_canon(game: str, fn: Canon) -> None:
  """Registers game's canonical deal form.

  A game without a registered form uses its GameSpec.canon.

  Args:
    game: the game name.
    fn: fn(info [N] (numpy leaves), labels [N]) -> uint8[N, k], one row per
      deal.

  Raises:
    ValueError: game has another canonical form.
  """
  if _CANON.get(game, fn) is not fn:
    raise ValueError(f"game {game!r} already has a canonical deal form")
  _CANON[game] = fn


def canon_of(game: str) -> Canon:
  """The canonical deal form of `game`.

  The registered one, else its GameSpec.canon.

  Args:
    game: the game name.

  Returns:
    The Canon.

  Raises:
    KeyError: the game has neither.
  """
  if game in _CANON:
    return _CANON[game]
  spec = env_registry.game(game) if game in env_registry.names() else None
  if spec is not None and spec.canon is not None:
    return env_registry.resolve(spec.canon)
  raise KeyError(
      f"no canonical deal form for {game!r} (registered {sorted(_CANON)}): "
      "register one with sources.deals.register_canon"
  )


def deal_hashes(
    game: str, rules_sha: str, info: Any, labels: Mapping[str, Any]
) -> list[str]:
  """The canonical hash (hex) of each row's deal.

  Args:
    game: the game name.
    rules_sha: the rules' sha.
    info: the InfoState [N] at the first ply of the hand.
    labels: the labels [N] at that ply.

  Returns:
    The hex hashes, one per row.
  """
  canon = canon_of(game)(jax.device_get(info), jax.device_get(dict(labels)))
  pre = HASH_TAG + game.encode() + b"\0" + rules_sha.encode() + b"\0"
  return [hashlib.sha256(pre + row.tobytes()).hexdigest() for row in canon]


def row_hand_starts(steps: types.SeatStep, valid) -> tuple[Any, Any, Any]:
  """Every hand start (a new_hand ply) of a batch [N, T].

  Args:
    steps: the SeatSteps [N, T].
    valid: bool[N, T].

  Returns:
    (rows i64[H], plies i64[H], unaligned bool[N]): unaligned rows have a valid
      ply before their first hand start (their deal is not in the row).
  """
  nh = np.asarray(steps.new_hand, bool)
  v = np.asarray(valid, bool)
  rows, ts = np.nonzero(nh)
  before = np.cumsum(nh, axis=1) == 0
  return rows.astype(np.int64), ts.astype(np.int64), (v & before).any(axis=1)


def every_ply(game: str) -> bool:
  """Whether `game`'s labels carry the whole deal on every ply (module doc).

  Args:
    game: the game name.

  Returns:
    True for an every-ply game.
  """
  return bool(getattr(canon_of(game), "every_ply", False))


def batch_hashes(
    game: str, rules_sha: str, batch: types.TrajectoryBatch
) -> tuple[Any, Any, Any]:
  """(rows i64[H], hashes [H], unaligned bool[N]) of a TrajectoryBatch.

  The hand starts' deals, or for an every-ply game every valid ply's (never
  unaligned).

  Args:
    game: the game name.
    rules_sha: the rules' sha.
    batch: the TrajectoryBatch.

  Returns:
    (rows, hashes, unaligned).
  """
  if every_ply(game):
    rows, ts = np.nonzero(np.asarray(batch.valid, bool))
    rows, ts = rows.astype(np.int64), ts.astype(np.int64)
    unaligned = np.zeros((int(np.shape(batch.valid)[0]),), bool)
  else:
    rows, ts, unaligned = row_hand_starts(batch.steps, batch.valid)
  if rows.size == 0:
    return rows, [], unaligned

  def pick(x):
    """The entries at the picked (row, ply) pairs.

    Args:
      x: A [rows, plies, ...] array.

    Returns:
      The picked array.
    """
    return np.asarray(x)[rows, ts]

  info = jax.tree_util.tree_map(pick, batch.steps.info)
  labels = {k: pick(v) for k, v in batch.labels.items()}
  return rows, deal_hashes(game, rules_sha, info, labels), unaligned


def eval_deal_states(env: types.Env, seed: int, num_pairs: int):
  """The env states of hand 0 of the Evaluator's num_pairs deals.

  The deals the Evaluator plays for jax.random.key(seed) (duplicate: one deal
  per seat-swapped pair).

  Args:
    env: the env.
    seed: the evaluation seed.
    num_pairs: the deal count.

  Returns:
    The env states [num_pairs].
  """
  dk, _ = arena_engine.deal_keys(
      jax.random.key(int(seed)), int(num_pairs), True
  )
  dk = dk[::2]  # rows (i, 0) and (i, 1) share deal i's key
  states, _ = jax.jit(
      lambda k: arena_lib.deal_states(env, k, jnp.zeros(k.shape[:1], jnp.int64))
  )(dk)
  return states


def state_hashes(env: types.Env, rules_sha: str, states) -> list[str]:
  """Canonical hashes of env states [N] (seat 0's InfoState and labels).

  Args:
    env: the env.
    rules_sha: the rules' sha.
    states: env states [N].

  Returns:
    The hex hashes.
  """
  seat = jnp.zeros((jax.tree_util.tree_leaves(states)[0].shape[0],), jnp.int32)
  info = jax.vmap(env.info_state)(states, seat)
  labels = jax.vmap(env.labels)(states, seat)
  return deal_hashes(env.name, rules_sha, info, labels)


class DealIndex:
  """A game's evaluation deal hashes (module doc).

  Attributes:
    game: the game name.
    rules_sha: the rules' sha.
    hashes: the deal hashes (hex), a frozenset.
    sources: provenance records of the deals.
  """

  def __init__(
      self,
      game: str,
      rules_sha: str,
      hashes: Sequence[str],
      sources: Sequence[Mapping[str, Any]] = (),
  ):
    """Builds the index.

    Args:
      game: the game name.
      rules_sha: the rules' sha.
      hashes: the deal hashes (hex).
      sources: provenance records of the deals.
    """
    self.game, self.rules_sha = str(game), str(rules_sha)
    self.hashes = frozenset(str(h) for h in hashes)
    self.sources = tuple(dict(s) for s in sources)

  def to_json(self) -> dict[str, Any]:
    """The index as JSON types.

    Returns:
      The JSON dict.
    """
    return dict(
        format=INDEX_FORMAT,
        version=1,
        game=self.game,
        rules_sha=self.rules_sha,
        sources=list(self.sources),
        hashes=sorted(self.hashes),
    )

  @property
  def sha256(self) -> str:
    """The sha256 of the canonical JSON of the index."""
    return hashlib.sha256(
        json.dumps(self.to_json(), sort_keys=True).encode()
    ).hexdigest()

  def save(self, path: str) -> str:
    """Writes the index as JSON.

    Args:
      path: the file path.

    Returns:
      The index's sha256.
    """
    p = epath.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(self.to_json(), sort_keys=True, indent=1))
    return self.sha256

  @classmethod
  def load(cls, path: str) -> "DealIndex":
    """Reads an index written by save.

    Args:
      path: the file path.

    Returns:
      The DealIndex.

    Raises:
      ValueError: the file is not an index of version 1.
    """
    d = json.loads(epath.Path(path).read_text())
    if d.get("format") != INDEX_FORMAT or d.get("version") != 1:
      raise ValueError(f"{path}: not a {INDEX_FORMAT} of version 1")
    return cls(d["game"], d["rules_sha"], d["hashes"], d.get("sources", ()))

  @classmethod
  def from_labels(
      cls,
      game: str,
      rules_sha: str,
      labels: Mapping[str, Any],
      sources: Sequence[Mapping[str, Any]] = (),
      info: Any = None,
  ) -> "DealIndex":
    """The index of deals given as rows [N] of what the game's canon reads.

    E.g. bridge: {"deal_holder": int8[N, 52]}: for a game whose eval deals the
    engine that owns them supplies (a stub env has no dynamics to replay).

    Args:
      game: the game name.
      rules_sha: the rules' sha.
      labels: the deals' labels [N].
      sources: provenance records of the deals.
      info: the InfoState [N], when the canon reads it.

    Returns:
      The DealIndex.
    """
    return cls(
        game,
        rules_sha,
        sorted(set(deal_hashes(game, rules_sha, info, labels))),
        sources,
    )

  @classmethod
  def from_eval_sets(
      cls,
      env: types.Env,
      sets: Sequence[tuple[str, int, int]],
      rules_sha: Optional[str] = None,
  ) -> "DealIndex":
    """The index of every evaluation set, given as (name, seed, num_pairs).

    Args:
      env: the env.
      sets: (name, seed, num_pairs) per evaluation set.
      rules_sha: the rules' sha (default the env's).

    Returns:
      The DealIndex.
    """
    rs = rules_sha or env_registry.rules_sha(env)
    hashes, sources = set(), []
    for name, seed, n in sets:
      h = state_hashes(env, rs, eval_deal_states(env, int(seed), int(n)))
      hashes.update(h)
      sources.append(
          dict(
              name=str(name),
              seed=int(seed),
              num_pairs=int(n),
              deals=len(set(h)),
          )
      )
    return cls(env.name, rs, sorted(hashes), sources)

  def hits(self, hashes: Sequence[str]) -> np.ndarray:
    """Which hashes are in the index.

    Args:
      hashes: hex hashes.

    Returns:
      bool[N].
    """
    return np.asarray([h in self.hashes for h in hashes], bool)
