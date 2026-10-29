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

"""The vocabulary that search operators declare, and their reserved aux keys.

A search operator (policies.operators) is a policy whose outputs are training
targets. Its class declares who decides its action, how its targets depend on
the network it improves, and what it emits; testing.check_operator asserts
the declarations against these vocabularies. Importing this module registers
the operator aux keys with core.auxiliary and reserves the "search_" prefix,
so the Arena checks their dtype, shape and finiteness, and no learned aux head
may use the prefix.

  ov = operator_vocab
  ov.check_prior_dependence("puct_prior+value")
  aux = {ov.AUX_SEARCH_VALUE: v, ov.AUX_SEARCH_Q: q}   # f32[B], f32[B, A]

Values are from the searching seat's view. An operator never emits
core.types.AUX_TARGET_VALUE, which the Arena treats as a teacher's value.
"""

from typing import Mapping

from open_spiel.python.scale.core import auxiliary as aux
from open_spiel.python.scale.core import types

#: Who chooses the action (PolicySpec.decider; "forced" is an outcome of play,
#: recorded in artifact manifests, not a decider a policy declares).
DECIDERS = types.POLICY_DECIDERS

#: How an operator's targets depend on the network it improves. A value may
#: carry the suffix "+sampler_head" when the operator's belief sampler is a
#: head of the same network.
PRIOR_DEPENDENCE: Mapping[str, str] = {
    "puct_prior+value": (
        "PUCT tree search: the network's policy is the prior over children and "
        "its value scores the leaves (AlphaZero, MuZero and Gumbel MuZero)"
    ),
    "mmd_anchor+rollout": (
        "an MMD step anchored at the network's logits over rollout Q "
        "(pi ~ exp((logit + eta q) / (1 + eta tau)))"
    ),
    "rollout_only": (
        "actions scored by rollouts alone; the network does not enter the "
        "target"
    ),
    "topk_pruning": (
        "the network's policy prunes the candidate actions to its top k; the "
        "survivors are scored without it"
    ),
    "topk_pruning+rollout": (
        "the network's policy prunes the candidates to its top k and the "
        "survivors are scored by rollouts that play the network's policy"
    ),
    "tie_break_only": (
        "the network breaks ties between actions the search scores equal"
    ),
}
PRIOR_DEPENDENCE_SUFFIX = "+sampler_head"

#: An operator's value bias, computed from its configuration: "unbiased", or
#: "fused_upper_bound" when it searches worlds the acting seat cannot tell
#: apart, so its values overestimate (strategy fusion).
VALUE_BIASES = ("unbiased", "fused_upper_bound")

RESERVED_AUX_PREFIX = "search_"
AUX_SEARCH_VALUE = "search_value"  # f32[B]: root value
AUX_SEARCH_Q = "search_q"  # f32[B, A]: root Q per action, 0 where unvisited
AUX_SEARCH_Q_SE = "search_q_se"  # f32[B, A]: standard error of each Q
AUX_SEARCH_DECIDED = "search_decided"  # bool[B]: the search decided this row

#: key -> (dtype name, trailing shape after B).
AUX_SPECS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    types.AUX_TARGET_POLICY: ("float32", ("A",)),
    AUX_SEARCH_VALUE: ("float32", ()),
    AUX_SEARCH_Q: ("float32", ("A",)),
    AUX_SEARCH_Q_SE: ("float32", ("A",)),
    AUX_SEARCH_DECIDED: ("bool", ()),
}


def is_reserved_aux(name: str) -> bool:
  """Whether an aux or aux-term name lies in the reserved search namespace.

  Args:
    name: the name.

  Returns:
    bool.
  """
  return str(name).startswith(RESERVED_AUX_PREFIX)


def check_prior_dependence(v: str) -> str:
  """Validates a prior_dependence value.

  Args:
    v: a PRIOR_DEPENDENCE key, optionally with the +sampler_head suffix.

  Returns:
    `v`.

  Raises:
    ValueError: any other value.
  """
  base = (
      v[: -len(PRIOR_DEPENDENCE_SUFFIX)]
      if v.endswith(PRIOR_DEPENDENCE_SUFFIX)
      else v
  )
  if base not in PRIOR_DEPENDENCE:
    raise ValueError(
        f"prior_dependence {v!r} not in {sorted(PRIOR_DEPENDENCE)} "
        "(optionally + "
        f"{PRIOR_DEPENDENCE_SUFFIX!r})"
    )
  return v


for _name, (_dtype, _shape) in AUX_SPECS.items():
  aux.register_aux_key(
      _name,
      _shape,
      _dtype,
      writer="policy"
      if _name == types.AUX_TARGET_POLICY
      else "search operator",
  )
aux.reserve_prefix(RESERVED_AUX_PREFIX)
