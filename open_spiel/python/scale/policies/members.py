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

"""Assembly of K same-architecture (policy, params) members into one slot.

A slot of K members is one policy object with params stacked [K, ...]. For a
policy whose params are values (every JAX policy) that is a stack of the K
trees. For a policy whose weights live outside its params (a host-weights
policy, whose params are handles into storage the object owns), a handle
means nothing on another object: stacking handles of separately loaded
policies under the first object would play the first object's weights K
times. Here every member's weights are first copied into the owner
(`adopt`), then the owner's handles are stacked. RoutedPolicy recurses part
by part.

  params = members.stack(owner, [(policy_a, params_a), (policy_b, params_b)])
  frozen = members.freeze(owner, live_params)  # keeps playing current weights
  members.release(owner, frozen)               # its storage may be reused

Host-weights policies implement three methods (HostWeightsPolicy):

  export_member(params) -> blob      the weights one member's params play;
  import_member(blob) -> params      a member of this object holding a copy
                                     of blob, in released storage if any;
  release_member(params) -> None     the member's storage is no longer used.

Every place that puts several loaded policies in one slot or pool goes
through this module, so a pool of host-weights policies holds at most its
live members' weights.

The module imports no array framework at import time: jax is imported when
a JAX stack is built. A routed policy is recognised by its declared
`routed_parts` (the names of its two parts, RoutedPolicy's ("base",
"override")), without importing its module.
"""

from typing import Any, Sequence

import numpy as np

from open_spiel.python.scale.core import types


def _arch(policy: types.Policy) -> tuple[str, dict[str, Any]]:
  """The policy's (arch name, arch kwargs).

  Args:
    policy: The policy.

  Returns:
    The pair.
  """
  return (policy.spec.arch_name, dict(policy.spec.arch_kwargs))


def _is_routed(policy: Any) -> bool:
  """Whether `policy` is a routed (base, override) policy.

  Args:
    policy: The policy.

  Returns:
    True for one.
  """
  return getattr(policy, "routed_parts", None) == ("base", "override")


def holds_host_weights(policy: types.Policy) -> bool:
  """Whether the policy's params are handles into weights the object holds.

  Args:
    policy: the policy.

  Returns:
    True when it has export_member / import_member / release_member.
  """
  return all(
      callable(getattr(policy, m, None))
      for m in ("export_member", "import_member", "release_member")
  )


def adopt(owner: types.Policy, policy: types.Policy, params: Any) -> Any:
  """One member's `params` of `policy` as params of `owner`.

  The same values for value params; for host weights, an owner member
  holding a copy of the member's weights (the params themselves when
  `policy` is `owner`).

  Args:
    owner: the policy whose params are wanted.
    policy: the policy `params` belong to.
    params: one member's params of `policy`.

  Returns:
    Params of `owner`.

  Raises:
    ValueError: owner and policy have different architectures.
  """
  if policy is owner:
    return params
  if _arch(owner) != _arch(policy):
    raise ValueError(
        f"adopt: owner is {owner.spec.arch_name} "
        f"{dict(owner.spec.arch_kwargs)}, "
        f"member is {policy.spec.arch_name} {dict(policy.spec.arch_kwargs)}"
    )
  if _is_routed(owner):
    return owner.make_params(
        adopt(owner.base, policy.base, params["base"]),
        adopt(owner.override, policy.override, params["override"]),
    )
  if holds_host_weights(owner):
    return owner.import_member(policy.export_member(params))
  return params


def freeze(owner: types.Policy, params: Any) -> Any:
  """Params of `owner` that keep playing what `params` plays now.

  They keep playing it however the member they name changes later: a copy for
  host weights, the params themselves for (immutable) values.

  Args:
    owner: the policy.
    params: its params.

  Returns:
    The frozen params.
  """
  if _is_routed(owner):
    return owner.make_params(
        freeze(owner.base, params["base"]),
        freeze(owner.override, params["override"]),
    )
  if holds_host_weights(owner):
    return owner.import_member(owner.export_member(params))
  return params


def release(owner: types.Policy, params: Any) -> None:
  """Frees the weights of the member(s) `params` name, which nobody plays.

  For value params there is nothing to free.

  Args:
    owner: the policy.
    params: its params.
  """
  if _is_routed(owner):
    release(owner.base, params["base"])
    release(owner.override, params["override"])
    return
  if holds_host_weights(owner):
    owner.release_member(params)


def _stack_tree(members: Sequence[Any], host: bool) -> Any:
  """The members' PyTrees stacked leafwise on a new leading axis.

  Args:
    members: the PyTrees, all of one structure.
    host: whether to stack as numpy (else jax) arrays.

  Returns:
    The stacked PyTree.
  """
  # Optional dependency: jax is imported here, not at the top, so the league's
  # control plane imports this module where jax cannot be imported (module
  # doc; league/tests/jax_free_control_test.py).
  import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  if host:
    return jax.tree_util.tree_map(
        lambda *xs: np.stack([np.asarray(x) for x in xs]), *members
    )
  # Optional dependency: jax, as above.
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return jax.tree_util.tree_map(
      lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *members
  )


def stack(
    owner: types.Policy, members: Sequence[tuple[types.Policy, Any]]
) -> Any:
  """The slot params of `owner` whose member k plays members[k].

  Args:
    owner: the slot's policy.
    members: (policy, one member's params) per member.

  Returns:
    Params stacked [K, ...] (jnp for a JAX owner, numpy for any other
    framework), or the list of K params for a HOST_DRIVEN owner (the form
    core.host_arena takes).

  Raises:
    ValueError: no members, or a member of another architecture.
  """
  if not members:
    raise ValueError("stack needs at least one member")
  adopted = [adopt(owner, pol, params) for pol, params in members]
  if owner.spec.capability == types.Capability.HOST_DRIVEN:
    return adopted
  return _stack_tree(adopted, host=owner.spec.framework != "jax")
