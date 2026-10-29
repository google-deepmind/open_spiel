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

"""f64_sites / check_no_f64: every float64 value in a traced program.

The library runs with jax_enable_x64 on. The contract: int64 counters are fine;
the ONE decide (core.decide) computes in f64 by contract; everything else is
f32, or bf16 where a network configures it. One operand built without an
explicit dtype (a python float array, jnp.arange, jax.nn.one_hot,
jnp.zeros(...)) is 64-bit under x64 and promotes every op downstream of it to
f64, which is much slower than f32 or bf16 on most accelerators and keeps the
op off bf16 tensor cores. This module walks a ClosedJaxpr (including every
nested jaxpr: scan / while / cond / pjit / shard_map / custom_jvp / custom_vjp
bodies) and reports each equation that reads or writes a float64 value,
with its primitive, the user source line that traced it and its name stack.

decide runs under jax.named_scope(decide.scope()) (core.decide), so its
equations are recognized by name stack, not by a source-path guess. Anything
else that is f64 must be named in an explicit allow-list
({source substring: justification}); check_no_f64 raises on the rest.
"""

import contextlib
import functools
import os
import re
import sysconfig
from typing import Any, Iterator, Mapping, NamedTuple, Optional

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.testing.errors import require

# Source paths are reported relative to the folder that holds the package.
_PACKAGE_PARENT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
F64 = jnp.dtype(jnp.float64)


class _Frame(NamedTuple):
  file_name: str
  start_line: int
  function_name: str


# The files that are not user code (JAX's own rule for an equation's user
# frame): the jax package and the standard library.
_NOT_USER = (
    os.path.dirname(os.path.abspath(jax.__file__)) + os.sep,
    sysconfig.get_path("stdlib"),
    os.path.dirname(contextlib.__file__),
)


def _user_frame(traceback: Any) -> Optional[_Frame]:
  """The innermost frame of an equation's traceback outside jax and stdlib.

  A *_test.py file always counts as user code (JAX's rule), read from the
  traceback's public frames.

  Args:
    traceback: The equation's source traceback, or None.

  Returns:
    The frame, or None when every frame is in jax or the stdlib.
  """
  for frame in traceback.frames if traceback is not None else ():
    path = frame.file_name
    if path.endswith("_test.py") or not path.startswith(_NOT_USER):
      return _Frame(path, frame.line_num, frame.function_name)
  return None


@functools.cache
def _jaxpr_types() -> tuple[type[Any], type[Any]]:
  """(ClosedJaxpr, Jaxpr), the types jax.make_jaxpr returns.

  Returns:
    (ClosedJaxpr, Jaxpr).
  """
  closed = jax.make_jaxpr(lambda x: x)(0)
  return type(closed), type(closed.jaxpr)


class Site(NamedTuple):
  """One float64 site of a program.

  Attributes:
    primitive: The equation's primitive.
    where: The user frame that traced the equation, "file:line (fn)".
    scope: The name stack, outermost first, "/"-joined.
    dtypes: "in -> out" avals of the equation (dtype and shape).
  """

  primitive: str
  where: str  # the user frame that traced the equation, "file:line (fn)"
  scope: str  # the name stack, outermost first, "/"-joined
  dtypes: str  # "in -> out" avals of the equation (dtype and shape)

  @property
  def in_decide(self) -> bool:
    """Whether the site is inside decide's scope."""
    # A scope component is the scope name, or wrapped by a transform:
    # "vmap(policy_decide)", "jvp(policy_decide)".
    name = re.escape(decide_lib.scope())
    return any(
        re.fullmatch(rf"(\w+\()*{name}\)*", c) for c in self.scope.split("/")
    )

  def __str__(self) -> str:
    return (
        f"{self.primitive:<22s} {self.dtypes:<48s} {self.where}  [{self.scope}]"
    )


def _sub_jaxprs(eqn) -> Iterator[Any]:
  for v in eqn.params.values():
    for x in v if isinstance(v, (list, tuple)) else (v,):
      closed_jaxpr, jaxpr = _jaxpr_types()
      if isinstance(x, closed_jaxpr):
        yield x.jaxpr
      elif isinstance(x, jaxpr):
        yield x


def equations(jaxpr) -> Iterator[Any]:
  """Yields every equation of `jaxpr`, then of each nested jaxpr, depth first.

  Containers such as jit / scan / cond are included.

  Args:
    jaxpr: The jaxpr.

  Yields:
    The equations.
  """
  for eqn in jaxpr.eqns:
    yield eqn
    for sub in _sub_jaxprs(eqn):
      yield from equations(sub)


def _dtype(v):
  """The dtype of a jaxpr variable, if any.

  Args:
    v: A jaxpr variable.

  Returns:
    The dtype, or None.
  """
  return getattr(getattr(v, "aval", None), "dtype", None)


def _where(eqn, parent: str = "") -> str:
  """The user source location of an equation.

  Args:
    eqn: The equation.
    parent: The enclosing equation's location.

  Returns:
    path:line (function).
  """
  frame = _user_frame(eqn.source_info.traceback)
  if frame is None:  # traced inside jax itself: the enclosing equation's frame
    return f"{parent} (inside)" if parent else "<no user frame>"
  path = frame.file_name
  if path.startswith(_PACKAGE_PARENT + os.sep):
    path = path[len(_PACKAGE_PARENT) + 1 :]
  return f"{path}:{frame.start_line} ({frame.function_name})"


def _aval_str(v) -> str:
  """The short aval string of a variable.

  Args:
    v: A jaxpr variable.

  Returns:
    The string, or '' without a dtype.
  """
  aval = getattr(v, "aval", None)
  if aval is None or not hasattr(aval, "dtype"):
    return ""
  return aval.str_short(short_dtypes=True)


def _walk(jaxpr, prefix: str, parent: str) -> Iterator[Site]:
  """The f64 sites of `jaxpr` and its sub-jaxprs.

  Args:
    jaxpr: The jaxpr.
    prefix: The enclosing name scope.
    parent: The enclosing equation's location.

  Yields:
    The Sites.
  """
  for eqn in jaxpr.eqns:
    stack = str(eqn.source_info.name_stack)
    scope = "/".join(s for s in (prefix, stack) if s)
    where = _where(eqn, parent)
    ins = [_dtype(v) for v in eqn.invars]
    outs = [_dtype(v) for v in eqn.outvars]
    subs = list(_sub_jaxprs(eqn))
    # A container (jit / scan / cond / while ...) is reported by its body's
    # equations, not again itself.
    if not subs and any(d == F64 for d in ins + outs):

      def fmt(vs):
        """The avals of `vs`, comma-joined.

        Args:
          vs: Jaxpr variables.

        Returns:
          The string.
        """
        return ",".join(a for a in map(_aval_str, vs) if a)

      yield Site(
          eqn.primitive.name,
          where,
          scope,
          f"{fmt(eqn.invars)} -> {fmt(eqn.outvars)}",
      )
    for sub in subs:
      yield from _walk(sub, scope, where.replace(" (inside)", ""))


def f64_sites(closed) -> list[Site]:
  """Returns every equation of `closed` that reads or writes a float64 value.

  Nested jaxprs are included; f64 program inputs and constants come first as
  primitive "<input>" / "<const>".

  Args:
    closed: A ClosedJaxpr.

  Returns:
    The sites.
  """
  require(
      jax.config.read("jax_enable_x64"),
      "f64_sites must run with jax_enable_x64 on",
  )
  jaxpr = closed.jaxpr
  out = [
      Site("<input>", f"invar {i}", "", str(_dtype(v)))
      for i, v in enumerate(jaxpr.invars)
      if _dtype(v) == F64
  ]
  out += [
      Site("<const>", f"constvar {i}", "", str(_dtype(v)))
      for i, v in enumerate(jaxpr.constvars)
      if _dtype(v) == F64
  ]
  out += list(_walk(jaxpr, "", ""))
  return out


def outside(sites, allow: Mapping[str, str]) -> list[Site]:
  """Returns the sites neither inside decide nor matched by an allow-list entry.

  Every allow-list entry needs a justification.

  Args:
    sites: The sites.
    allow: {substring of Site.where: justification}.

  Returns:
    The sites outside.
  """
  for k, why in allow.items():
    require(
        isinstance(why, str) and len(why.strip()) >= 20,
        f"allow-list entry {k!r} needs a written justification",
    )
  return [
      s
      for s in sites
      if not s.in_decide and not any(k in s.where for k in allow)
  ]


def check_no_f64(
    closed, allow: Mapping[str, str] = None, what: str = "program"
) -> list[Site]:
  """Raises ContractError listing every float64 site outside decide and `allow`.

  Args:
    closed: A ClosedJaxpr.
    allow: {substring of Site.where: justification}.
    what: What the program is, for the message.

  Returns:
    All sites (decide's included), for logging.
  """
  sites = f64_sites(closed)
  bad = outside(sites, dict(allow or {}))
  require(
      not bad,
      f"{len(bad)} float64 value(s) in the {what} outside decide and the "
      "allow-list:\n  "
      + "\n  ".join(str(s) for s in bad[:60]),
  )
  return sites
