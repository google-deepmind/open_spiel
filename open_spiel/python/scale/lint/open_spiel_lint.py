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

"""Pylint checks of the package lint contract that pylint does not ship.

Load it with `load-plugins=open_spiel_lint` in the package .pylintrc, with
the directory that holds this file on PYTHONPATH. Two messages:

  bare-generic              `Callable`, `dict`, `list`, `tuple`, `set`, `type`,
                            `subprocess.CompletedProcess` (and `Popen`,
                            `re.Match`, `re.Pattern`) written without
                            parameters in an annotation or inside a type
                            alias's parameters. Write `Callable[..., Any]`,
                            `dict[str, Any]`,
                            `subprocess.CompletedProcess[str]`.
                            A type alias is `X: TypeAlias = ...`, `type X =
                            ...`, or a module- or class-level `X = G[...]`
                            with G a typing generic; a value such as
                            `isinstance(x, dict)` or `factory = dict` is not
                            checked.
  blanket-type-suppression  a `# type: ignore` or `# pyrefly: ignore` with
                            no `[code]`, or a `# pytype: disable` with no
                            `=name`. Name the error being suppressed.
"""

from __future__ import annotations

import re
import tokenize
import typing

import astroid
from astroid import nodes
from pylint.checkers import BaseChecker
from pylint.checkers import BaseTokenChecker
from pylint.checkers import utils

if typing.TYPE_CHECKING:
  from pylint.lint import PyLinter

# Generics that are bare when written as a plain name; the attribute names
# that are bare under any module (typing.Callable, collections.abc.Callable,
# subprocess.CompletedProcess); and dotted names bare as written.
BARE_NAMES = frozenset(
    ["Callable", "dict", "list", "tuple", "set", "type", "CompletedProcess"]
)
BARE_ATTRIBUTES = frozenset(["Callable", "CompletedProcess"])
BARE_DOTTED = frozenset(["subprocess.Popen", "re.Match", "re.Pattern"])
# Subscripted generics whose module- or class-level assignment is a type
# alias (`Runner = Callable[..., subprocess.CompletedProcess[str]]`).
ALIAS_GENERICS = frozenset([
    "Annotated",
    "Callable",
    "Dict",
    "FrozenSet",
    "Generator",
    "Iterable",
    "Iterator",
    "List",
    "Mapping",
    "MutableMapping",
    "Optional",
    "Sequence",
    "Set",
    "Tuple",
    "Type",
    "Union",
    "dict",
    "frozenset",
    "list",
    "set",
    "tuple",
    "type",
])
# Subscripts whose parameters are not all types: Literal's are values (a
# string there is not a forward reference), Annotated's after the first are
# metadata.
_LITERAL = frozenset(["Literal"])
_ANNOTATED = frozenset(["Annotated"])

_TYPE_IGNORE = re.compile(
    r"#\s*(?:type|pyrefly):\s*ignore(?P<codes>\[[^\]]*\])?"
)
_PYTYPE_DISABLE = re.compile(
    r"#\s*pytype:\s*disable(?P<codes>\s*=\s*[\w-]+(?:\s*,\s*[\w-]+)*)?"
)


def _subscript_name(node: nodes.NodeNG) -> str | None:
  """The bare name of a subscript's value (`Literal` for `typing.Literal`).

  Args:
    node: The value of a subscript.

  Returns:
    The name or last attribute name, or None for any other expression.
  """
  if isinstance(node, nodes.Name):
    return node.name
  if isinstance(node, nodes.Attribute):
    return node.attrname
  return None


def bare_generics(annotation: nodes.NodeNG) -> list[nodes.NodeNG]:
  """The nodes of an annotation that name a generic without parameters.

  A string (a forward reference) is parsed and its findings are reported on
  the string itself.

  Args:
    annotation: An annotation expression.

  Returns:
    Each bare generic, in source order.
  """
  if isinstance(annotation, nodes.Name):
    return [annotation] if annotation.name in BARE_NAMES else []
  if isinstance(annotation, nodes.Attribute):
    bare = (
        annotation.attrname in BARE_ATTRIBUTES
        or annotation.as_string() in BARE_DOTTED
    )
    return [annotation] if bare else []
  if isinstance(annotation, nodes.Const):
    if not isinstance(annotation.value, str):
      return []
    try:
      parsed = astroid.extract_node(annotation.value.strip())
    except (astroid.AstroidSyntaxError, ValueError):
      return []
    return [annotation] if bare_generics(parsed) else []
  if isinstance(annotation, nodes.Subscript):
    name = _subscript_name(annotation.value)
    if name in _LITERAL:
      return []
    params = annotation.slice
    if name in _ANNOTATED and isinstance(params, nodes.Tuple) and params.elts:
      return bare_generics(params.elts[0])
    return bare_generics(params)
  found = []
  for child in annotation.get_children():
    found += bare_generics(child)
  return found


class BareGenericChecker(BaseChecker):
  """Flags parameterless generics in annotations (bare-generic)."""

  name = "open-spiel-bare-generic"
  msgs = {
      "C9701": (
          "Generic '%s' used in an annotation without parameters",
          "bare-generic",
          (
              "Callable, dict, list, tuple, set, type,"
              " subprocess.CompletedProcess (and Popen, re.Match, re.Pattern)"
              " carry their parameters in an annotation or type alias:"
              " Callable[..., Any], dict[str, Any],"
              " subprocess.CompletedProcess[str]."
          ),
      ),
  }

  def _check(self, annotation: nodes.NodeNG | None) -> None:
    """Reports every bare generic in one annotation.

    Args:
      annotation: An annotation expression, or None when there is none.
    """
    if annotation is None:
      return
    for node in bare_generics(annotation):
      self.add_message("bare-generic", node=node, args=(node.as_string(),))

  @utils.only_required_for_messages("bare-generic")
  def visit_functiondef(self, node: nodes.FunctionDef) -> None:
    """Checks a function's parameter and return annotations.

    Args:
      node: The function definition.
    """
    args = node.args
    for annotation in (
        *args.posonlyargs_annotations,
        *args.annotations,
        args.varargannotation,
        *args.kwonlyargs_annotations,
        args.kwargannotation,
        node.returns,
    ):
      self._check(annotation)

  visit_asyncfunctiondef = visit_functiondef

  @utils.only_required_for_messages("bare-generic")
  def visit_annassign(self, node: nodes.AnnAssign) -> None:
    """Checks a variable annotation, and an explicit type alias's value.

    `x: dict = {}` is an annotation; `X: TypeAlias = dict[str, list]` an
    explicit type alias.

    Args:
      node: The annotated assignment.
    """
    self._check(node.annotation)
    if _subscript_name(node.annotation) == "TypeAlias":
      self._check(node.value)

  @utils.only_required_for_messages("bare-generic")
  def visit_assign(self, node: nodes.Assign) -> None:
    """Checks an implicit type alias's value.

    An implicit alias is `X = G[...]` at module or class level, with G a
    typing generic (ALIAS_GENERICS).

    Args:
      node: The assignment.
    """
    if not isinstance(node.parent, (nodes.Module, nodes.ClassDef)):
      return
    value = node.value
    if (
        isinstance(value, nodes.Subscript)
        and _subscript_name(value.value) in ALIAS_GENERICS
    ):
      self._check(value)

  @utils.only_required_for_messages("bare-generic")
  def visit_typealias(self, node: nodes.TypeAlias) -> None:
    """Checks a `type X = ...` statement's value.

    Args:
      node: The type alias statement.
    """
    self._check(node.value)


def blanket_suppressions(comment: str) -> list[str]:
  """The type-checker suppressions in a comment that name no error.

  Args:
    comment: One comment token, `#` included.

  Returns:
    The text of each blanket suppression, in order.
  """
  found = []
  for m in _TYPE_IGNORE.finditer(comment):
    codes = m.group("codes")
    if codes is None or not codes.strip("[] "):
      found.append(m.group(0))
  for m in _PYTYPE_DISABLE.finditer(comment):
    if m.group("codes") is None:
      found.append(m.group(0))
  return found


class BlanketSuppressionChecker(BaseTokenChecker):
  """Flags type suppressions without an error code."""

  name = "open-spiel-blanket-type-suppression"
  msgs = {
      "C9702": (
          "Type suppression '%s' names no error",
          "blanket-type-suppression",
          (
              "Write `# type: ignore[code]`, `# pyrefly: ignore[code]` or `#"
              " pytype: disable=name` so the suppression covers one error, not"
              " every error on the line."
          ),
      ),
  }

  def process_tokens(self, tokens: list[tokenize.TokenInfo]) -> None:
    """Checks every comment token of a module.

    Args:
      tokens: The module's tokens.
    """
    for tok in tokens:
      if tok.type != tokenize.COMMENT:
        continue
      for text in blanket_suppressions(tok.string):
        self.add_message(
            "blanket-type-suppression",
            line=tok.start[0],
            col_offset=tok.start[1],
            args=(text,),
        )


def register(linter: PyLinter) -> None:
  """Registers both checkers (pylint's plugin entry point).

  Args:
    linter: The running linter.
  """
  linter.register_checker(BareGenericChecker(linter))
  linter.register_checker(BlanketSuppressionChecker(linter))
