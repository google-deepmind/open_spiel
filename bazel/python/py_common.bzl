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

"""Attribute adaptation shared by the OpenSpiel Python rule shims.

The BUILD files are written for Google's own build; these helpers turn
their Python rule attributes into ones rules_python accepts.
"""

load(":py_deps_table.bzl", "MANUAL_PACKAGES", "OPTIONAL_DEP_GROUPS")

# Attributes of Google's Python rules that rules_python does not have.
# They only drive static checks that run at Google, so they are dropped.
_UNSUPPORTED_ATTRS = [
    "pytype_deps",
    "pytype_srcs",
    "strict_deps",
]

_PY_DEPS_PREFIXES = ["@py_deps//", "@@py_deps//"]

def _py_deps_packages(deps):
    """Returns the top-level @py_deps packages named in `deps`, in order."""
    packages = []

    # `deps` given as (or concatenated with) a select() cannot be inspected at
    # loading time, so @py_deps labels inside it get no group gate: list
    # optional Python packages outside of select().
    if type(deps) != type([]):
        return packages
    for dep in deps:
        for prefix in _PY_DEPS_PREFIXES:
            if dep.startswith(prefix):
                package = dep[len(prefix):].split(":")[0].split("/")[0]
                if package not in packages:
                    packages.append(package)
    return packages

def _group_enabled_constraint(group):
    """Returns a target_compatible_with value requiring optional `group`."""
    return select({
        Label("//bazel/python:{}_enabled".format(group)): [],
        "//conditions:default": [Label("@platforms//:incompatible")],
    })

def adapt_py_kwargs(kwargs):
    """Returns a copy of Python rule `kwargs` that rules_python accepts.

    Drops attributes rules_python does not support. A target depending directly on packages
    of optional dependency groups is made incompatible unless those groups
    are enabled (and tagged with the group names); one depending on
    MANUAL_PACKAGES is tagged "manual".

    Args:
      kwargs: attributes passed to an OpenSpiel Python rule macro.

    Returns:
      A new dict of attributes.
    """
    adapted = {k: v for k, v in kwargs.items() if k not in _UNSUPPORTED_ATTRS}
    packages = _py_deps_packages(adapted.get("deps", []))
    groups = []
    for package in packages:
        group = OPTIONAL_DEP_GROUPS.get(package)
        if group and group not in groups:
            groups.append(group)
    extra_tags = list(groups)
    if [p for p in packages if p in MANUAL_PACKAGES]:
        extra_tags.append("manual")
    if extra_tags:
        tags = list(adapted.get("tags", []))
        adapted["tags"] = tags + [tag for tag in extra_tags if tag not in tags]
    if groups:
        compatible_with = adapted.get("target_compatible_with", [])
        for group in groups:
            compatible_with = compatible_with + _group_enabled_constraint(group)
        adapted["target_compatible_with"] = compatible_with
    return adapted

def adapt_py_executable_kwargs(kwargs):
    """`adapt_py_kwargs` for py_binary and py_test.

    Also turns off implicit `__init__.py` creation in runfiles by default
    (`legacy_create_init = 0`): a generated empty `__init__.py` can be written
    through the runfiles symlink of a real one, emptying the source file. The
    package `__init__.py` files that matter are listed in `srcs`; other
    directories import as namespace packages.

    Args:
      kwargs: attributes passed to an OpenSpiel py_binary or py_test macro.

    Returns:
      A new dict of attributes.
    """
    adapted = adapt_py_kwargs(kwargs)
    adapted.setdefault("legacy_create_init", 0)
    return adapted
