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

"""`py_test` for OpenSpiel BUILD files: rules_python's, with adapted attributes."""

load("@rules_python//python:py_test.bzl", _py_test = "py_test")
load(":py_common.bzl", "adapt_py_executable_kwargs")

# Optional C++ dependencies, each enabled by --//open_spiel:build_with_<dep>.
# Tests learn which are built from OPEN_SPIEL_BUILD_WITH_<DEP>=ON, the
# environment variables the CMake build sets.
_OPTIONAL_CC_DEPS = [
    "acpc",
    "gamut",
    "hanabi",
    "libnop",
    "libtorch",
    "ortools",
    "roshambo",
    "xinxin",
]

def _optional_cc_deps_env():
    """Returns a configurable `env` announcing the enabled optional C++ deps."""
    env = {}
    for dep in _OPTIONAL_CC_DEPS:
        env = env | select({
            Label("//open_spiel:build_with_{}_enabled".format(dep)): {
                "OPEN_SPIEL_BUILD_WITH_" + dep.upper(): "ON",
            },
            "//conditions:default": {},
        })
    return env

def py_test(**kwargs):
    """rules_python `py_test`; see `adapt_py_executable_kwargs`.

    Also sets OPEN_SPIEL_BUILD_WITH_<DEP>=ON for enabled optional C++ deps.

    Args:
      **kwargs: `py_test` attributes.
    """
    adapted = adapt_py_executable_kwargs(kwargs)
    adapted["env"] = adapted.get("env", {}) | _optional_cc_deps_env()
    _py_test(**adapted)
