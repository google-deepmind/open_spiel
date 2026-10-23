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

"""`py_test` for OpenSpiel BUILD files: rules_python's, with adapted attributes.

The test file runs through a generated main (from `py_test_main.py`) that fails
the test if the file defines tests but never runs them, e.g. pytest-style
`test_*` functions without `absltest.main()`; use `pytest.bzl` for those.
"""

load("@rules_python//python:py_test.bzl", _py_test = "py_test")
load(":py_common.bzl", "adapt_py_executable_kwargs")
load(":test_main.bzl", "test_main")

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

def rules_python_test(**kwargs):
    """rules_python `py_test`; see `adapt_py_executable_kwargs`.

    Also sets OPEN_SPIEL_BUILD_WITH_<DEP>=ON for enabled optional C++ deps. For
    the shims of this directory; BUILD files use `py_test` or `pytest_test`.

    Args:
      **kwargs: `py_test` attributes.
    """
    adapted = adapt_py_executable_kwargs(kwargs)
    adapted["env"] = adapted.get("env", {}) | _optional_cc_deps_env()
    _py_test(**adapted)

def _main_name(main):
    """Returns the file name of a `main` label given relative to the package."""
    return main.split(":")[-1]

def py_test(name, srcs = [], main = None, **kwargs):
    """`rules_python_test` that fails if the test file runs no tests.

    Args:
      name: name of the test.
      srcs: Python sources; the test file is `main`, by default `<name>.py`.
      main: the test file, if not `<name>.py`.
      **kwargs: other `py_test` attributes.
    """
    checked_main = "_{}_checked_main.py".format(name)
    test_main(
        name = checked_main[:-len(".py")],
        out = checked_main,
        main = _main_name(main) if main else name + ".py",
        srcs = srcs,
        template = Label(":py_test_main.py"),
        testonly = True,
        tags = ["manual"],
        visibility = ["//visibility:private"],
    )
    rules_python_test(
        name = name,
        srcs = srcs + [checked_main],
        main = checked_main,
        **kwargs
    )
