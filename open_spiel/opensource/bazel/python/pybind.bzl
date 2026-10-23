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

"""pybind11 rules for OpenSpiel BUILD files.

`pybind_library` and `pybind_library_test` are pybind11_bazel's. OpenSpiel's
`pybind_extension` differs from pybind11_bazel's: the target named `name` is a
`py_library` (not an alias of the shared object), so Python targets can list it
in `deps` and `import <name>` works, as it does in Google's own build.
"""

load(
    "@pybind11_bazel//:build_defs.bzl",
    "PYBIND_COPTS",
    "PYBIND_DEPS",
    "PYBIND_FEATURES",
    _pybind_library = "pybind_library",
    _pybind_library_test = "pybind_library_test",
)
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_python//python:py_library.bzl", "py_library")

pybind_library = _pybind_library
pybind_library_test = _pybind_library_test

def pybind_extension(
        name,
        copts = [],
        features = [],
        linkopts = [],
        deps = [],
        tags = [],
        visibility = None,
        testonly = None,
        **kwargs):
    """Builds the Python extension module `name` (Linux and macOS).

    Defines `<name>.so`, the extension's shared object, and `<name>`, a
    `py_library` that puts it on the Python path.

    Args:
      name: module name; Python code does `import <name>`.
      copts: compiler options.
      features: C++ toolchain features.
      linkopts: linker options.
      deps: C++ dependencies.
      tags: tags of both targets.
      visibility: visibility of both targets.
      testonly: whether both targets are testonly.
      **kwargs: further `cc_binary` attributes.
    """

    # GCC/Clang flags: Windows (MSVC, .pyd) needs its own branch once supported.
    cc_binary(
        name = name + ".so",
        copts = copts + PYBIND_COPTS + ["-fvisibility=hidden"],
        features = features + PYBIND_FEATURES,
        linkopts = linkopts + select({
            "@platforms//os:osx": ["-undefined", "dynamic_lookup"],
            "//conditions:default": ["-Wl,-Bsymbolic"],
        }),
        linkshared = True,
        deps = deps + PYBIND_DEPS,
        tags = tags,
        visibility = visibility,
        testonly = testonly,
        **kwargs
    )
    py_library(
        name = name,
        data = [":" + name + ".so"],
        imports = ["."],
        tags = tags,
        visibility = visibility,
        testonly = testonly,
    )
