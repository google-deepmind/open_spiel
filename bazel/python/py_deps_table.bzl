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

"""Table of the Python packages OpenSpiel BUILD files depend on.

BUILD files name Python dependencies with labels of the form
`@py_deps//<package>[:<target>]` (for example `@py_deps//absl/testing:absltest`).
`py_deps.bzl` creates the `@py_deps` repository: every such label found in
OpenSpiel's BUILD files becomes a `py_library` forwarding to pip packages of
the `@open_spiel_pip` hub, whose pinned versions live in `requirements_lock.txt`.

A label is resolved, in order, by:
  1. its exact entry in `PY_DEPS`;
  2. its top-level package (`jax` in `@py_deps//jax/experimental`) in
     `LOCAL_PY_DEPS` or `PY_PACKAGES`;
  3. the pip package named like its top-level package, if that package is in
     `requirements_lock.txt`.
So most labels need no entry here, only a pin in `requirements.in`. A label
that resolves to nothing fails the build of its dependents with a message
saying what to add.

`OPTIONAL_DEP_GROUPS` mirrors the OPEN_SPIEL_ENABLE_{JAX,PYTORCH,PYTHON_MISC}
switches of the CMake build: targets needing a disabled group are skipped.
"""

# Labels `<package>[:<target>]` under @py_deps -> normalized pip package names,
# where the top-level package rule below does not fit. Labels listed here are
# always generated, also when no BUILD file uses them (e.g. ones added by the
# rule macros of this directory).
PY_DEPS = {
    "etils/epath": ["etils", "typing_extensions", "zipp"],
    "pytest": ["pytest"],
}

# Top-level @py_deps package -> normalized pip package names, for packages
# whose import name differs from their pip name (or that need several).
PY_PACKAGES = {
    "absl": ["absl_py"],
    "attr": ["attrs"],
    "haiku": ["dm_haiku"],
    "jax": ["jax", "jaxlib"],
    "orbax": ["orbax_checkpoint"],
    "tree": ["dm_tree"],
    "yaml": ["pyyaml"],
}

# Top-level @py_deps package -> OpenSpiel target providing it (relative to the
# OpenSpiel repository), for modules OpenSpiel builds itself.
LOCAL_PY_DEPS = {
    "pyspiel": "//open_spiel/python/pybind11:pyspiel",
}

# Top-level @py_deps package -> optional dependency group. Each group has a
# flag, --//bazel/python:enable_<group> (default off; see BUILD.bazel). A py target that depends
# directly on a package of a disabled group is incompatible, and so is every
# target depending on it: `bazel build/test //...` skips them.
OPTIONAL_DEP_GROUPS = {
    "chex": "jax",
    "distrax": "jax",
    "flax": "jax",
    "haiku": "jax",
    "jax": "jax",
    "mctx": "jax",
    "optax": "jax",
    "orbax": "jax",
    "rlax": "jax",
    "torch": "pytorch",
    "cma": "python_misc",
    "cvxpy": "python_misc",
    "matplotlib": "python_misc",
    "nashpy": "python_misc",
    "networkx": "python_misc",
    "pandas": "python_misc",
    "pyarrow": "python_misc",
    "pygraphviz": "python_misc",
}

# Top-level @py_deps packages whose dependents are tagged "manual", i.e. built
# only on request: pygraphviz ships no wheels and needs the Graphviz headers.
MANUAL_PACKAGES = ["pygraphviz"]
