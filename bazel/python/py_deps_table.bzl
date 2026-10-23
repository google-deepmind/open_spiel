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
`py_deps.bzl` creates the `@py_deps` repository from `PY_DEPS`: every entry
becomes a `py_library` forwarding to pip packages of the `@open_spiel_pip` hub,
whose pinned versions live in `requirements_lock.txt`.

`OPTIONAL_DEP_GROUPS` mirrors the OPEN_SPIEL_ENABLE_{JAX,PYTORCH,PYTHON_MISC}
switches of the CMake build: targets needing a disabled group are skipped.
"""

# `<package>[:<target>]` under @py_deps -> list of normalized pip package names.
PY_DEPS = {
    "absl:app": ["absl_py"],
    "absl/flags": ["absl_py"],
    "absl/logging": ["absl_py"],
    "absl/testing:absltest": ["absl_py"],
    "absl/testing:parameterized": ["absl_py"],
    "attr": ["attrs"],
    "chex": ["chex"],
    "clu/metric_writers": ["clu"],
    "cma": ["cma"],
    "cvxpy": ["cvxpy"],
    "distrax": ["distrax"],
    "etils/epath": ["etils", "typing_extensions", "zipp"],
    "flax/nnx:v1": ["flax"],
    "haiku": ["dm_haiku"],
    "jax": ["jax", "jaxlib"],
    "jax/example_libraries:optimizers": ["jax", "jaxlib"],
    "matplotlib": ["matplotlib"],
    "matplotlib:matplotlib_interactive": ["matplotlib"],
    "ml_collections": ["ml_collections"],
    "ml_collections/config_dict": ["ml_collections"],
    "mock": ["mock"],
    "nashpy": ["nashpy"],
    "networkx": ["networkx"],
    "numpy": ["numpy"],
    "optax": ["optax"],
    "orbax/checkpoint": ["orbax_checkpoint"],
    "pandas": ["pandas"],
    "pokerkit": ["pokerkit"],
    "pyarrow": ["pyarrow"],
    "pygraphviz": ["pygraphviz"],
    "rlax": ["rlax"],
    "scipy": ["scipy"],
    "tensorflow": ["tensorflow"],
    "torch": ["torch"],
    "torch:pytorch": ["torch"],
    "tree": ["dm_tree"],
}

# Top-level @py_deps package -> optional dependency group. Each group has a
# flag, --//bazel/python:enable_<group> (default off; see BUILD.bazel). A py target that depends
# directly on a package of a disabled group is incompatible, and so is every
# target depending on it: `bazel build/test //...` skips them.
OPTIONAL_DEP_GROUPS = {
    "chex": "jax",
    "clu": "jax",
    "distrax": "jax",
    "flax": "jax",
    "haiku": "jax",
    "jax": "jax",
    "optax": "jax",
    "orbax": "jax",
    "rlax": "jax",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
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
