# Python in the Bazel build

OpenSpiel's `BUILD.bazel` files are the source of truth for both the Bazel and
the CMake builds. This directory has what their Python targets need under
Bazel; CMake and `pip install .` do not use it.

| File | Purpose |
|---|---|
| `python.MODULE.bazel` | Included by `MODULE.bazel`: rules_python, the Python 3.12 toolchain, the `@open_spiel_pip` pip hub (`pypi` is taken by pybind11_abseil) and the `@py_deps` repository. |
| `requirements.in` | Every Python package used by a target, pinned like `requirements.txt` and `scripts/python_extra_deps.sh`. |
| `requirements_lock.txt` | Hash-pinned lock (Linux + macOS). Regenerate with `update_requirements_lock.sh` after editing `requirements.in`. |
| `py_deps_table.bzl` | `@py_deps//<package>[:<target>]` labels used by BUILD files, mapped to pip packages; optional dependency groups. |
| `py_deps.bzl` | Repository rule creating `@py_deps` from the table. |
| `py_library.bzl`, `py_binary.bzl`, `py_test.bzl`, `pytype.bzl` | Python rules used by BUILD files: rules_python's, minus attributes that only Google's own build knows. |
| `pybind.bzl` | pybind11_bazel's rules, with a `pybind_extension` that Python targets can depend on (`import pyspiel`). |

## Optional dependencies

Targets that need JAX, PyTorch, TensorFlow or the miscellaneous packages
(cvxpy, nashpy, matplotlib, ...) are skipped unless their group is enabled,
like the `OPEN_SPIEL_ENABLE_*` switches of the CMake build:

```shell
bazel test //open_spiel/python/...                                  # core
bazel test //open_spiel/python/... --//bazel/python:enable_jax      # + JAX
```

The groups are `jax`, `pytorch`, `python_misc` and `tensorflow`. Pip packages
are only downloaded for targets that are built.

## Adding a Python dependency

1.  Add the package to `requirements.in` and run `update_requirements_lock.sh`
    (needs [uv](https://docs.astral.sh/uv/)).
2.  Add its label to `PY_DEPS` in `py_deps_table.bzl` (and to
    `OPTIONAL_DEP_GROUPS` if it is optional).
3.  Depend on `@py_deps//<package>` in the BUILD file.
