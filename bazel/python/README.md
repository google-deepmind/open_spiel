# Python in the Bazel build

OpenSpiel's `BUILD.bazel` files are the source of truth for both the Bazel and
the CMake builds. This directory has what their Python targets need under
Bazel; CMake and `pip install .` do not use it.

| File | Purpose |
|---|---|
| `python.MODULE.bazel` | Included by `MODULE.bazel`: rules_python, the Python 3.12 toolchain, the `@open_spiel_pip` pip hub (`pypi` is taken by pybind11_abseil) and the `@py_deps` repository. |
| `requirements.in` | Every Python package used by a target, pinned like `requirements.txt` and `scripts/python_extra_deps.sh`. |
| `requirements_lock.txt` | Hash-pinned lock (Linux + macOS). Regenerate with `update_requirements_lock.sh` after editing `requirements.in`. |
| `py_deps_table.bzl` | How `@py_deps//<package>[:<target>]` labels map to pip packages where the names differ; optional dependency groups. |
| `py_deps.bzl` | Repository rule creating `@py_deps` for the labels the BUILD files use. |
| `py_library.bzl`, `py_binary.bzl`, `py_test.bzl`, `pytype.bzl` | Python rules used by BUILD files: rules_python's, minus attributes that only Google's own build knows. |
| `py_test_main.py`, `test_main.bzl` | Generated main of every `py_test`: runs the test file and fails the test if it defines tests but runs none (e.g. pytest-style tests without `absltest.main()`). |
| `pytest.bzl`, `pytest_main.py` | `pytest_test`: runs test files under pytest; collecting no tests fails the test. |
| `pybind.bzl` | pybind11_bazel's rules, with a `pybind_extension` that Python targets can depend on (`import pyspiel`). |

## Optional dependencies

Targets that need JAX, PyTorch or the miscellaneous packages
(cvxpy, nashpy, matplotlib, ...) are skipped unless their group is enabled,
like the `OPEN_SPIEL_ENABLE_*` switches of the CMake build:

```shell
bazel test //open_spiel/python/...                                  # core
bazel test //open_spiel/python/... --//bazel/python:enable_jax      # + JAX
```

The groups are `jax`, `pytorch` and `python_misc`. Pip packages
are only downloaded for targets that are built.

## Adding a Python dependency

1.  Add the package to `requirements.in` and run `update_requirements_lock.sh`
    (needs [uv](https://docs.astral.sh/uv/)). Pins also in
    `scripts/python_extra_deps.sh` must agree with it:
    `python3 open_spiel/scripts/bazel_to_cmake/check_python_pins.py` checks.
2.  Depend on `@py_deps//<package>[:<target>]` in the BUILD file. Any label
    whose top-level package is also the pip package's name (e.g.
    `@py_deps//jax/experimental` needs no entry beyond `jax`'s, `@py_deps//mctx`
    just works once `mctx` is locked) resolves to that pip package. If the
    names differ (`yaml` is `pyyaml`), add the top-level package to
    `PY_PACKAGES` in `py_deps_table.bzl`.
3.  If the package is optional, add its top-level package to
    `OPTIONAL_DEP_GROUPS` there and to `OPTIONAL_PYTHON_GROUPS` in
    `open_spiel/scripts/bazel_to_cmake/label_map.py`.
