# Bazel build

OpenSpiel's `BUILD.bazel` files are the source of truth for both the Bazel and
the CMake builds (`open_spiel/scripts/generate_cmake.sh` derives the CMake
files from them). CMake (`install.sh` + `cmake`) and `pip install .` do not
need Bazel.

| Directory | Purpose |
|---|---|
| `python/` | Python rules, the pinned pip requirements and the `@py_deps` labels; see [python/README.md](python/README.md). |
| `third_party/` | BUILD files and repository rules for C++ dependencies that do not ship Bazel files. |
| `tests/consumer/` | Example project depending on OpenSpiel, built by CI (job `bazel_consumer`). |

## Using OpenSpiel from another Bazel project

OpenSpiel is not in the Bazel Central Registry; depend on a commit with
`git_override` (or on a local checkout with `local_path_override`). Nothing
else from OpenSpiel's `MODULE.bazel` needs to be copied: OpenSpiel brings its
own C++ dependencies, Python toolchains and pip packages.

`MODULE.bazel`:

```starlark
bazel_dep(name = "open_spiel")
git_override(
    module_name = "open_spiel",
    commit = "<commit>",
    remote = "https://github.com/google-deepmind/open_spiel.git",
)
# Or, for a local checkout:
# local_path_override(module_name = "open_spiel", path = "/path/to/open_spiel")

bazel_dep(name = "rules_cc", version = "0.2.27")
bazel_dep(name = "rules_python", version = "2.4.4")
```

`.bazelrc` (OpenSpiel is C++20; Bazel's default C++ toolchain compiles C++17):

```
build --cxxopt=-std=c++20 --host_cxxopt=-std=c++20
```

`BUILD.bazel`:

```starlark
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_python//python:py_binary.bzl", "py_binary")

cc_binary(
    name = "my_tool",
    srcs = ["my_tool.cc"],  # #include "open_spiel/spiel.h"
    deps = [
        "@open_spiel//open_spiel",  # the core library
        "@open_spiel//open_spiel/games/tic_tac_toe",  # or open_spiel/games:all_games
    ],
)

py_binary(
    name = "my_script",
    srcs = ["my_script.py"],  # import pyspiel; from open_spiel.python import ...
    deps = [
        "@open_spiel//open_spiel/python/pybind11:pyspiel",
        "@open_spiel//open_spiel/python:rl_environment",
    ],
)
```

Notes:

*   Labels and include paths are the ones of OpenSpiel's own BUILD files
    (`@open_spiel//open_spiel/...`, `#include "open_spiel/..."`,
    `import pyspiel`, `from open_spiel.python import ...`). Depend on these
    targets rather than on the `open_spiel` pip package, which would bring a
    second copy of `pyspiel`.
*   The Python version is chosen by your root module (rules_python's default
    if it sets none). OpenSpiel's pip packages are set up for Python 3.12,
    3.13 and 3.14 (a few optional ones have no 3.14 wheels yet); OpenSpiel's
    own CI tests 3.12, which you get with
    `python.defaults(python_version = "3.12")` from the
    `@rules_python//python/extensions:python.bzl` extension.
*   Optional dependencies are off by default, as in OpenSpiel's own build. They
    are flags of the `open_spiel` repository, e.g.
    `--@open_spiel//open_spiel:build_with_hanabi` or
    `--@open_spiel//bazel/python:enable_jax`.
*   `git_override` and `local_path_override` are honored only in your root
    module. A module that depends on OpenSpiel and is itself used as a
    dependency needs its own root module to repeat the override.
