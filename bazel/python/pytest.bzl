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

"""`pytest_test` for OpenSpiel BUILD files: runs test files under pytest."""

load(":py_test.bzl", "rules_python_test")
load(":test_main.bzl", "test_main")

def pytest_test(name, tests = [], srcs = [], args = ["-s"], deps = [], deselect = [], **kwargs):
    """A test running pytest on `tests`, like `cd <package>; pytest <tests>`.

    The test fails if pytest fails, including when it collects no tests.

    Args:
      name: name of the test.
      tests: files containing the tests; all of the package if empty.
      srcs: other Python files the tests need.
      args: pytest command-line arguments.
      deps: other dependencies of the tests.
      deselect: test id prefixes, relative to the package, to leave out.
      **kwargs: other `py_test` attributes.
    """
    main = "_{}_pytest_main.py".format(name)
    test_main(
        name = main[:-len(".py")],
        out = main,
        tests = tests,
        template = Label(":pytest_main.py"),
        testonly = True,
        tags = ["manual"],
        visibility = ["//visibility:private"],
    )
    package = native.package_name()
    rules_python_test(
        name = name,
        srcs = [main] + tests + srcs,
        main = main,
        args = args + [
            "--deselect={}/{}".format(package, prefix)
            for prefix in deselect
        ],
        deps = deps + ["@py_deps//pytest"],
        **kwargs
    )
