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

"""Generates the main file of a Python test from a template.

The template gets two substitutions, both paths relative to the runfiles
directory of the main repository (the working directory of a test):
  * `%main_file%`: the generated file itself;
  * `%test_files%`: space-separated test files: the files of `tests`, else the
    file of `srcs` named `main` (picked like rules_python picks the main file
    of a py_test), else the package directory.
"""

def _path_endswith(path, suffix):
    """Whether `path` ends with the path components of `suffix`."""
    return ("/" + path).endswith("/" + suffix)

def _test_main_impl(ctx):
    package_dir = ctx.outputs.out.short_path.rpartition("/")[0]
    if ctx.files.tests:
        test_files = [f.short_path for f in ctx.files.tests]
    elif ctx.attr.main:
        test_files = [
            f.short_path
            for f in ctx.files.srcs
            if _path_endswith(f.short_path, ctx.attr.main)
        ]
        if len(test_files) != 1:
            fail("{}: expected one file '{}' in srcs, found {}".format(
                ctx.label,
                ctx.attr.main,
                test_files,
            ))
    else:
        test_files = [package_dir]
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        substitutions = {
            "%main_file%": ctx.outputs.out.short_path,
            "%test_files%": " ".join(test_files),
        },
    )

test_main = rule(
    implementation = _test_main_impl,
    doc = "Writes the main .py file of a Python test from `template`.",
    attrs = {
        "main": attr.string(
            doc = "Name of the test file in `srcs`, used if `tests` is empty.",
        ),
        "out": attr.output(
            doc = "The main file to write.",
            mandatory = True,
        ),
        "srcs": attr.label_list(
            doc = "Sources of the test, among which `main` is looked up.",
            allow_files = True,
        ),
        "template": attr.label(
            doc = "Template of the main file.",
            allow_single_file = [".py"],
            mandatory = True,
        ),
        "tests": attr.label_list(
            doc = "Test files, if given explicitly.",
            allow_files = [".py"],
        ),
    },
)
