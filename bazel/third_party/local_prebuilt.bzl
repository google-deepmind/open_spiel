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

"""Repository rule for optional precompiled dependencies (libtorch, OR-Tools).

Like `new_local_repository`, but if the directory does not exist (install.sh
was not run with the dependency enabled) the repository still exists and every
target in it is incompatible with all platforms. `bazel build //...` then skips
the targets that need the dependency instead of failing.
"""

_MISSING_BUILD = """\
# Generated: {path} does not exist. Run install.sh with the dependency enabled.
package(default_visibility = ["//visibility:public"])
"""

_MISSING_TARGET = """
filegroup(
    name = "{name}",
    target_compatible_with = ["@platforms//:incompatible"],
)
"""

def _local_prebuilt_repository_impl(rctx):
    # Relative to the OpenSpiel checkout, not to rctx.workspace_root: that is
    # the root module's directory when another project depends on OpenSpiel.
    # The checkout root is not a package, so go up from this file's package.
    open_spiel_root = rctx.path(Label("//bazel/third_party:local_prebuilt.bzl")).dirname.dirname.dirname
    path = open_spiel_root.get_child(rctx.attr.path)

    # Refetch when the directory appears or disappears (install.sh run later).
    rctx.watch(path)
    if path.exists:
        for child in path.readdir():
            rctx.symlink(child, child.basename)
        rctx.symlink(rctx.attr.build_file, "BUILD.bazel")
    else:
        content = _MISSING_BUILD.format(path = rctx.attr.path)
        for target in rctx.attr.targets:
            content += _MISSING_TARGET.format(name = target)
        rctx.file("BUILD.bazel", content)
    rctx.file("REPO.bazel", "")

local_prebuilt_repository = repository_rule(
    implementation = _local_prebuilt_repository_impl,
    attrs = {
        "build_file": attr.label(mandatory = True, allow_single_file = True),
        "path": attr.string(
            mandatory = True,
            doc = "Directory relative to the OpenSpiel checkout.",
        ),
        "targets": attr.string_list(
            mandatory = True,
            doc = "Targets of build_file, defined as incompatible stubs if path is missing.",
        ),
    },
    local = True,
)
