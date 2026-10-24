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

"""Repository rule: an archive whose BUILD files get repository names rewritten.

Used for Bazel modules that OpenSpiel needs at a commit newer than their Bazel
Central Registry release. A `bazel_dep` plus `archive_override` would do that
only when OpenSpiel is the root module (Bazel ignores overrides elsewhere), so
the archive is fetched as a plain repository instead. Such a repository sees
OpenSpiel's dependencies under OpenSpiel's names, so labels the archive spells
differently (e.g. `@com_google_absl`) are rewritten.
"""

_BUILD_FILE_NAMES = ["BUILD", "BUILD.bazel"]
_MAX_DEPTH = 8

def _label_rewriting_archive_impl(rctx):
    rctx.download_and_extract(
        url = rctx.attr.urls,
        integrity = rctx.attr.integrity,
        stripPrefix = rctx.attr.strip_prefix,
    )
    dirs = [rctx.path(".")]
    for _ in range(_MAX_DEPTH):
        subdirs = []
        for directory in dirs:
            for child in directory.readdir():
                if child.is_dir:
                    subdirs.append(child)
                elif child.basename in _BUILD_FILE_NAMES:
                    content = rctx.read(child)
                    for before, after in rctx.attr.rewrites.items():
                        content = content.replace(before, after)
                    rctx.file(child, content)
        dirs = subdirs

label_rewriting_archive = repository_rule(
    implementation = _label_rewriting_archive_impl,
    attrs = {
        "integrity": attr.string(mandatory = True),
        "rewrites": attr.string_dict(
            mandatory = True,
            doc = "Substring replacements applied to every BUILD file.",
        ),
        "strip_prefix": attr.string(),
        "urls": attr.string_list(mandatory = True),
    },
)
