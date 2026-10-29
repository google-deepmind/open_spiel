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

# Precompiled OR-Tools downloaded by install.sh into open_spiel/ortools.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "ortools",
    srcs = glob([
        "lib/libortools.*",
        "lib64/libortools.*",
    ]),
    hdrs = glob(["include/**"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

alias(
    name = "linear_solver_glop",
    actual = ":ortools",
    visibility = ["//visibility:public"],
)

alias(
    name = "linear_solver_lib",
    actual = ":ortools",
    visibility = ["//visibility:public"],
)
