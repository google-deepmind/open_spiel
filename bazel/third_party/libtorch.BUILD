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

# Precompiled libtorch (PyTorch C++ API) downloaded by install.sh into
# open_spiel/libtorch/libtorch.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "torch_headers",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.cuh",
    ]),
    includes = [
        "include",
        "include/torch/csrc/api/include",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtorch",
    srcs = glob([
        "lib/libc10.*",
        "lib/libtorch.*",
        "lib/libtorch_cpu.*",
    ]),
    visibility = ["//visibility:public"],
    deps = [":torch_headers"],
)
