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

# Double-dummy solver (https://github.com/jblespiau/dds), exposed under the same
# include prefix as the install.sh checkout in open_spiel/games/bridge.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "dds",
    srcs = glob(["src/*.cpp"]),
    hdrs = glob([
        "include/*.h",
        "src/*.h",
    ]),
    copts = ["-w"],
    defines = ["DDS_NO_STATIC_INIT"],
    include_prefix = "open_spiel/games/bridge/double_dummy_solver",
    linkopts = select({
        "@platforms//os:linux": ["-lpthread"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
