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

# ACPC poker server (https://github.com/jblespiau/project_acpc_server),
# exposed under the include prefix of the install.sh checkout in
# open_spiel/games/universal_poker.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "acpc",
    srcs = [
        "project_acpc_server/game.cc",
        "project_acpc_server/net.cc",
        "project_acpc_server/rng.cc",
    ],
    hdrs = [
        "project_acpc_server/evalHandTables",
        "project_acpc_server/game.h",
        "project_acpc_server/net.h",
        "project_acpc_server/rng.h",
    ],
    copts = [
        "-w",
        "-D__STDC_FORMAT_MACROS=1",
        # net.cc writes "%"PRIu16, which clang rejects in C++11 and later even
        # with -w (it is an error, not a warning, unless explicitly disabled).
        "-Wno-reserved-user-defined-literal",
    ],
    include_prefix = "open_spiel/games/universal_poker/acpc",
    visibility = ["//visibility:public"],
)
