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

# Hanabi Learning Environment
# (https://github.com/google-deepmind/hanabi-learning-environment).

load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "hanabi",
    srcs = glob(["hanabi_learning_environment/hanabi_lib/*.cc"]),
    hdrs = glob(["hanabi_learning_environment/hanabi_lib/*.h"]),
    copts = ["-w"],
    strip_include_prefix = "hanabi_learning_environment",
    visibility = ["//visibility:public"],
)
