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

# RoShamBo competition bots (https://github.com/jhtschultz/roshambo), exposed
# under the include prefix of the install.sh checkout in open_spiel/bots.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "rsb_bot",
    hdrs = ["BotClasses/rsb_bot.h"],
    include_prefix = "open_spiel/bots/roshambo/roshambo",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "roshambo",
    srcs = ["bot_map.cc"] + glob(["BotClasses/*.cc"]),
    hdrs = ["bot_map.h"] + glob(
        ["BotClasses/*.h"],
        exclude = ["BotClasses/rsb_bot.h"],
    ),
    copts = ["-w"],
    include_prefix = "open_spiel/bots/roshambo/roshambo",
    visibility = ["//visibility:public"],
    deps = [":rsb_bot"],
)
