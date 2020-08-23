// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/gamut/gamut.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/games/nfg_game.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace gamut {
namespace {
constexpr const char* kDefaultJavaPath = "java";
}  // namespace

GamutGenerator::GamutGenerator(const std::string& jar_path)
    : java_path_(kDefaultJavaPath), jar_path_(jar_path) {}

GamutGenerator::GamutGenerator(const std::string& java_path,
                               const std::string& jar_path)
    : java_path_(java_path), jar_path_(jar_path) {}

std::shared_ptr<const Game> GamutGenerator::GenerateGame(
    const std::string& cmdline_args) {
  return GenerateGame(absl::StrSplit(cmdline_args, ' '));
}

std::shared_ptr<const Game> GamutGenerator::GenerateGame(
    const std::vector<std::string>& cmdline_args) {
  // Check that there's no -f and no -output in the command-line args. The get
  // added by this generator.
  for (const auto& arg : cmdline_args) {
    if (arg == "-f") {
      SpielFatalError("Do not use -f in the command-line arguments.");
    } else if (arg == "-output") {
      SpielFatalError("Do not use -output in the command-line arguments.");
    }
  }

  std::vector<std::string> arguments = cmdline_args;
  arguments.push_back("-output");
  arguments.push_back("GambitOutput");

  // Lock here to prevent concurrently writing / removal.
  std::shared_ptr<const Game> game;
  {
    absl::MutexLock lock(&generation_mutex_);
    // Get a temprary file and add it to the arguments.
    std::string tmp_filename = tmpnam(nullptr);
    arguments.push_back("-f");
    arguments.push_back(tmp_filename);
    std::string full_cmd = absl::StrCat(java_path_, " -jar ", jar_path_, " ",
                                        absl::StrJoin(arguments, " "));
    system(full_cmd.c_str());
    game = LoadGame("nfg_game", {{"filename", GameParameter(tmp_filename)}});
    remove(tmp_filename.c_str());
  }
  return game;
}

}  // namespace gamut
}  // namespace open_spiel
