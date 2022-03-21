// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/gamut/gamut.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/games/nfg_game.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace gamut {
namespace {
constexpr const char* kDefaultJavaPath = "java";
constexpr const int kNumTmpfileRetries = 16;
constexpr const int kNumRandomChars = 32;
constexpr const char* kAlphaChars =
    "abcdefghijklmnopqrstuvwxyxABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
}  // namespace

GamutGenerator::GamutGenerator(const std::string& jar_path, int tmpfile_seed)
    : GamutGenerator(kDefaultJavaPath, jar_path, tmpfile_seed) {}

GamutGenerator::GamutGenerator(const std::string& java_path,
                               const std::string& jar_path, int tmpfile_seed)
    : java_path_(java_path),
      jar_path_(jar_path),
      rng_(tmpfile_seed == 0 ? time(nullptr) : tmpfile_seed),
      rand_string_(kAlphaChars) {}

std::shared_ptr<const Game> GamutGenerator::GenerateGame(
    const std::string& cmdline_args) {
  return GenerateGame(absl::StrSplit(cmdline_args, ' '));
}

std::string GamutGenerator::TmpFile() {
  for (int retries = 0; retries < kNumTmpfileRetries; ++retries) {
    // Try random files until we find one that does not exist.
    absl::c_shuffle(rand_string_, rng_);
    std::string candidate =
        absl::StrCat(file::GetTmpDir(), "/gamut_tmpgame_",
                     rand_string_.substr(0, kNumRandomChars));
    if (!file::Exists(candidate)) {
      return candidate;
    }
  }

  SpielFatalError(absl::StrCat("Could not get a temporary file after ",
                               kNumTmpfileRetries, " tries."));
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
    // Get a temporary file and add it to the arguments.
    std::string tmp_filename = TmpFile();
    arguments.push_back("-f");
    arguments.push_back(tmp_filename);
    std::string full_cmd = absl::StrCat(java_path_, " -jar ", jar_path_, " ",
                                        absl::StrJoin(arguments, " "));
    system(full_cmd.c_str());
    game = LoadGame("nfg_game", {{"filename", GameParameter(tmp_filename)}});
    file::Remove(tmp_filename);
  }
  return game;
}

std::shared_ptr<const matrix_game::MatrixGame>
GamutGenerator::GenerateMatrixGame(
    const std::vector<std::string>& cmdline_args) {
  return std::dynamic_pointer_cast<const matrix_game::MatrixGame>(
      GenerateGame(cmdline_args));
}

std::shared_ptr<const tensor_game::TensorGame>
GamutGenerator::GenerateTensorGame(
    const std::vector<std::string>& cmdline_args) {
  return std::dynamic_pointer_cast<const tensor_game::TensorGame>(
      GenerateGame(cmdline_args));
}

}  // namespace gamut
}  // namespace open_spiel
