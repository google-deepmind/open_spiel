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

#ifndef OPEN_SPIEL_GAMES_GAMUT_GAMUT_H_
#define OPEN_SPIEL_GAMES_GAMUT_GAMUT_H_

#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/matrix_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tensor_game.h"

namespace open_spiel {
namespace gamut {

// A wrapper class to the GAMUT (http://gamut.stanford.edu) game generator.
// See also, "Run the GAMUT: A Comprehensive Approach to Evaluating
// Game-Theoretic Algorithms." Eugene Nudelman, Jennifer Wortman,
// Kevin Leyton-Brown, Yoav Shoham. AAMAS-2004.
class GamutGenerator {
 public:
  // Create a game generator with the specified java executable and GAMUT jar
  // file. The seed is used for random file names (if 0, uses the current time).
  GamutGenerator(const std::string& java_path, const std::string& jar_path,
                 int tmpfile_seed = 0);

  // Create a game generator using the default path to java executable, defined
  // in gamut.cc. The seed is used for random file names (if 0, uses the
  // current time).
  GamutGenerator(const std::string& jar_path, int tmpfile_seed = 0);

  // Generate a game using GAMUT command-line arguments. Do not use -f nor
  // -output, as they are added to the command-line arguments inside this
  // function.
  std::shared_ptr<const Game> GenerateGame(const std::string& cmdline_args);
  std::shared_ptr<const Game> GenerateGame(
      const std::vector<std::string>& cmdline_args);

  // Same as above; returns a MatrixGame subtype for two-player games.
  std::shared_ptr<const matrix_game::MatrixGame> GenerateMatrixGame(
      const std::vector<std::string>& cmdline_args);

  // Same as above; returns a MatrixGame subtype for games with >= 2 players.
  std::shared_ptr<const tensor_game::TensorGame> GenerateTensorGame(
      const std::vector<std::string>& cmdline_args);

 private:
  std::string TmpFile();

  std::string java_path_;
  std::string jar_path_;
  absl::Mutex generation_mutex_;
  std::mt19937 rng_;
  std::string rand_string_;  // Random string used for temp file names.
};

}  // namespace gamut
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GAMUT_GAMUT_H_
