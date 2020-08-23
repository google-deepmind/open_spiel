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

#ifndef OPEN_SPIEL_GAMES_GAMUT_GAMUT_H_
#define OPEN_SPIEL_GAMES_GAMUT_GAMUT_H_

#include <cstring>
#include <memory>
#include <string>

#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace gamut {

// A wrapper class to the GAMUT (http://gamut.stanford.edu) game generator.
// See also, "Run the GAMUT: A Comprehensive Approach to Evaluating
// Game-Theoretic Algorithms." Eugene Nudelman, Jennifer Wortman,
// Kevin Leyton-Brown, Yoav Shoham. AAMAS-2004.
class GamutGenerator {
 public:
  // Create a game generator with the specified java executable and GAMUT jar
  // file.
  GamutGenerator(const std::string& java_path, const std::string& jar_path);

  // Create a game generator using the default path to java executable, defined
  // in gamut.cc
  GamutGenerator(const std::string& jar_path);

  // Generate a game using GAMUT command-line arguments. Do not use -f nor
  // -output, as they are added to the command-line arguments inside this
  // function.
  std::shared_ptr<const Game> GenerateGame(const std::string& cmdline_args);
  std::shared_ptr<const Game> GenerateGame(
      const std::vector<std::string>& cmdline_args);

 private:
  std::string java_path_;
  std::string jar_path_;
  absl::Mutex generation_mutex_;
};

}  // namespace gamut
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GAMUT_GAMUT_H_
