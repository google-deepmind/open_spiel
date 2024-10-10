// Copyright 2024 DeepMind Technologies Limited
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

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "open_spiel/games/german_whist_foregame/german_whist_foregame.h"

int main() {
  std::vector<std::vector<uint32_t>> bin_coeffs =
      open_spiel::german_whist_foregame::BinCoeffs(
          2 * open_spiel::german_whist_foregame::kNumRanks);
  const uint32_t hard_threads =
      8;  // set this to take advantage of more cores on your machine//
  open_spiel::german_whist_foregame::vectorNa tablebase =
      open_spiel::german_whist_foregame::BuildTablebase(bin_coeffs,
                                                        hard_threads);
  std::random_device rd;
  int num_samples = 100;
  if (open_spiel::german_whist_foregame::TestTablebase(num_samples, rd(),
                                                       tablebase, bin_coeffs)) {
    std::cout << "Tablebase accurate" << std::endl;
  } else {
    std::cout << "Tablebase inaccurate" << std::endl;
  }
  std::cout << "Starting Saving Tablebase" << std::endl;
  open_spiel::german_whist_foregame::StoreTTable("TTable13.txt", tablebase);
  std::cout << "Finished Saving Tablebase" << std::endl;

  return 0;
}
