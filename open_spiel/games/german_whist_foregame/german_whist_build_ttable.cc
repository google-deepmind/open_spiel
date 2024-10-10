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
