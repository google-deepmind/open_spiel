// Copyright 2022 DeepMind Technologies Limited
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

// This generates strategically interesting instances of Colored Trails
// according to the criteria of Sec 5 of Jong et al', 2011, Metastrategies in
// the Colored Trails Game.
// https://www.ifaamas.org/Proceedings/aamas2011/papers/C4_R57.pdf

#include <numeric>
#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/games/bargaining.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/init.h"

ABSL_FLAG(int, seed, 0, "Seed to use");
ABSL_FLAG(int, num_instances, 1000, "Number of boards to generate.");
ABSL_FLAG(std::string, filename, "/tmp/instances.txt",
          "File to save boards to.");

namespace open_spiel {
namespace bargaining {
namespace {

// TODO(author5): the efficiency can be greatly improved :)
Instance GenerateInstance(std::mt19937* rng) {
  Instance instance;
  bool valid = false;
  while (!valid) {
    valid = true;
    for (int i = 0; i < kNumItemTypes; ++i) {
      instance.pool[i] = absl::Uniform<int>(*rng, 1, kPoolMaxNumItems + 1);
    }
    int num_pool_items =
        std::accumulate(instance.pool.begin(), instance.pool.end(), 0);
    if (!(num_pool_items >= kPoolMinNumItems &&
          num_pool_items <= kPoolMaxNumItems)) {
      valid = false;
      continue;
    }

    // total value to each user is 10
    // every item has nonzero value to at least one player
    // some items have nonzero value to both players
    bool exists_valuable_to_both = false;
    std::array<int, 2> total_values = {0, 0};
    for (int i = 0; i < kNumItemTypes && valid; ++i) {
      for (Player p : {0, 1}) {
        instance.values[p][i] =
            absl::Uniform<int>(*rng, 0, kTotalValueAllItems + 1);
      }

      if (instance.values[0][i] == 0 && instance.values[1][i] == 0) {
        valid = false;
        break;
      } else if (instance.values[0][i] > 0 && instance.values[1][i] > 0) {
        exists_valuable_to_both = true;
      }

      for (Player p : {0, 1}) {
        total_values[p] += instance.values[p][i] * instance.pool[i];
        if (total_values[p] > kTotalValueAllItems) {
          valid = false;
          break;
        }
      }
    }

    if (!valid) {
      continue;
    }

    if (!(total_values[0] == kTotalValueAllItems &&
          total_values[1] == kTotalValueAllItems && exists_valuable_to_both)) {
      valid = false;
    }
  }

  return instance;
}

void GenerateInstances(int num) {
  std::string filename = absl::GetFlag(FLAGS_filename);
  int seed = absl::GetFlag(FLAGS_seed);
  std::mt19937 rng(seed);

  std::cout << "Opening file: " << filename << std::endl;
  open_spiel::file::File outfile(filename, "w");
  for (int i = 0; i < num; ++i) {
    Instance instance = GenerateInstance(&rng);
    std::string instance_line = instance.ToString();
    instance_line.push_back('\n');
    std::cout << i << std::endl << instance_line;
    outfile.Write(instance_line);
  }
  std::cout << "Wrote to file: " << filename << std::endl;
}

}  // namespace
}  // namespace bargaining
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, false);
  absl::ParseCommandLine(argc, argv);
  open_spiel::bargaining::GenerateInstances(absl::GetFlag(FLAGS_num_instances));
}
