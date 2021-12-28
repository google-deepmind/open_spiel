// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/contrib/tf_trajectories.h"

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

void SimpleTFTrajectoryExample(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  TFBatchTrajectoryRecorder recorder(*game, "/tmp/graph.pb", 1024);
  recorder.Record();
}

void DoubleRecordTFTrajectoryExample(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  TFBatchTrajectoryRecorder recorder(*game, "/tmp/graph.pb", 1024);
  recorder.Record();
  recorder.Record();
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) {
  // Batch size 32:
  //   32 games with uniform policy (no tensorflow): 5 ms
  //   32 games with TF policy: 180 ms  (~178 episodes / sec)
  // Batch size 1024:
  //   1024 games with TF policy: 1.24 sec (~832 episodes / sec)
  algorithms::SimpleTFTrajectoryExample("breakthrough");
  algorithms::DoubleRecordTFTrajectoryExample("breakthrough");
}
