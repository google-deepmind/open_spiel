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

#ifndef OPEN_SPIEL_CONTRIB_TF_TRAJECTORIES_H_
#define OPEN_SPIEL_CONTRIB_TF_TRAJECTORIES_H_

#include <string>

#include "open_spiel/spiel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/public/session.h"

// Important note: TF_Trajectories is an unsupported part of OpenSpiel. It has
// not tested with CMake and public Tensorflow. We do not anticipate any
// problems, but cannot support it officially at this time. We might officially
// support it in the future, in which case it would be moved into the core part
// of the library.
//
// This is a class to generate a batch of trajectories entirely in C++ using
// Tensorflow policies:
// - The graph is created in Python and serialized into a file (using
//   tf.train.write_graph). See contrib/python/export_graph.py.
// - The graph is loaded in C++ and we use the TF C++ API to execute ops.
//
// This code has been adapted from the Travis Ebesu's blog post:
// https://tebesu.github.io/posts/Training-a-TensorFlow-graph-in-C++-API

namespace open_spiel {
namespace algorithms {

class TFBatchTrajectoryRecorder {
 public:
  TFBatchTrajectoryRecorder(const Game& game, const std::string& graph_filename,
                            int batch_size);

  // Reset all the games to their initial states and clears the terminal flags.
  // The random number generator is *not* reset.
  void Reset();

  // Record batch-size trajectories. Currently the data is not sent anywhere,
  // but this can be easily modified to fill one of the BatchedTrajectory
  // structures (see algorithms/trajectories.{h,cc}).
  void Record();

 protected:
  void ApplyActions();

  int batch_size_;
  std::vector<std::unique_ptr<State>> states_;

  // This is a vector<int> as subclasses access it from multiple threads, which
  // isn't possible with a vector<bool>, as vector<bool> is implemented as a
  // series of bytes.
  std::vector<int> terminal_flags_;
  tensorflow::Tensor tf_inputs_;
  tensorflow::Tensor tf_legal_mask_;

  void FillInputsAndMasks();
  void RunInference();
  void GetNextStatesTF();
  int num_terminals_;
  std::vector<tensorflow::Tensor> tf_outputs_;

 private:
  void ResetInitialStates();
  void SampleChance(int idx);
  void GetNextStatesUniform();

  void InitTF();

  std::shared_ptr<const Game> game_;
  std::string graph_filename_;

  std::mt19937 rng_;
  std::uniform_real_distribution<double> dist_;

  // Tensorflow variables
  int flat_input_size_;
  int num_actions_;
  tensorflow::Session* tf_session_ = nullptr;
  tensorflow::GraphDef graph_def_;
  tensorflow::SessionOptions tf_opts_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_CONTRIB_TF_TRAJECTORIES_H_
