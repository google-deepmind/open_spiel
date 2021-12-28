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

#ifndef OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_ALPHA_ZERO_H_
#define OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_ALPHA_ZERO_H_

#include "open_spiel/utils/thread.h"
#include "open_spiel/utils/json.h"

namespace open_spiel::algorithms {

struct AlphaZeroConfig {
  std::string game;
  std::string path;
  std::string graph_def;
  std::string nn_model;
  int nn_width;
  int nn_depth;
  std::string devices;

  double learning_rate;
  double weight_decay;
  int train_batch_size;
  int inference_batch_size;
  int inference_threads;
  int inference_cache;
  int replay_buffer_size;
  int replay_buffer_reuse;
  int checkpoint_freq;
  int evaluation_window;

  double uct_c;
  int max_simulations;
  double policy_alpha;
  double policy_epsilon;
  double temperature;
  double temperature_drop;
  double cutoff_probability;
  double cutoff_value;

  int actors;
  int evaluators;
  int eval_levels;
  int max_steps;

  json::Object ToJson() const {
    return json::Object({
        {"game", game},
        {"path", path},
        {"graph_def", graph_def},
        {"nn_model", nn_model},
        {"nn_width", nn_width},
        {"nn_depth", nn_depth},
        {"devices", devices},
        {"learning_rate", learning_rate},
        {"weight_decay", weight_decay},
        {"train_batch_size", train_batch_size},
        {"inference_batch_size", inference_batch_size},
        {"inference_threads", inference_threads},
        {"inference_cache", inference_cache},
        {"replay_buffer_size", replay_buffer_size},
        {"replay_buffer_reuse", replay_buffer_reuse},
        {"checkpoint_freq", checkpoint_freq},
        {"evaluation_window", evaluation_window},
        {"uct_c", uct_c},
        {"max_simulations", max_simulations},
        {"policy_alpha", policy_alpha},
        {"policy_epsilon", policy_epsilon},
        {"temperature", temperature},
        {"temperature_drop", temperature_drop},
        {"cutoff_probability", cutoff_probability},
        {"cutoff_value", cutoff_value},
        {"actors", actors},
        {"evaluators", evaluators},
        {"eval_levels", eval_levels},
        {"max_steps", max_steps},
    });
  }
};

bool AlphaZero(AlphaZeroConfig config, StopToken* stop);

}  // namespace open_spiel::algorithms

#endif  // OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_ALPHA_ZERO_H_
