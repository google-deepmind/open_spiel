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

#ifndef OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_ALPHA_ZERO_H_
#define OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_ALPHA_ZERO_H_

#include <iostream>
#include <string>
#include <vector>

#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/thread.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

struct AlphaZeroConfig {
  std::string game;
  std::string path;
  std::string graph_def;
  std::string nn_model;
  int nn_width;
  int nn_depth;
  std::string devices;

  bool explicit_learning;
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
        {"explicit_learning", explicit_learning},
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

  void FromJson(const json::Object& config_json) {
    game = config_json.at("game").GetString();
    path = config_json.at("path").GetString();
    graph_def = config_json.at("graph_def").GetString();
    nn_model = config_json.at("nn_model").GetString();
    nn_width = config_json.at("nn_width").GetInt();
    nn_depth = config_json.at("nn_depth").GetInt();
    devices = config_json.at("devices").GetString();
    explicit_learning = config_json.at("explicit_learning").GetBool();
    learning_rate = config_json.at("learning_rate").GetDouble();
    weight_decay = config_json.at("weight_decay").GetDouble();
    train_batch_size = config_json.at("train_batch_size").GetInt();
    inference_batch_size = config_json.at("inference_batch_size").GetInt();
    inference_threads = config_json.at("inference_threads").GetInt();
    inference_cache = config_json.at("inference_cache").GetInt();
    replay_buffer_size = config_json.at("replay_buffer_size").GetInt();
    replay_buffer_reuse = config_json.at("replay_buffer_reuse").GetInt();
    checkpoint_freq = config_json.at("checkpoint_freq").GetInt();
    evaluation_window = config_json.at("evaluation_window").GetInt();
    uct_c = config_json.at("uct_c").GetDouble();
    max_simulations = config_json.at("max_simulations").GetInt();
    policy_alpha = config_json.at("policy_alpha").GetDouble();
    policy_epsilon = config_json.at("policy_epsilon").GetDouble();
    temperature = config_json.at("temperature").GetDouble();
    temperature_drop = config_json.at("temperature_drop").GetDouble();
    cutoff_probability = config_json.at("cutoff_probability").GetDouble();
    cutoff_value = config_json.at("cutoff_value").GetDouble();
    actors = config_json.at("actors").GetInt();
    evaluators = config_json.at("evaluators").GetInt();
    eval_levels = config_json.at("eval_levels").GetInt();
    max_steps = config_json.at("max_steps").GetInt();
  }
};

bool AlphaZero(AlphaZeroConfig config, StopToken* stop, bool resuming);

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_ALPHA_ZERO_H_
