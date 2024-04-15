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
  int min_simulations;
  int max_simulations;
  double policy_alpha;
  double policy_epsilon;
  double temperature;
  double temperature_drop;
  double cutoff_probability;
  double cutoff_value;
  double td_lambda;
  int td_n_steps;

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
        {"min_simulations", min_simulations},
        {"max_simulations", max_simulations},
        {"policy_alpha", policy_alpha},
        {"policy_epsilon", policy_epsilon},
        {"temperature", temperature},
        {"temperature_drop", temperature_drop},
        {"cutoff_probability", cutoff_probability},
        {"cutoff_value", cutoff_value},
        {"td_lambda", td_lambda},
        {"td_n_steps", td_n_steps},
        {"actors", actors},
        {"evaluators", evaluators},
        {"eval_levels", eval_levels},
        {"max_steps", max_steps},
    });
  }

  void FromJsonWithDefaults(const json::Object& config_json,
                            const json::Object& defaults_json) {
    json::Object merged;
    merged.insert(config_json.begin(), config_json.end());
    merged.insert(defaults_json.begin(), defaults_json.end());
    game = merged.at("game").GetString();
    path = merged.at("path").GetString();
    graph_def = merged.at("graph_def").GetString();
    nn_model = merged.at("nn_model").GetString();
    nn_width = merged.at("nn_width").GetInt();
    nn_depth = merged.at("nn_depth").GetInt();
    devices = merged.at("devices").GetString();
    explicit_learning = merged.at("explicit_learning").GetBool();
    learning_rate = merged.at("learning_rate").GetDouble();
    weight_decay = merged.at("weight_decay").GetDouble();
    train_batch_size = merged.at("train_batch_size").GetInt();
    inference_batch_size = merged.at("inference_batch_size").GetInt();
    inference_threads = merged.at("inference_threads").GetInt();
    inference_cache = merged.at("inference_cache").GetInt();
    replay_buffer_size = merged.at("replay_buffer_size").GetInt();
    replay_buffer_reuse = merged.at("replay_buffer_reuse").GetInt();
    checkpoint_freq = merged.at("checkpoint_freq").GetInt();
    evaluation_window = merged.at("evaluation_window").GetInt();
    uct_c = merged.at("uct_c").GetDouble();
    min_simulations = merged.at("min_simulations").GetInt();
    max_simulations = merged.at("max_simulations").GetInt();
    policy_alpha = merged.at("policy_alpha").GetDouble();
    policy_epsilon = merged.at("policy_epsilon").GetDouble();
    temperature = merged.at("temperature").GetDouble();
    temperature_drop = merged.at("temperature_drop").GetDouble();
    cutoff_probability = merged.at("cutoff_probability").GetDouble();
    cutoff_value = merged.at("cutoff_value").GetDouble();
    td_lambda = merged.at("td_lambda").GetDouble();
    td_n_steps = merged.at("td_n_steps").GetInt();
    actors = merged.at("actors").GetInt();
    evaluators = merged.at("evaluators").GetInt();
    eval_levels = merged.at("eval_levels").GetInt();
    max_steps = merged.at("max_steps").GetInt();
  }
};

bool AlphaZero(AlphaZeroConfig config, StopToken* stop, bool resuming);

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_ALPHA_ZERO_H_
