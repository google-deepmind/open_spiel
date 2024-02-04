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

#include "open_spiel/algorithms/alpha_zero_torch/alpha_zero.h"

#include <memory>
#include <utility>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {
namespace {

AlphaZeroConfig MakeConfig() {
  AlphaZeroConfig config;
  config.game = "game";
  config.path = "path";
  config.graph_def = "graph_def";
  config.nn_model = "nn_model";
  config.nn_width = 1;
  config.nn_depth = 2;
  config.devices = "devices";
  config.explicit_learning = true;
  config.learning_rate = 3.0;
  config.weight_decay = 4.0;
  config.train_batch_size = 5;
  config.inference_batch_size = 6;
  config.inference_threads = 7;
  config.inference_cache = 8;
  config.replay_buffer_size = 9;
  config.replay_buffer_reuse = 10;
  config.checkpoint_freq = 11;
  config.evaluation_window = 12;
  config.uct_c = 13.0;
  config.min_simulations = 14;
  config.max_simulations = 15;
  config.policy_alpha = 16.0;
  config.policy_epsilon = 17.0;
  config.temperature = 18.0;
  config.temperature_drop = 19.0;
  config.cutoff_probability = 20.0;
  config.cutoff_value = 21.0;
  config.td_lambda = 22.0;
  config.td_n_steps = 23;
  config.actors = 24;
  config.evaluators = 25;
  config.eval_levels = 26;
  config.max_steps = 27;
  return config;
}

void AlphaZeroTest_Config_ToJson() {
  AlphaZeroConfig config = MakeConfig();
  json::Object expected;
  expected.emplace("game", "game");
  expected.emplace("path", "path");
  expected.emplace("graph_def", "graph_def");
  expected.emplace("nn_model", "nn_model");
  expected.emplace("nn_width", 1);
  expected.emplace("nn_depth", 2);
  expected.emplace("devices", "devices");
  expected.emplace("explicit_learning", true);
  expected.emplace("learning_rate", 3.0);
  expected.emplace("weight_decay", 4.0);
  expected.emplace("train_batch_size", 5);
  expected.emplace("inference_batch_size", 6);
  expected.emplace("inference_threads", 7);
  expected.emplace("inference_cache", 8);
  expected.emplace("replay_buffer_size", 9);
  expected.emplace("replay_buffer_reuse", 10);
  expected.emplace("checkpoint_freq", 11);
  expected.emplace("evaluation_window", 12);
  expected.emplace("uct_c", 13.0);
  expected.emplace("min_simulations", 14);
  expected.emplace("max_simulations", 15);
  expected.emplace("policy_alpha", 16.0);
  expected.emplace("policy_epsilon", 17.0);
  expected.emplace("temperature", 18.0);
  expected.emplace("temperature_drop", 19.0);
  expected.emplace("cutoff_probability", 20.0);
  expected.emplace("cutoff_value", 21.0);
  expected.emplace("td_lambda", 22.0);
  expected.emplace("td_n_steps", 23);
  expected.emplace("actors", 24);
  expected.emplace("evaluators", 25);
  expected.emplace("eval_levels", 26);
  expected.emplace("max_steps", 27);

  std::string config_str = json::ToString(config.ToJson());
  std::string expected_str = json::ToString(expected);
  std::cout << "Config: " << config_str << std::endl
      << "Expected: " << expected_str << std::endl;
  SPIEL_CHECK_TRUE(config_str == expected_str);
}

void AlphaZeroTest_Config_FromJsonWithDefaults() {
  AlphaZeroConfig defaults = MakeConfig();
  {
    // When settings are empty, all values come from defaults.
    AlphaZeroConfig config;
    json::Object settings;
    config.FromJsonWithDefaults(settings, defaults.ToJson());
    AlphaZeroConfig expected = defaults;

    std::string config_str = json::ToString(config.ToJson());
    std::string expected_str = json::ToString(expected.ToJson());
    std::cout << "Config: " << config_str << std::endl
        << "Expected: " << expected_str << std::endl;
    SPIEL_CHECK_TRUE(config_str == expected_str);
  }
  {
    // Values contained in settings persist, others get populated from defaults.
    AlphaZeroConfig config;
    json::Object settings;
    settings.emplace("game", "another_game");
    config.FromJsonWithDefaults(settings, defaults.ToJson());
    AlphaZeroConfig expected = defaults;
    expected.game = "another_game";

    std::string config_str = json::ToString(config.ToJson());
    std::string expected_str = json::ToString(expected.ToJson());
    std::cout << "Config: " << config_str << std::endl
        << "Expected: " << expected_str << std::endl;
    SPIEL_CHECK_TRUE(config_str == expected_str);
  }
}

}  // namespace
}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::torch_az::AlphaZeroTest_Config_ToJson();
  open_spiel::algorithms::torch_az::AlphaZeroTest_Config_FromJsonWithDefaults();
}
