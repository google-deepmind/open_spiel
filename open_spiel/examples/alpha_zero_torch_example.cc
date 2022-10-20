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

#include <signal.h>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/alpha_zero_torch/alpha_zero.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/thread.h"

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");
ABSL_FLAG(std::string, path, "/tmp/az", "Where to output the logs.");
ABSL_FLAG(std::string, graph_def, "",
          ("Where to get the graph. This could be from export_model.py, or "
           "from a checkpoint. If this is empty it'll create one."));
ABSL_FLAG(std::string, nn_model, "resnet",
          "Model torso type, can be resnet or mlp.");
ABSL_FLAG(int, nn_width, 128, "Width of the model, passed to export_model.py.");
ABSL_FLAG(int, nn_depth, 10, "Depth of the model, passed to export_model.py.");
ABSL_FLAG(double, uct_c, 2, "UCT exploration constant.");
ABSL_FLAG(double, temperature, 1,
          "Temperature for final move selection for early moves in training.");
ABSL_FLAG(double, temperature_drop, 10,  // Smaller than AZ due to short games.
          "Drop the temperature to 0 after this many moves.");
ABSL_FLAG(double, cutoff_probability, 0.8,
          ("Cut off rollouts early when above the cutoff value with this "
           "probability."));
ABSL_FLAG(double, cutoff_value, 0.95,
          "Cut off rollouts early when above this value.");
ABSL_FLAG(double, learning_rate, 0.0001, "Learning rate.");
ABSL_FLAG(double, weight_decay, 0.0001, "Weight decay.");
ABSL_FLAG(double, policy_alpha, 1, "What dirichlet noise alpha to use.");
ABSL_FLAG(double, policy_epsilon, 0.25, "What dirichlet noise epsilon to use.");
ABSL_FLAG(int, replay_buffer_size, 1 << 16,
          "How many states to store in the replay buffer.");
ABSL_FLAG(double, replay_buffer_reuse, 3,
          "How many times to reuse each state in the replay buffer.");
ABSL_FLAG(int, checkpoint_freq, 100, "Save a checkpoint every N steps.");
ABSL_FLAG(int, min_simulations, 0, "How many simulations to run (min).");
ABSL_FLAG(int, max_simulations, 300, "How many simulations to run (max).");
ABSL_FLAG(int, train_batch_size, 1 << 10,
          "How many states to learn from per batch.");
ABSL_FLAG(int, inference_batch_size, 1,
          "How many threads to wait for for inference.");
ABSL_FLAG(int, inference_threads, 0, "How many threads to run inference.");
ABSL_FLAG(int, inference_cache, 1 << 18,
          "Whether to cache the results from inference.");
ABSL_FLAG(std::string, devices, "/cpu:0",
          "Comma separated list of devices. The first device listed is used "
          "also as the learner. Allowable device names: cpu, cuda:0, cuda:1, "
          "cuda:2, cuda:3, ... Where cuda:n implies the n'th GPU resource.");
ABSL_FLAG(bool, explicit_learning, false,
          "The way the first device handles learning, either 'false' "
          "(while learning, the first device also takes on inference "
          "requests), or 'true' (while learning, the first device does "
          "not take on inference requests) which can only be used when "
          "multiple devices are available).");
ABSL_FLAG(bool, verbose, false, "Show the MCTS stats of possible moves.");
ABSL_FLAG(int, actors, 4, "How many actors to run.");
ABSL_FLAG(int, evaluators, 2, "How many evaluators to run.");
ABSL_FLAG(int, eval_levels, 7,
          ("Play evaluation games vs MCTS+Solver, with max_simulations*10^(n/2)"
           " simulations for n in range(eval_levels). Default of 7 means "
           "running mcts with up to 1000 times more simulations."));
ABSL_FLAG(int, max_steps, 0, "How many learn steps to run.");
ABSL_FLAG(int, evaluation_window, 100,
          "Number of games to average results over.");
ABSL_FLAG(double, td_lambda, 1.0,
          "Exponential base for weighting future returns of a trajectory.");
ABSL_FLAG(int, td_n_steps, 0,
          "When greater than zero, the maximum number of steps forward to"
          "consider when calculating td(lambda).");

open_spiel::StopToken stop_token;

void signal_handler(int s) {
  if (stop_token.StopRequested()) {
    exit(1);
  } else {
    stop_token.Stop();
  }
}

void signal_installer() {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = signal_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, nullptr);
}

int main(int argc, char** argv) {
  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);
  signal_installer();

  bool resuming;
  open_spiel::algorithms::torch_az::AlphaZeroConfig config;

  // Initialize config with defaults
  config.game = absl::GetFlag(FLAGS_game);
  config.path = absl::GetFlag(FLAGS_path);
  config.graph_def = absl::GetFlag(FLAGS_graph_def);
  config.nn_model = absl::GetFlag(FLAGS_nn_model);
  config.nn_width = absl::GetFlag(FLAGS_nn_width);
  config.nn_depth = absl::GetFlag(FLAGS_nn_depth);
  config.devices = absl::GetFlag(FLAGS_devices);
  config.explicit_learning = absl::GetFlag(FLAGS_explicit_learning);
  config.learning_rate = absl::GetFlag(FLAGS_learning_rate);
  config.weight_decay = absl::GetFlag(FLAGS_weight_decay);
  config.train_batch_size = absl::GetFlag(FLAGS_train_batch_size);
  config.inference_batch_size = absl::GetFlag(FLAGS_inference_batch_size);
  config.inference_threads = absl::GetFlag(FLAGS_inference_threads);
  config.inference_cache = absl::GetFlag(FLAGS_inference_cache);
  config.replay_buffer_size = absl::GetFlag(FLAGS_replay_buffer_size);
  config.replay_buffer_reuse = absl::GetFlag(FLAGS_replay_buffer_reuse);
  config.checkpoint_freq = absl::GetFlag(FLAGS_checkpoint_freq);
  config.evaluation_window = absl::GetFlag(FLAGS_evaluation_window);
  config.uct_c = absl::GetFlag(FLAGS_uct_c);
  config.min_simulations = absl::GetFlag(FLAGS_min_simulations);
  config.max_simulations = absl::GetFlag(FLAGS_max_simulations);
  config.policy_alpha = absl::GetFlag(FLAGS_policy_alpha);
  config.policy_epsilon = absl::GetFlag(FLAGS_policy_epsilon);
  config.temperature = absl::GetFlag(FLAGS_temperature);
  config.temperature_drop = absl::GetFlag(FLAGS_temperature_drop);
  config.cutoff_probability = absl::GetFlag(FLAGS_cutoff_probability);
  config.cutoff_value = absl::GetFlag(FLAGS_cutoff_value);
  config.td_lambda = absl::GetFlag(FLAGS_td_lambda);
  config.td_n_steps = absl::GetFlag(FLAGS_td_n_steps);
  config.actors = absl::GetFlag(FLAGS_actors);
  config.evaluators = absl::GetFlag(FLAGS_evaluators);
  config.eval_levels = absl::GetFlag(FLAGS_eval_levels);
  config.max_steps = absl::GetFlag(FLAGS_max_steps);

  if (positional_args.size() > 1) {
    // Resume training from a checkpoint: overwrite config defaults from file.
    resuming = true;

    if (positional_args.size() > 2) {
      open_spiel::SpielFatalError(
          "Specify only a path to a config.json to resume training.");
    }

    open_spiel::file::File config_file(positional_args[1], "r");
    std::string config_string = config_file.ReadContents();
    open_spiel::json::Object config_json = open_spiel::json::FromString(
        config_string).value().GetObject();
    config.FromJsonWithDefaults(config_json, config.ToJson());
  } else {
    // Start training from scratch.
    resuming = false;
  }

  return !AlphaZero(config, &stop_token, resuming);
}
