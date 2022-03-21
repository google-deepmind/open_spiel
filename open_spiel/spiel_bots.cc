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

#include "open_spiel/spiel_bots.h"

#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

class UniformRandomBot : public Bot {
 public:
  UniformRandomBot(Player player_id, int seed)
      : player_id_(player_id), rng_(seed) {}
  ~UniformRandomBot() = default;

  void RestartAt(const State&) override {}
  Action Step(const State& state) override {
    return StepWithPolicy(state).second;
  }
  bool ProvidesPolicy() override { return true; }
  ActionsAndProbs GetPolicy(const State& state) override {
    ActionsAndProbs policy;
    auto legal_actions = state.LegalActions(player_id_);
    const int num_legal_actions = legal_actions.size();
    const double p = 1.0 / num_legal_actions;
    for (auto action : legal_actions) policy.emplace_back(action, p);
    return policy;
  }

  std::pair<ActionsAndProbs, Action> StepWithPolicy(
      const State& state) override {
    ActionsAndProbs policy = GetPolicy(state);
    const int num_legal_actions = policy.size();

    int selection =
        absl::uniform_int_distribution<int>(0, num_legal_actions - 1)(rng_);
    return std::make_pair(policy, policy[selection].first);
  }

 private:
  const Player player_id_;
  std::mt19937 rng_;
};

// A UniformRandomBot that keeps a copy of the state up to date. This exists
// primarily to verify that InformAction is called correctly by the run loop.
class StatefulRandomBot : public UniformRandomBot {
 public:
  StatefulRandomBot(const Game& game, Player player_id, int seed)
      : UniformRandomBot(player_id, seed), state_(game.NewInitialState()) {}

  void Restart() override { state_ = state_->GetGame()->NewInitialState(); }
  void RestartAt(const State& state) override { state_ = state.Clone(); }
  void InformAction(const State& state, Player player_id,
                    Action action) override {
    CheckStatesEqual(state, *state_);
    state_->ApplyAction(action);
  }
  ActionsAndProbs GetPolicy(const State& state) override {
    CheckStatesEqual(state, *state_);
    return UniformRandomBot::GetPolicy(*state_);
  }
  std::pair<ActionsAndProbs, Action> StepWithPolicy(
      const State& state) override {
    std::pair<ActionsAndProbs, Action> ret =
        UniformRandomBot::StepWithPolicy(*state_);
    state_->ApplyAction(ret.second);
    return ret;
  }

 private:
  void CheckStatesEqual(const State& state1, const State& state2) const {
    SPIEL_CHECK_EQ(state1.History(), state2.History());
    SPIEL_CHECK_EQ(state1.CurrentPlayer(), state2.CurrentPlayer());
    SPIEL_CHECK_EQ(state1.LegalActions(), state2.LegalActions());
    if (!state1.IsChanceNode()) {
      SPIEL_CHECK_EQ(state1.ObservationTensor(), state2.ObservationTensor());
    }
  }
  std::unique_ptr<State> state_;
};

class PolicyBot : public Bot {
 public:
  PolicyBot(int seed, std::shared_ptr<Policy> policy)
      : Bot(), rng_(seed), policy_(std::move(policy)) {}
  ~PolicyBot() = default;

  void RestartAt(const State&) override {}
  Action Step(const State& state) override {
    return StepWithPolicy(state).second;
  }
  bool ProvidesPolicy() override { return true; }
  ActionsAndProbs GetPolicy(const State& state) override {
    return policy_->GetStatePolicy(state);
  }

  std::pair<ActionsAndProbs, Action> StepWithPolicy(
      const State& state) override {
    ActionsAndProbs actions_and_probs = GetPolicy(state);
    return {actions_and_probs, SampleAction(actions_and_probs, rng_).first};
  }

 private:
  std::mt19937 rng_;
  std::shared_ptr<Policy> policy_;
};

class FixedActionPreferenceBot : public Bot {
 public:
  FixedActionPreferenceBot(Player player_id, const std::vector<Action>& actions)
      : Bot(), player_id_(player_id), actions_(actions) {}
  ~FixedActionPreferenceBot() = default;

  void RestartAt(const State&) override {}
  Action Step(const State& state) override {
    return StepWithPolicy(state).second;
  }
  bool ProvidesPolicy() override { return true; }
  ActionsAndProbs GetPolicy(const State& state) override {
    std::vector<Action> legal_actions = state.LegalActions(player_id_);
    std::unordered_set<Action> legal_actions_set =
        std::unordered_set<Action>(legal_actions.begin(), legal_actions.end());
    for (Action action : actions_) {
      if (legal_actions_set.count(action) == 1) {
        return {{action, 1.0}};
      }
    }
    SpielFatalError("No legal actions in action list.");
  }

  std::pair<ActionsAndProbs, Action> StepWithPolicy(
      const State& state) override {
    ActionsAndProbs actions_and_probs = GetPolicy(state);
    return {actions_and_probs, actions_and_probs[0].first};
  }

 private:
  const Player player_id_;
  std::vector<Action> actions_;
};

}  // namespace

// A uniform random bot, for test purposes.
std::unique_ptr<Bot> MakeUniformRandomBot(Player player_id, int seed) {
  return std::make_unique<UniformRandomBot>(player_id, seed);
}
namespace {
class UniformRandomBotFactory : public BotFactory {
 public:
  ~UniformRandomBotFactory() = default;

  bool CanPlayGame(const Game& game, Player player_id) const override {
    return true;
  }
  std::unique_ptr<Bot> Create(std::shared_ptr<const Game> game,
                              Player player_id,
                              const GameParameters& bot_params) const override {
    int seed = 0;
    if (IsParameterSpecified(bot_params, "seed")) {
      const GameParameter& seed_param = bot_params.at("seed");
      seed = seed_param.int_value();
    } else {
      absl::BitGen gen;
      seed = absl::Uniform<int>(gen, std::numeric_limits<int>::min(),
                                std::numeric_limits<int>::max());
    }
    return MakeUniformRandomBot(player_id, seed);
  }
};
REGISTER_SPIEL_BOT("uniform_random", UniformRandomBotFactory);
}  // namespace

// A bot that samples from a policy.
std::unique_ptr<Bot> MakePolicyBot(int seed, std::shared_ptr<Policy> policy) {
  return std::make_unique<PolicyBot>(seed, std::move(policy));
}
std::unique_ptr<Bot> MakePolicyBot(const Game& game, Player player_id, int seed,
                                   std::shared_ptr<Policy> policy) {
  return MakePolicyBot(seed, std::move(policy));
}
// A bot with a fixed action preference, for test purposes.
// Picks the first legal action found in the list of actions.
std::unique_ptr<Bot> MakeFixedActionPreferenceBot(
    Player player_id, const std::vector<Action>& actions) {
  return std::make_unique<FixedActionPreferenceBot>(player_id, actions);
}
namespace {
std::vector<Action> ActionsFromStr(const absl::string_view& str,
                                   const absl::string_view& delim) {
  std::vector<Action> actions;
  for (absl::string_view token : absl::StrSplit(str, delim)) {
    int v;
    SPIEL_CHECK_TRUE(absl::SimpleAtoi(token, &v));
    actions.push_back(v);
  }
  return actions;
}

class FixedActionPreferenceFactory : public BotFactory {
 public:
  ~FixedActionPreferenceFactory() = default;

  bool CanPlayGame(const Game& game, Player player_id) const override {
    return true;
  }
  std::unique_ptr<Bot> Create(std::shared_ptr<const Game> game,
                              Player player_id,
                              const GameParameters& bot_params) const override {
    std::vector<Action> actions{0, 1, 2, 3, 4, 5, 6, 7};
    if (IsParameterSpecified(bot_params, "actions")) {
      const GameParameter& actions_param = bot_params.at("actions");
      actions = ActionsFromStr(actions_param.string_value(), ":");
    }
    return MakeFixedActionPreferenceBot(player_id, actions);
  }
};
REGISTER_SPIEL_BOT("fixed_action_preference", FixedActionPreferenceFactory);
}  // namespace

std::unique_ptr<Bot> MakeStatefulRandomBot(const Game& game, Player player_id,
                                           int seed) {
  return std::make_unique<StatefulRandomBot>(game, player_id, seed);
}

BotRegisterer::BotRegisterer(const std::string& bot_name,
                             std::unique_ptr<BotFactory> factory) {
  RegisterBot(bot_name, std::move(factory));
}

std::unique_ptr<Bot> BotRegisterer::CreateByName(
    const std::string& bot_name, std::shared_ptr<const Game> game,
    Player player_id, const GameParameters& params) {
  auto iter = factories().find(bot_name);
  if (iter == factories().end()) {
    SpielFatalError(absl::StrCat("Unknown bot '", bot_name,
                                 "'. Available bots are:\n",
                                 absl::StrJoin(RegisteredBots(), "\n")));

  } else {
    const std::unique_ptr<BotFactory>& factory = iter->second;
    return factory->Create(std::move(game), player_id, params);
  }
}

std::vector<std::string> BotRegisterer::BotsThatCanPlayGame(const Game& game,
                                                            Player player_id) {
  std::vector<std::string> bot_names;
  for (const auto& key_val : factories()) {
    if (key_val.second->CanPlayGame(game, player_id)) {
      bot_names.push_back(key_val.first);
    }
  }
  return bot_names;
}

std::vector<std::string> BotRegisterer::BotsThatCanPlayGame(const Game& game) {
  std::vector<std::string> bot_names;
  for (const auto& key_val : factories()) {
    bool can_play_for_all = true;
    for (int player_id = 0; player_id < game.NumPlayers(); ++player_id) {
      if (!key_val.second->CanPlayGame(game, player_id)) {
        can_play_for_all = false;
        break;
      }
    }
    if (can_play_for_all) bot_names.push_back(key_val.first);
  }
  return bot_names;
}

void BotRegisterer::RegisterBot(const std::string& bot_name,
                                std::unique_ptr<BotFactory> factory) {
  factories()[bot_name] = std::move(factory);
}

std::vector<std::string> BotRegisterer::RegisteredBots() {
  std::vector<std::string> names;
  for (const auto& key_val : factories()) names.push_back(key_val.first);
  return names;
}

std::vector<std::string> RegisteredBots() {
  return BotRegisterer::RegisteredBots();
}

bool BotRegisterer::IsBotRegistered(const std::string& bot_name) {
  return factories().find(bot_name) != factories().end();
}

bool IsBotRegistered(const std::string& bot_name) {
  return BotRegisterer::IsBotRegistered(bot_name);
}

std::unique_ptr<Bot> LoadBot(const std::string& bot_name,
                             const std::shared_ptr<const Game>& game,
                             Player player_id) {
  GameParameters params = GameParametersFromString(bot_name);

  // We use the "name" parameter, as that is the "short_name", which is what we
  // want. Otherwise, this will use the "long name", which includes the config.
  // e.g. if the bot_name is "my_bot(parameter=value)", then we want the
  // bot_name here to be "my_bot", not "my_bot(parameter=value)".
  return LoadBot(params["name"].string_value(), game, player_id, params);
}

std::unique_ptr<Bot> LoadBot(const std::string& bot_name,
                             const std::shared_ptr<const Game>& game,
                             Player player_id, const GameParameters& params) {
  std::unique_ptr<Bot> result =
      BotRegisterer::CreateByName(bot_name, game, player_id, params);
  if (result == nullptr) {
    SpielFatalError(absl::StrCat("Unable to create bot: ", bot_name));
  }
  return result;
}

std::vector<std::string> BotsThatCanPlayGame(const Game& game,
                                             Player player_id) {
  return BotRegisterer::BotsThatCanPlayGame(game, player_id);
}

std::vector<std::string> BotsThatCanPlayGame(const Game& game) {
  return BotRegisterer::BotsThatCanPlayGame(game);
}

}  // namespace open_spiel
