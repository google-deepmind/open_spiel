// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/bots/human/human_bot.h"

#include <iostream>
#include <sstream>
#include <string>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

Action GetActionFromString(const State& state,
                           const std::string& action_string) {
  for (Action action : state.LegalActions(state.CurrentPlayer())) {
    if (action_string == state.ActionToString(action)) {
      return action;
    }
  }
  return kInvalidAction;
}

void TerminalActionTest() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();

  // Apply actions to get a terminal state.
  state->ApplyAction(0);
  state->ApplyAction(3);
  state->ApplyAction(1);
  state->ApplyAction(4);
  state->ApplyAction(2);

  HumanBot human_bot;
  Action action = human_bot.Step(*state);

  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(action == kInvalidAction);
}

void LegalStringActionTest() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  std::string action_string = "x(0,0)";

  // Put action_string in cin stream so that the bot can receive it as input.
  std::istringstream action_string_stream(action_string);
  std::cin.rdbuf(action_string_stream.rdbuf());

  HumanBot human_bot;
  Action bot_action = human_bot.Step(*state);
  Action action = GetActionFromString(*state, action_string);

  SPIEL_CHECK_TRUE(bot_action == action);
}

void LegalIntActionTest() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  std::string action_string = "0";

  // Put action_string in cin stream so that the bot can receive it as input.
  std::istringstream action_string_stream(action_string);
  std::cin.rdbuf(action_string_stream.rdbuf());

  HumanBot human_bot;
  Action bot_action = human_bot.Step(*state);
  Action action = std::stoi(action_string);

  SPIEL_CHECK_TRUE(bot_action == action);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::TerminalActionTest();
  open_spiel::LegalStringActionTest();
  open_spiel::LegalIntActionTest();
}
