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

#include "open_spiel/bots/human/human_bot.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

Action GetActionFromString(const State &state,
                           const std::string &action_string) {
  for (Action action : state.LegalActions(state.CurrentPlayer())) {
    if (action_string == state.ActionToString(action)) {
      return action;
    }
  }
  return kInvalidAction;
}

Action StepHumanBotWithInputs(HumanBot &human_bot,
                              const std::vector<std::string> &inputs,
                              const State &state) {
  // Add a newline character to each input.
  std::string human_bot_inputs = absl::StrJoin(inputs, "\n") + "\n";
  std::istringstream human_bot_input_stream(human_bot_inputs);

  // Allow the human bot to access the input through std::cin.
  std::cin.rdbuf(human_bot_input_stream.rdbuf());

  return human_bot.Step(state);
}

void EmptyActionTest() {
  HumanBot human_bot;
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  std::string empty_action_string = "";
  std::string legal_action_string = "0";

  // Have the human bot receive a empty action, then a legal action.
  Action human_bot_action = StepHumanBotWithInputs(
      human_bot, {empty_action_string, legal_action_string}, *state);
  Action legal_action = kInvalidAction;
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(legal_action_string, &legal_action));
  SPIEL_CHECK_TRUE(human_bot_action == legal_action);
}

void TerminalActionTest() {
  HumanBot human_bot;
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();

  // Apply actions to get a terminal state.
  state->ApplyAction(0);
  state->ApplyAction(3);
  state->ApplyAction(1);
  state->ApplyAction(4);
  state->ApplyAction(2);

  // Ensure the human bot handles the terminal state case before trying to parse
  // an action.
  Action human_bot_action = StepHumanBotWithInputs(human_bot, {""}, *state);

  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(human_bot_action == kInvalidAction);
}

void LegalStringActionTest() {
  HumanBot human_bot;
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  std::string legal_action_string = "x(0,0)";

  // Have the human bot receive a legal string action.
  Action human_bot_action =
      StepHumanBotWithInputs(human_bot, {legal_action_string}, *state);
  Action legal_action = GetActionFromString(*state, legal_action_string);

  SPIEL_CHECK_TRUE(human_bot_action == legal_action);
}

void LegalIntActionTest() {
  HumanBot human_bot;
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  std::string legal_action_string = "0";

  // Have the human bot receive a legal integer action.
  Action human_bot_action =
      StepHumanBotWithInputs(human_bot, {legal_action_string}, *state);
  Action legal_action = kInvalidAction;
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(legal_action_string, &legal_action));
  SPIEL_CHECK_TRUE(human_bot_action == legal_action);
}

void IllegalStringActionTest() {
  HumanBot human_bot;
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  std::string illegal_action_string = "illegal_action_string";
  std::string legal_action_string = "x(0,0)";

  // Have the human bot first receive an illegal string action, then a legal
  // string action.
  Action human_bot_action = StepHumanBotWithInputs(
      human_bot, {illegal_action_string, legal_action_string}, *state);
  Action legal_action = GetActionFromString(*state, legal_action_string);

  SPIEL_CHECK_TRUE(human_bot_action == legal_action);
}

void IllegalIntActionTest() {
  HumanBot human_bot;
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  std::string illegal_action_string = "12345";
  std::string legal_action_string = "0";

  // Have the human bot first receive an illegal integer action, then a legal
  // integer action.
  Action human_bot_action = StepHumanBotWithInputs(
      human_bot, {illegal_action_string, legal_action_string}, *state);
  Action legal_action = kInvalidAction;
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(legal_action_string, &legal_action));
  SPIEL_CHECK_TRUE(human_bot_action == legal_action);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::EmptyActionTest();
  open_spiel::TerminalActionTest();
  open_spiel::LegalStringActionTest();
  open_spiel::LegalIntActionTest();
  open_spiel::IllegalStringActionTest();
  open_spiel::IllegalIntActionTest();
}
