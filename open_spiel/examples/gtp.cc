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

#include<functional>
#include <map>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");

std::string Success() { return "=\n\n"; }
std::string Success(const std::string& s) {
  return absl::StrCat("= ", s, "\n\n");
}
std::string Failure(const std::string& s) {
  return absl::StrCat("? ", s, "\n\n");
}

std::unique_ptr<open_spiel::Bot> MakeBot(
    const open_spiel::Game& game,
    std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator) {
  return std::make_unique<open_spiel::algorithms::MCTSBot>(
      game, std::move(evaluator), /*uct_c=*/2, /*max_simulations=*/1000,
      /*max_memory_mb=*/0, /*solve=*/true, /*seed=*/0, /*verbose=*/false);
}

// Implements the Go Text Protocol, GTP, which is a text based protocol for
// communication with computer Go programs
// (https://www.lysator.liu.se/~gunnar/gtp/). This offers the open_spiel games
// and the mcts bot as a command line gtp server, which can be played against
// third party programs, or used on the command line directly.
int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string game_name = absl::GetFlag(FLAGS_game);
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_name);

  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  auto evaluator =
      std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
      /*n_rollouts=*/1, /*seed=*/0);
  std::unique_ptr<open_spiel::Bot> bot = MakeBot(*game, evaluator);

  using Args = std::vector<std::string>;
  std::map<std::string, std::function<std::string(const Args&)>> cmds = {
    {"name", [](const Args&) { return Success("open_spiel"); }},
    {"version", [](const Args&) { return Success("unknown"); }},
    {"protocol_version", [](const Args&) { return Success("2"); }},
    {"quit", [](const Args&) { return Success(); }},
    {"list_commands", [&cmds](const Args& args) {
      std::vector<std::string> keys;
      keys.reserve(cmds.size());
      for (auto const& item : cmds) {
        keys.push_back(item.first);
      }
      return Success(absl::StrJoin(keys, " "));
    }},
    {"known_command", [&cmds](const Args& args) {
      if (args.empty()) {
        return Failure("Not enough args");
      }
      return Success(cmds.find(args[0]) == cmds.end() ? "false" : "true");
    }},
    {"known_games", [](const Args& args) {
      return Success(absl::StrJoin(open_spiel::RegisteredGames(), " "));
    }},
    {"game", [&bot, &game, &state, &evaluator](const Args& args) {
      if (args.empty()) {
        return Success(game->ToString());
      }
      game = open_spiel::LoadGame(args[0]);
      state = game->NewInitialState();
      bot = MakeBot(*game, evaluator);
      return Success(game->ToString());
    }},
    {"boardsize", [&bot, &game, &state, &evaluator](const Args& args) {
      open_spiel::GameParameters params = game->GetParameters();
      if (params.find("board_size") == params.end()) {
        return Failure("Game doesn't support setting the board size");
      }
      if (args.empty()) {
        return Success(params["board_size"].ToString());
      }
      int board_size;
      if (!absl::SimpleAtoi(args[0], &board_size)) {
        return Failure("Failed to parse first arg as an int");
      }
      params["board_size"] = open_spiel::GameParameter(board_size);
      game = open_spiel::LoadGame(game->GetType().short_name, params);
      state = game->NewInitialState();
      bot = MakeBot(*game, evaluator);
      return Success();
    }},
    {"play", [&bot, &state](const Args& args) {
      if (args.size() < 2) {
        return Failure("Not enough args");
      }
      // Ignore player arg, assume it's always the current player.
      const std::string& action_str = args[1];
      for (const open_spiel::Action action : state->LegalActions()) {
        if (action_str == state->ActionToString(action)) {
          bot->InformAction(*state, state->CurrentPlayer(), action);
          state->ApplyAction(action);
          return Success();
        }
      }
      return Failure("Invalid action");
    }},
    {"genmove", [&bot, &state](const Args& args) {
      if (state->IsTerminal()) {
        return Failure("Game is already over");
      }
      // Ignore player arg, assume it's always the current player.
      open_spiel::Action action = bot->Step(*state);
      std::string action_str = state->ActionToString(action);
      state->ApplyAction(action);
      return Success(action_str);
    }},
    {"clear_board", [&bot, &game, &state](const Args& args) {
      state = game->NewInitialState();
      bot->Restart();
      return Success();
    }},
    {"undo", [&bot, &game, &state](const Args& args) {
      std::vector<open_spiel::Action> history = state->History();
      int count = 1;
      if (!args.empty() && !absl::SimpleAtoi(args[0], &count)) {
        return Failure("Failed to parse first arg as an int");
      }
      if (history.size() < count) {
        return Failure(absl::StrCat(
            "Can't undo ", count, " moves from game of length ",
            history.size()));
      }
      state = game->NewInitialState();
      bot->Restart();
      for (int i = 0; i < history.size() - count; ++i) {
        bot->InformAction(*state, state->CurrentPlayer(), history[i]);
        state->ApplyAction(history[i]);
      }
      return Success();
    }},
    {"showboard", [&state](const Args& args) {
      return Success("\n" + state->ToString());
    }},
    {"history", [&state](const Args& args) {
      return Success(state->HistoryString());
    }},
    {"is_terminal", [&state](const Args& args) {
      return Success(state->IsTerminal() ? "true" : "false");
    }},
    {"current_player", [&state](const Args& args) {
      return Success(absl::StrCat(state->CurrentPlayer()));
    }},
    {"returns", [&state](const Args& args) {
      return Success(absl::StrJoin(state->Returns(), " "));
    }},
    {"legal_actions", [&state](const Args& args) {
      std::vector<std::string> actions;
      std::vector<open_spiel::Action> legal_actions = state->LegalActions();
      actions.reserve(legal_actions.size());
      for (const open_spiel::Action action : legal_actions) {
        actions.push_back(state->ActionToString(action));
      }
      return Success(absl::StrJoin(actions, " "));
    }},
  };

  std::cerr << "Welcome to OpenSpiel GTP interface. Try `list_commands`."
            << std::endl << std::endl;
  for (std::string line; std::getline(std::cin, line);) {
    std::vector<std::string> parts = absl::StrSplit(line, ' ');
    if (parts.empty()) continue;
    std::string& cmd = parts[0];

    auto cmd_it = cmds.find(cmd);
    if (cmd_it == cmds.end()) {
      std::cout << Failure("unknown command");
      continue;
    }

    Args args(parts.begin() + 1, parts.end());
    std::cout << cmd_it->second(args);
    if (cmd == "quit") {
      break;
    }
  }
  return 0;
}
