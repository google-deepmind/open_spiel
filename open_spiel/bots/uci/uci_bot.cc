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

#include "open_spiel/bots/uci/uci_bot.h"

#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace uci {

UCIBot::UCIBot(const std::string& bot_binary_path, int search_limit_value,
               bool ponder, const Options& options,
               SearchLimitType search_limit_type,
               bool use_game_history_for_position)
    : ponder_(ponder),
      use_game_history_for_position_(use_game_history_for_position) {
  SPIEL_CHECK_GT(search_limit_value, 0);
  SPIEL_CHECK_GT(bot_binary_path.size(), 0);
  search_limit_type_ = search_limit_type;
  search_limit_value_ = search_limit_value;
  if (search_limit_type_ == SearchLimitType::kMoveTime) {
    search_limit_string_ = "movetime " + std::to_string(search_limit_value_);
  } else if (search_limit_type_ == SearchLimitType::kNodes) {
    search_limit_string_ = "nodes " + std::to_string(search_limit_value_);
  } else if (search_limit_type_ == SearchLimitType::kDepth) {
    search_limit_string_ = "depth " + std::to_string(search_limit_value_);
  } else {
    SpielFatalError("Unsupported search limit type");
  }

  StartProcess(bot_binary_path);
  Uci();
  for (auto const& [name, value] : options) {
    SetOption(name, value);
  }
  IsReady();
  UciNewGame();
}

UCIBot::~UCIBot() {
  Quit();
  int status;
  while (waitpid(pid_, &status, 0) == -1) {
    // Do nothing.
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::cerr << "Uci sub-process failed" << std::endl;
  }

  // Close the input stream
  fclose(input_stream_);
  // Free the input stream buffer allocated in ReadLine
  free(input_stream_buffer_);
  // Close the output pipe
  close(output_fd_);
}

void UCIBot::PositionFromState(const chess::ChessState& state,
                               const std::vector<std::string>& extra_moves) {
  if (use_game_history_for_position_) {
    std::pair<std::string, std::vector<std::string>> fen_and_moves =
        state.ExtractFenAndMaybeMoves();
    fen_and_moves.second.insert(fen_and_moves.second.end(),
                                extra_moves.begin(), extra_moves.end());
    Position(fen_and_moves.first, fen_and_moves.second);
  } else {
    Position(state.Board().ToFEN(), extra_moves);
  }
}

Action UCIBot::Step(const State& state) { return StepVerbose(state).first; }

std::pair<Action, std::string> UCIBot::StepVerbose(const State& state) {
  std::string move_str;
  std::string info_str;  // Contains the last info string from the bot.
  auto chess_state = down_cast<const chess::ChessState&>(state);
  auto chess_game = down_cast<const chess::ChessGame*>(state.GetGame().get());
  if (ponder_ && ponder_move_) {
    if (!was_ponder_hit_) {
      Stop();
      PositionFromState(chess_state);
      tie(move_str, ponder_move_) = Go(&info_str);
    } else {
      tie(move_str, ponder_move_) = ReadBestMove(&info_str);
    }
  } else {
    PositionFromState(chess_state);
    tie(move_str, ponder_move_) = Go(&info_str);
  }
  was_ponder_hit_ = false;
  auto move = chess_state.Board().ParseLANMove(move_str,
                                               chess_game->IsChess960());
  if (!move) {
    SpielFatalError("Uci sub-process returned an illegal or invalid move");
  }

  if (ponder_ && ponder_move_) {
    PositionFromState(chess_state, {move_str, *ponder_move_});
    GoPonder();
  }

  Action action = chess::MoveToAction(*move, chess_state.BoardSize());
  return {action, info_str};
}

void UCIBot::Restart() {
  ponder_move_ = absl::nullopt;
  was_ponder_hit_ = false;
  UciNewGame();
}

void UCIBot::RestartAt(const State& state) {
  ponder_move_ = absl::nullopt;
  was_ponder_hit_ = false;
  auto chess_state = down_cast<const chess::ChessState&>(state);
  PositionFromState(chess_state);
}

void UCIBot::InformAction(const State& state, Player player_id, Action action) {
  auto chess_state = down_cast<const chess::ChessState&>(state);
  auto chess_game = down_cast<const chess::ChessGame*>(state.GetGame().get());
  chess::Move move = chess::ActionToMove(action, chess_state.Board());
  std::string move_str = move.ToLAN(chess_game->IsChess960(),
                                    &chess_state.Board());
  if (ponder_ && move_str == ponder_move_) {
    PonderHit();
    was_ponder_hit_ = true;
  }
}

void UCIBot::StartProcess(const std::string& bot_binary_path) {
  int output_pipe[2];
  int input_pipe[2];

  if (pipe(output_pipe) || pipe(input_pipe)) {
    SpielFatalError("Creating pipes failed");
  }

  pid_ = fork();
  if (pid_ < 0) {
    SpielFatalError("Forking failed");
  }

  if (pid_ > 0) {  // parent
    close(output_pipe[0]);
    close(input_pipe[1]);

    output_fd_ = output_pipe[1];
    input_stream_ = fdopen(input_pipe[0], "r");
    if (input_stream_ == nullptr) {
      SpielFatalError("Opening the UCI input pipe as a file stream failed");
    }

  } else {  // child
    dup2(output_pipe[0], STDIN_FILENO);
    dup2(input_pipe[1], STDOUT_FILENO);
    dup2(input_pipe[1], STDERR_FILENO);

    close(output_pipe[1]);
    close(input_pipe[0]);

    std::string real_binary_path = open_spiel::file::RealPath(bot_binary_path);
    execlp(real_binary_path.c_str(), real_binary_path.c_str(), (char*)nullptr);
    // See /usr/include/asm-generic/errno-base.h for error codes.
    switch (errno) {
      case ENOENT:
        SpielFatalError(
            absl::StrCat("Executing uci bot sub-process failed: file '",
                         real_binary_path, "' not found."));
      default:
        SpielFatalError(absl::StrCat(
            "Executing uci bot sub-process failed: Error ", errno));
    }
  }
}

void UCIBot::Uci() {
  Write("uci");
  while (true) {
    std::string response = ReadLine();
    if (!response.empty()) {
      if (absl::StartsWith(response, "id") ||
          absl::StartsWith(response, "option")) {
        continue;  // Don't print options and ids
      }
      if (absl::StrContains(response, "uciok")) {
        return;
      } else {
        std::cerr << "Bot: " << response << std::endl;
      }
    }
  }
}

void UCIBot::SetOption(const std::string& name, const std::string& value) {
  std::string msg = "setoption name " + name + " value " + value;
  Write(msg);
}

void UCIBot::UciNewGame() { Write("ucinewgame"); }

void UCIBot::IsReady() {
  Write("isready");
  while (true) {
    std::string response = ReadLine();
    if (!response.empty()) {
      if (absl::StrContains(response, "readyok")) {
        return;
      } else {
        std::cerr << "Bot: " << response << std::endl;
      }
    }
  }
}

void UCIBot::Position(const std::string& fen,
                      const std::vector<std::string>& moves) {
  std::string msg = "position fen " + fen;
  if (!moves.empty()) {
    std::string moves_str = absl::StrJoin(moves, " ");
    msg += " moves " + moves_str;
  }
  Write(msg);
}

std::pair<std::string, absl::optional<std::string>> UCIBot::Go(
    absl::optional<std::string*> info_string) {
  Write("go " + search_limit_string_);
  return ReadBestMove(info_string);
}

void UCIBot::GoPonder() { Write("go ponder " + search_limit_string_); }

void UCIBot::PonderHit() { Write("ponderhit"); }

std::pair<std::string, absl::optional<std::string>> UCIBot::Stop() {
  Write("stop");
  return ReadBestMove();
}

void UCIBot::Quit() { Write("quit"); }

std::pair<std::string, absl::optional<std::string>> UCIBot::ReadBestMove(
    absl::optional<std::string*> info_string) {
  while (true) {
    // istringstream can't use a string_view so we need to copy to a string.
    std::string response = ReadLine();
    // Save the most recent info string if requested. Specifying that the string
    // contains the number of nodes makes sure that we don't save strings of the
    // form "info depth 30 currmove c2c1 currmovenumber 22", we want the ones
    // with metadata about the search.
    if (info_string.has_value() && absl::StartsWith(response, "info") &&
        absl::StrContains(response, "nodes")) {
      *info_string.value() = response;
    }
    std::istringstream response_line(response);
    std::string token;
    std::string move_str;
    absl::optional<std::string> ponder_str = absl::nullopt;
    response_line >> std::skipws;
    while (response_line >> token) {
      if (token == "bestmove") {
        response_line >> move_str;
      } else if (token == "ponder") {
        response_line >> token;
        ponder_str = token;
      }
    }
    if (!move_str.empty()) {
      return std::make_pair(move_str, ponder_str);
    }
  }
}

void UCIBot::Write(const std::string& msg) const {
  if (write(output_fd_, (msg + "\n").c_str(), msg.size() + 1) !=
      msg.size() + 1) {
    SpielFatalError("Sending a command to uci sub-process failed");
  }
}

std::string UCIBot::ReadLine() {
  if (auto bytes_read = ::getline(&input_stream_buffer_,
                                  &input_stream_buffer_size_, input_stream_);
      bytes_read != -1) {
    absl::string_view response =
        absl::string_view(input_stream_buffer_, bytes_read);
    // Remove the trailing newline that getline left in the string.
    // Using a string_view as input saves us from copying the string.
    return std::string(absl::StripTrailingAsciiWhitespace(response));
  }
  std::cerr << "Failed to read from input stream: " << std::strerror(errno)
            << "\n";
  SpielFatalError("Reading a line from uci sub-process failed");
}

std::unique_ptr<Bot> MakeUCIBot(const std::string& bot_binary_path,
                                int search_limit_value, bool ponder,
                                const Options& options,
                                SearchLimitType search_limit_type,
                                bool use_game_history_for_position) {
  return std::make_unique<UCIBot>(bot_binary_path, search_limit_value, ponder,
                                  options, search_limit_type,
                                  use_game_history_for_position);
}

}  // namespace uci
}  // namespace open_spiel
