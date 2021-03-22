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

#include "open_spiel/games/chess/pgn.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace pgn {
namespace {

template <typename T>
void WriteTag(
    std::ostream& os, const std::string& key, const T& value) {
  os << "[" << key << " \"" << value << "\']" << std::endl;
}

void WriteComment(std::ostream& os, const std::string& comment) {
  os << "{" << comment << "}";
}

std::string GetResultString(const State &state) {
  std::string result_str;
  if (state.IsTerminal()) {
    std::vector<std::string> results;
    for (double ret : state.Returns()) {
      if (ret == chess::WinUtility()) {
        results.emplace_back("1");
      } else if (ret == chess::DrawUtility()) {
        results.emplace_back("1/2");
      } else if (ret == chess::LossUtility()) {
        results.emplace_back("0");
      } else {
        SpielFatalError("Invalid chess utility");
      }
    }
    result_str = results[chess::ColorToPlayer(chess::Color::kWhite)] + "-" +
                 results[chess::ColorToPlayer(chess::Color::kBlack)];
  } else {
    result_str = "*";
  }
  return result_str;
}

void WriteHeader(std::ostream& os, const std::string& event, int round,
                 const std::string& white, const std::string& black,
                 const State& state,
                 const std::vector<std::pair<std::string, std::string>> &tags = {}) {

  WriteTag(os, "Event", event);
  WriteTag(os, "Site", "OpenSpiel");
  WriteTag(os, "Round", round);

  std::string date_str;
  if (state.IsTerminal()) {
    time_t ttime = time(nullptr);
    char buffer[11];
    strftime(buffer,sizeof(buffer),"%Y.%m.%d", localtime(&ttime));
    date_str = buffer;
  } else {
    date_str = "????.??.??";
  }

  WriteTag(os, "Date", date_str);
  WriteTag(os, "White", white);
  WriteTag(os, "Black", black);
  WriteTag(os, "Result", GetResultString(state));
  for (const auto &[key, value] : tags) {
    WriteTag(os, key, value);
  }
}

std::string CheckTypeToPGNString(kriegspiel::KriegspielCheckType check_type) {

  switch (check_type) {
    case kriegspiel::KriegspielCheckType::kNoCheck:
      return "";
    case kriegspiel::KriegspielCheckType::kFile:
      return "F";
    case kriegspiel::KriegspielCheckType::kRank:
      return "R";
    case kriegspiel::KriegspielCheckType::kLongDiagonal:
      return "L";
    case kriegspiel::KriegspielCheckType::kShortDiagonal:
      return "S";
    case kriegspiel::KriegspielCheckType::kKnight:
      return "N";
    default:
      SpielFatalError("Unknown check type");
  }
}

std::string CheckTypesToPGNString(
    const std::pair<kriegspiel::KriegspielCheckType,
    kriegspiel::KriegspielCheckType>& check_type) {

  std::string str;
  str += CheckTypeToPGNString(check_type.first);
  if (check_type.second != kriegspiel::KriegspielCheckType::kNoCheck) {
    std::string second_check_str = CheckTypeToPGNString(check_type.second);
    if (second_check_str < str) {
      str = second_check_str + str;
    }
    else {
      str += second_check_str;
    }
  }
  return str;
}

} // namespace


ChessPGN::ChessPGN(const chess::ChessState &state, const std::string& event,
                        int round, const std::string& white,
                        const std::string& black) :
    state_(state),
    event_(event),
    round_(round),
    white_(white),
    black_(black) {}


void ChessPGN::WriteHeader(std::ostream& os) const {
  pgn::WriteHeader(os, event_, round_, white_, black_, state_);
}

void ChessPGN::WriteMoves(std::ostream& os) const {
  chess::ChessBoard board = state_.StartBoard();

  for (const chess::Move& move : state_.MovesHistory()) {
    if (board.ToPlay() == chess::Color::kWhite) {
      os << board.Movenumber() << ". ";
    }
    else {
      os << "   ";
      if (board.Movenumber() > 9) os << " ";
      if (board.Movenumber() > 99) os << " ";
    }
    os << move.ToSAN(board) << std::endl;
    board.ApplyMove(move);
  }
}

void ChessPGN::WriteResult(std::ostream& os) const {
  os << GetResultString(state_) << std::endl;
}

KriegspielPGN::KriegspielPGN(const kriegspiel::KriegspielState &state,
                             const std::string& event,
                             int round,
                             const std::string& white,
                             const std::string& black,
                             chess::Color filtered) :
    state_(state),
    event_(event),
    round_(round),
    white_(white),
    black_(black),
    filtered_(filtered) {}

void KriegspielPGN::WriteHeader(std::ostream &os) const {
  std::vector<std::pair<std::string, std::string>> tags
      {std::make_pair("Variant", "ICC"),
       std::make_pair("Filtered", GetFilteredString())};

  pgn::WriteHeader(os, event_, round_, white_, black_, state_, tags);
}

void KriegspielPGN::WriteMoves(std::ostream &os) const {
  chess::ChessBoard board = state_.StartBoard();

  std::vector<std::string> illegal_moves;
  for (auto &[move, msg] : state_.MoveMsgHistory()) {
    std::string move_str = move.ToSAN(board, true);
    if (msg.illegal) {
      illegal_moves.emplace_back(move_str);
      continue;
    }
    if (board.ToPlay() == chess::Color::kWhite) {
      os << board.Movenumber() << ". ";
    }
    else {
      os << "   ";
      if (board.Movenumber() > 9) os << " ";
      if (board.Movenumber() > 99) os << " ";
    }
    if (filtered_ == chess::Color::kEmpty || filtered_ == board.ToPlay()) {
      os << move_str << " ";
    }
    else {
      os << "??" << " ";
    }

    WriteUmpireComment(os, illegal_moves, msg);
    os << std::endl;

    board.ApplyMove(move);
    illegal_moves.clear();
  }
}

void KriegspielPGN::WriteResult(std::ostream &os) const {
  os << GetResultString(state_) << std::endl;
}

std::string KriegspielPGN::GetFilteredString() const {
  if (filtered_ == chess::Color::kEmpty) {
    return "no";
  }
  return chess::ColorToString(filtered_);
}

void KriegspielPGN::WriteUmpireComment(
    std::ostream &os, const std::vector<std::string> &illegal_moves,
    const kriegspiel::KriegspielUmpireMessage& msg) const {

  std::string str;
  if (msg.capture_type != kriegspiel::KriegspielCaptureType::kNoCapture) {
    str += chess::SquareToString(msg.square);
  }
  std::string check_types_str = CheckTypesToPGNString(msg.check_types);
  if (!check_types_str.empty()) {
    if (!str.empty()) {
      str += ",";
    }
    str += check_types_str;
  }
  str += ":";
  if (filtered_ == chess::Color::kEmpty ||
      filtered_ == chess::OppColor(msg.to_move)) {
    str += absl::StrJoin(illegal_moves.rbegin(), illegal_moves.rend(), ",");
  } else {
    str += std::to_string(illegal_moves.size());
  }

  WriteComment(os, str);
}

std::ostream& operator<<(std::ostream& os, const PGN& pgn) {

  pgn.WriteHeader(os);
  os << std::endl;
  pgn.WriteMoves(os);
  pgn.WriteResult(os);

  // There is a possibility to write multiple games in one file. Those games
  // should be separated by multiple empty lines.
  os << std::endl << std::endl;
  return os;
}
}  // namespace pgn
}  // namespace open_spiel
