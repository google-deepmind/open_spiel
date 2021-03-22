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

#ifndef OPEN_SPIEL_GAMES_IMPL_CHESS_PGN_H_
#define OPEN_SPIEL_GAMES_IMPL_CHESS_PGN_H_

#include <array>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/chess/chess_common.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/kriegspiel.h"

namespace open_spiel {
namespace pgn {

class PGN {
 public:
  virtual void WriteHeader(std::ostream& os) const = 0;
  virtual void WriteMoves(std::ostream& os) const = 0;
  virtual void WriteResult(std::ostream& os) const = 0;
};

// Wrapper class for a ChessState to write a game to an output stream in the PGN
// format: https://cs.wikipedia.org/wiki/PGN
class ChessPGN : public PGN {
 public:
  ChessPGN(const chess::ChessState& state, const std::string& event, int round,
           const std::string& white, const std::string& black);

  void WriteHeader(std::ostream& os) const override;
  void WriteMoves(std::ostream& os) const override;
  void WriteResult(std::ostream& os) const override;

 private:
  const chess::ChessState& state_;
  const std::string& event_;
  int round_;
  const std::string& white_;
  const std::string& black_;
};

// Wrapper class for a KriegspielState to write a game to an output stream
// in the Kriegspiel PGN format, which is an extension of classic chess PGN
// http://w01fe.com/berkeley/kriegspiel/notation.html
class KriegspielPGN : public PGN {
 public:
  KriegspielPGN(const kriegspiel::KriegspielState &state,
                const std::string &event, int round, const std::string &white,
                const std::string &black, chess::Color filtered);

  void WriteHeader(std::ostream& os) const override;
  void WriteMoves(std::ostream& os) const override;
  void WriteResult(std::ostream& os) const override;

 private:

  std::string GetFilteredString() const;
  void WriteUmpireComment(std::ostream& os,
                          const std::vector<std::string>& illegal_moves,
                          const kriegspiel::KriegspielUmpireMessage& msg) const;

  kriegspiel::KriegspielState state_;
  const std::string& event_;
  int round_;
  const std::string& white_;
  const std::string& black_;
  chess::Color filtered_;
};

std::ostream& operator<<(std::ostream& os, const PGN& pgn);

}  // namespace pgn
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_IMPL_CHESS_PGN_H_
