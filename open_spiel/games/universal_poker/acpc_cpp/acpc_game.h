// Copyright 2019 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_ACPC_GAME_H
#define OPEN_SPIEL_ACPC_GAME_H

#include <memory>
#include <string>

#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/game.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace acpc_cpp {

struct RawACPCGame : public ::project_acpc_server::Game {};
struct RawACPCState : public ::project_acpc_server::State {};
struct RawACPCAction : public ::project_acpc_server::Action {};

class ACPCGame {
 public:
  explicit ACPCGame(const std::string& gameDef);
  explicit ACPCGame(const ACPCGame& other);

  std::string ToString() const;

  bool IsLimitGame() const {
    return acpc_game_.bettingType == project_acpc_server::limitBetting;
  }

  // The total number of betting rounds.
  int NumRounds() const { return acpc_game_.numRounds; }
  int GetNbPlayers() const { return acpc_game_.numPlayers; }

  // Returns the number of private cards for each player in this game.
  uint8_t GetNbHoleCardsRequired() const { return acpc_game_.numHoleCards; }
  uint8_t GetNbBoardCardsRequired(uint8_t round) const;
  uint8_t NumSuitsDeck() const { return acpc_game_.numSuits; }
  uint8_t NumRanksDeck() const { return acpc_game_.numRanks; }
  uint8_t NumBoardCards(int round) const {
    return acpc_game_.numBoardCards[round];
  }
  uint32_t StackSize(uint8_t player) const;
  // Returns the money amount that is used in the game (sum of all stacks).
  uint32_t TotalMoney() const;
  uint32_t BlindSize(uint8_t player) const;
  uint8_t GetTotalNbBoardCards() const;

  // Accessors.
  ::project_acpc_server::Game* MutableGame() const { return &acpc_game_; }
  const project_acpc_server::Game& Game() const { return acpc_game_; }
  uint32_t HandId() const { return handId_; }
  absl::Span<const int32_t> blinds() const {
    return absl::Span<const int32_t>(acpc_game_.blind, acpc_game_.numPlayers);
  }

  // Checks that the underlying acpc_game_ structs have all their fields equal.
  bool operator==(const ACPCGame& other) const;
  bool operator!=(const ACPCGame& other) const { return !(*this == other); }

 private:
  uint32_t handId_ = 0;
  mutable project_acpc_server::Game acpc_game_;
};

class ACPCState {
 public:
  enum ACPCActionType { ACPC_FOLD, ACPC_CALL, ACPC_RAISE, ACPC_INVALID };

  explicit ACPCState(const ACPCGame* game);
  explicit ACPCState(const ACPCState& other)
      : game_(other.game_), acpcState_(other.acpcState_) {}

  void SetHoleAndBoardCards(uint8_t holeCards[10][3], uint8_t boardCards[7],
                            uint8_t nbHoleCards[10], uint8_t nbBoardCards);

  // The current player is the first player in a new round, or the next player
  // within a round.
  uint8_t CurrentPlayer() const;
  uint8_t NumFolded() const;
  uint32_t Money(const uint8_t player) const;
  uint32_t Ante(const uint8_t player) const;
  uint32_t TotalSpent() const;
  uint32_t CurrentSpent(const uint8_t player) const;
  std::string ToString() const;
  std::string BettingSequence(uint8_t round) const;
  int RaiseIsValid(int32_t* minSize, int32_t* maxSize) const;
  int IsValidAction(const ACPCActionType actionType, const int32_t size) const;
  void DoAction(const ACPCActionType actionType, const int32_t size);
  double ValueOfState(const uint8_t player) const;

  // Trivial methods.
  bool IsFinished() const { return stateFinished(&acpcState_); }
  uint32_t MaxSpend() const { return acpcState_.maxSpent; }
  uint8_t hole_cards(int player_index, int card_index) const {
    SPIEL_CHECK_LT(player_index, MAX_PLAYERS);
    SPIEL_CHECK_LT(card_index, MAX_HOLE_CARDS);
    return acpcState_.holeCards[player_index][card_index];
  }
  uint8_t board_cards(int card_index) const {
    SPIEL_CHECK_LT(card_index, MAX_BOARD_CARDS);
    return acpcState_.boardCards[card_index];
  }

  void AddHoleCard(int player_index, int card_index, uint8_t card) {
    SPIEL_CHECK_LT(player_index, MAX_PLAYERS);
    SPIEL_CHECK_LT(card_index, MAX_HOLE_CARDS);
    acpcState_.holeCards[player_index][card_index] = card;
  }

  void AddBoardCard(int card_index, uint8_t card) {
    SPIEL_CHECK_LT(card_index, MAX_BOARD_CARDS);
    acpcState_.boardCards[card_index] = card;
  }

  // Set the spent amounts uniformly for each player.
  // Must be divisible by the number of players!
  void SetPotSize(int pot_size) {
    SPIEL_CHECK_GE(pot_size, 0);
    SPIEL_CHECK_LE(pot_size, game_->TotalMoney());
    SPIEL_CHECK_EQ(pot_size % game_->GetNbPlayers(), 0);
    const int num_players = game_->GetNbPlayers();
    for (int pl = 0; pl < num_players; ++pl) {
      acpcState_.spent[pl] = pot_size / num_players;
    }
  }

  // Returns the current round 0-indexed round id (<= game.NumRounds() - 1).
  // A showdown is still in game.NumRounds()-1, not a separate round
  int GetRound() const { return acpcState_.round; }

  const ACPCGame* game() const { return game_; }
  const RawACPCState& raw_state() const { return acpcState_; }
  RawACPCState* mutable_state() { return &acpcState_; }

 private:
  std::string ActionToString(const project_acpc_server::Action& action) const;

  const ACPCGame* game_;
  mutable RawACPCState acpcState_;
};

}  // namespace acpc_cpp
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ACPC_GAME_H
