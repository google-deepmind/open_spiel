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

#ifndef OPEN_SPIEL_ACPC_GAME_H
#define OPEN_SPIEL_ACPC_GAME_H

#include <memory>
#include <string>

namespace open_spiel {
namespace universal_poker {
namespace acpc_cpp {

struct RawACPCGame;
struct RawACPCState;
struct RawACPCAction;

class ACPCGame;

class ACPCState {
 public:
  enum ACPCActionType { ACPC_FOLD, ACPC_CALL, ACPC_RAISE, ACPC_INVALID };

 public:
  ACPCState(ACPCGame* game);
  ACPCState(const ACPCState& other);
  virtual ~ACPCState();

  void SetHoleAndBoardCards(uint8_t holeCards[10][3], uint8_t boardCards[7],
                            uint8_t nbHoleCards[10],
                            uint8_t nbBoardCards) const;

  uint8_t CurrentPlayer() const;

  bool IsFinished() const;
  int RaiseIsValid(int32_t* minSize, int32_t* maxSize) const;
  int IsValidAction(const ACPCActionType actionType, const int32_t size) const;
  void DoAction(const ACPCActionType actionType, const int32_t size);
  double ValueOfState(const uint8_t player) const;
  uint32_t MaxSpend() const;
  uint8_t GetRound() const;
  uint8_t NumFolded() const;
  uint32_t Money(const uint8_t player) const;
  uint32_t Ante(const uint8_t player) const;
  std::string ToString() const;
  std::string BettingSequence(uint8_t round) const;

  ACPCGame* game_;
  std::unique_ptr<RawACPCState> acpcState_;
};

class ACPCGame {
 public:
  ACPCGame(const std::string& gameDef);
  ACPCGame(const ACPCGame& other);
  virtual ~ACPCGame();

  std::string ToString() const;
  bool IsLimitGame() const;
  uint8_t GetNbRounds() const;
  uint8_t GetNbPlayers() const;
  uint8_t GetNbHoleCardsRequired() const;
  uint8_t GetNbBoardCardsRequired(uint8_t round) const;
  uint8_t NumSuitsDeck() const;
  uint8_t NumRanksDeck() const;
  uint32_t StackSize(uint8_t player) const;
  uint32_t BlindSize(uint8_t player) const;
  uint8_t GetTotalNbBoardCards() const;

  uint32_t handId_;
  std::unique_ptr<RawACPCGame> acpc_game_;
};

}  // namespace acpc_cpp
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ACPC_GAME_H
