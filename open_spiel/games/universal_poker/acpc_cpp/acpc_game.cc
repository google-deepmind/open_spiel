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

#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <numeric>
#include <ostream>
#include <sstream>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/game.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace acpc_cpp {

constexpr int kStringBuffersize = 4096;

namespace {
project_acpc_server::Action GetAction(ACPCState::ACPCActionType type,
                                      int32_t size) {
  project_acpc_server::Action acpc_action;

  acpc_action.size = size;
  switch (type) {
    case ACPCState::ACPC_CALL:
      acpc_action.type = project_acpc_server::a_call;
      break;
    case ACPCState::ACPC_FOLD:
      acpc_action.type = project_acpc_server::a_fold;
      break;
    case ACPCState::ACPC_RAISE:
      acpc_action.type = project_acpc_server::a_raise;
      break;
    default:
      acpc_action.type = project_acpc_server::a_invalid;
      break;
  }
  return acpc_action;
}

void readGameToStruct(const std::string &gameDef,
                      project_acpc_server::Game *acpc_game) {
  char buf[kStringBuffersize];
  gameDef.copy(buf, kStringBuffersize);

  FILE *f = fmemopen(&buf, kStringBuffersize, "r");
  project_acpc_server::Game *game = project_acpc_server::readGame(f);
  memcpy(acpc_game, game, sizeof(project_acpc_server::Game));

  free(game);
  fclose(f);
}

}  // namespace

ACPCGame::ACPCGame(const std::string &gameDef) {
  readGameToStruct(gameDef, &acpc_game_);
}

ACPCGame::ACPCGame(const ACPCGame &other)
    : handId_(other.handId_), acpc_game_(other.acpc_game_) {}

// We compare the values for all the fields. For arrays, note that only the
// first `numPlayers` or `numRounds` values are meaningful, the rest being
// non-initialized.
bool ACPCGame::operator==(const ACPCGame &other) const {
  // See project_acpc_server/game.h:42. 12 fields total.
  // int32_t stack[ MAX_PLAYERS ];
  // int32_t blind[ MAX_PLAYERS ];
  // int32_t raiseSize[ MAX_ROUNDS ];
  // enum BettingType bettingType;
  // uint8_t numPlayers;
  // uint8_t numRounds;
  // uint8_t firstPlayer[ MAX_ROUNDS ];
  // uint8_t maxRaises[ MAX_ROUNDS ];
  // uint8_t numSuits;
  // uint8_t numRanks;
  // uint8_t numHoleCards;
  // uint8_t numBoardCards[ MAX_ROUNDS ];
  const project_acpc_server::Game *first = &acpc_game_;
  const project_acpc_server::Game &second = other.Game();
  const int num_players = first->numPlayers;
  const int num_rounds = first->numRounds;

  // We check for `raiseSize` only for limit betting.
  if (first->bettingType != second.bettingType) {
    return false;
  }
  if (first->bettingType == project_acpc_server::limitBetting) {
    if (!std::equal(first->raiseSize, first->raiseSize + num_rounds,
                    second.raiseSize)) {
      return false;
    }
  }
  return (  // new line
      first->numPlayers == second.numPlayers &&
      first->numRounds == second.numRounds &&
      std::equal(first->stack, first->stack + num_players, second.stack) &&
      std::equal(first->blind, first->blind + num_players, second.blind) &&
      std::equal(first->firstPlayer, first->firstPlayer + num_rounds,
                 second.firstPlayer) &&
      std::equal(first->maxRaises, first->maxRaises + num_rounds,
                 second.maxRaises) &&
      first->numSuits == second.numSuits &&
      first->numRanks == second.numRanks &&
      first->numHoleCards == second.numHoleCards &&
      std::equal(first->numBoardCards, first->numBoardCards + num_rounds,
                 second.numBoardCards));
}

std::string ACPCGame::ToString() const {
  char buf[kStringBuffersize];
  memset(buf, 0, kStringBuffersize);
  FILE *f = fmemopen(&buf, kStringBuffersize, "w");
  project_acpc_server::printGame(f, &acpc_game_);
  std::ostringstream result;
  rewind(f);
  result << buf;
  fclose(f);
  return result.str();
}

uint8_t ACPCGame::GetNbBoardCardsRequired(uint8_t round) const {
  SPIEL_CHECK_LT(round, acpc_game_.numRounds);

  uint8_t nbCards = 0;
  for (int r = 0; r <= round; ++r) {
    nbCards += acpc_game_.numBoardCards[r];
  }
  return nbCards;
}

uint8_t ACPCGame::GetTotalNbBoardCards() const {
  uint8_t nbCards = 0;
  for (int r = 0; r < acpc_game_.numRounds; ++r) {
    nbCards += acpc_game_.numBoardCards[r];
  }

  return nbCards;
}


uint32_t ACPCGame::StackSize(uint8_t player) const {
  SPIEL_CHECK_LE(0, player);
  SPIEL_CHECK_LT(player, GetNbPlayers());
  return acpc_game_.stack[player];
}

uint32_t ACPCGame::BlindSize(uint8_t player) const {
  SPIEL_CHECK_LE(0, player);
  SPIEL_CHECK_LT(player, GetNbPlayers());
  return acpc_game_.blind[player];
}
uint32_t ACPCGame::TotalMoney() const {
  int money_pool = 0;
  for (int pl = 0; pl < acpc_game_.numPlayers; ++pl) {
    money_pool += acpc_game_.stack[pl];
  }
  return money_pool;
}

std::string ACPCState::ToString() const {
  char buf[kStringBuffersize];
  project_acpc_server::printState(game_->MutableGame(), &acpcState_,
                                  kStringBuffersize, buf);
  std::ostringstream out;

  out << buf << std::endl << "Spent: [";
  for (int p = 0; p < game_->GetNbPlayers(); ++p) {
    out << "P" << p << ": " << acpcState_.spent[p] << "  ";
  }
  out << "]" << std::endl;

  return out.str();
}


double ACPCState::ValueOfState(const uint8_t player) const {
  SPIEL_CHECK_TRUE(stateFinished(&acpcState_));
  return project_acpc_server::valueOfState(game_->MutableGame(), &acpcState_,
                                           player);
}
int ACPCState::RaiseIsValid(int32_t *minSize, int32_t *maxSize) const {
  return raiseIsValid(game_->MutableGame(), &acpcState_, minSize, maxSize);
}

uint8_t ACPCState::NumFolded() const {
  return project_acpc_server::numFolded(game_->MutableGame(), &acpcState_);
}

uint8_t ACPCState::CurrentPlayer() const {
  return project_acpc_server::currentPlayer(game_->MutableGame(), &acpcState_);
}

ACPCState::ACPCState(const ACPCGame *game)
    // This is necessary as we need to value-initialize acpcState_.
    : game_(game), acpcState_() {
  project_acpc_server::initState(game_->MutableGame(),
                                 game_->HandId()
                                 /*TODO this make a unit test fail++*/,
                                 &acpcState_);
}

void ACPCState::DoAction(const ACPCState::ACPCActionType actionType,
                         const int32_t size) {
  project_acpc_server::Action a = GetAction(actionType, size);
  SPIEL_CHECK_TRUE(project_acpc_server::isValidAction(game_->MutableGame(),
                                                      &acpcState_, false, &a));
  project_acpc_server::doAction(game_->MutableGame(), &a, &acpcState_);
}

int ACPCState::IsValidAction(const ACPCState::ACPCActionType actionType,
                             const int32_t size) const {
  project_acpc_server::Action a = GetAction(actionType, size);
  return project_acpc_server::isValidAction(game_->MutableGame(), &acpcState_,
                                            false, &a);
}

uint32_t ACPCState::Money(const uint8_t player) const {
  SPIEL_CHECK_LE(player, game_->GetNbPlayers());
  return game_->StackSize(player) - acpcState_.spent[player];
}

uint32_t ACPCState::Ante(const uint8_t player) const {
  SPIEL_CHECK_LE(player, game_->GetNbPlayers());
  return acpcState_.spent[player];
}

uint32_t ACPCState::TotalSpent() const {
  return static_cast<uint32_t>(absl::c_accumulate(acpcState_.spent, 0));
}

uint32_t ACPCState::CurrentSpent(const uint8_t player) const {
  SPIEL_CHECK_LE(player, game_->GetNbPlayers());
  return acpcState_.spent[player];
}

std::string ACPCState::ActionToString(
    const project_acpc_server::Action &action) const {
  switch (action.type) {
    case ACPCState::ACPC_CALL:
      return "c";
    case ACPCState::ACPC_FOLD:
      return "f";
    case ACPCState::ACPC_RAISE:
      if (game_->IsLimitGame()) return "r";
      return absl::StrCat("r", action.size);
    default:
      SpielFatalError("Should never happen.");
  }
}

std::string ACPCState::BettingSequence(uint8_t round) const {
  SPIEL_CHECK_LT(round, game_->NumRounds());
  std::string out;
  for (int a = 0; a < acpcState_.numActions[round]; a++) {
    absl::StrAppend(&out, ActionToString(acpcState_.action[round][a]));
  }
  return out;
}

void ACPCState::SetHoleAndBoardCards(uint8_t holeCards[10][3],
                                     uint8_t boardCards[7],
                                     uint8_t nbHoleCards[10],
                                     uint8_t nbBoardCards) {
  for (int p = 0; p < game_->GetNbPlayers(); ++p) {
    SPIEL_CHECK_EQ(nbHoleCards[p], game_->GetNbHoleCardsRequired());
    for (int c = 0; c < nbHoleCards[p]; ++c) {
      acpcState_.holeCards[p][c] = holeCards[p][c];
    }
  }

  SPIEL_CHECK_EQ(nbBoardCards, game_->GetNbBoardCardsRequired(GetRound()));
  for (int c = 0; c < nbBoardCards; ++c) {
    acpcState_.boardCards[c] = boardCards[c];
  }
}

}  // namespace acpc_cpp
}  // namespace universal_poker
}  // namespace open_spiel
