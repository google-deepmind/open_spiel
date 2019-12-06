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

#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <ostream>
#include <sstream>


#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace acpc_cpp {

static const int STRING_BUFFERSIZE = 4096;


namespace {
RawACPCAction GetAction(ACPCState::ACPCActionType type, int32_t size) {
  RawACPCAction acpc_action;

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

void readGameToStruct(const std::string &gameDef, RawACPCGame *acpc_game) {
  char buf[STRING_BUFFERSIZE];
  gameDef.copy(buf, STRING_BUFFERSIZE);

  FILE *f = fmemopen(&buf, STRING_BUFFERSIZE, "r");
  project_acpc_server::Game *game = project_acpc_server::readGame(f);

  memcpy(acpc_game, game, sizeof(RawACPCGame));

  free(game);
  fclose(f);
}

}  // namespace

ACPCGame::ACPCGame(const std::string &gameDef)
    // check this make unique.
    : handId_(0), acpc_game_(std::make_unique<RawACPCGame>()) {
  readGameToStruct(gameDef, acpc_game_.get());
}

ACPCGame::ACPCGame(const ACPCGame &other)
    : handId_(other.handId_),
      acpc_game_(std::make_unique<RawACPCGame>(*other.acpc_game_)) {}

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
  const RawACPCGame *first = acpc_game_.get();
  const RawACPCGame *second = other.acpc_game_.get();
  const int num_players = first->numPlayers;
const int num_rounds = first->numRounds;
  return (  // new line
      first->numPlayers == second->numPlayers &&
      first->numRounds == second->numRounds &&
      std::equal(first->stack, first->stack + num_players, second->stack) &&
      std::equal(first->blind, first->blind + num_players, second->blind) &&
      std::equal(first->raiseSize, first->raiseSize + num_rounds,
                 second->raiseSize) &&
      first->bettingType == second->bettingType &&
      std::equal(first->firstPlayer, first->firstPlayer + num_rounds,
                 second->firstPlayer) &&
      std::equal(first->maxRaises, first->maxRaises + num_rounds,
                 second->maxRaises) &&
      first->numSuits == second->numSuits &&
      first->numRanks == second->numRanks &&
      first->numHoleCards == second->numHoleCards &&
      std::equal(first->numBoardCards, first->numBoardCards + num_rounds,
                 second->numBoardCards));
}

std::string ACPCGame::ToString() const {
  char buf[STRING_BUFFERSIZE];
  FILE *f = fmemopen(&buf, STRING_BUFFERSIZE, "w");
  project_acpc_server::printGame(f, acpc_game_.get());
  std::ostringstream result;
  rewind(f);
  result << buf;
  fclose(f);

  return result.str();
}

bool ACPCGame::IsLimitGame() const {
  return acpc_game_->bettingType == project_acpc_server::limitBetting;
}

int ACPCGame::GetNbPlayers() const { return acpc_game_->numPlayers; }

uint8_t ACPCGame::GetNbHoleCardsRequired() const {
  return acpc_game_->numHoleCards;
}

uint8_t ACPCGame::GetNbBoardCardsRequired(uint8_t round) const {
  SPIEL_CHECK_LT(round, acpc_game_->numRounds);

  uint8_t nbCards = 0;
  for (int r = 0; r <= round; ++r) {
    nbCards += acpc_game_->numBoardCards[r];
  }
  return nbCards;
}

uint8_t ACPCGame::GetTotalNbBoardCards() const {
  uint8_t nbCards = 0;
  for (int r = 0; r < acpc_game_->numRounds; ++r) {
    nbCards += acpc_game_->numBoardCards[r];
  }

  return nbCards;
}

uint8_t ACPCGame::NumSuitsDeck() const { return acpc_game_->numSuits; }

uint8_t ACPCGame::NumRanksDeck() const { return acpc_game_->numRanks; }

uint32_t ACPCGame::StackSize(uint8_t player) const {
  SPIEL_CHECK_LE(0, player);
  SPIEL_CHECK_LT(player, acpc_game_->numPlayers);
  return acpc_game_->stack[player];
}

int ACPCGame::NumRounds() const { return acpc_game_->numRounds; }

uint32_t ACPCGame::BlindSize(uint8_t player) const {
  SPIEL_CHECK_LE(0, player);
  SPIEL_CHECK_LT(player, acpc_game_->numPlayers);
  return acpc_game_->blind[player];
}

std::string ACPCState::ToString() const {
  char buf[STRING_BUFFERSIZE];
  project_acpc_server::printState(game_->acpc_game_.get(), acpcState_.get(),
                                  STRING_BUFFERSIZE, buf);
  std::ostringstream out;

  out << buf << std::endl << "Spent: [";
  for (int p = 0; p < game_->acpc_game_->numPlayers; ++p) {
    out << "P" << p << ": " << acpcState_->spent[p] << "  ";
  }
  out << "]" << std::endl;

  return out.str();
}

int ACPCState::RaiseIsValid(int32_t *minSize, int32_t *maxSize) const {
  return raiseIsValid(game_->acpc_game_.get(), acpcState_.get(), minSize,
                      maxSize);
}

double ACPCState::ValueOfState(const uint8_t player) const {
  assert(stateFinished(acpcState_.get()));
  return project_acpc_server::valueOfState(game_->acpc_game_.get(),
                                           acpcState_.get(), player);
}

bool ACPCState::IsFinished() const { return stateFinished(acpcState_.get()); }

uint32_t ACPCState::MaxSpend() const { return acpcState_->maxSpent; }

int ACPCState::GetRound() const { return acpcState_->round; }

uint8_t ACPCState::NumFolded() const {
  return project_acpc_server::numFolded(game_->acpc_game_.get(),
                                        acpcState_.get());
}

uint8_t ACPCState::CurrentPlayer() const {
  return project_acpc_server::currentPlayer(game_->acpc_game_.get(),
                                            acpcState_.get());
}

ACPCState::ACPCState(const ACPCGame *game)
    : game_(game), acpcState_(std::make_unique<RawACPCState>()) {
  project_acpc_server::initState(
      game_->acpc_game_.get(),
      game_->handId_ /*TODO this make a unit test fail++*/, acpcState_.get());
}

ACPCState::ACPCState(const ACPCState &other)
    : game_(other.game_),
      acpcState_(std::make_unique<RawACPCState>(*other.acpcState_)) {}

void ACPCState::DoAction(const ACPCState::ACPCActionType actionType,
                         const int32_t size) {
  RawACPCAction a = GetAction(actionType, size);
  assert(project_acpc_server::isValidAction(game_->acpc_game_.get(),
                                            acpcState_.get(), false, &a));
  project_acpc_server::doAction(game_->acpc_game_.get(), &a, acpcState_.get());
}

int ACPCState::IsValidAction(const ACPCState::ACPCActionType actionType,
                             const int32_t size) const {
  RawACPCAction a = GetAction(actionType, size);
  return project_acpc_server::isValidAction(game_->acpc_game_.get(),
                                            acpcState_.get(), false, &a);
}

uint32_t ACPCState::Money(const uint8_t player) const {
  assert(player < game_->acpc_game_->numPlayers);
  return game_->acpc_game_->stack[player] - acpcState_->spent[player];
}

uint32_t ACPCState::Ante(const uint8_t player) const {
  assert(player < game_->acpc_game_->numPlayers);
  return acpcState_->spent[player];
}

std::string ACPCState::BettingSequence(uint8_t round) const {
  assert(round < game_->acpc_game_->numRounds);
  std::ostringstream out;
  char buf[10];
  for (int a = 0; a < acpcState_->numActions[round]; a++) {
    project_acpc_server::Action *action = &acpcState_->action[round][a];
    printAction(game_->acpc_game_.get(), action, 10, buf);
    out << buf;
  }

  return out.str();
}

void ACPCState::SetHoleAndBoardCards(uint8_t holeCards[10][3],
                                     uint8_t boardCards[7],
                                     uint8_t nbHoleCards[10],
                                     uint8_t nbBoardCards) const {
  for (int p = 0; p < game_->GetNbPlayers(); ++p) {
    assert(nbHoleCards[p] == game_->GetNbHoleCardsRequired());
    for (int c = 0; c < nbHoleCards[p]; ++c) {
      acpcState_->holeCards[p][c] = holeCards[p][c];
    }
  }

  assert(nbBoardCards == game_->GetNbBoardCardsRequired(GetRound()));
  for (int c = 0; c < nbBoardCards; ++c) {
    acpcState_->boardCards[c] = boardCards[c];
  }
}

ACPCState::~ACPCState() = default;
ACPCGame::~ACPCGame() = default;
}  // namespace acpc_cpp
}  // namespace universal_poker
}  // namespace open_spiel
