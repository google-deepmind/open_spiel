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

#include <ostream>
#include <sstream>

extern "C" {
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/game.h"
};

namespace open_spiel::universal_poker::acpc_cpp {
struct Game : public ::Game {};
struct State : public ::State {};
struct Action : public ::Action {};

Action GetAction(ACPCGame::ACPCState::ACPCActionType type, int32_t size) {
  Action acpcAction;

  acpcAction.size = size;
  switch (type) {
    case ACPCGame::ACPCState::ACPC_CALL:
      acpcAction.type = a_call;
      break;
    case ACPCGame::ACPCState::ACPC_FOLD:
      acpcAction.type = a_fold;
      break;
    case ACPCGame::ACPCState::ACPC_RAISE:
      acpcAction.type = a_raise;
      break;
    default:
      acpcAction.type = a_invalid;
      break;
  }

  return acpcAction;
}

void readGameToStruct(const std::string &gameDef, Game *acpcGame) {
  char buf[STRING_BUFFERSIZE];
  gameDef.copy(buf, STRING_BUFFERSIZE);

  FILE *f = fmemopen(&buf, STRING_BUFFERSIZE, "r");
  ::Game *game = readGame(f);

  memcpy(acpcGame, game, sizeof(Game));

  free(game);
  fclose(f);
}

ACPCGame::ACPCGame(const std::string &gameDef)
    : handId_(0), acpcGame_(std::make_unique<Game>()) {
  readGameToStruct(gameDef, acpcGame_.get());
}

ACPCGame::ACPCGame(const ACPCGame &other)
    : handId_(other.handId_),
      acpcGame_(std::make_unique<Game>(*other.acpcGame_)) {}

std::string ACPCGame::ToString() const {
  char buf[STRING_BUFFERSIZE];
  FILE *f = fmemopen(&buf, STRING_BUFFERSIZE, "w");
  printGame(f, acpcGame_.get());
  std::ostringstream result;
  rewind(f);
  result << buf;
  fclose(f);

  return result.str();
}

bool ACPCGame::IsLimitGame() const {
  return acpcGame_->bettingType == limitBetting;
}

uint8_t ACPCGame::GetNbPlayers() const { return acpcGame_->numPlayers; }

uint8_t ACPCGame::GetNbHoleCardsRequired() const {
  return acpcGame_->numHoleCards;
}

uint8_t ACPCGame::GetNbBoardCardsRequired(uint8_t round) const {
  assert(round < acpcGame_->numRounds);

  uint8_t nbCards = 0;
  for (int r = 0; r <= round; r++) {
    nbCards += acpcGame_->numBoardCards[r];
  }

  return nbCards;
}

uint8_t ACPCGame::GetTotalNbBoardCards() const {
  uint8_t nbCards = 0;
  for (int r = 0; r < acpcGame_->numRounds; r++) {
    nbCards += acpcGame_->numBoardCards[r];
  }

  return nbCards;
}

uint8_t ACPCGame::NumSuitsDeck() const { return acpcGame_->numSuits; }

uint8_t ACPCGame::NumRanksDeck() const { return acpcGame_->numRanks; }

uint32_t ACPCGame::StackSize(uint8_t player) const {
  assert(player < acpcGame_->numPlayers);

  return acpcGame_->stack[player];
}

uint8_t ACPCGame::GetNbRounds() const { return acpcGame_->numRounds; }

uint32_t ACPCGame::BlindSize(uint8_t player) const {
  assert(player < acpcGame_->numPlayers);

  return acpcGame_->blind[player];
}

std::string ACPCGame::ACPCState::ToString() const {
  char buf[STRING_BUFFERSIZE];
  printState(game_->acpcGame_.get(), acpcState_.get(), STRING_BUFFERSIZE, buf);
  std::ostringstream out;

  out << buf << std::endl << "Spent: [";
  for (int p = 0; p < game_->acpcGame_->numPlayers; p++) {
    out << "P" << p << ": " << acpcState_->spent[p] << "\t";
  }
  out << "]" << std::endl;

  return out.str();
}

int ACPCGame::ACPCState::RaiseIsValid(int32_t *minSize,
                                      int32_t *maxSize) const {
  return raiseIsValid(game_->acpcGame_.get(), acpcState_.get(), minSize,
                      maxSize);
}

double ACPCGame::ACPCState::ValueOfState(const uint8_t player) const {
  assert(stateFinished(acpcState_.get()));
  return valueOfState(game_->acpcGame_.get(), acpcState_.get(), player);
}

bool ACPCGame::ACPCState::IsFinished() const {
  return stateFinished(acpcState_.get());
}

uint32_t ACPCGame::ACPCState::MaxSpend() const { return acpcState_->maxSpent; }

uint8_t ACPCGame::ACPCState::GetRound() const { return acpcState_->round; }

uint8_t ACPCGame::ACPCState::NumFolded() const {
  return numFolded(game_->acpcGame_.get(), acpcState_.get());
}

uint8_t ACPCGame::ACPCState::CurrentPlayer() const {
  return currentPlayer(game_->acpcGame_.get(), acpcState_.get());
}

ACPCGame::ACPCState::ACPCState(ACPCGame *game)
    : game_(game), acpcState_(std::make_unique<State>()) {
  initState(game_->acpcGame_.get(),
            game_->handId_ /*TODO this make a unit test fail++*/,
            acpcState_.get());
}

ACPCGame::ACPCState::ACPCState(const ACPCGame::ACPCState &other)
    : game_(other.game_),
      acpcState_(std::make_unique<State>(*other.acpcState_)) {}

void ACPCGame::ACPCState::DoAction(
    const ACPCGame::ACPCState::ACPCActionType actionType, const int32_t size) {
  Action a = GetAction(actionType, size);
  assert(isValidAction(game_->acpcGame_.get(), acpcState_.get(), false, &a));
  doAction(game_->acpcGame_.get(), &a, acpcState_.get());
}

int ACPCGame::ACPCState::IsValidAction(
    const ACPCGame::ACPCState::ACPCActionType actionType,
    const int32_t size) const {
  Action a = GetAction(actionType, size);
  return isValidAction(game_->acpcGame_.get(), acpcState_.get(), false, &a);
}

uint32_t ACPCGame::ACPCState::Money(const uint8_t player) const {
  assert(player < game_->acpcGame_->numPlayers);
  return game_->acpcGame_->stack[player] - acpcState_->spent[player];
}

uint32_t ACPCGame::ACPCState::Ante(const uint8_t player) const {
  assert(player < game_->acpcGame_->numPlayers);
  return acpcState_->spent[player];
}

std::string ACPCGame::ACPCState::BettingSequence(uint8_t round) const {
  assert(round < game_->acpcGame_->numRounds);
  std::ostringstream out;
  char buf[10];
  for (int a = 0; a < acpcState_->numActions[round]; a++) {
    ::Action *action = &acpcState_->action[round][a];
    printAction(game_->acpcGame_.get(), action, 10, buf);
    out << buf;
  }

  return out.str();
}

void ACPCGame::ACPCState::SetHoleAndBoardCards(uint8_t holeCards[10][3],
                                               uint8_t boardCards[7],
                                               uint8_t nbHoleCards[10],
                                               uint8_t nbBoardCards) const {
  for (int p = 0; p < game_->GetNbPlayers(); p++) {
    assert(nbHoleCards[p] == game_->GetNbHoleCardsRequired());
    for (int c = 0; c < nbHoleCards[p]; c++) {
      acpcState_->holeCards[p][c] = holeCards[p][c];
    }
  }

  assert(nbBoardCards == game_->GetNbBoardCardsRequired(GetRound()));
  for (int c = 0; c < nbBoardCards; c++) {
    acpcState_->boardCards[c] = boardCards[c];
  }
}

ACPCGame::ACPCState::~ACPCState() = default;
ACPCGame::~ACPCGame() = default;
}  // namespace open_spiel::universal_poker::acpc_cpp
