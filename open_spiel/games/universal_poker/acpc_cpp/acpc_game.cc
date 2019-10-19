//
// Created by dj on 10/19/19.
//

#include "acpc_game.h"
#include <stdio.h>
#include <sstream>
#include <assert.h>

extern "C"
{
#include "open_spiel/games/universal_poker/acpc/game.h"
};

struct open_spiel::universal_poker::acpc_cpp::Game : public ::Game{};
struct open_spiel::universal_poker::acpc_cpp::State : public ::State{};
struct open_spiel::universal_poker::acpc_cpp::Action : public ::Action{};



open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCGame(const std::string& gameDef)
    : handId_(0)
{
    char buf[STRING_BUFFERSIZE];
    gameDef.copy(buf, STRING_BUFFERSIZE);

    FILE* f = fmemopen(&buf, STRING_BUFFERSIZE, "r");
    acpcGame_ = (Game*)readGame(f);
    fclose(f);
}

open_spiel::universal_poker::acpc_cpp::ACPCGame::~ACPCGame() {
    free(acpcGame_);
}

std::string open_spiel::universal_poker::acpc_cpp::ACPCGame::ToString() const {
    char buf[STRING_BUFFERSIZE];
    FILE* f = fmemopen(&buf, STRING_BUFFERSIZE, "w");
    printGame(f, acpcGame_);
    std::ostringstream result;
    rewind(f);
    result << buf;
    fclose(f);

    return result.str();
}

bool open_spiel::universal_poker::acpc_cpp::ACPCGame::IsLimitGame() const {
    return acpcGame_->bettingType == limitBetting;
}

uint8_t open_spiel::universal_poker::acpc_cpp::ACPCGame::GetNbPlayers() const {
    return acpcGame_->numPlayers;
}

uint8_t open_spiel::universal_poker::acpc_cpp::ACPCGame::GetNbHoleCardsRequired() const {
    return acpcGame_->numHoleCards;
}

uint8_t open_spiel::universal_poker::acpc_cpp::ACPCGame::GetNbBoardCardsRequired(uint8_t round) const {
    assert(round < acpcGame_->numRounds);

    uint8_t nbCards = 0;
    for( int r = 0; r <= round; r++){
        nbCards += acpcGame_->numBoardCards[r];
    }

    return nbCards;
}



open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::ACPCState(
        open_spiel::universal_poker::acpc_cpp::ACPCGame &game)
        :game_(game)
        {

    acpcState_ = new State();
    initState(game.acpcGame_, game.handId_++, acpcState_);
}

open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::~ACPCState() {
    delete acpcState_;
}

std::string open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::ToString() const {
    char buf[STRING_BUFFERSIZE];
    printState(game_.acpcGame_, acpcState_, STRING_BUFFERSIZE, buf);
    return std::string(buf);
}

int open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::RaiseIsValid(int32_t *minSize, int32_t *maxSize) const{
    return raiseIsValid(game_.acpcGame_, acpcState_, minSize, maxSize);
}

int open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::IsValidAction(const int tryFixing,
                                                                              const ACPCAction& action) const{
    return isValidAction(game_.acpcGame_, acpcState_, tryFixing, action.acpcAction_);
}

void open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::DoAction(
        const open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::ACPCAction &action) {

    doAction(game_.acpcGame_, action.acpcAction_, acpcState_ );
}

double open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::ValueOfState(const uint8_t player) const{
    valueOfState(game_.acpcGame_, acpcState_, player );
}

int open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::IsFinished() const{
    return stateFinished(acpcState_);
}

uint32_t open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::MaxSpend() const {
    return acpcState_->maxSpent;
}

uint8_t open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::GetRound() const {
    return acpcState_->round;
}

uint8_t open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::NumFolded() const {
    return numFolded(game_.acpcGame_, acpcState_);
}


open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::ACPCAction::ACPCAction(
        open_spiel::universal_poker::acpc_cpp::ACPCGame* game,
        ACPCAction::ACPCActionType type, int32_t size)
        :game_(game) {

    acpcAction_ = new Action();
    SetActionAndSize(type, size);
}


open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::ACPCAction::~ACPCAction() {
    delete(acpcAction_);
}

std::string open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::ACPCAction::ToString() const{
    char buf[STRING_BUFFERSIZE];
    printAction(game_->acpcGame_, acpcAction_, STRING_BUFFERSIZE, buf);
    return std::string(buf);
}

void open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::ACPCAction::SetActionAndSize(
        open_spiel::universal_poker::acpc_cpp::ACPCGame::ACPCState::ACPCAction::ACPCActionType type, int32_t size) {
    acpcAction_->size = size;
    switch(type) {
        case ACTION_CALL:
            acpcAction_->type = a_call;
            break;
        case ACTION_FOLD:
            acpcAction_->type = a_fold;
            break;
        case ACTION_RAISE:
            acpcAction_->type = a_raise;
            break;
        default:
            acpcAction_->type = a_invalid;
            break;
    }
}
