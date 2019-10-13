//
// Created by dennis on 01.09.19.
//

#include <iostream>
#include <algorithm>
#include <assert.h>
#include <bitset>
#include "PokerGameState.h"
#include "PokerGame.h"

#define NUM_ACTIONS_FCPA 4

#define ACTION_FOLD 0
#define ACTION_CALL 1
#define ACTION_BET_POT 2
#define ACTION_ALL_IN 3

char BET_NAME[5] = "fcpa";


uint32_t PokerGameState::handId = 0;

using I = uint64_t ;

auto dump(I v) { return std::bitset<sizeof(I) * __CHAR_BIT__>(v); }
I bit_twiddle_permute(I v) {
    I t = v | (v - 1);
    I w = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctzl(v) + 1));

    return w;
}

PokerGameState::PokerGameState(Game* game1, PokerGameState* parent, GameAction action)
:game(game1), actionsAllowed()
{
    for (double &r : totalReward) {
        r = 0;
    }

    if(parent == nullptr){
        //This is a really fresh game
        initState(game, handId++, &handState);
        for( auto &card : handState.boardCards) card=255;
        for (auto &holeCard : handState.holeCards) {
            for (auto &card : holeCard) card = 255;
        }

        initDeck();
        name = "STATE:Initialized";
        bettingHistory = "R";
    }
    else {
        //Copy the state from parent
        handState = parent->handState;
        deck = parent->deck;
        bettingHistory = parent->bettingHistory;

        for( uint8_t p = 0; p<this->game->numPlayers; p++) {
            handCards[p] = parent->handCards[p];
        }
        for( uint8_t r = 0; r<this->game->numRounds; r++) {
            boardCards[r] = parent->boardCards[r];
        }

        executeAction(action);
    }

    actionsAllowed = calculateActionsAllowed();
}

void PokerGameState::initDeck() {
    for (int r = MAX_RANKS - 1; r >= MAX_RANKS - game->numRanks; r--) {
        for (int s = MAX_SUITS - 1; s >= MAX_SUITS - game->numSuits; s--) {
            deck.addCard(makeCard(r, s));
        }
    }
}

std::string PokerGameState::getName() {
    return name;
}

std::vector<PokerGameState::GameAction> PokerGameState::getActionsAllowed() {
    return actionsAllowed;
}

std::vector<CardSet> PokerGameState::sampleFromDeck(uint8_t nbCards)
{
    std::vector<CardSet> combinations;

    uint64_t p = 0;
    for (int i = 0; i < nbCards; i++)
    {
        p += (1<<i);
    }

    for (I n = bit_twiddle_permute(p); n>p; p = n, n = bit_twiddle_permute(p)) {
        uint64_t combo = n & deck.cs.cards;
        if( __builtin_popcountl(combo) == nbCards ){
            CardSet cs;
            cs.cs.cards = combo;
            combinations.emplace_back(cs);
        }
    }

    //std::cout << "combinations.size() " << combinations.size() << std::endl;
    return combinations;

}

void PokerGameState::executeAction(GameAction a) {
    //This is the chance action
    if( a.actionName == 'd'){
        assert( a.cards.cs.cards != 0 );
        if( a.cardsForPlayer == PLAYER_DEALER){
            assert(boardCards[a.boardCardsForRound].cs.cards == 0);
            boardCards[a.boardCardsForRound] = a.cards;

            std::vector<int> cards = boardCards[a.boardCardsForRound].toCardArray();
            int offset = 0;
            for(int r = 0; r < a.boardCardsForRound; r++){
                offset += game->numBoardCards[r];
            }
            for(int i = 0, c=offset; i < cards.size(); i++, c++){
                deck.removeCard(static_cast<uint8_t>(cards[i]));
                handState.boardCards[c] = static_cast<uint8_t>(cards[i]);
            }
        }
        else {
            assert(handCards[a.cardsForPlayer].cs.cards == 0);
            handCards[a.cardsForPlayer] = a.cards;

            std::vector<int> cards = handCards[a.cardsForPlayer].toCardArray();
            for(int i = 0; i < cards.size(); i++){
                deck.removeCard(static_cast<uint8_t>(cards[i]));
                handState.holeCards[a.cardsForPlayer][i] = static_cast<uint8_t>(cards[i]);
            }
        }
    }

    //This is a choice action
    if( isValidAction(game, &handState, 0, &a.action) ){
        doAction(game, &a.action, &handState);
    }

    bettingHistory += a.actionName;
    char buf[MAX_LINE_LEN];
    printState(game, &handState, MAX_LINE_LEN, buf);
    name = buf;
}

std::vector<PokerGameState::GameAction> PokerGameState::calculateActionsAllowed() {

    std::vector<PokerGameState::GameAction> result;

    //Check if there are cards to be dealt
    if( handState.round == 0 ){
        for( uint8_t p = 0; p<this->game->numPlayers; p++ ){
            if(handCards[p].cs.cards == 0){
                nodeType = BettingNode::CHANCE_NODE;
                for( auto combo : sampleFromDeck(game->numHoleCards)){
                    result.emplace_back(GameAction({a_invalid, 0}, 'd', combo, p, 0));
                }
                return result;
            }
        }
    }
    else{
        for( uint8_t r = 0; r <= handState.round; r++ ){
            if(boardCards[r].cs.cards == 0 && game->numBoardCards[r] > 0){
                nodeType = BettingNode::CHANCE_NODE;
                for( auto combo : sampleFromDeck(game->numBoardCards[r])){
                    result.emplace_back(GameAction({a_invalid, 0}, 'd', combo, PLAYER_DEALER, r));
                }
                return result;
            }
        }
    }

    //Check if game is finished
    if( handState.finished ){
        if( numFolded(game, &handState) >= game->numPlayers - 1 ){
            nodeType = BettingNode::TERMINAL_FOLD_NODE;
        }
        else{
            nodeType =BettingNode::TERMINAL_SHOWDOWN_NODE;
        }

        for( uint8_t p=0; p < game->numPlayers; p++ ){

            totalReward[p] = valueOfState(game, &handState, p) / (double)game->blind[p] ;
        }


        return result;

    }

    //Check if there are choice actions to be done
    Action actions[NUM_ACTIONS_FCPA]{{a_fold, 0},{a_call, 0},{a_raise, 0},{a_invalid, 0}} ;
    if( game->bettingType == noLimitBetting ){
        int32_t opponentBet = handState.maxSpent;
        int32_t minRaise = 0, maxRaise = 0;
        if( raiseIsValid(game, &handState, &minRaise, &maxRaise) ) {
            int32_t potBet = opponentBet + opponentBet * 2;
            if( potBet < minRaise || potBet >= maxRaise ){
                potBet = maxRaise;
                actions[ACTION_ALL_IN] = {a_raise, potBet};
            }
            else {
                actions[ACTION_BET_POT] = {a_raise, potBet};
            }

            if( maxRaise > potBet ) {
                actions[ACTION_ALL_IN] = {a_raise, maxRaise};
            }
        }
    }
    for (int i = 0; i < NUM_ACTIONS_FCPA; i++) {
        if(isValidAction(game, &handState, 0, &actions[i]) ){
            nodeType = BettingNode::CHOICE_NODE;
            result.emplace_back(GameAction(actions[i], BET_NAME[i]));
        }
    }

    return result;

}

BettingNode::BettingNodeType PokerGameState::getType() {
    return nodeType;
}

uint32_t PokerGameState::getNbActions() {
    return static_cast<uint32_t>(actionsAllowed.size());
}

uint32_t PokerGameState::getPlayer() {
    if( nodeType == BettingNode::CHOICE_NODE ) {
        return currentPlayer(game, &handState);
    }

    return PLAYER_DEALER;
}

double PokerGameState::getTotalReward(int player) {
    assert(player >= 0);
    assert(player < game->numPlayers);
    assert(nodeType == BettingNode::TERMINAL_SHOWDOWN_NODE || nodeType == BettingNode::TERMINAL_FOLD_NODE);
    return this->totalReward[player];
}

uint64_t PokerGameState::getCardState(int player) {
    assert(player >= 0);
    assert(player < game->numPlayers || player == PLAYER_DEALER);

    uint64_t cardState = 0;
    if( player == PLAYER_DEALER) {
        for (int r = 0; r <= handState.round; r++) {
            cardState += boardCards[r].cs.cards;
        }
    }
    else {
        cardState += handCards[player].cs.cards;
    }

    return cardState;
}

uint64_t PokerGameState::getBetSize(int player) {
    assert(player >= 0);
    assert(player < game->numPlayers || player == PLAYER_DEALER);

    uint64_t betSize = 0;
    if( player == PLAYER_DEALER) {
       betSize += handState.maxSpent;
    }
    else {
        betSize += handState.spent[player];
    }


    return betSize;
}

uint64_t PokerGameState::getCardsInDeck() {
    return __builtin_popcountl( deck.cs.cards );
}

std::string PokerGameState::getBettingHistory() {
    return bettingHistory;
};

