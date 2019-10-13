//
// Created by dennis on 01.09.19.
//

#ifndef DEEPSTACK_CPP_POKERGAMESTATE_H
#define DEEPSTACK_CPP_POKERGAMESTATE_H


#include "open_spiel/games/universal_poker/CardTree/CardSet.h"
#include "open_spiel/games/universal_poker/BettingTree/BettingNode.h"

#define PLAYER_DEALER 255

class PokerGameState {
private:
    static uint32_t handId;

public:
    struct GameAction {
        Action action;
        char actionName;
        CardSet cards;
        uint8_t cardsForPlayer;
        uint8_t boardCardsForRound;

        GameAction(Action a, char n, CardSet c, uint8_t p, uint8_t r)
        :action(a), actionName(n), cards(c), cardsForPlayer(p), boardCardsForRound(r) {}

        GameAction(Action a, char n)
        :action(a), actionName(n), cards(), cardsForPlayer(PLAYER_DEALER), boardCardsForRound(0){}

        GameAction()
        :action({a_invalid, 0}), cards(), cardsForPlayer(PLAYER_DEALER), boardCardsForRound(0) {}
    };

private:
    Game* game;
    State handState;
    CardSet deck;
    CardSet handCards[MAX_PLAYERS];
    CardSet boardCards[MAX_ROUNDS];

    std::string name;
    std::vector<GameAction> actionsAllowed;
    double totalReward[MAX_PLAYERS];

    BettingNode::BettingNodeType nodeType;
    std::string bettingHistory;

public:
    PokerGameState(Game* game, PokerGameState* parent, GameAction action);
    std::string getName();
    BettingNode::BettingNodeType getType();
    uint32_t getNbActions();
    uint32_t getPlayer();
    double getTotalReward(int player);
    uint64_t getCardState(int player);
    uint64_t getBetSize(int player);
    std::string getBettingHistory();
    uint64_t getCardsInDeck();

    std::vector<GameAction> getActionsAllowed();

protected:
    void initDeck();
    std::vector<CardSet> sampleFromDeck(uint8_t nbCards);

    void executeAction(GameAction a);
    std::vector<PokerGameState::GameAction> calculateActionsAllowed();
};


#endif //DEEPSTACK_CPP_POKERGAMESTATE_H
