//
// Created by Dennis Jöst on 03.05.18.
//

#ifndef DEEPSTACK_CPP_GAMETREE_H
#define DEEPSTACK_CPP_GAMETREE_H

#define MAX_GAME_NODES (16*1024*1024)

#include "open_spiel/games/universal_poker/BettingTree/BettingTree.h"
#include "open_spiel/games/universal_poker/CardTree/CardTree.h"
#include <random>
#include <memory>
#include <mutex>
#include "GameNode.h"

class GameTree {
private:
    Game game;
    std::unique_ptr<BettingTree> bettingTree;
    std::unique_ptr<CardTree> cardTree;

    GameNode* root;
    GameNode gameNodes[MAX_GAME_NODES];
    size_t gameNodeCount;

    std::string gameName;

    std::mt19937 generator;

public:
    explicit GameTree(Game* game, std::string gameName);

    size_t getGameNodeCount() const;
    void buildGameTree(CardSet initialBoard, int bettingNodeIdx, int maxDepth);
    void buildGameTreeWithSeq(CardSet initialBoard, std::string bettingSequence, int maxDepth);
    void buildRandomTree(int round, int maxDepth);

    GameNode* getRoot();

    std::vector<std::string> getBettingSequencesForRound(uint8_t round);
    std::vector<CardSet> getCardSetsForRound(uint8_t round);

    BettingTree *getBettingTree() const;
    CardTree *getCardTree() const;
    std::string getMatrixHead();
    std::string getTreeAsString();
    std::string getGameName();
    uint8_t getPlayerCount();
    float getBlind(uint8_t player);
    uint32_t getHandCount();
    uint8_t getRounds();

    void writeToDot( const char* filename );

    static std::unique_ptr<GameTree> createFromGamedef(const char* fileName);

    uint32_t getBettingNodeCount();
    uint32_t getCardNodeCount();

    BettingNode *getBettingNode(uint32_t index);
    CardNode *getCardNode(uint32_t index);

private:
    GameNode *createGameNode(BettingNode *bettingNode, CardNode *cardNode, int maxDepth);
    std::string nodeToGraphviz(GameNode* node);
};


#endif //DEEPSTACK_CPP_GAMETREE_H
