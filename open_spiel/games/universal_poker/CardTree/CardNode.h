//
// Created by Dennis Jöst on 01.05.18.
//

#ifndef DEEPSTACK_CPP_TERMINALEQUITY_H
#define DEEPSTACK_CPP_TERMINALEQUITY_H

#ifdef PYBIND_ENABLED
#include <pybind11/eigen.h>
#else
#include <eigen3/Eigen/Dense>
#endif

#include "open_spiel/games/universal_poker/CardTree/CardSet.h"
#include "open_spiel/games/universal_poker/CardTree/CardSetIndex.h"


typedef  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
        Eigen::OuterStride<Eigen::Dynamic> > MatrixMapped;

class CardTree;

class CardNode {
private:
    uint32_t index;
    CardSet boardCards;
    CardSetIndex* handCards;
    CardTree* cardTree;
    size_t handCount;
    uint8_t round;

    bool callMatrixCalculated;
    Eigen::MatrixXf callMatrix;

    bool foldMatrixCalculated;
    Eigen::MatrixXf foldMatrix;

    bool handMaskCalculated;
    Eigen::VectorXf handMask;

public:
    CardNode();
    CardNode(CardSet boardCards, CardSetIndex* handCards, CardTree* cardTree, uint8_t round, uint32_t index);

    size_t getHandCount();
    CardSet getBoardCards();
    uint8_t getRound();

    Eigen::MatrixXf viewCallMatrix();
    Eigen::MatrixXf viewFoldMatrix();
    Eigen::VectorXf viewHandMask();

    CardSetIndex *getHandCards() const;
    uint32_t getIndex() const;

protected:
    void calculateCallMatrixLastRound();
    void calculateCallMatrix();
    void calculateFoldMatrix();

    bool handsBlocking(int p1, int p2);

    void calculateHandMask();
};



#endif //DEEPSTACK_CPP_TERMINALEQUITY_H
