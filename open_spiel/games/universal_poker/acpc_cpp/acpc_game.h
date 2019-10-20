//
// Created by dj on 10/19/19.
//

#ifndef OPEN_SPIEL_ACPC_GAME_H
#define OPEN_SPIEL_ACPC_GAME_H

static const int STRING_BUFFERSIZE = 4096;

#include <string>
#include <memory>

namespace open_spiel::universal_poker::acpc_cpp {
    struct Game;
    struct State;
    struct Action;

    class ACPCGame {
    public:
        class ACPCState {
        public:
            enum ACPCActionType {
                ACPC_FOLD, ACPC_CALL, ACPC_RAISE, ACPC_INVALID
            };

        public:
            ACPCState(ACPCGame* game);
            ACPCState(const ACPCState& game);
            virtual ~ACPCState();

            void SetHoleAndBoardCards(uint8_t holeCards[10][3], uint8_t boardCards[7], uint8_t nbHoleCards[10], uint8_t nbBoardCards) const;

            uint8_t CurrentPlayer() const;

            virtual bool IsFinished() const;
            int RaiseIsValid(int32_t *minSize, int32_t *maxSize) const;
            int IsValidAction(const ACPCActionType actionType, const int32_t size) const;
            void DoAction(const ACPCActionType actionType, const int32_t size);
            double ValueOfState( const uint8_t player ) const;
            uint32_t MaxSpend() const;
            uint8_t GetRound() const;
            uint8_t NumFolded() const;
            uint32_t Money(const uint8_t player) const;
            uint32_t Ante(const uint8_t player) const;
            std::string ToString() const;
            std::string BettingSequence(uint8_t round) const;

        private:
            ACPCGame* game_;
            std::unique_ptr<State> acpcState_;
        };


    public:
        ACPCGame(const std::string &gameDef);
        ACPCGame(const ACPCGame& game);
        virtual ~ACPCGame();

        std::string ToString() const;
        bool IsLimitGame() const;
        uint8_t GetNbPlayers() const;
        uint8_t GetNbHoleCardsRequired() const;
        uint8_t GetNbBoardCardsRequired(uint8_t round) const;
        uint8_t NumSuitsDeck() const;
        uint8_t NumRanksDeck() const;
        uint32_t StackSize(uint8_t player) const;
        uint8_t GetTotalNbBoardCards() const;
    private:
        std::unique_ptr<Game> acpcGame_;
        uint32_t handId_;


    };


}


#endif //OPEN_SPIEL_ACPC_GAME_H
