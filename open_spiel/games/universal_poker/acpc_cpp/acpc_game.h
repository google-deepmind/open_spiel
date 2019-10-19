//
// Created by dj on 10/19/19.
//

#ifndef OPEN_SPIEL_ACPC_GAME_H
#define OPEN_SPIEL_ACPC_GAME_H

static const int STRING_BUFFERSIZE = 2048;

#include <string>

namespace open_spiel::universal_poker::acpc_cpp {
    struct Game;
    struct State;
    struct Action;

    class ACPCGame {
    public:
        class ACPCState {
        public:
            class ACPCAction {
                friend ACPCState;
            public:
                enum ACPCActionType {
                    ACTION_FOLD, ACTION_CALL, ACTION_RAISE, ACTION_INVALID
                };

            private:
                ACPCGame* game_ ;
                Action* acpcAction_;

            public:
                explicit ACPCAction(ACPCGame* game, ACPCAction::ACPCActionType type, int32_t size);
                ACPCAction(const ACPCAction&) = delete;
                virtual ~ACPCAction();
                std::string ToString() const;
                void SetActionAndSize(ACPCAction::ACPCActionType type, int32_t size);
            };




        public:
            explicit ACPCState(ACPCGame &game);
            ACPCState(const ACPCState&) = delete;

            virtual ~ACPCState();

            int IsFinished() const;
            int RaiseIsValid(int32_t *minSize, int32_t *maxSize) const;
            int IsValidAction(const int tryFixing, const ACPCAction& action) const;
            void DoAction(const ACPCAction& action);
            double ValueOfState( const uint8_t player ) const;
            uint32_t MaxSpend() const;
            uint8_t GetRound() const;
            uint8_t NumFolded() const;

            std::string ToString() const;
        private:
            ACPCGame &game_;
            State *acpcState_;
        };


    public:
        explicit ACPCGame(const std::string &gameDef);
        ACPCGame(const ACPCGame&) = delete;

        virtual ~ACPCGame();

        std::string ToString() const;
        bool IsLimitGame() const;
        uint8_t GetNbPlayers() const;
        uint8_t GetNbHoleCardsRequired() const;
        uint8_t GetNbBoardCardsRequired(uint8_t round) const;

    private:
        Game *acpcGame_;
        uint32_t handId_;

    };


}


#endif //OPEN_SPIEL_ACPC_GAME_H
