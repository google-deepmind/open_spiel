set(HEADER_FILES
        BettingTree/BettingNode.h
        BettingTree/BettingTree.h

        GameTree/GameTree.h
        GameTree/GameNode.h

        CardTree/CardSet.h
        CardTree/CardSetIndex.h
        CardTree/CardTree.h
        CardTree/CardNode.h

        PokerGame/PokerGame.h
        PokerGame/PokerGameState.h

        acpc_cpp/acpc_state.h)

set(CLIB_FILES
        acpc/game.h
        acpc/game.c
        acpc/rng.h
        acpc/rng.c)

set(SOURCE_FILES
        BettingTree/BettingNode.cpp
        BettingTree/BettingTree.cpp

        GameTree/GameTree.cpp
        GameTree/GameNode.cpp

        CardTree/CardSet.cpp
        CardTree/CardSetIndex.cpp
        CardTree/CardTree.cpp
        CardTree/CardNode.cpp

        PokerGame/PokerGameState.cpp
        PokerGame/PokerGame.cpp

        acpc_cpp/acpc_state.cc
        acpc_cpp/acpc_game.cc)

add_library(universal_poker_clib ${CLIB_FILES} )
set_target_properties(universal_poker_clib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The library contains header and source files.
add_library(universal_poker_lib STATIC
    ${SOURCE_FILES}
    ${HEADER_FILES}
)