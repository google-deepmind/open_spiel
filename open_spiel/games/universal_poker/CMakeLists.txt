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
        )

set(CLIB_FILES
        ACPC/game.h
        ACPC/game.c
        ACPC/rng.h
        ACPC/rng.c
        )

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
        )

add_library(universal_poker_clib ${CLIB_FILES} )
set_target_properties(universal_poker_clib PROPERTIES POSITION_INDEPENDENT_CODE ON)

include_directories(.)

# The library contains header and source files.
add_library(universal_poker_lib STATIC
    ${SOURCE_FILES}
    ${HEADER_FILES}
)