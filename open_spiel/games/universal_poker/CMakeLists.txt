project(${CMAKE_PROJECT_NAME}_lib)


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

add_library(${CMAKE_PROJECT_NAME}_clib ${CLIB_FILES} )
set_target_properties(${CMAKE_PROJECT_NAME}_clib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The library contains header and source files.
add_library(${CMAKE_PROJECT_NAME}_lib STATIC
    ${SOURCE_FILES}
    ${HEADER_FILES}
)