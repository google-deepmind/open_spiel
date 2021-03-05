add_library(stockfish OBJECT
        stockfish_bot.cc
        stockfish_bot.h
        )

add_executable (stockfish_bot_test stockfish_bot_test.cc ${OPEN_SPIEL_OBJECTS}
        $<TARGET_OBJECTS:tests>)
add_test(stockfish_bot_test stockfish_bot_test)