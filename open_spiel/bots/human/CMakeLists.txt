add_library(human_bot OBJECT
    human_bot.cc
    human_bot.h
)

add_executable (human_bot_test human_bot_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests> $<TARGET_OBJECTS:human_bot>)
add_test(human_bot_test human_bot_test)
