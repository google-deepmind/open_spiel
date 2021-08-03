add_executable (human_bot_test human_bot_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(human_bot_test human_bot_test)
