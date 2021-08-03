
add_executable(random_uci_bot random_uci_bot.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:bots>)

add_executable(uci_bot_test uci_bot_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(NAME uci_bot_test
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/uci_bot_test
         --binary ${CMAKE_CURRENT_BINARY_DIR}/random_uci_bot)
