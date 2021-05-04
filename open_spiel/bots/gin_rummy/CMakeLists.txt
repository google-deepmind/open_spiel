add_executable (simple_bot_example simple_bot_example.cc ${OPEN_SPIEL_OBJECTS})

add_executable (simple_bot_test simple_bot_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:tests>)
add_test(simple_bot_test simple_bot_test)
