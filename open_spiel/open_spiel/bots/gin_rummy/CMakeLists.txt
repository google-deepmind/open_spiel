add_executable (simple_gin_rummy_bot_example simple_gin_rummy_bot_example.cc
                ${OPEN_SPIEL_OBJECTS})

add_executable (simple_gin_rummy_bot_test simple_gin_rummy_bot_test.cc
                ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:tests>)
add_test(simple_gin_rummy_bot_test simple_gin_rummy_bot_test)
