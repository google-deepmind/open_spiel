
add_library (higc OBJECT
        base64.cc
        base64.h
        subprocess.hpp
        utils.cc
        utils.h
        referee.cc
        referee.h
        fdstream.hpp
        )

find_package(Threads REQUIRED)
add_executable(tournament tournament.cc ${OPEN_SPIEL_OBJECTS})
target_link_libraries(tournament Threads::Threads)

add_executable(random_bot bots/random_bot.cc ${OPEN_SPIEL_OBJECTS})

add_executable(example_read example_read.cc ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:tests>)
add_executable(referee_test referee_test.cc ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:tests>)
add_test(referee_test referee_test --bots_dir=${CMAKE_CURRENT_SOURCE_DIR}/bots)
