set(HEADER_FILES
  acpc_cpp/acpc_game.h
  logic/betting_tree.h
  logic/card_set.h
)

set(CLIB_FILES
  acpc/project_acpc_server/game.h
  acpc/project_acpc_server/game.c
  acpc/project_acpc_server/rng.h
  acpc/project_acpc_server/rng.c
)

set(SOURCE_FILES
  acpc_cpp/acpc_game.cc
  logic/betting_tree.cc
  logic/card_set.cc
)

add_library(universal_poker_clib OBJECT ${CLIB_FILES} )
set_target_properties(universal_poker_clib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The library contains header and source files.
add_library(universal_poker_lib OBJECT
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

add_executable(universal_poker_acpc_cpp_test acpc_cpp/acpc_game_test.cc ${SOURCE_FILES}
        $<TARGET_OBJECTS:tests>)
target_link_libraries(universal_poker_acpc_cpp_test universal_poker_clib)

add_test(universal_poker_acpc_cpp_test universal_poker_acpc_cpp_test)

add_executable(universal_poker_betting_tree_test logic/betting_tree_test.cc ${SOURCE_FILES}
        $<TARGET_OBJECTS:tests>)
target_link_libraries(universal_poker_betting_tree_test universal_poker_clib)

add_test(universal_poker_betting_tree_test universal_poker_betting_tree_test)

add_executable(universal_poker_card_set_test logic/card_set_test.cc ${SOURCE_FILES}
        $<TARGET_OBJECTS:tests>)
target_link_libraries(universal_poker_card_set_test universal_poker_clib)

add_test(universal_poker_card_set_test universal_poker_card_set_test)

