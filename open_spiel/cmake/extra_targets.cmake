# Tests and binaries that exist only in the CMake build. This file is
# hand-written and included at the end of the root CMakeLists.txt, after all
# generated targets are defined.

## cc_binary targets that CMake also runs as smoke tests. The test names get a
## `_binary` suffix so they differ from the generated cc_test names (several
## examples have both a binary and a test with the same base name).
macro(openspiel_binary_test TEST_NAME TARGET)
  add_test(NAME ${TEST_NAME}_binary COMMAND ${TARGET} ${ARGN})
endmacro()

openspiel_binary_test(benchmark_game_test open_spiel_examples_benchmark_game
  --game=tic_tac_toe --sims=100 --attempts=2)
openspiel_binary_test(cfr_example_test open_spiel_examples_cfr_example)
openspiel_binary_test(example_test open_spiel_examples_example
  --game=tic_tac_toe --seed=0)
openspiel_binary_test(matrix_example_test open_spiel_examples_matrix_example)
openspiel_binary_test(mcts_example_test open_spiel_examples_mcts_example)
openspiel_binary_test(minimax_example_test open_spiel_examples_minimax_example)
openspiel_binary_test(policy_iteration_example_test
  open_spiel_examples_policy_iteration_example)
openspiel_binary_test(value_iteration_example_test
  open_spiel_examples_value_iteration_example)
if (OPEN_SPIEL_BUILD_WITH_ACPC)
  openspiel_binary_test(universal_poker_mccfr_acpc_gamedef_example_test
    open_spiel_examples_universal_poker_mccfr_acpc_gamedef_example)
endif()

## Binaries without a BUILD target.
add_executable(is_mcts_gwhist examples/is_mcts_gwhist.cc)
target_link_libraries(is_mcts_gwhist PRIVATE open_spiel_full)

## The shared library (see BUILD_SHARED_LIB in the root CMakeLists.txt).
if (BUILD_SHARED_LIB)
  add_executable(shared_lib_test tests/shared_lib_test.cc)
  add_executable(shared_library_example examples/shared_library_example.cc)
  foreach(target IN ITEMS shared_lib_test shared_library_example)
    target_link_libraries(${target} PRIVATE open_spiel)
    if (WIN32)
      # The DLL exports no symbols, so link the objects directly as well.
      target_link_libraries(${target} PRIVATE open_spiel_full)
    endif()
  endforeach()
  add_test(shared_lib_test shared_lib_test)
endif()
