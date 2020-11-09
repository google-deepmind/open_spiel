add_library (tests OBJECT
  basic_tests.h
  basic_tests.cc
)
target_include_directories (tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(spiel_test spiel_test.cc
               $<TARGET_OBJECTS:tests> ${OPEN_SPIEL_OBJECTS})
add_test(spiel_test spiel_test)

add_executable(action_view_test action_view_test.cc ${OPEN_SPIEL_OBJECTS}
  $<TARGET_OBJECTS:tests>)
add_test(action_view_test action_view_test)
