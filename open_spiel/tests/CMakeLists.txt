add_library (tests OBJECT
  basic_tests.h
  basic_tests.cc
)
target_include_directories (tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(spiel_test spiel_test.cc
               $<TARGET_OBJECTS:tests> ${OPEN_SPIEL_OBJECTS})
add_test(spiel_test spiel_test)
