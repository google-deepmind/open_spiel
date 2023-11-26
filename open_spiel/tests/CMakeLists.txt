add_library (tests OBJECT
  basic_tests.h
  basic_tests.cc
  console_play_test.h
  console_play_test.cc
)
target_include_directories (tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_open_spiel_test(spiel_test spiel_test.cc)

add_open_spiel_test(action_view_test action_view_test.cc)

if (WIN32)
  add_executable(shared_lib_test shared_lib_test.cc ${OPEN_SPIEL_OBJECTS})
else()
  add_executable(shared_lib_test shared_lib_test.cc)
endif()
target_link_libraries(shared_lib_test open_spiel)
add_test(shared_lib_test shared_lib_test)
