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

if (BUILD_SHARED_LIB)
  add_executable(shared_lib_test shared_lib_test.cc)
  target_link_libraries(shared_lib_test open_spiel)
  add_test(shared_lib_test shared_lib_test)
endif()

