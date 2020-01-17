add_library (utils OBJECT
  tensor_view.h
)
target_include_directories (utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(tensor_view_test tensor_view_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(tensor_view_test tensor_view_test)
