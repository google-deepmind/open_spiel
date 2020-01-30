add_library (utils OBJECT
  circular_buffer.h
  tensor_view.h
)
target_include_directories (utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(circular_buffer_test circular_buffer_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(circular_buffer_test circular_buffer_test)

add_executable(tensor_view_test tensor_view_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(tensor_view_test tensor_view_test)
