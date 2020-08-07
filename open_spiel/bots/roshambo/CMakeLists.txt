add_library(roshambo OBJECT
  roshambo/bot_map.cc
  roshambo/bot_map.h
  roshambo/rsb-ts1-modified.c
  roshambo_bot.cc
  roshambo_bot.h
)
target_compile_options(roshambo PUBLIC -w)
target_include_directories(roshambo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(roshambo_bot_test roshambo_bot_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(roshambo_bot_test roshambo_bot_test)
