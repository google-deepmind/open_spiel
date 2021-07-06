add_subdirectory(human)

if (OPEN_SPIEL_BUILD_WITH_ROSHAMBO)
  add_subdirectory(roshambo)
endif()

if (OPEN_SPIEL_BUILD_WITH_XINXIN)
  add_subdirectory(xinxin)
endif()

add_subdirectory(gin_rummy)

add_library (bots OBJECT
  gin_rummy/simple_gin_rummy_bot.cc
  gin_rummy/simple_gin_rummy_bot.h
)

