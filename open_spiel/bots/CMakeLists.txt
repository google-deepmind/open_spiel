if (BUILD_WITH_ROSHAMBO)
  add_subdirectory(roshambo)
endif()
if (BUILD_WITH_XINXIN)
  add_subdirectory(xinxin)
endif()

add_subdirectory(gin_rummy)

add_library (bots OBJECT
  gin_rummy/simple_bot.cc
  gin_rummy/simple_bot.h
)

