add_subdirectory(human)

if (OPEN_SPIEL_BUILD_WITH_ROSHAMBO)
  add_subdirectory(roshambo)
endif()

if (OPEN_SPIEL_BUILD_WITH_XINXIN)
  add_subdirectory(xinxin)
endif()
