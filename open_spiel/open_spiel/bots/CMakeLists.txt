set (BOT_SOURCES
  gin_rummy/simple_gin_rummy_bot.cc
  gin_rummy/simple_gin_rummy_bot.h
  human/human_bot.cc
  human/human_bot.h
)

if (NOT WIN32)
  # UCI bot not supported on Windows.
  set (BOT_SOURCES ${BOT_SOURCES}
       uci/uci_bot.cc
       uci/uci_bot.h
  )
endif()

add_library (bots OBJECT ${BOT_SOURCES})

add_subdirectory(gin_rummy)
add_subdirectory(human)

if (NOT WIN32)
  # UCI bot not supported on Windows.
  add_subdirectory(uci)
endif()

if (OPEN_SPIEL_BUILD_WITH_ROSHAMBO)
  add_subdirectory(roshambo)
endif()

if (OPEN_SPIEL_BUILD_WITH_XINXIN)
  add_subdirectory(xinxin)
endif()

