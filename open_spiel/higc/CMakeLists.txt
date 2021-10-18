add_library (higc OBJECT
  channel.cc
  channel.h
  subprocess.h
  utils.cc
  utils.h
  referee.cc
  referee.h
)

add_executable(tournament tournament.cc ${OPEN_SPIEL_OBJECTS})
add_executable(random_bot bots/random_bot.cc ${OPEN_SPIEL_OBJECTS})
add_executable(referee_test referee_test.cc ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:tests>)

# This is causing sporadic failures. Disabling until we fix it (see #715)
# add_test(referee_test referee_test
#        --bots_dir=${CMAKE_CURRENT_SOURCE_DIR}/bots)
