# Now we can use #include "open_spiel/spiel.h"
include_directories(../..)

set(PUBLIC_STATES_CORE_FILES
  policy.h
  public_states.h
  public_states.cc
  )

# Just the minimal API: no games or algorithms.
add_library(public_states_core OBJECT ${PUBLIC_STATES_CORE_FILES})

add_subdirectory(games)
add_subdirectory(algorithms)
add_subdirectory(game_transforms)
