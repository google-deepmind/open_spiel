# Now we can use #include "open_spiel/spiel.h"
include_directories(../..)

set(PUBLIC_STATES_CORE_FILES
  public_states.h
  public_states.cc
  )

add_library(public_states_core OBJECT ${PUBLIC_STATES_CORE_FILES})

# Just the minimal API: no games or algorithms.
set(PUBLIC_STATES_CORE_OBJECTS $<TARGET_OBJECTS:public_states_core>)

set(PUBLIC_STATES_OBJECTS
  $<TARGET_OBJECTS:public_states_core>
  $<TARGET_OBJECTS:public_states_games>
  $<TARGET_OBJECTS:public_states_algorithms>
  $<TARGET_OBJECTS:public_states_game_transforms>
  )

add_library(public_states OBJECT ${PUBLIC_STATES_OBJECTS})

add_subdirectory(games)
add_subdirectory(algorithms)
add_subdirectory(game_transforms)
