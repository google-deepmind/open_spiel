# The interface to GAMUT is optional. To enable it, set the flag in
# scripts/global_variables.sh.

add_library(gamut OBJECT
  gamut.cc
  gamut.h
)

add_executable (gamut_test gamut_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(gamut_test gamut_test)
