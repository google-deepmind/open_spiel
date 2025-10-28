find_package(JlCxx REQUIRED)

add_library(spieljl SHARED wrapper/spieljl.cc ${OPEN_SPIEL_OBJECTS})
target_link_libraries(spieljl PRIVATE JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl)
set_target_properties(spieljl PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)

install(TARGETS spieljl
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Disabling due to how long it takes to run.
# https://github.com/google-deepmind/open_spiel/issues/1377
# add_test(NAME julia_test COMMAND julia --project=${CMAKE_CURRENT_SOURCE_DIR}
#                                       -e "using Pkg; Pkg.build(); Pkg.test()")
