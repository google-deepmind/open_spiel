find_package(JlCxx)

IF (${JlCxx_FOUND})
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
ELSE (${JlCxx_FOUND})
    MESSAGE(WARNING "BUILD_WITH_JULIA is set to ON, but JlCxx not found. Skipped building the Julia wrapper.")
ENDIF (${JlCxx_FOUND})