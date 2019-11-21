find_package(JlCxx REQUIRED)

add_library(spieljl SHARED spieljl.cc ${OPEN_SPIEL_OBJECTS})
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