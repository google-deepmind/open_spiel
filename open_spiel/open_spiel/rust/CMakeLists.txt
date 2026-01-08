set(RUST_BINDINGS ${RUST_BINDINGS}
  src/rust_open_spiel.cc
  src/rust_open_spiel.h
)

set(RUST_API_FILES
  Cargo.toml
  build.rs
  src/rust_open_spiel.rs
  src/example.rs
)

# Note: needs to be SHARED rather than MODULE to work on MacOS
add_library(rust_spiel SHARED ${RUST_BINDINGS} ${OPEN_SPIEL_OBJECTS})

# Copy the files keeping the directories intact
foreach(rust_api_file IN LISTS RUST_API_FILES)
  get_filename_component(file_dir ${rust_api_file} DIRECTORY)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${file_dir})
  file(COPY ${rust_api_file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${file_dir})
endforeach(rust_api_file)

add_custom_target(rust_bindgen ALL $ENV{HOME}/.cargo/bin/bindgen ${CMAKE_CURRENT_SOURCE_DIR}/src/rust_open_spiel.h -o ${CMAKE_CURRENT_BINARY_DIR}/src/open_spiel_bindings.rs
                  DEPENDS rust_spiel)

add_custom_target(rust_open_spiel ALL cargo build
                  DEPENDS rust_spiel rust_bindgen
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME rust_example_test COMMAND cargo run --example example)
set_property(TEST rust_example_test
    PROPERTY ENVIRONMENT
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR};
    TEST_SRCDIR=${CMAKE_CURRENT_BINARY_DIR})
