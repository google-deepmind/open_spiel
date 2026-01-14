# Note: GO API is disabled in ../CMakeLists.txt for now due to failing tests:
# # openspiel_test
# [openspiel_test]
# ./example_leduc_test.go:14:1: ExampleLeduc refers to unknown identifier: Leduc
# ./example_test.go:10:1: ExampleTicTacToe refers to unknown identifier: TicTacToe
# ./example_test.go:138:1: ExampleLoadParametrizedGame refers to unknown identifier: LoadParametrizedGame
#    FAIL openspiel [build failed]

set(GO_BINDINGS ${GO_BINDINGS}
  go_open_spiel.cc
  go_open_spiel.h
)

set(GO_API_FILES ${GO_API_FILES}
  examples/example.go
  example_leduc_test.go
  example_test.go
  go_open_spiel.h
  openspiel.go
)

# Note: needs to be SHARED rather than MODULE to work on MacOS
add_library(gospiel SHARED ${GO_BINDINGS} ${OPEN_SPIEL_OBJECTS})

# Copy the files keeping the directories intact
foreach(go_api_file IN LISTS GO_API_FILES)
  get_filename_component(file_dir ${go_api_file} DIRECTORY)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${file_dir})
  file(COPY ${go_api_file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${file_dir})
endforeach(go_api_file)

execute_process(COMMAND go mod init openspiel
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})


# add_test(NAME gospiel_test COMMAND go test -v)
# set_property(TEST gospiel_test
#     PROPERTY ENVIRONMENT
#     LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR};
#     TEST_SRCDIR=${CMAKE_CURRENT_BINARY_DIR})
