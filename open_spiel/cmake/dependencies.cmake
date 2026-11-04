# External dependencies of OpenSpiel. This file is hand-written.
#
# Every external Bazel label that the generated CMakeLists.txt files refer to
# maps to one of the targets defined here (see
# scripts/bazel_to_cmake/label_map.py). The sources of the dependencies are
# fetched by install.sh into the directories used below.

## Abseil.
set (ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory (abseil-cpp)
# Both the canonical absl/... include paths and the vendored
# open_spiel/abseil-cpp/absl/... ones work.
include_directories (abseil-cpp)

## nlohmann/json.
add_subdirectory (json)
include_directories (json/include)

## Python and pybind11.
if (OPEN_SPIEL_BUILD_WITH_PYTHON)
  if (OPEN_SPIEL_BUILDING_WHEEL)
    # When building a wheel, need to use this. See:
    # https://github.com/joerick/cibuildwheel/issues/639#issuecomment-817872369
    message(NOTICE "Building Wheel Detected. Finding Python Interpeter Development.Module")
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  else()
    message(NOTICE "Not building wheel. Finding Python normally...")
    find_package(Python3 COMPONENTS Interpreter Development)
  endif()
  message(NOTICE "Python executable: ${Python3_EXECUTABLE}")
  message(NOTICE "Python include dirs: ${Python3_INCLUDE_DIRS}")
  message(NOTICE "Python library dirs: ${Python3_LIBRARY_DIRS}")
  if(Python3_VERSION VERSION_LESS "3.8.0")
    message(FATAL_ERROR
      "Python found ${Python3_VERSION} < 3.8.0")
  endif()

  # pybind11 headers and the Python headers. Linked by every pybind_library and
  # pybind_extension target.
  add_library (open_spiel_ext_pybind11 INTERFACE)
  add_library (open_spiel_ext::pybind11 ALIAS open_spiel_ext_pybind11)
  target_include_directories (open_spiel_ext_pybind11 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../pybind11/include
    ${Python3_INCLUDE_DIRS})
  target_link_directories (open_spiel_ext_pybind11 INTERFACE
    ${Python3_LIBRARY_DIRS})
  if (WIN32)
    # Symbols must be resolved at link time on Windows.
    target_link_libraries (open_spiel_ext_pybind11 INTERFACE Python3::Module)
  endif()
endif()

# The pybind11_abseil and pybind11_json headers. Their include directories are
# also added globally, as some non-pybind11 targets include their headers.
include_directories (pybind11_abseil pybind11_json/include)

add_library (open_spiel_ext_pybind11_abseil INTERFACE)
add_library (open_spiel_ext::pybind11_abseil ALIAS open_spiel_ext_pybind11_abseil)
target_include_directories (open_spiel_ext_pybind11_abseil INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11_abseil)

add_library (open_spiel_ext_pybind11_json INTERFACE)
add_library (open_spiel_ext::pybind11_json ALIAS open_spiel_ext_pybind11_json)
target_include_directories (open_spiel_ext_pybind11_json INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11_json/include)
target_link_libraries (open_spiel_ext_pybind11_json INTERFACE
  nlohmann_json::nlohmann_json)

## Bridge double dummy solver (always built: the bridge game uses it).
set (DDS_DIR games/bridge/double_dummy_solver)
add_library (open_spiel_ext_dds OBJECT
  ${DDS_DIR}/src/ABsearch.cpp
  ${DDS_DIR}/src/ABsearch.h
  ${DDS_DIR}/src/ABstats.cpp
  ${DDS_DIR}/src/ABstats.h
  ${DDS_DIR}/src/CalcTables.cpp
  ${DDS_DIR}/src/CalcTables.h
  ${DDS_DIR}/src/dds.cpp
  ${DDS_DIR}/src/DealerPar.cpp
  ${DDS_DIR}/src/dump.cpp
  ${DDS_DIR}/src/File.cpp
  ${DDS_DIR}/src/File.h
  ${DDS_DIR}/src/Init.cpp
  ${DDS_DIR}/src/Init.h
  ${DDS_DIR}/src/LaterTricks.cpp
  ${DDS_DIR}/src/LaterTricks.h
  ${DDS_DIR}/src/Memory.cpp
  ${DDS_DIR}/src/Memory.h
  ${DDS_DIR}/src/Moves.cpp
  ${DDS_DIR}/src/Moves.h
  ${DDS_DIR}/src/Par.cpp
  ${DDS_DIR}/src/PBN.cpp
  ${DDS_DIR}/src/PBN.h
  ${DDS_DIR}/src/PlayAnalyser.cpp
  ${DDS_DIR}/src/PlayAnalyser.h
  ${DDS_DIR}/src/QuickTricks.cpp
  ${DDS_DIR}/src/QuickTricks.h
  ${DDS_DIR}/src/Scheduler.cpp
  ${DDS_DIR}/src/Scheduler.h
  ${DDS_DIR}/src/SolveBoard.cpp
  ${DDS_DIR}/src/SolveBoard.h
  ${DDS_DIR}/src/SolverIF.cpp
  ${DDS_DIR}/src/SolverIF.h
  ${DDS_DIR}/src/System.cpp
  ${DDS_DIR}/src/System.h
  ${DDS_DIR}/src/ThreadMgr.cpp
  ${DDS_DIR}/src/ThreadMgr.h
  ${DDS_DIR}/src/Timer.cpp
  ${DDS_DIR}/src/Timer.h
  ${DDS_DIR}/src/TimerGroup.cpp
  ${DDS_DIR}/src/TimerGroup.h
  ${DDS_DIR}/src/TimerList.cpp
  ${DDS_DIR}/src/TimerList.h
  ${DDS_DIR}/src/TimeStat.cpp
  ${DDS_DIR}/src/TimeStat.h
  ${DDS_DIR}/src/TimeStatList.cpp
  ${DDS_DIR}/src/TimeStatList.h
  ${DDS_DIR}/src/TransTable.h
  ${DDS_DIR}/src/TransTableL.cpp
  ${DDS_DIR}/src/TransTableL.h
  ${DDS_DIR}/src/TransTableS.cpp
  ${DDS_DIR}/src/TransTableS.h
)
target_compile_definitions (open_spiel_ext_dds PUBLIC DDS_NO_STATIC_INIT)
add_library (open_spiel_ext::dds INTERFACE IMPORTED GLOBAL)
target_link_libraries (open_spiel_ext::dds INTERFACE
  open_spiel_ext_dds $<TARGET_OBJECTS:open_spiel_ext_dds>)

## Optional dependencies.
if (OPEN_SPIEL_BUILD_WITH_ACPC)
  set (ACPC_DIR games/universal_poker/acpc/project_acpc_server)
  add_library (open_spiel_ext_acpc OBJECT
    ${ACPC_DIR}/game.h
    ${ACPC_DIR}/game.cc
    ${ACPC_DIR}/rng.h
    ${ACPC_DIR}/rng.cc
  )
  add_library (open_spiel_ext::acpc INTERFACE IMPORTED GLOBAL)
  target_link_libraries (open_spiel_ext::acpc INTERFACE
    open_spiel_ext_acpc $<TARGET_OBJECTS:open_spiel_ext_acpc>)
endif()

if (OPEN_SPIEL_BUILD_WITH_HANABI)
  set (HANABI_DIR
       games/hanabi/hanabi-learning-environment/hanabi_learning_environment)
  add_library (open_spiel_ext_hanabi OBJECT
    ${HANABI_DIR}/hanabi_lib/canonical_encoders.cc
    ${HANABI_DIR}/hanabi_lib/canonical_encoders.h
    ${HANABI_DIR}/hanabi_lib/hanabi_card.cc
    ${HANABI_DIR}/hanabi_lib/hanabi_card.h
    ${HANABI_DIR}/hanabi_lib/hanabi_game.cc
    ${HANABI_DIR}/hanabi_lib/hanabi_game.h
    ${HANABI_DIR}/hanabi_lib/hanabi_hand.cc
    ${HANABI_DIR}/hanabi_lib/hanabi_hand.h
    ${HANABI_DIR}/hanabi_lib/hanabi_history_item.cc
    ${HANABI_DIR}/hanabi_lib/hanabi_history_item.h
    ${HANABI_DIR}/hanabi_lib/hanabi_move.cc
    ${HANABI_DIR}/hanabi_lib/hanabi_move.h
    ${HANABI_DIR}/hanabi_lib/hanabi_observation.cc
    ${HANABI_DIR}/hanabi_lib/hanabi_observation.h
    ${HANABI_DIR}/hanabi_lib/hanabi_state.cc
    ${HANABI_DIR}/hanabi_lib/hanabi_state.h
    ${HANABI_DIR}/hanabi_lib/observation_encoder.h
    ${HANABI_DIR}/hanabi_lib/util.cc
    ${HANABI_DIR}/hanabi_lib/util.h
  )
  target_include_directories (open_spiel_ext_hanabi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/${HANABI_DIR})
  add_library (open_spiel_ext::hanabi INTERFACE IMPORTED GLOBAL)
  target_link_libraries (open_spiel_ext::hanabi INTERFACE
    open_spiel_ext_hanabi $<TARGET_OBJECTS:open_spiel_ext_hanabi>)
endif()

if (OPEN_SPIEL_BUILD_WITH_ROSHAMBO)
  set (ROSHAMBO_DIR bots/roshambo/roshambo)
  add_library (open_spiel_ext_roshambo OBJECT
    ${ROSHAMBO_DIR}/BotClasses/multibot.cc
    ${ROSHAMBO_DIR}/BotClasses/phasenbott.cc
    ${ROSHAMBO_DIR}/BotClasses/shofar.cc
    ${ROSHAMBO_DIR}/bot_map.cc
    ${ROSHAMBO_DIR}/bot_map.h
  )
  # The RoShamBo bots are third-party code: silence their warnings.
  if (MSVC)
    target_compile_options (open_spiel_ext_roshambo PRIVATE /w)
  else()
    target_compile_options (open_spiel_ext_roshambo PRIVATE -w)
  endif()
  target_include_directories (open_spiel_ext_roshambo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/bots/roshambo)
  add_library (open_spiel_ext::roshambo INTERFACE IMPORTED GLOBAL)
  target_link_libraries (open_spiel_ext::roshambo INTERFACE
    open_spiel_ext_roshambo $<TARGET_OBJECTS:open_spiel_ext_roshambo>)
endif()

if (OPEN_SPIEL_BUILD_WITH_XINXIN)
  set (XINXIN_DIR bots/xinxin/hearts)
  add_library (open_spiel_ext_xinxin OBJECT
    ${XINXIN_DIR}/Algorithm.cpp
    ${XINXIN_DIR}/Algorithm.h
    ${XINXIN_DIR}/CardGameState.cpp
    ${XINXIN_DIR}/CardGameState.h
    ${XINXIN_DIR}/CardProbabilityData.cpp
    ${XINXIN_DIR}/CardProbabilityData.h
    ${XINXIN_DIR}/Game.cpp
    ${XINXIN_DIR}/Game.h
    ${XINXIN_DIR}/GameState.cpp
    ${XINXIN_DIR}/GameState.h
    ${XINXIN_DIR}/Hearts.cpp
    ${XINXIN_DIR}/Hearts.h
    ${XINXIN_DIR}/HeartsGameData.cpp
    ${XINXIN_DIR}/HeartsGameData.h
    ${XINXIN_DIR}/HeartsGameHistories.cpp
    ${XINXIN_DIR}/HeartsGameHistories.h
    ${XINXIN_DIR}/Player.cpp
    ${XINXIN_DIR}/Player.h
    ${XINXIN_DIR}/ProblemState.cpp
    ${XINXIN_DIR}/ProblemState.h
    ${XINXIN_DIR}/States.cpp
    ${XINXIN_DIR}/States.h
    ${XINXIN_DIR}/Timer.cpp
    ${XINXIN_DIR}/Timer.h
    ${XINXIN_DIR}/UCT.cpp
    ${XINXIN_DIR}/UCT.h
    ${XINXIN_DIR}/algorithmStates.cpp
    ${XINXIN_DIR}/algorithmStates.h
    ${XINXIN_DIR}/fpUtil.cpp
    ${XINXIN_DIR}/fpUtil.h
    ${XINXIN_DIR}/hash.cpp
    ${XINXIN_DIR}/hash.h
    ${XINXIN_DIR}/iiGameState.cpp
    ${XINXIN_DIR}/iiGameState.h
    ${XINXIN_DIR}/iiMonteCarlo.cpp
    ${XINXIN_DIR}/iiMonteCarlo.h
    ${XINXIN_DIR}/mt_random.cpp
    ${XINXIN_DIR}/mt_random.h
  )
  target_include_directories (open_spiel_ext_xinxin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/${XINXIN_DIR})
  add_library (open_spiel_ext::xinxin_hearts INTERFACE IMPORTED GLOBAL)
  target_link_libraries (open_spiel_ext::xinxin_hearts INTERFACE
    open_spiel_ext_xinxin $<TARGET_OBJECTS:open_spiel_ext_xinxin>)
endif()

if (OPEN_SPIEL_BUILD_WITH_LIBNOP)
  add_library (open_spiel_ext::libnop INTERFACE IMPORTED GLOBAL)
  target_include_directories (open_spiel_ext::libnop INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/libnop/libnop/include)
endif()

if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  list (APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/libtorch")
  find_package (Torch REQUIRED)
  add_library (open_spiel_ext::torch INTERFACE IMPORTED GLOBAL)
  target_include_directories (open_spiel_ext::torch INTERFACE
    ${TORCH_INCLUDE_DIRS})
  target_link_libraries (open_spiel_ext::torch INTERFACE ${TORCH_LIBRARIES})
endif()

if (OPEN_SPIEL_BUILD_WITH_ORTOOLS)
  # Compile with OR-Tools headers and link against binary distribution,
  # downloaded from https://developers.google.com/optimization/install/cpp/linux
  # and assumed to be in $HOME/or-tools.
  # The flags were taken from the compilation of linear_programming.cc after
  # running make test_cc.
  set (ORTOOLS_HOME "${CMAKE_CURRENT_SOURCE_DIR}/ortools")
  set (ORTOOLS_LIBS z rt pthread ortools)
  add_library (open_spiel_ext::ortools INTERFACE IMPORTED GLOBAL)
  target_compile_definitions (open_spiel_ext::ortools INTERFACE
    USE_BOP USE_GLOP USE_CBC USE_CLP USE_SCIP)
  # GCC/Clang flag: OR-Tools is only supported on Linux.
  target_compile_options (open_spiel_ext::ortools INTERFACE -pthread)
  target_include_directories (open_spiel_ext::ortools INTERFACE
    ${ORTOOLS_HOME} ${ORTOOLS_HOME}/include)
  target_link_directories (open_spiel_ext::ortools INTERFACE
    ${ORTOOLS_HOME}/lib ${ORTOOLS_HOME}/lib64)
  target_link_libraries (open_spiel_ext::ortools INTERFACE ${ORTOOLS_LIBS})
endif()
