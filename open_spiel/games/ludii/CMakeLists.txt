set (JDK_HOME /usr/lib/jvm/java-8-openjdk-amd64)

add_library (ludii OBJECT
  chunk_set.h
  chunk_set.cc
  container_state.h
  container_state.cc
  context.h
  context.cc
  game.h
  game.cc
  game_loader.h
  game_loader.cc
  jni_utils.h
  jni_utils.cc
  mode.h
  mode.cc
  move.h
  move.cc
  moves.h
  moves.cc
  region.h
  region.cc
  state.h
  state.cc
  trial.h
  trial.cc
)
target_include_directories (ludii PUBLIC ${JDK_HOME}/include/linux ${JDK_HOME}/include)
target_link_directories (ludii PUBLIC ${JDK_HOME}/jre/lib/amd64/server)
target_link_libraries (ludii jvm)

add_executable(ludii_demo ludii_demo.cc $<TARGET_OBJECTS:ludii>)
target_include_directories (ludii_demo PUBLIC ${JDK_HOME}/include/linux ${JDK_HOME}/include)
target_link_directories (ludii_demo PUBLIC ${JDK_HOME}/jre/lib/amd64/server)
target_link_libraries (ludii_demo jvm)
