if (BUILD_WITH_TENSORFLOW_CC)
  add_executable(tf_trajectories_example tf_trajectories_example.cc tf_trajectories.cc ${OPEN_SPIEL_OBJECTS}
      $<TARGET_OBJECTS:tests>)
  add_test(tf_trajectories_example tf_trajectories_example)
  target_link_libraries(tf_trajectories_example TensorflowCC::TensorflowCC)
endif()
