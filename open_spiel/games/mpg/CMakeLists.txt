# To enable C++ AlphaZero, you will need to set OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC. See:
# https://github.com/deepmind/open_spiel/blob/master/docs/alpha_zero.md
if (OPEN_SPIEL_BUILD_WITH_BOOST)
  set(MPG_FILES
    mpg/mpg.cc
    mpg/mpg.h
    mpg/mpg_generator.cc
    mpg/mpg_generator.h
    mpg/string_compress.h
    mpg/string_compress.cc
         PARENT_SCOPE )

  set( BOOST_DEPENDENCIES
    boost
    PARENT_SCOPE)
  set( BOOST_MODULE_DEPENDENCIES
    boost::boost
    PARENT_SCOPE)

  add_executable(string_compress_test string_compress_test.cc string_compress.cc)
    add_test(string_compress_test string_compress_test.cc string_compress.cc)
    target_link_libraries(string_compress_test Boost::iostreams)

endif()
