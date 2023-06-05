# ==============================================================================
# MIT License
# Copyright 2022 Institute for Automotive Engineering of RWTH Aachen University.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

# find dependencies
find_package(Protobuf REQUIRED)

# find include directories
find_path(INCLUDE_DIR tensorflow/core/public/session.h PATH_SUFFIXES tensorflow)
list(APPEND INCLUDE_DIRS ${INCLUDE_DIR})
if(INCLUDE_DIR)
    list(APPEND INCLUDE_DIRS ${INCLUDE_DIR}/src)
endif()

# find libraries
find_library(LIBRARY libtensorflow_cc.so PATH_SUFFIXES tensorflow)
find_library(LIBRARY_FRAMEWORK libtensorflow_framework.so PATH_SUFFIXES tensorflow)

# handle the QUIETLY and REQUIRED arguments and set *_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorflowCC DEFAULT_MSG)
mark_as_advanced(INCLUDE_DIRS LIBRARY LIBRARY_FRAMEWORK)

# set INCLUDE_DIRS and LIBRARIES
if(TensorflowCC_FOUND)
    set(TensorflowCC_INCLUDE_DIRS ${INCLUDE_DIRS})
    add_library(TensorflowCC::TensorflowCC INTERFACE IMPORTED)
    target_link_libraries(TensorflowCC::TensorflowCC INTERFACE libtensorflow_cc.so libtensorflow_framework.so)
    if(LIBRARY_FRAMEWORK)
        set(TensorflowCC_LIBRARIES ${LIBRARY} ${LIBRARY_FRAMEWORK} ${Protobuf_LIBRARY})
    else()
        set(TensorflowCC_LIBRARIES ${LIBRARY} ${Protobuf_LIBRARY})
    endif()
endif()