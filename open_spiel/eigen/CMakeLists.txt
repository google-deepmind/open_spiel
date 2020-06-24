# Now we can use #include "open_spiel/spiel.h"
include_directories(../..)
# Now we can use #include "Eigen/Dense"
include_directories(libeigen)

set(EIGEN_SOURCES
        libeigen/Eigen/Cholesky
        libeigen/Eigen/CholmodSupport
        libeigen/Eigen/Core
        libeigen/Eigen/Dense
        libeigen/Eigen/Eigen
        libeigen/Eigen/Eigenvalues
        libeigen/Eigen/Geometry
        libeigen/Eigen/Householder
        libeigen/Eigen/IterativeLinearSolvers
        libeigen/Eigen/Jacobi
        libeigen/Eigen/LU
        libeigen/Eigen/MetisSupport
        libeigen/Eigen/OrderingMethods
        libeigen/Eigen/PardisoSupport
        libeigen/Eigen/PaStiXSupport
        libeigen/Eigen/QR
        libeigen/Eigen/QtAlignedMalloc
        libeigen/Eigen/Sparse
        libeigen/Eigen/SparseCholesky
        libeigen/Eigen/SparseCore
        libeigen/Eigen/SparseLU
        libeigen/Eigen/SparseQR
        libeigen/Eigen/SPQRSupport
        libeigen/Eigen/StdDeque
        libeigen/Eigen/StdList
        libeigen/Eigen/StdVector
        libeigen/Eigen/SuperLUSupport
        libeigen/Eigen/SVD
        libeigen/Eigen/UmfPackSupport
        )

set(EIGEN_OPENSPIEL_USES ${EIGEN_SOURCES}
        pyeig.h
        )

add_library(eigen OBJECT ${EIGEN_OPENSPIEL_USES})
target_include_directories(eigen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(eigen PUBLIC libeigen/Eigen)


# ------ TESTS --------

# Add a basic eigen test
add_executable(eigen_basic_test
        eigen_basic_test.cc ${OPEN_SPIEL_OBJECTS}
        $<TARGET_OBJECTS:tests>)
add_test(eigen_basic_test eigen_basic_test)
