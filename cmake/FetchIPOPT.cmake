# cmake/FetchIPOPT.cmake
#
# Builds IPOPT 3.14 + ThirdParty-Mumps from source into
# ${CMAKE_BINARY_DIR}/ipopt_local.  Re-uses the existing build if the
# libraries are already present (ExternalProject stamps handle this).
#
# BUILD_BYPRODUCTS is required for Ninja: it tells the generator that
# libipopt.so / libcoinmumps.so will be produced by the external build,
# so the linker step is not considered broken when they don't yet exist.

include(ExternalProject)

set(IPOPT_LOCAL_PREFIX "${CMAKE_BINARY_DIR}/ipopt_local")
set(IPOPT_INCLUDE_DIR "${IPOPT_LOCAL_PREFIX}/include/coin-or")
set(IPOPT_LIB_DIR "${IPOPT_LOCAL_PREFIX}/lib")

ExternalProject_Add(ThirdPartyMumps
        GIT_REPOSITORY "https://github.com/coin-or-tools/ThirdParty-Mumps.git"
        GIT_TAG "stable/3.0"
        GIT_SHALLOW TRUE
        BUILD_IN_SOURCE TRUE
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND
        sh -c "cd <SOURCE_DIR> && ./get.Mumps && ./configure --prefix=${IPOPT_LOCAL_PREFIX}"
        BUILD_COMMAND make -j 8
        INSTALL_COMMAND make install
        BUILD_BYPRODUCTS
        "${IPOPT_LIB_DIR}/libcoinmumps.so"
        LOG_DOWNLOAD TRUE
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_INSTALL TRUE
)

ExternalProject_Add(IpoptProject
        GIT_REPOSITORY "https://github.com/coin-or/Ipopt.git"
        GIT_TAG "releases/3.14.14"
        GIT_SHALLOW TRUE
        DEPENDS ThirdPartyMumps
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND
        <SOURCE_DIR>/configure
        --prefix=${IPOPT_LOCAL_PREFIX}
        --with-mumps-cflags=-I${IPOPT_LOCAL_PREFIX}/include/coin-or/mumps
        --with-mumps-lflags=-L${IPOPT_LOCAL_PREFIX}/lib\ -lcoinmumps
        --with-lapack-lflags=-lblas\ -llapack
        BUILD_COMMAND make -j4
        INSTALL_COMMAND make install
        BUILD_BYPRODUCTS
        "${IPOPT_LIB_DIR}/libipopt.so"
        LOG_DOWNLOAD TRUE
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_INSTALL TRUE
)

file(MAKE_DIRECTORY "${IPOPT_INCLUDE_DIR}")
add_library(IPOPT::ipopt INTERFACE IMPORTED)
target_include_directories(IPOPT::ipopt INTERFACE "${IPOPT_INCLUDE_DIR}")
target_link_libraries(IPOPT::ipopt INTERFACE
        "${IPOPT_LIB_DIR}/libipopt.so"
        "${IPOPT_LIB_DIR}/libcoinmumps.so"
        blas lapack
)
