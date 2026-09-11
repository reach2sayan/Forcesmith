find_package(Eigen3 3.4 REQUIRED NO_MODULE)

find_package(TBB CONFIG REQUIRED)
option(FORCESMITH_USE_MKL "Use system Intel MKL as Eigen's BLAS/LAPACK backend" ON)

add_library(forcesmith_eigen INTERFACE)
add_library(forcesmith::eigen ALIAS forcesmith_eigen)
target_link_libraries(forcesmith_eigen INTERFACE Eigen3::Eigen)

if (FORCESMITH_USE_MKL)
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        # IMPORTED_TARGET → PkgConfig::MKL carries the include dir (/usr/include/mkl)
        # and the full MKL link line. Prefer the TBB-threaded layer
        # (-lmkl_tbb_thread -ltbb): it parallelises MKL GEMM and the LAPACKE-backed
        # dense decompositions (e.g. Eigen's ColPivHouseholderQR → dgeqp3, the LM
        # solver hotspot) across cores, and shares the *same* TBB runtime the
        # parallel Jacobian already uses — no OpenMP+TBB oversubscription. Fall back
        # to the single-threaded sequential layer if the TBB module isn't present.
        set(MKL_THREADING_LAYER "tbb")
        set(_mkl_pc "mkl-dynamic-lp64-tbb")
        pkg_check_modules(MKL QUIET ${_mkl_pc})
        if (NOT MKL_FOUND)
            set(MKL_THREADING_LAYER "sequential")
            set(_mkl_pc "mkl-dynamic-lp64-seq")
            pkg_check_modules(MKL QUIET ${_mkl_pc})
        endif ()
        if (MKL_FOUND)
            # Resolve every MKL .so from the single libdir the .pc advertises, so
            # the interface/threading/core libs all come from ONE consistent MKL
            # install. Without this, find_library() inside the IMPORTED_TARGET
            # pass can mix a system /usr/lib mkl_core (which lacks mkl_tbb_thread)
            # with an oneAPI mkl_tbb_thread → version skew → undefined references.
            # This only happens when the oneAPI environment (setvars.sh) is absent
            # — e.g. when CLion drives the build instead of a sourced shell.
            # CMAKE_LIBRARY_PATH is searched ahead of the implicit system dirs, so
            # prepending the .pc's own libdir makes resolution environment-independent.
            pkg_get_variable(_mkl_libdir ${_mkl_pc} libdir)
            get_filename_component(_mkl_libdir "${_mkl_libdir}" REALPATH)
            list(PREPEND CMAKE_LIBRARY_PATH "${_mkl_libdir}")
            pkg_check_modules(MKL QUIET IMPORTED_TARGET ${_mkl_pc})
        endif ()
    endif ()
endif ()

if (FORCESMITH_USE_MKL AND MKL_FOUND)
    target_link_libraries(forcesmith_eigen INTERFACE PkgConfig::MKL)
    target_compile_definitions(forcesmith_eigen INTERFACE EIGEN_USE_MKL_ALL)
    message(STATUS "Eigen backend: system Intel MKL (lp64, ${MKL_THREADING_LAYER}) via pkg-config")
elseif (FORCESMITH_USE_MKL)
    message(STATUS "Eigen backend: built-in kernels (system MKL pkg-config 'mkl-dynamic-lp64-{tbb,seq}' not found)")
else ()
    message(STATUS "Eigen backend: built-in kernels (FORCESMITH_USE_MKL=OFF)")
endif ()

include(FetchContent)

# ── GoogleTest ───────────────────────────────────────────────────────────────
# Declared here, ahead of ddx: ddx declares a googletest of its own when its
# dependency module is included, and FetchContent honours the FIRST declaration.
# Consumed (MakeAvailable) in tests/CMakeLists.txt, which owns the test targets.
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
)

# ── Boost 1.92 (one CMake superproject) ──────────────────────────────────────
# One Boost for everything. The system 1.83 packages cannot supply Boost.Parser
# (>= 1.87), which ddx and einsum require under Boost::headers, and mixing two
# versions of one library is not an option. The official CMake tarball gives
# Boost::headers, Boost::serialization and Boost::program_options out of a
# single tree; every header-only library used here (parser, math, leaf, mp11,
# describe, pfr, type_erasure, container, hana, hof, bimap, unordered, signals2,
# multi_index) rides on Boost::headers. Declaring it before
# ddx/einsum also makes their `if (NOT TARGET Boost::headers)` guards skip a
# second Boost download.
set(BOOST_ENABLE_CMAKE ON)
set(BOOST_INCLUDE_LIBRARIES
        serialization program_options parser math leaf mp11 describe pfr
        type_erasure container hana hof bimap unordered signals2 multi_index
        preprocessor)
FetchContent_Declare(Boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.92.0/boost-1.92.0-cmake.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        EXCLUDE_FROM_ALL
        SYSTEM
)
FetchContent_MakeAvailable(Boost)

# ── ddx: compile-time symbolic differentiation (header-only) ─────────────────
# Consumed through its own CMake. Only the header-only compile-time target
# ddx::ddx is linked — never ddx::rt / ddx::jit. Its include root is flat, so
# the include is always spelled "ddx.hpp"; never reach into its subdirectories.
FetchContent_Declare(ddx
        GIT_REPOSITORY https://github.com/reach2sayan/ddx.git
        GIT_TAG 13c8c9a
        EXCLUDE_FROM_ALL
        SYSTEM
)
FetchContent_MakeAvailable(ddx)

# ── einsum: subscript-driven tensor contractions (header-only) ───────────────
set(EINSUM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(einsum
        GIT_REPOSITORY https://github.com/reach2sayan/einsum.git
        GIT_TAG 666591d
        EXCLUDE_FROM_ALL
        SYSTEM
)
FetchContent_MakeAvailable(einsum)

FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.15.3  # bundled fmt < 11 fails consteval format strings on Clang >= 19
        GIT_SHALLOW TRUE
)
FetchContent_GetProperties(spdlog)
if (NOT spdlog_POPULATED)
    if (POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif ()
    FetchContent_Populate(spdlog)
endif ()
add_library(spdlog INTERFACE)
target_include_directories(spdlog SYSTEM INTERFACE "${spdlog_SOURCE_DIR}/include")

# IPOPT + MUMPS built from source into the build tree (no system install needed).
# Defines the INTERFACE IMPORTED target IPOPT::ipopt and the ExternalProject
# target IpoptProject (depended on by forcesmith_engine so the libs exist before link).
include(FetchIPOPT)
# $ORIGIN-relative RPATH so the CLI/test binaries find libipopt/libcoinmumps in the
# build tree at runtime, regardless of where the build tree lives on disk.
set(CMAKE_BUILD_RPATH "$ORIGIN/ipopt_local/lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
