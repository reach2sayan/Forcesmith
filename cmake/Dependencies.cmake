# ── Dependencies ──────────────────────────────────────────────────────────────

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# CMP0167 NEW: use Boost's own CMake config files (FindBoost module removed).
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif ()
find_package(Boost 1.83 CONFIG REQUIRED COMPONENTS serialization program_options)

find_package(TBB CONFIG REQUIRED)
option(POTFIT_USE_MKL "Use system Intel MKL as Eigen's BLAS/LAPACK backend" ON)

add_library(potfit_eigen INTERFACE)
add_library(potfit::eigen ALIAS potfit_eigen)
target_link_libraries(potfit_eigen INTERFACE Eigen3::Eigen)

if (POTFIT_USE_MKL)
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

if (POTFIT_USE_MKL AND MKL_FOUND)
    target_link_libraries(potfit_eigen INTERFACE PkgConfig::MKL)
    target_compile_definitions(potfit_eigen INTERFACE EIGEN_USE_MKL_ALL)
    message(STATUS "Eigen backend: system Intel MKL (lp64, ${MKL_THREADING_LAYER}) via pkg-config")
elseif (POTFIT_USE_MKL)
    message(STATUS "Eigen backend: built-in kernels (system MKL pkg-config 'mkl-dynamic-lp64-{tbb,seq}' not found)")
else ()
    message(STATUS "Eigen backend: built-in kernels (POTFIT_USE_MKL=OFF)")
endif ()

include(FetchContent)
FetchContent_Declare(boost_parser
        GIT_REPOSITORY https://github.com/boostorg/parser.git
        GIT_TAG boost-1.91.0
        GIT_SHALLOW TRUE
)
# Populate only (do not add_subdirectory — the parser has no CMake install rules
# we want, and we create the INTERFACE target ourselves for SYSTEM include control).
FetchContent_GetProperties(boost_parser)
if (NOT boost_parser_POPULATED)
    # CMP0169 OLD: allow the explicit FetchContent_Populate call used here.
    if (POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif ()
    FetchContent_Populate(boost_parser)
endif ()
add_library(boost_parser INTERFACE)
target_include_directories(boost_parser SYSTEM INTERFACE "${boost_parser_SOURCE_DIR}/include")
target_link_libraries(boost_parser INTERFACE Boost::boost)

FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# boost::math (header-only) — system Boost 1.83 lacks differential_evolution.hpp
# which was added in 1.84.  Fetch just the math headers at 1.87.
FetchContent_Declare(boost_math
        GIT_REPOSITORY https://github.com/boostorg/math.git
        GIT_TAG boost-1.91.0
        GIT_SHALLOW TRUE
)
FetchContent_GetProperties(boost_math)
if (NOT boost_math_POPULATED)
    if (POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif ()
    FetchContent_Populate(boost_math)
endif ()
add_library(boost_math INTERFACE)
target_include_directories(boost_math SYSTEM INTERFACE "${boost_math_SOURCE_DIR}/include")
target_link_libraries(boost_math INTERFACE Boost::boost)

# IPOPT + MUMPS built from source into the build tree (no system install needed).
# Defines the INTERFACE IMPORTED target IPOPT::ipopt and the ExternalProject
# target IpoptProject (depended on by potfit_engine so the libs exist before link).
include(FetchIPOPT)
# $ORIGIN-relative RPATH so the CLI/test binaries find libipopt/libcoinmumps in the
# build tree at runtime, regardless of where the build tree lives on disk.
set(CMAKE_BUILD_RPATH "$ORIGIN/ipopt_local/lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
