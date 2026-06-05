cmake_minimum_required(VERSION 3.28)
project(Forcesmith CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Compiler cache (ccache) ──────────────────────────────────────────────────
# Speeds up rebuilds by caching object files. Used only when ccache is on PATH,
# so a machine without it still configures cleanly.
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
else ()
    message(STATUS "ccache not found; building without compiler cache")
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

# ── Link-time optimization (LTO/IPO) ─────────────────────────────────────────
# Enable interprocedural optimization for optimized builds only; LTO would just
# slow Debug compiles down without buying anything. Probe support first so a
# toolchain without working IPO degrades gracefully instead of hard-failing.
include(CheckIPOSupported)
check_ipo_supported(RESULT FORCESMITH_IPO_SUPPORTED OUTPUT FORCESMITH_IPO_ERROR)
if (FORCESMITH_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON)
    message(STATUS "LTO/IPO enabled for optimized builds")
else ()
    message(STATUS "LTO/IPO not supported: ${FORCESMITH_IPO_ERROR}")
endif ()

add_library(forcesmith_warnings INTERFACE)
target_compile_options(forcesmith_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
        $<$<CONFIG:Debug>:-g -fsanitize=address,undefined>
        $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)

target_link_options(forcesmith_warnings INTERFACE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)

# ── Static engine library ──────────────────────────────────────────────────
# Everything below the public API. Must NOT depend on the Forcesmith API (the shared
# `forcesmith` lib sits on top of this). io/loaders.cpp is a thin API front-end and
# therefore lives with the shared lib, not here.
set(FORCESMITH_ENGINE_SOURCES
        src/core/checkpoint.cpp
        src/core/neighbor_list.cpp
        src/core/rescale.cpp
        src/force/adp_force.cpp
        src/force/pair_force.cpp
        src/force/angular_force.cpp
        src/force/eam_force.cpp
        src/force/stiweb_force.cpp
        src/force/tersoff_force.cpp
        src/force/ml_force.cpp
        src/force/descriptor_layout.cpp
        src/force/acsf.cpp
        src/force/lmbtr.cpp
        src/force/soap.cpp
        src/force/evaluate.cpp
        src/force/smoothness.cpp
        src/io/config_reader.cpp
        src/io/force_model_reader.cpp
        src/io/logging.cpp
        src/io/output_writer.cpp
        src/io/potential_reader.cpp
        src/io/write_model.cpp
        src/optimization/line_search.cpp
        src/optimization/optimizer.cpp
        src/optimization/forcesmith_functor.cpp
        src/optimization/solver.cpp
        src/optimization/ipopt_solver.cpp
        src/potentials/analytic_param_defs.cpp
        src/potentials/spline.cpp
)

add_library(forcesmith_engine STATIC ${FORCESMITH_ENGINE_SOURCES})
target_include_directories(forcesmith_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(forcesmith_engine PUBLIC
        forcesmith::eigen
        Boost::serialization
        Boost::boost
        boost_parser
        boost_math
        nlohmann_json::nlohmann_json
        TBB::tbb
        IPOPT::ipopt
)
# spdlog is only included by src/io/logging.cpp (the public header is
# spdlog-free), so keep its include path off the engine's public interface.
target_link_libraries(forcesmith_engine PRIVATE forcesmith_warnings spdlog)
# Ipopt+MUMPS are built by the IpoptProject ExternalProject; ensure they exist
# before the engine compiles/links the TU that includes the Ipopt headers.
add_dependencies(forcesmith_engine IpoptProject)

# Cross-check the spline evaluation cache (eval_at/deriv_at and the hand-rolled
# makima slopes) against boost makima at every call via BOOST_ASSERT. PUBLIC so
# the macro reaches every TU that includes spline.hpp (it toggles a member of
# SplinePotential — the layout must agree across the library and the tests).
# Slow; intended for debug/CI verification, OFF in release.
option(FORCESMITH_SPLINE_VERIFY
        "Assert the spline eval cache matches boost makima (slow; debug only)" OFF)
if (FORCESMITH_SPLINE_VERIFY)
    target_compile_definitions(forcesmith_engine PUBLIC FORCESMITH_SPLINE_VERIFY)
endif ()
# Its objects get archived into the shared `forcesmith` lib, so they need -fPIC.
set_target_properties(forcesmith_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── First-class shared API library ─────────────────────────────────────────
# The programmatic Forcesmith API (forcesmith::Forcesmith) plus the thin file-driven loaders
# that drive it. This is the product's public surface; the CLI and the tests are
# ordinary clients that link it.
add_library(forcesmith SHARED
        src/api/forcesmith.cpp
        src/io/loaders.cpp
)
target_include_directories(forcesmith PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(forcesmith PUBLIC forcesmith_engine)
target_link_libraries(forcesmith PRIVATE forcesmith_warnings)

# ── CLI ─────────────────────────────────────────────────────────────────────
# Target renamed to forcesmith_cli so the `forcesmith` name is free for the library; the
# user-facing binary is still called `forcesmith` (libforcesmith.so vs forcesmith — no clash).
# main.cpp is a thin entry point; the CLI logic lives in src/cli/, with its
# headers under include/forcesmith/cli/ (resolved as "forcesmith/cli/<file>.hpp").
add_executable(forcesmith_cli
        src/main.cpp
        src/cli/options.cpp
        src/cli/evaluate_report.cpp
        src/cli/init.cpp
        src/cli/app.cpp)
set_target_properties(forcesmith_cli PROPERTIES OUTPUT_NAME forcesmith)
target_include_directories(forcesmith_cli PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(forcesmith_cli PRIVATE forcesmith_warnings forcesmith Boost::program_options)

enable_testing()
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(UNIT_TEST_SOURCES
        tests/unit/test_adp_force.cpp
        tests/unit/test_analytic_functions.cpp
        tests/unit/test_angular_force.cpp
        tests/unit/test_boundary_conditions.cpp
        tests/unit/test_checkpoint.cpp
        tests/unit/test_config_reader.cpp
        tests/unit/test_eam_force.cpp
        tests/unit/test_forcesmith.cpp
        tests/unit/test_force_calc.cpp
        tests/unit/test_force_model_reader.cpp
        tests/unit/test_neighbor_list.cpp
        tests/unit/test_line_search.cpp
        tests/unit/test_optimizer.cpp
        tests/unit/test_solver_potentials.cpp
        tests/unit/test_potential_reader.cpp
        tests/unit/test_rescale.cpp
        tests/unit/test_species.cpp
        tests/unit/test_spline.cpp
        tests/unit/test_stiweb_force.cpp
        tests/unit/test_tersoff_force.cpp
        tests/unit/test_ml_force.cpp
        tests/unit/test_ml_cache.cpp
        tests/unit/test_ml_jacobian.cpp
        tests/unit/test_soap.cpp
        tests/unit/test_acsf.cpp
        tests/unit/test_lmbtr.cpp
        tests/unit/test_ml_fit.cpp
        tests/unit/test_ml_rerank.cpp
        tests/unit/test_types.cpp
        tests/unit/test_init.cpp
)

# test_init.cpp drives the real scaffolder (src/cli/init.cpp), which is not part
# of the forcesmith library, so compile it into the test binary and link the
# program_options it parses with.
add_executable(forcesmith_tests ${UNIT_TEST_SOURCES} src/cli/init.cpp)
target_include_directories(forcesmith_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(forcesmith_tests PRIVATE forcesmith_warnings forcesmith GTest::gtest_main Boost::program_options)

include(GoogleTest)
gtest_discover_tests(forcesmith_tests)

file(GLOB INTEGRATION_TEST_SOURCES "tests/integration/*.cpp")
if (INTEGRATION_TEST_SOURCES)
    add_executable(forcesmith_integration_tests ${INTEGRATION_TEST_SOURCES})
    target_include_directories(forcesmith_integration_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(forcesmith_integration_tests PRIVATE forcesmith_warnings forcesmith GTest::gtest_main)
    gtest_discover_tests(forcesmith_integration_tests)
endif ()

# ── UNEP EAM fit driver (programmatic-API parallel of fit_eam.py) ────────────
# A single `unep_fit` executable built from tests/unep/fit_eam_api.cpp. The
# element is a runtime flag (--element Cu), so there is one binary rather than
# one per metal. Off by default; enable with -DFORCESMITH_BUILD_UNEP_FITS=ON.
option(FORCESMITH_BUILD_UNEP_FITS "Build the UNEP EAM fit executable (unep_fit)" OFF)
if (FORCESMITH_BUILD_UNEP_FITS)
    add_executable(unep_fit tests/unep/fit_eam_api.cpp)
    target_include_directories(unep_fit PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(unep_fit PRIVATE forcesmith_warnings forcesmith Boost::program_options)
endif ()

# ── ML fit driver (parallel of the UNEP EAM fits) ────────────────────────────
# A single `ml_fit` executable built from tests/ml/ml_fit.cpp. The descriptor,
# energy head and element are runtime flags (--descriptor soap|symfunc,
# --head nn|linear, --element Cu), so there is one binary rather than one per
# metal. Off by default (heavy: FD optimizer Jacobian × FD-gradient descriptor);
# enable with -DFORCESMITH_BUILD_ML_FITS=ON.
option(FORCESMITH_BUILD_ML_FITS "Build the ML fit executable (ml_fit)" OFF)
if (FORCESMITH_BUILD_ML_FITS)
    add_executable(ml_fit tests/ml/ml_fit.cpp)
    target_include_directories(ml_fit PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(ml_fit PRIVATE forcesmith_warnings forcesmith Boost::program_options)
endif ()

include(ClangFormat)
