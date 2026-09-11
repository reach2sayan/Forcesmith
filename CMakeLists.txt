cmake_minimum_required(VERSION 3.28)
project(Forcesmith CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

include(CheckIPOSupported)
# LANGUAGES CXX: the project is C++ only, but dependencies (Boost, IPOPT) enable
# C, and probing every enabled language would disable LTO for our targets
# whenever the C toolchain cannot do IPO.
check_ipo_supported(RESULT FORCESMITH_IPO_SUPPORTED OUTPUT FORCESMITH_IPO_ERROR LANGUAGES CXX)
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
        # Boost 1.92, one library target at a time. In the CMake superproject
        # Boost::headers is only libs/headers/include — it does NOT aggregate
        # the other libraries — so a missing target here silently falls through
        # to whatever Boost the system happens to have in /usr/include.
        Boost::serialization
        Boost::headers
        Boost::parser
        Boost::math
        Boost::leaf
        Boost::mp11
        Boost::describe
        Boost::pfr
        Boost::type_erasure
        Boost::container
        Boost::hana
        Boost::hof
        Boost::bimap
        Boost::unordered
        Boost::signals2
        Boost::multi_index
        ddx::ddx
        einsum::einsum
        nlohmann_json::nlohmann_json
        TBB::tbb
        IPOPT::ipopt
)

target_link_libraries(forcesmith_engine PRIVATE forcesmith_warnings spdlog)


add_dependencies(forcesmith_engine IpoptProject)

option(FORCESMITH_SPLINE_VERIFY
        "Assert the spline eval cache matches boost makima (slow; debug only)" OFF)
if (FORCESMITH_SPLINE_VERIFY)
    target_compile_definitions(forcesmith_engine PUBLIC FORCESMITH_SPLINE_VERIFY)
endif ()
# Its objects get archived into the shared `forcesmith` lib, so they need -fPIC.
set_target_properties(forcesmith_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The programmatic Forcesmith API (forcesmith::Forcesmith)
add_library(forcesmith SHARED src/api/forcesmith.cpp src/io/loaders.cpp)
target_include_directories(forcesmith PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(forcesmith PUBLIC forcesmith_engine)
target_link_libraries(forcesmith PRIVATE forcesmith_warnings)

add_executable(forcesmith_cli
        src/main.cpp
        src/cli/options.cpp
        src/cli/evaluate_report.cpp
        src/cli/init.cpp
        src/cli/solver_registry.cpp
        src/cli/app.cpp)
set_target_properties(forcesmith_cli PROPERTIES OUTPUT_NAME forcesmith)
target_include_directories(forcesmith_cli PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(forcesmith_cli PRIVATE forcesmith_warnings forcesmith Boost::program_options)

# Testing, integration tests and the optional fit drivers live in tests/.
# enable_testing() stays here (top level) so ctest registers from the build root.
enable_testing()
add_subdirectory(tests)

include(ClangFormat)
