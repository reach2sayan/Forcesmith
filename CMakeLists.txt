cmake_minimum_required(VERSION 3.28)
project(Potfit CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

add_library(potfit_warnings INTERFACE)
target_compile_options(potfit_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
        $<$<CONFIG:Debug>:-g -fsanitize=address,undefined>
        $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)

target_link_options(potfit_warnings INTERFACE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)

set(POTFIT_LIB_SOURCES
        src/api/fit_session.cpp
        src/core/checkpoint.cpp
        src/core/neighbor_list.cpp
        src/core/rescale.cpp
        src/force/adp_force.cpp
        src/force/pair_force.cpp
        src/force/angular_force.cpp
        src/force/eam_force.cpp
        src/force/stiweb_force.cpp
        src/force/tersoff_force.cpp
        src/force/evaluate.cpp
        src/io/config_reader.cpp
        src/io/force_model_reader.cpp
        src/io/loaders.cpp
        src/io/output_writer.cpp
        src/io/potential_reader.cpp
        src/io/write_model.cpp
        src/optimization/line_search.cpp
        src/optimization/optimizer.cpp
        src/optimization/potfit_functor.cpp
        src/optimization/solver.cpp
        src/potentials/spline.cpp
)

add_library(potfit_lib STATIC ${POTFIT_LIB_SOURCES})
target_include_directories(potfit_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(potfit_lib PUBLIC
        potfit::eigen
        Boost::serialization
        Boost::boost
        boost_parser
        boost_math
        nlohmann_json::nlohmann_json
        TBB::tbb
)
target_link_libraries(potfit_lib PRIVATE potfit_warnings)

add_executable(potfit src/main.cpp)
target_include_directories(potfit PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(potfit PRIVATE potfit_warnings potfit_lib Boost::program_options)

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
        tests/unit/test_fit_session.cpp
        tests/unit/test_force_calc.cpp
        tests/unit/test_force_model_reader.cpp
        tests/unit/test_neighbor_list.cpp
        tests/unit/test_line_search.cpp
        tests/unit/test_optimizer.cpp
        tests/unit/test_potential_reader.cpp
        tests/unit/test_rescale.cpp
        tests/unit/test_species.cpp
        tests/unit/test_spline.cpp
        tests/unit/test_stiweb_force.cpp
        tests/unit/test_tersoff_force.cpp
        tests/unit/test_types.cpp
)

add_executable(potfit_tests ${UNIT_TEST_SOURCES})
target_include_directories(potfit_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(potfit_tests PRIVATE potfit_warnings potfit_lib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(potfit_tests)

file(GLOB INTEGRATION_TEST_SOURCES "tests/integration/*.cpp")
if (INTEGRATION_TEST_SOURCES)
    add_executable(potfit_integration_tests ${INTEGRATION_TEST_SOURCES})
    target_include_directories(potfit_integration_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(potfit_integration_tests PRIVATE potfit_warnings potfit_lib GTest::gtest_main)
    gtest_discover_tests(potfit_integration_tests)
endif ()

# ── Per-element UNEP EAM fit drivers (programmatic-API parallel of fit_eam.py) ──
# One executable per metal, the element baked in via POTFIT_UNEP_ELEMENT. All are
# built from the single self-contained source tests/unep/fit_eam_api.cpp.
# Off by default; enable with -DPOTFIT_BUILD_UNEP_FITS=ON. The aggregate target
# `unep_fits` builds every element at once.
option(POTFIT_BUILD_UNEP_FITS "Build the per-element UNEP EAM fit executables" OFF)
if (POTFIT_BUILD_UNEP_FITS)
    set(POTFIT_UNEP_ELEMENTS
            Ag Al Au Cr Cu Mg Mo Ni Pb Pd Pt Ta Ti V W Zr)
    add_custom_target(unep_fits)
    foreach (el IN LISTS POTFIT_UNEP_ELEMENTS)
        string(TOLOWER "${el}" el_lc)
        set(tgt "unep_fit_${el_lc}")
        add_executable(${tgt} tests/unep/fit_eam_api.cpp)
        target_include_directories(${tgt} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_compile_definitions(${tgt} PRIVATE POTFIT_UNEP_ELEMENT="${el}")
        target_link_libraries(${tgt} PRIVATE potfit_warnings potfit_lib Boost::program_options)
        add_dependencies(unep_fits ${tgt})
    endforeach ()
endif ()

include(ClangFormat)
