cmake_minimum_required(VERSION 3.28)
project(Potfit CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Dependencies)

# ── Compiler options ──────────────────────────────────────────────────────────

add_library(potfit_warnings INTERFACE)
target_compile_options(potfit_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
        $<$<CONFIG:Debug>:-g -fsanitize=address,undefined>
        $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
)
target_link_options(potfit_warnings INTERFACE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

# ── Implementation library (linked by both executable and tests) ──────────────

set(POTFIT_LIB_SOURCES
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
        src/io/output_writer.cpp
        src/io/potential_reader.cpp
        src/optimization/line_search.cpp
        src/optimization/optimizer.cpp
        src/optimization/potfit_functor.cpp
        src/optimization/solver.cpp
        src/potentials/spline.cpp
)

add_library(potfit_lib STATIC ${POTFIT_LIB_SOURCES})

target_include_directories(potfit_lib PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(potfit_lib PUBLIC
        Eigen3::Eigen
        Boost::serialization
        Boost::boost
        boost_parser
        boost_math
        nlohmann_json::nlohmann_json
        TBB::tbb
)
target_link_libraries(potfit_lib PRIVATE
        potfit_warnings
)

# ── Main executable ───────────────────────────────────────────────────────────

add_executable(potfit src/main.cpp)

target_include_directories(potfit PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(potfit PRIVATE
        potfit_warnings
        potfit_lib
        Boost::program_options
)

# ── Tests ─────────────────────────────────────────────────────────────────────

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
        tests/unit/test_force_calc.cpp
        tests/unit/test_force_model_reader.cpp
        tests/unit/test_neighbor_list.cpp
        tests/unit/test_line_search.cpp
        tests/unit/test_optimizer.cpp
        tests/unit/test_potential_reader.cpp
        tests/unit/test_rescale.cpp
        tests/unit/test_spline.cpp
        tests/unit/test_stiweb_force.cpp
        tests/unit/test_tersoff_force.cpp
        tests/unit/test_types.cpp
)

add_executable(potfit_tests ${UNIT_TEST_SOURCES})

target_include_directories(potfit_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(potfit_tests PRIVATE
        potfit_warnings
        potfit_lib
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(potfit_tests)

# ── Integration tests ─────────────────────────────────────────────────────────
file(GLOB INTEGRATION_TEST_SOURCES "tests/integration/*.cpp")
if (INTEGRATION_TEST_SOURCES)
    add_executable(potfit_integration_tests ${INTEGRATION_TEST_SOURCES})
    target_include_directories(potfit_integration_tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(potfit_integration_tests PRIVATE
            potfit_warnings
            potfit_lib
            GTest::gtest_main
    )
    gtest_discover_tests(potfit_integration_tests)
endif ()

# ── Evaluation-parity harness (port vs. original C potfit) ────────────────────
# Each case converts a potfit text potential to the port's JSON, evaluates it
# with the `potfit` binary, and diffs the forces against a committed oracle
# reference (tests/integration/harness/reference/<case>/). Needs python3; the
# references are pre-generated (gen_reference.sh builds the upstream C potfit and
# is NOT part of the test). Registered only when python3 is available.
find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
    set(_harness "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/harness")
    set(_data "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/data/potfit_upstream")
    # case-dir → ntypes
    foreach (_case_spec "Al_eam_sc:1" "Al_eam_gsc:1")
        string(REPLACE ":" ";" _parts "${_case_spec}")
        list(GET _parts 0 _case)
        list(GET _parts 1 _nt)
        add_test(
                NAME parity_${_case}
                COMMAND bash "${_harness}/run_parity.sh"
                "$<TARGET_FILE:potfit>" "${_data}/${_case}" "${_nt}"
        )
        set_tests_properties(parity_${_case} PROPERTIES LABELS "parity")
    endforeach ()
endif ()

include(ClangFormat)
