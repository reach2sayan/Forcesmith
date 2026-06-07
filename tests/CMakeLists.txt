# ── Unit and integration tests ───────────────────────────────────────────────
# Pulled out of the top-level CMakeLists. enable_testing() is invoked by the
# parent (so ctest registers from the build root); this file owns GoogleTest and
# the test targets. Sources are relative to this directory; library sources and
# headers are referenced via ${PROJECT_SOURCE_DIR}.

include(FetchContent)
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(UNIT_TEST_SOURCES
        unit/test_adp_force.cpp
        unit/test_analytic_functions.cpp
        unit/test_angular_force.cpp
        unit/test_boundary_conditions.cpp
        unit/test_checkpoint.cpp
        unit/test_config_reader.cpp
        unit/test_eam_force.cpp
        unit/test_forcesmith.cpp
        unit/test_force_calc.cpp
        unit/test_force_model_reader.cpp
        unit/test_neighbor_list.cpp
        unit/test_line_search.cpp
        unit/test_optimizer.cpp
        unit/test_solver_potentials.cpp
        unit/test_potential_reader.cpp
        unit/test_rescale.cpp
        unit/test_species.cpp
        unit/test_spline.cpp
        unit/test_stiweb_force.cpp
        unit/test_tersoff_force.cpp
        unit/test_ml_force.cpp
        unit/test_ml_cache.cpp
        unit/test_ml_jacobian.cpp
        unit/test_soap.cpp
        unit/test_acsf.cpp
        unit/test_lmbtr.cpp
        unit/test_ml_fit.cpp
        unit/test_ml_rerank.cpp
        unit/test_types.cpp
        unit/test_init.cpp
)

# test_init.cpp drives the real scaffolder (src/cli/init.cpp), which is not part
# of the forcesmith library, so compile it into the test binary and link the
# program_options it parses with.
add_executable(forcesmith_tests ${UNIT_TEST_SOURCES} ${PROJECT_SOURCE_DIR}/src/cli/init.cpp)
target_include_directories(forcesmith_tests PRIVATE "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(forcesmith_tests PRIVATE forcesmith_warnings forcesmith GTest::gtest_main Boost::program_options)

include(GoogleTest)
gtest_discover_tests(forcesmith_tests)

file(GLOB INTEGRATION_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp")
if (INTEGRATION_TEST_SOURCES)
    add_executable(forcesmith_integration_tests ${INTEGRATION_TEST_SOURCES})
    target_include_directories(forcesmith_integration_tests PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(forcesmith_integration_tests PRIVATE forcesmith_warnings forcesmith GTest::gtest_main)
    gtest_discover_tests(forcesmith_integration_tests)
endif ()
