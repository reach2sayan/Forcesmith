# ── Dependencies ──────────────────────────────────────────────────────────────

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# CMP0167 NEW: use Boost's own CMake config files (FindBoost module removed).
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif ()
find_package(Boost 1.83 CONFIG REQUIRED COMPONENTS serialization program_options)

# Intel oneTBB: drives the std::execution::par parallel algorithms (libstdc++
# dispatches the parallel STL to TBB) and provides the task_arena / global_control
# / enumerable_thread_specific used to bound and compose the nested parallelism.
find_package(TBB CONFIG REQUIRED)

# boost::parser is header-only and not yet in Ubuntu Boost packages;
# fetch from the official Boost Git mirror.
include(FetchContent)
FetchContent_Declare(boost_parser
        GIT_REPOSITORY https://github.com/boostorg/parser.git
        GIT_TAG boost-1.87.0
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
        GIT_TAG boost-1.87.0
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
