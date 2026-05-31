# ── clang-format ──────────────────────────────────────────────────────────────
# `format`       — rewrite every source/header in place using .clang-format.
# `format-check` — fail if anything is not formatted (CI-friendly; no rewrite).
# Only the project's own code under src/, include/ and tests/ is touched; the
# FetchContent build tree and cmake-build-* dirs are deliberately excluded.
# Toggle with -DPOTFIT_CLANG_FORMAT=OFF to skip creating the targets entirely.
option(POTFIT_CLANG_FORMAT "Provide 'format' / 'format-check' clang-format targets" ON)
find_program(CLANG_FORMAT_EXE NAMES clang-format clang-format-18 clang-format-17)
if (POTFIT_CLANG_FORMAT AND CLANG_FORMAT_EXE)
    file(GLOB_RECURSE POTFIT_FORMAT_FILES CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp"
    )

    add_custom_target(format
            COMMAND ${CLANG_FORMAT_EXE} -i --style=file ${POTFIT_FORMAT_FILES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running clang-format in place on ${CLANG_FORMAT_EXE}"
            VERBATIM
    )

    add_custom_target(format-check
            COMMAND ${CLANG_FORMAT_EXE} --style=file --dry-run -Werror ${POTFIT_FORMAT_FILES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Checking formatting with ${CLANG_FORMAT_EXE} (no files changed)"
            VERBATIM
    )
elseif (POTFIT_CLANG_FORMAT)
    message(STATUS "clang-format not found; 'format' / 'format-check' targets disabled")
endif ()
