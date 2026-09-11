# FindIpopt.cmake — locate the Ipopt NLP solver
# Provides imported target Ipopt::Ipopt
# Hints: IPOPT_ROOT, PKG_CONFIG_PATH

include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(PC_IPOPT QUIET ipopt)
endif ()

find_path(IPOPT_INCLUDE_DIR
        NAMES IpIpoptApplication.hpp
        PATH_SUFFIXES coin-or coin
        HINTS
        ${IPOPT_ROOT}/include
        ${PC_IPOPT_INCLUDE_DIRS}
        /usr/include/coin
        /usr/local/include/coin
        /usr/include/coin-or
        /usr/local/include/coin-or
)

# ipopt.dll / ipopt-3: the import-library names of conda-forge's Windows build.
find_library(IPOPT_LIBRARY
        NAMES ipopt ipopt.dll ipopt-3
        HINTS
        ${IPOPT_ROOT}/lib
        ${PC_IPOPT_LIBRARY_DIRS}
        /usr/lib
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu
)

find_package_handle_standard_args(Ipopt
        REQUIRED_VARS IPOPT_LIBRARY IPOPT_INCLUDE_DIR
        VERSION_VAR PC_IPOPT_VERSION
)

if (Ipopt_FOUND AND NOT TARGET Ipopt::Ipopt)
    add_library(Ipopt::Ipopt UNKNOWN IMPORTED)
    set_target_properties(Ipopt::Ipopt PROPERTIES
            IMPORTED_LOCATION "${IPOPT_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${IPOPT_INCLUDE_DIR}"
    )
endif ()

mark_as_advanced(IPOPT_INCLUDE_DIR IPOPT_LIBRARY)
