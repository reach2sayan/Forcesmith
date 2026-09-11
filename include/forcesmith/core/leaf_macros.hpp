#pragma once

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored                                               \
    "-Wgnu-statement-expression-from-macro-expansion"
#endif

#include <boost/leaf/handle_errors.hpp>
#include <boost/leaf/result.hpp>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
