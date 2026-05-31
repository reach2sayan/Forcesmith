#pragma once

#include "potfit/core/atom.hpp"
#include <Eigen/Core>
#include <concepts>
#include <cstdint>

namespace potfit {

template <typename T>
concept ForceCalculatorModel =
    requires(T calc, Configuration &cfg, Eigen::VectorXd &v, std::size_t off) {
        { calc.eval_forces(cfg) }       -> std::same_as<void>;
        { calc.param_count() }          -> std::same_as<std::size_t>;
        { calc.gather_params(v, off) }  -> std::same_as<void>;
        { calc.scatter_params(v, off) } -> std::same_as<void>;
        { calc.max_cutoff() }           -> std::same_as<double>;
    };

template <typename Derived> struct ForceCalculatorBase {
    std::size_t ntypes = 1;
    std::uint64_t conf_index = 0;
};

} // namespace potfit
