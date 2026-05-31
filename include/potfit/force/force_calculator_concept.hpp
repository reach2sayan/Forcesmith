#pragma once

#include "potfit/core/atom.hpp"
#include <Eigen/Core>
#include <concepts>
#include <cstdint>

namespace potfit {

template <typename T>
concept ForceCalculatorModel =
    requires(T calc, Configuration &cfg, Eigen::VectorXd &v, int off) {
        { calc.eval_forces(cfg) }       -> std::same_as<void>;
        { calc.param_count() }          -> std::same_as<int>;
        { calc.gather_params(v, off) }  -> std::same_as<void>;
        { calc.scatter_params(v, off) } -> std::same_as<void>;
        { calc.max_cutoff() }           -> std::same_as<double>;
    };

template <typename Derived> struct ForceCalculatorBase {
    int ntypes = 1;
    std::uint64_t conf_index = 0;
};

} // namespace potfit
