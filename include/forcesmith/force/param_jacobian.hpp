#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/fields.hpp"
#include "forcesmith/core/types.hpp"
#include "forcesmith/core/voigt.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

namespace forcesmith::force {

struct TableColumns {
  std::vector<std::vector<int>> of;
  std::vector<int> start;
  int total = 0;

  [[nodiscard]] int width(std::size_t t) const {
    return (t + 1 < start.size() ? start[t + 1] : total) - start[t];
  }
};

template <class Model>
[[nodiscard]] TableColumns table_columns(const Model &m) {
  TableColumns tc;
  for_each_table(m, [&](const auto &table, std::string_view) {
    tc.start.push_back(tc.total);
    std::vector<int> cols;
    for (const auto &p : table) {
      cols.push_back(tc.total);
      tc.total += static_cast<int>(p.param_count());
    }
    tc.of.push_back(std::move(cols));
  });
  return tc;
}

template <class Model>
[[nodiscard]] bool analytic_jacobian_available(const Model &m) {
  bool ok = true;
  for_each_table(m, [&](const auto &table, std::string_view) {
    for (const auto &p : table) {
      ok = ok && p.has_param_jacobian();
    }
  });
  if constexpr (requires { m.globals; }) {
    for (const auto &g : m.globals) {
      ok = ok && g.value.fixed;
    }
  }
  return ok;
}

struct JacRows {
  int force0;
  int energy;
  int stress0; // -1 when stress is not fitted
  int limit;

  JacRows(const Configuration &cfg, int row0, double stress_weight) {
    force0 = row0;
    energy = row0 + 3 * static_cast<int>(cfg.atoms.size());
    const bool with_stress = stress_weight > 0.0;
    stress0 = with_stress ? energy + 1 : -1;
    limit = energy + 1 + (with_stress ? static_cast<int>(kVoigtCount) : 0);
  }
};

struct Partials {
  std::vector<double> value, deriv;
  std::size_t n = 0;

  template <class Pot> void take(const Pot &pot, double x) {
    n = pot.param_count();
    value.resize(n);
    deriv.resize(n);
    if (n > 0) {
      pot.param_grad(x, value);
      pot.dderiv_dparam(x, deriv);
    }
  }
  [[nodiscard]] explicit operator bool() const { return n > 0; }
  [[nodiscard]] auto slots() const { return std::views::iota(std::size_t{0}, n); }
};

inline void add_force_column(Eigen::MatrixXd &J, const JacRows &rows,
                             int atom_row, int col, const Vec3 &d,
                             const Vec3 &df, double denergy,
                             double energy_weight, double stress_weight,
                             double inv_volume) {
  for (int a = 0; a < 3; ++a) {
    J(atom_row + a, col) += df[a];
  }
  if (denergy != 0.0) {
    J(rows.energy, col) += energy_weight * denergy;
  }
  if (rows.stress0 >= 0) {
    int v = 0;
    for (const auto &[a, b] : kVoigt6) {
      J(rows.stress0 + v++, col) -=
          stress_weight * 0.5 * d[a] * df[b] * inv_volume;
    }
  }
}

inline void add_radial_bond(Eigen::MatrixXd &J, const JacRows &rows,
                            int atom_row, int col0, const Vec3 &d, double inv_r,
                            const Partials &p, double scale, bool with_energy,
                            double energy_weight, double stress_weight,
                            double inv_volume) {
  for (const std::size_t k : p.slots()) {
    add_force_column(J, rows, atom_row, col0 + static_cast<int>(k), d,
                     (scale * p.deriv[k] * inv_r) * d,
                     with_energy ? 0.5 * p.value[k] : 0.0, energy_weight,
                     stress_weight, inv_volume);
  }
}

} // namespace forcesmith::force
