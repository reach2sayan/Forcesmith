#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/fit_params.hpp"
#include "forcesmith/core/neighbor_list.hpp" // build_neighbor_list, bc_volume
#include "forcesmith/core/param.hpp"
#include "forcesmith/core/strong.hpp"
#include "forcesmith/core/voigt.hpp"
#include "forcesmith/events/signals.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/ml_energy_heads.hpp" // HeadConcept, HeadParams, EnergyHead
#include "forcesmith/force/potential_table.hpp" // TypeArray

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <execution>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace forcesmith {

struct TypeIndexTag;
using TypeIndex = Strong<std::size_t, TypeIndexTag, true>;

using DescriptorGrad = Eigen::Matrix<double, Eigen::Dynamic, 3>;
struct DescriptorValue {
  Eigen::VectorXd values; // D_i, length = descriptor_size
  bool has_grad = false;

  DescriptorGrad grad_self;               // dD_i/dr_i (S×3)
  std::vector<DescriptorGrad> grad_neigh; // dD_i/dr_j per neighbor jj
};

struct LinearHead : HeadParams<LinearHead> {
  std::vector<Param> coeffs;
  Param bias{0.0, true};

  double energy(const Eigen::VectorXd &D) const;
  Eigen::VectorXd grad(const Eigen::VectorXd &) const;
  constexpr bool constant_grad() const { return true; }
  bool has_param_jacobian() const { return true; }
  Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const;
  Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const;
  auto param_range() {
    return std::views::iota(std::size_t{0}, coeffs.size() + 1) |
           std::views::transform([this](std::size_t i) -> Param & {
             return i < coeffs.size() ? coeffs[i] : bias;
           });
  }
  auto param_range() const {
    return std::views::iota(std::size_t{0}, coeffs.size() + 1) |
           std::views::transform([this](std::size_t i) -> const Param & {
             return i < coeffs.size() ? coeffs[i] : bias;
           });
  }

  [[nodiscard]] LinearHead
  remapped(const std::vector<std::optional<Eigen::Index>> &map) const;
  [[nodiscard]] static LinearHead zero_like(Eigen::Index n);

  constexpr std::string type_tag() const { return "linear"; }
  constexpr std::vector<int> architecture() const {
    return {static_cast<int>(coeffs.size())};
  }
};

nlohmann::json build_json_from_head(const LinearHead &lh);

static_assert(CHead<LinearHead>);

struct DescriptorNeighbor {
  std::size_t slot; // element-type channel
  double r;         // |d|
};

[[nodiscard]] FORCE_INLINE std::optional<DescriptorNeighbor>
descriptor_neighbor(const NeighborEntry &nb, double rcut, std::size_t ntypes,
                    double r_eps) {
  const double r = nb.dist.norm();
  if (r < r_eps || r >= rcut) {
    return std::nullopt;
  }
  const auto slot = static_cast<long long>(nb.neighbor->type.index);
  if (slot < 0 || static_cast<std::size_t>(slot) >= ntypes) {
    return std::nullopt;
  }
  return DescriptorNeighbor{static_cast<std::size_t>(slot), r};
}

// What MLBase needs of the descriptor it is given. Naming it turns "MLBase
// silently failed to find get_descriptor on your model" into one line at the
// point of definition — the CRTP base called these four through
// static_cast<const Derived&> with nothing stating the requirement.
template <class T>
concept CDescriptorModel = requires(const T &t, const Atom &a) {
  { t.get_descriptor(a) };
  { t.descriptor_cutoff() } -> std::convertible_to<double>;
  { t.analytic_grads() } -> std::convertible_to<bool>;
  { t.descriptor_size() } -> std::convertible_to<std::size_t>;
};

template <typename Derived> struct MLBase : ForceCalculatorBase<Derived> {
  TypeArray<EnergyHead> heads;

  [[nodiscard]] std::vector<std::optional<Eigen::Index>>
  descriptor_index_map(const SpeciesRegistry &old_reg,
                       const SpeciesRegistry &new_reg) const
    requires requires(const Derived &d, std::size_t S) { d.layout_for(S); }
  {
    const std::size_t S_old = forcesmith::ntypes(old_reg);
    const std::size_t S_new = forcesmith::ntypes(new_reg);
    const Derived &self = static_cast<const Derived &>(*this);
    return remap_layout(self.layout_for(S_old).d_, self.layout_for(S_new).d_,
                        old_slot_of_new(old_reg, new_reg), S_old, S_new);
  }

  struct AtomCache {
    TypeIndex type{};       // element slot → head index
    Eigen::VectorXd values; // D_i  (descriptor_size S)

    Eigen::MatrixXd grad_all;
    std::vector<std::size_t> neigh_idx; // neighbor atom index in the config
    std::vector<Vec3> neigh_dist;       // bond r_j − r_i (for the virial)
  };
  struct ForceGroup {
    TypeIndex type{};
    Eigen::MatrixXd
        grad; // S × (3·nblocks): block b occupies columns [3b, 3b+3)
    std::vector<int> target; // [nblocks] atom of block b's forces
    std::vector<Vec3> bond;  // [nblocks] bond vector for the virial
  };
  struct CacheData {
    std::vector<std::vector<AtomCache>> rows; // [config][atom] (energy/values)
    std::vector<std::vector<ForceGroup>> groups; // [config][type]
    std::vector<double> volume;                  // per-config cell volume
    std::vector<Eigen::MatrixXd> dgrad_param;
  };
  mutable std::shared_ptr<const CacheData> cache_;
  void invalidate_cache() const { cache_.reset(); }
  [[nodiscard]] constexpr bool has_cache() const {
    return static_cast<bool>(cache_);
  }

  [[nodiscard]] bool has_param_jacobian() const {
    return heads.size() > 0 &&
           std::ranges::all_of(heads, [](const EnergyHead &h) {
             return h.has_param_jacobian();
           });
  }

  [[nodiscard]] bool all_constant_grad() const {
    return heads.size() > 0 &&
           std::ranges::all_of(
               heads, [](const EnergyHead &h) { return h.constant_grad(); });
  }

  [[nodiscard]] boost::leaf::result<Derived>
  remap(const SpeciesRegistry &old_reg, const SpeciesRegistry &new_reg) const {
    Derived out = self();
    const std::vector<std::optional<Eigen::Index>> idx =
        self().descriptor_index_map(old_reg, new_reg);
    const std::vector<std::optional<std::size_t>> old_of_new =
        old_slot_of_new(old_reg, new_reg);
    const std::size_t S_new = forcesmith::ntypes(new_reg);
    out.ntypes = S_new;
    const auto new_size = static_cast<Eigen::Index>(out.descriptor_size());

    TypeArray<EnergyHead> new_heads;
    new_heads.reserve(S_new);
    for (std::size_t t = 0; t < S_new; ++t) {
      if (const auto s_old = old_of_new[t]) {
        new_heads.emplace_back(heads[*s_old].remapped(idx));
      } else {
        new_heads.emplace_back(EnergyHead{LinearHead::zero_like(new_size)});
      }
    }
    out.heads = std::move(new_heads);

    out.mean_ = {};    // stale (old-layout); prepare() recomputes on next fit
    out.inv_std_ = {}; // (standardize_features flag is preserved by the copy)
    out.invalidate_cache();
    return out;
  }

  bool standardize_features = false;
  mutable TypeArray<Eigen::VectorXd> mean_; // per type, length descriptor_size
  mutable TypeArray<Eigen::VectorXd> inv_std_; // per type, 1/σ (0 for dead)

  [[nodiscard]] bool has_standardization() const {
    return standardize_features && inv_std_.size() > 0;
  }

private:
  using Worklist = std::vector<std::pair<std::size_t, std::size_t>>;

  static Worklist step_atom_worklist(std::span<Configuration> configs) {
    return std::views::iota(std::size_t{0}, configs.size()) |
           std::views::transform([configs](std::size_t c) {
             return std::views::iota(std::size_t{0},
                                     configs[c].atoms.size()) |
                    std::views::transform([c](std::size_t a) {
                      return Worklist::value_type{c, a};
                    });
           }) |
           std::views::join | std::ranges::to<Worklist>();
  }

  CacheData step_allocate_cache(std::span<Configuration> configs) const {
    CacheData data;
    data.rows.resize(configs.size());
    data.volume.resize(configs.size());
    build_all_neighbor_lists(configs, max_cutoff());
    for (auto [cfg, row, vol] :
         std::views::zip(configs, data.rows, data.volume)) {
      row.resize(cfg.atoms.size());
      vol = bc_volume(cfg.bc);
    }
    return data;
  }

  void step_warm_up_descriptors(std::span<Configuration> configs) const {
    for (auto &cfg : configs) {
      if (!cfg.atoms.empty()) {
        (void)self().get_descriptor(cfg.atoms[0]);
        break;
      }
    }
  }

  CacheData step_fill_cache(std::span<Configuration> configs,
                            const Worklist &work, CacheData data) const {
    std::for_each(std::execution::par, work.begin(), work.end(),
                  [&](const std::pair<std::size_t, std::size_t> &ca) {
                    fill_atom_cache(configs[ca.first], ca.second,
                                    data.rows[ca.first][ca.second]);
                  });
    return data;
  }

  CacheData step_whiten_cache(const Worklist &work, CacheData data) const {
    if (!standardize_features) {
      return data;
    }
    compute_standardization(data);
    std::for_each(std::execution::par, work.begin(), work.end(),
                  [&](const std::pair<std::size_t, std::size_t> &ca) {
                    AtomCache &cc = data.rows[ca.first][ca.second];
                    standardize_cached(cc.type, cc.values, cc.grad_all);
                  });
    return data;
  }

  CacheData step_group_cache(CacheData data) const {
    data.groups.resize(data.rows.size());
    if (!all_constant_grad()) {
      return data; // keep per-atom AtomCache layout for non-linear heads
    }
    const auto T = static_cast<std::size_t>(this->ntypes);
    std::vector<std::size_t> cidx(data.rows.size());
    std::iota(cidx.begin(), cidx.end(), std::size_t{0});
    std::for_each(
        std::execution::par, cidx.begin(), cidx.end(), [&](std::size_t c) {
          auto &row = data.rows[c];
          std::vector<Eigen::Index> ncol(T, 0);
          Eigen::Index S = 0;
          for (const AtomCache &cc : row) {
            ncol[cc.type] += cc.grad_all.cols();
            S = cc.grad_all.rows();
          }
          std::vector<ForceGroup> &grps = data.groups[c];
          std::vector<int> gidx(T, -1);
          for (std::size_t t = 0; t < T; ++t) {
            if (ncol[t] > 0) {
              gidx[t] = static_cast<int>(grps.size());
              ForceGroup g;
              g.type = TypeIndex{t};
              g.grad.resize(S, ncol[t]);
              const auto nb = static_cast<std::size_t>(ncol[t] / 3);
              g.target.reserve(nb);
              g.bond.reserve(nb);
              grps.push_back(std::move(g));
            }
          }
          std::vector<Eigen::Index> off(T, 0);
          for (std::size_t a = 0; a < row.size(); ++a) {
            AtomCache &cc = row[a];
            ForceGroup &g = grps[static_cast<std::size_t>(gidx[cc.type])];
            Eigen::Index &o = off[cc.type];
            g.grad.middleCols(o, cc.grad_all.cols()) = cc.grad_all;
            o += cc.grad_all.cols();
            g.target.push_back(static_cast<int>(a)); // self block
            g.bond.emplace_back(Vec3::Zero());
            for (auto [ni, nd] : std::views::zip(cc.neigh_idx, cc.neigh_dist)) {
              g.target.push_back(static_cast<int>(ni));
              g.bond.push_back(nd);
            }
            cc.grad_all.resize(0, 0); // released: data now lives in the group
            cc.neigh_idx = {};
            cc.neigh_dist = {};
          }
        });

    Eigen::Index S = 0;
    for (const auto &row : data.rows) {
      if (!row.empty()) {
        S = row.front().values.size();
        break;
      }
    }
    data.dgrad_param.resize(T);
    if (S > 0) {
      for (std::size_t t = 0; t < T; ++t) {
        data.dgrad_param[t] = heads[t].dgrad_dparam(Eigen::VectorXd::Zero(S));
      }
    }
    return data;
  }

public:
  void prepare(std::span<Configuration> configs) const {
    const Worklist work = step_atom_worklist(configs);

    CacheData data = step_allocate_cache(configs); // neighbour lists + sizing
    step_warm_up_descriptors(configs); // serial lazy descriptor init
    data = step_fill_cache(configs, work, std::move(data)); // parallel fill
    data = step_whiten_cache(work, std::move(data));        // optional whiten
    data = step_group_cache(std::move(data)); // batch linear heads per type

    cache_ = std::make_shared<CacheData>(std::move(data));
  }

  void eval_cached(std::size_t cache_index, std::span<Vec3> forces,
                   double &energy, SymTens &stress) const {
    const std::shared_ptr<const CacheData> cache = cache_;
    const auto &rows = cache->rows[cache_index];
    energy = 0.0;
    stress = SymTens::Zero();
    std::ranges::for_each(forces, [](Vec3 &f) { f.setZero(); });

    for (const AtomCache &cc : rows) {
      energy += heads[cc.type].energy(cc.values);
    }

    const auto &groups = cache->groups[cache_index];
    if (!groups.empty()) {
      for (const ForceGroup &g : groups) {
        const EnergyHead &h = heads[g.type];
        const Eigen::VectorXd dEdD =
            h.grad(Eigen::VectorXd{}); // const for linear
        const Eigen::VectorXd contrib = g.grad.transpose() * dEdD;
        for (std::size_t b = 0; b < g.target.size(); ++b) {
          const Vec3 cb = contrib.segment<3>(3 * static_cast<Eigen::Index>(b));
          forces[static_cast<std::size_t>(g.target[b])] -= cb;
          stress -= g.bond[b] * cb.transpose(); // bond==0 for self → no stress
        }
      }
    } else {
      for (auto [cc, fi] : std::views::zip(rows, forces)) {
        const EnergyHead &h = heads[cc.type];
        const Eigen::VectorXd dEdD = h.grad(cc.values);
        const Eigen::VectorXd contrib = cc.grad_all.transpose() * dEdD;
        fi -= contrib.head<3>();
        Eigen::Index b = 1;
        for (const auto &[nidx, nd] :
             std::views::zip(cc.neigh_idx, cc.neigh_dist)) {
          const Vec3 f_on_j = -contrib.segment<3>(3 * b);
          forces[nidx] += f_on_j;
          stress += nd * f_on_j.transpose();
          ++b;
        }
      }
    }
    stress /= cache->volume[cache_index];
  }

  void eval_cached_jacobian(std::size_t cache_index, int row0,
                            const std::vector<int> &col_off,
                            double energy_weight, double stress_weight,
                            Eigen::MatrixXd &fjac) const {
    const std::shared_ptr<const CacheData> cache = cache_;
    const auto &rows = cache->rows[cache_index];
    const int na = static_cast<int>(rows.size());
    const int erow = row0 + 3 * na; // energy residual row
    const int srow = erow + 1;      // first stress row (if any)
    const double inv_vol = 1.0 / cache->volume[cache_index];

    for (int i = 0; i < na; ++i) {
      const AtomCache &cc = rows[i];
      const std::size_t t = cc.type;
      const int c0 = col_off[t];
      const int pc = col_off[t + 1] - c0;
      if (pc == 0) {
        continue;
      }
      const Eigen::VectorXd gE = heads[t].param_grad(cc.values);
      fjac.block(erow, c0, 1, pc).noalias() += energy_weight * gE.transpose();
    }

    const auto scatter_stress = [&](int c0, int pc, const Vec3 &nd,
                                    const Eigen::MatrixXd &fblk) {
      if (stress_weight > 0.0) {
        for (int s = 0; s < 6; ++s) {
          fjac.block(srow + s, c0, 1, pc) +=
              (stress_weight * inv_vol * nd[kVoigt6[s].first]) *
              fblk.row(kVoigt6[s].second);
        }
      }
    };

    const auto &groups = cache->groups[cache_index];
    if (!groups.empty()) {
      for (const ForceGroup &g : groups) {
        const std::size_t t = g.type;
        const int c0 = col_off[t];
        const int pc = col_off[t + 1] - c0;
        if (pc == 0) {
          continue;
        }
        const Eigen::MatrixXd M =
            heads[t].dgrad_dparam(Eigen::VectorXd::Zero(g.grad.rows()));
        const Eigen::MatrixXd blocks = g.grad.transpose() * M; // (3·nblocks)×pc
        for (std::size_t b = 0; b < g.target.size(); ++b) {
          const Eigen::MatrixXd fblk =
              -blocks.middleRows(3 * static_cast<Eigen::Index>(b), 3); // 3×pc
          fjac.block(row0 + 3 * g.target[b], c0, 3, pc) += fblk;
          scatter_stress(c0, pc, g.bond[b], fblk);
        }
      }
    } else {
      for (int i = 0; i < na; ++i) {
        const AtomCache &cc = rows[i];
        const std::size_t t = cc.type;
        const int c0 = col_off[t];
        const int pc = col_off[t + 1] - c0;
        if (pc == 0) {
          continue;
        }
        const Eigen::MatrixXd M = heads[t].dgrad_dparam(cc.values);
        const Eigen::MatrixXd blocks = cc.grad_all.transpose() * M;
        fjac.block(row0 + 3 * i, c0, 3, pc).noalias() -= blocks.topRows(3);
        Eigen::Index b = 1;
        for (const auto &[nidx, nd] :
             std::views::zip(cc.neigh_idx, cc.neigh_dist)) {
          const Eigen::MatrixXd fblk = -blocks.middleRows(3 * b, 3); // 3 × pc
          fjac.block(row0 + 3 * static_cast<int>(nidx), c0, 3, pc) += fblk;
          scatter_stress(c0, pc, nd, fblk);
          ++b;
        }
      }
    }
  }

  void eval_forces(Configuration &cfg) const {
    build_neighbor_list(cfg, max_cutoff());

    cfg.calc_energy = 0.0;
    cfg.calc_stress = SymTens::Zero();
    for (auto &a : cfg.atoms) {
      a.calc_force = Vec3::Zero();
    }

    for (auto &ai : cfg.atoms) {
      DescriptorValue d = descriptor_with_grad(ai);
      standardize_in_place(TypeIndex{ai.type.index}, d);
      const EnergyHead &h = heads[ai.type.index];
      cfg.calc_energy += h.energy(d.values);
      accumulate_atom_forces(cfg, ai, d, h.grad(d.values));
    }

    cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)

    events::on_force_eval(
        events::ForceEvalStats{this->conf_index, force_rms(cfg), cfg});
  }

  void eval_forces(Configuration &cfg, std::size_t cache_index) const {
    if (!cache_ || cache_index >= cache_->rows.size()) {
      eval_forces(cfg);
      return;
    }
    std::vector<Vec3> f(cfg.atoms.size());
    double e = 0.0;
    SymTens s = SymTens::Zero();
    eval_cached(cache_index, f, e, s);
    for (std::size_t a = 0; a < cfg.atoms.size(); ++a) {
      cfg.atoms[a].calc_force = f[a];
    }
    cfg.calc_energy = e;
    cfg.calc_stress = s;
    cfg.calc_limit = 0.0; // ML models have no embedding range penalty
    events::on_force_eval(
        events::ForceEvalStats{this->conf_index, force_rms(cfg), cfg});
  }

  constexpr std::size_t param_count() const {
    return std::ranges::fold_left(
        heads, std::size_t{0},
        [](std::size_t n, const auto &h) { return n + h.param_count(); });
  }

  std::vector<std::size_t> head_param_counts() const {
    return heads |
           std::views::transform([](const EnergyHead &h) {
             return h.param_count();
           }) |
           std::ranges::to<std::vector<std::size_t>>();
  }

  void gather_params(Eigen::VectorXd &dst, std::size_t off) const {
    std::for_each(heads.begin(), heads.end(), [&](const auto &h) {
      h.gather_params(dst, off);
      off += h.param_count();
    });
  }

  void scatter_params(const Eigen::VectorXd &src, std::size_t off) {
    for (auto &h : heads) {
      h.scatter_params(src, off);
      off += h.param_count();
    }
  }

  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const {
    std::for_each(heads.begin(), heads.end(), [&](const auto &h) {
      h.gather_bounds(lo, hi, off);
      off += h.param_count();
    });
  }

  constexpr double max_cutoff() const { return self().descriptor_cutoff(); }

private:
  const Derived &self() const { return static_cast<const Derived &>(*this); }

  DescriptorValue descriptor_with_grad(Atom &atom) const {
    DescriptorValue d = self().get_descriptor(atom);
    if (self().analytic_grads() && d.has_grad) {
      return d;
    }
    const Eigen::Index S = d.values.size();
    d.grad_neigh.assign(atom.neighbors.size(), DescriptorGrad::Zero(S, 3));
    d.grad_self = DescriptorGrad::Zero(S, 3);
    constexpr double h = 1e-4;
    for (auto [nb, gn] : std::views::zip(atom.neighbors, d.grad_neigh)) {
      DescriptorGrad g(S, 3);
      for (auto &&[cdim, nbdist] :
           nb.dist | std::views::take(3) |
               std::views::enumerate) { // int cdim = 0; cdim < 3; ++cdim) {
        const double x0 = nbdist;
        nbdist = x0 + h;
        const Eigen::VectorXd dp = self().get_descriptor(atom).values;
        nbdist = x0 - h;
        const Eigen::VectorXd dm = self().get_descriptor(atom).values;
        nbdist = x0;
        g.col(cdim) = (dp - dm) / (2.0 * h);
      }
      gn = g;
      d.grad_self -= g;
    }
    d.has_grad = true;
    return d;
  }

  void fill_atom_cache(Configuration &cfg, std::size_t a, AtomCache &cc) const {
    Atom &atom = cfg.atoms[a];
    cc.type = TypeIndex{atom.type.index};
    const DescriptorValue d = descriptor_with_grad(atom);
    cc.values = d.values;
    const Eigen::Index S = d.values.size();
    const auto K = static_cast<Eigen::Index>(d.grad_neigh.size());
    cc.grad_all.resize(S, 3 * (K + 1));
    cc.grad_all.leftCols(3) = d.grad_self;
    for (Eigen::Index k = 0; k < K; ++k) {
      cc.grad_all.middleCols(3 * (k + 1), 3) = d.grad_neigh[k];
    }
    cc.neigh_idx.resize(atom.neighbors.size());
    cc.neigh_dist.resize(atom.neighbors.size());
    for (auto [nb, nidx, nd] :
         std::views::zip(atom.neighbors, cc.neigh_idx, cc.neigh_dist)) {
      nidx = static_cast<std::size_t>(nb.neighbor - cfg.atoms.data());
      nd = nb.dist;
    }
  }

  void accumulate_atom_forces(Configuration &cfg, Atom &ai,
                              const DescriptorValue &d,
                              const Eigen::VectorXd &dEdD) const {
    ai.calc_force -= d.grad_self.transpose() * dEdD;
    for (const auto &[nb, gn] : std::views::zip(ai.neighbors, d.grad_neigh)) {
      Atom &aj = const_cast<Atom &>(*nb.neighbor);
      const Vec3 f_on_j = -(gn.transpose() * dEdD);
      aj.calc_force += f_on_j;
      cfg.calc_stress += nb.dist * f_on_j.transpose();
    }
  }

  void standardize_in_place(TypeIndex type, Eigen::VectorXd &values,
                            DescriptorGrad &grad_self,
                            std::vector<DescriptorGrad> &grad_neigh) const {
    if (!has_standardization()) {
      return;
    }
    const std::size_t t = type;
    const Eigen::VectorXd &mu = mean_[t];
    const Eigen::VectorXd &iv = inv_std_[t];
    if (mu.size() != values.size()) {
      return; // type had no training atoms → identity
    }
    values = (values - mu).cwiseProduct(iv);
    grad_self = iv.asDiagonal() * grad_self;
    for (auto &g : grad_neigh) {
      g = iv.asDiagonal() * g;
    }
  }
  void standardize_in_place(TypeIndex type, DescriptorValue &d) const {
    standardize_in_place(type, d.values, d.grad_self, d.grad_neigh);
  }

  void standardize_cached(TypeIndex type, Eigen::VectorXd &values,
                          Eigen::MatrixXd &grad_all) const {
    if (!has_standardization()) {
      return;
    }
    const std::size_t t = type;
    const Eigen::VectorXd &mu = mean_[t];
    const Eigen::VectorXd &iv = inv_std_[t];
    if (mu.size() != values.size()) {
      return; // type had no training atoms → identity
    }
    values = (values - mu).cwiseProduct(iv);
    grad_all = iv.asDiagonal() * grad_all;
  }

  void compute_standardization(const CacheData &data) const {
    Eigen::Index S = 0;
    for (const auto &row : data.rows) {
      if (!row.empty()) {
        S = row.front().values.size();
        break;
      }
    }
    const std::size_t T = static_cast<std::size_t>(this->ntypes);
    if (S == 0 || T == 0) {
      return; // nothing to standardize
    }
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(S);
    std::vector<Eigen::VectorXd> sum(T, zero);
    std::vector<Eigen::VectorXd> sumsq(T, zero);
    std::vector<std::size_t> count(T, 0);
    for (const auto &row : data.rows) {
      for (const AtomCache &cc : row) {
        const std::size_t t = cc.type;
        sum[t] += cc.values;
        sumsq[t] += cc.values.cwiseProduct(cc.values);
        ++count[t];
      }
    }
    constexpr double eps = 1e-12;
    TypeArray<Eigen::VectorXd> mean, inv_std;
    mean.reserve(T);
    inv_std.reserve(T);
    for (std::size_t t = 0; t < T; ++t) {
      if (count[t] == 0) {
        mean.emplace_back(Eigen::VectorXd{}); // empty → identity for this type
        inv_std.emplace_back(Eigen::VectorXd{});
        continue;
      }
      const double n = static_cast<double>(count[t]);
      const Eigen::VectorXd mu = sum[t] / n;
      const Eigen::VectorXd var = (sumsq[t] / n) - mu.array().square().matrix();

      mean.emplace_back(std::move(mu));
      inv_std.push_back(
          (var.array() > eps).select(var.array().rsqrt(), 0.0).matrix());
    }
    mean_ = std::move(mean);
    inv_std_ = std::move(inv_std);
  }
};

} // namespace forcesmith
