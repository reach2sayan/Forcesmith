#ifndef FORCESMITH_API_FORCE_MODEL_STRATEGY_HPP
#define FORCESMITH_API_FORCE_MODEL_STRATEGY_HPP

#include "forcesmith/api/forcesmith.hpp"
#include "forcesmith/force/adp_force.hpp"
#include "forcesmith/force/angular_force.hpp"
#include "forcesmith/force/eam_force.hpp"
#include "forcesmith/force/pair_force.hpp"
#include "forcesmith/force/potential_table.hpp"
#include "forcesmith/force/stiweb_force.hpp"
#include "forcesmith/force/tersoff_force.hpp"
#include "forcesmith/io/config_reader.hpp"
#include "forcesmith/potentials/acsf.hpp"
#include "forcesmith/potentials/lmbtr.hpp"
#include "forcesmith/potentials/soap.hpp"

#include <boost/leaf/result.hpp>

#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace forcesmith::detail {

namespace leaf = boost::leaf;

[[nodiscard]] FORCE_INLINE leaf::error_id err(std::string msg) {
  return leaf::new_error(io::ParseError{std::move(msg), 0});
}

template <class Map>
[[nodiscard]] leaf::result<typename Map::mapped_type>
require(const Map &m, const typename Map::key_type &key, std::string what) {
  if (auto it = m.find(key); it != m.end()) {
    return it->second;
  }
  return err("missing " + std::move(what));
}

using LambdaKey = std::tuple<std::string, std::string, std::string>;

// Mutable references to the symbol-keyed spec maps + globals, bundled so the
// free strategies can read (materialize) and write (decompose) them.
struct SpecRef {
  const SpeciesRegistry &registry;
  std::map<Forcesmith::PairKey, Potential> &pair;
  std::map<std::string, Potential> &density;
  std::map<std::string, Potential> &embedding;
  std::map<Forcesmith::PairKey, Potential> &dipole;
  std::map<Forcesmith::PairKey, Potential> &quadrupole;
  std::map<Forcesmith::PairKey, Potential> &radial;
  std::map<std::string, Potential> &angular;
  std::map<Forcesmith::PairKey, TersoffParams> &tersoff;
  std::map<Forcesmith::PairKey, SWParams> &stiweb;
  std::map<LambdaKey, Param> &lambda;
  const std::vector<GlobalParam> &globals;
};

[[nodiscard]] inline Forcesmith::PairKey norm_key(std::string_view a,
                                              std::string_view b) {
  std::string sa(a), sb(b);
  if (sb < sa) {
    std::swap(sa, sb);
  }
  return {std::move(sa), std::move(sb)};
}

template <class V>
[[nodiscard]] leaf::result<SymmetricMatrix<V>>
build_pair_table(const std::map<Forcesmith::PairKey, V> &src,
                 const SpeciesRegistry &reg, std::string_view what) {
  SymmetricMatrix<V> mat;
  mat.reserve(ntypes(reg));
  for (auto [ti, tj] : mat.indices()) {
    const auto key =
        norm_key(species_at(reg, ti).symbol, species_at(reg, tj).symbol);
    BOOST_LEAF_AUTO(
        v, require(src, key,
                   std::string(what) + " for " + key.first + "-" + key.second));
    mat.emplace_back(std::move(v));
  }
  return mat;
}

[[nodiscard]] inline leaf::result<PotentialArray>
build_type_array(const std::map<std::string, Potential> &src,
                 const SpeciesRegistry &reg, std::string_view what) {
  PotentialArray arr;
  const std::size_t n = ntypes(reg);
  arr.reserve(n);
  for (std::size_t t = 0; t < n; ++t) {
    const std::string sym(species_at(reg, t).symbol);
    BOOST_LEAF_AUTO(p, require(src, sym, std::string(what) + " for " + sym));
    arr.emplace_back(std::move(p));
  }
  return arr;
}

// Stiweb λ flat vector, in the calculator's own order: ti outer, then the
// unordered neighbour pair in upper-triangular slot order (== lambda_index).
[[nodiscard]] inline leaf::result<std::vector<Param>>
build_lambda(const std::map<LambdaKey, Param> &src,
             const SpeciesRegistry &reg) {
  const std::size_t n = ntypes(reg);
  std::vector<Param> lam;
  lam.reserve(n * n * (n + 1) / 2);
  for (std::size_t ti = 0; ti < n; ++ti) {
    const std::string ci(species_at(reg, ti).symbol);
    for (auto [tj, tk] : upper_triangle(n)) {
      const auto nb =
          norm_key(species_at(reg, tj).symbol, species_at(reg, tk).symbol);
      BOOST_LEAF_AUTO(v, require(src, LambdaKey{ci, nb.first, nb.second},
                                 "stiweb lambda for " + ci + ":" + nb.first +
                                     "-" + nb.second));
      lam.push_back(v);
    }
  }
  return lam;
}

template <class V>
void dump_pair_table(const SymmetricMatrix<V> &mat, const SpeciesRegistry &reg,
                     std::map<Forcesmith::PairKey, V> &dst) {
  for (auto [ti, tj] : upper_triangle(ntypes(reg))) {
    dst.insert_or_assign(
        norm_key(species_at(reg, ti).symbol, species_at(reg, tj).symbol),
        mat[ti, tj]);
  }
}

inline void dump_type_array(const PotentialArray &arr,
                            const SpeciesRegistry &reg,
                            std::map<std::string, Potential> &dst) {
  for (std::size_t t = 0; t < ntypes(reg); ++t) {
    dst.insert_or_assign(std::string(species_at(reg, t).symbol), arr[t]);
  }
}

inline void dump_lambda(const StiwebForceCalculator &c,
                        const SpeciesRegistry &reg,
                        std::map<LambdaKey, Param> &dst) {
  const std::size_t n = ntypes(reg);
  for (std::size_t ti = 0; ti < n; ++ti) {
    const std::string ci(species_at(reg, ti).symbol);
    for (auto [tj, tk] : upper_triangle(n)) {
      const auto nb =
          norm_key(species_at(reg, tj).symbol, species_at(reg, tk).symbol);
      dst.insert_or_assign(LambdaKey{ci, nb.first, nb.second},
                           c.lambda_at(ti, tj, tk));
    }
  }
}

inline constexpr char kGlobalsRerankError[] =
    "decomposing a seeded model with global parameters after a re-rank is not "
    "supported; set potentials programmatically";

template <class Calc> struct PotentialType;

template <> struct PotentialType<PairForceCalculator> {
  static constexpr bool applies(const SpecRef &) {
    return true;
  } // unconditional fallback

  static leaf::result<ForceCalculator> materialize(const SpecRef &s) {
    if (s.pair.empty()) {
      return err("no pair potentials set");
    }
    PairForceCalculator calc;
    calc.ntypes = ntypes(s.registry);
    BOOST_LEAF_AUTO(pt, build_pair_table(s.pair, s.registry, "pair potential"));
    calc.pair = std::move(pt);
    calc.set_globals(s.globals);
    calc.finalize_globals();
    return ForceCalculator{std::move(calc)};
  }
  static leaf::result<void> decompose(const PairForceCalculator &c,
                                      const SpeciesRegistry &reg, SpecRef &s) {
    if (!c.globals_empty()) {
      return err(kGlobalsRerankError);
    }
    dump_pair_table(c.pair, reg, s.pair);
    return {};
  }
};

template <> struct PotentialType<EAMForceCalculator> {
  static bool applies(const SpecRef &s) {
    return !s.density.empty() || !s.embedding.empty();
  }
  static leaf::result<ForceCalculator> materialize(const SpecRef &s) {
    EAMForceCalculator calc;
    calc.ntypes = ntypes(s.registry);
    BOOST_LEAF_AUTO(pt, build_pair_table(s.pair, s.registry, "pair potential"));
    calc.pair = std::move(pt);
    BOOST_LEAF_AUTO(d, build_type_array(s.density, s.registry,
                                        "density (transfer) function"));
    calc.density = std::move(d);
    BOOST_LEAF_AUTO(
        e, build_type_array(s.embedding, s.registry, "embedding function"));
    calc.embedding = std::move(e);
    calc.set_globals(s.globals);
    calc.finalize_globals();
    return ForceCalculator{std::move(calc)};
  }
  static leaf::result<void> decompose(const EAMForceCalculator &c,
                                      const SpeciesRegistry &reg, SpecRef &s) {
    if (!c.globals_empty()) {
      return err(kGlobalsRerankError);
    }
    dump_pair_table(c.pair, reg, s.pair);
    dump_type_array(c.density, reg, s.density);
    dump_type_array(c.embedding, reg, s.embedding);
    return {};
  }
};

template <> struct PotentialType<ADPForceCalculator> {
  static bool applies(const SpecRef &s) {
    return !s.dipole.empty() || !s.quadrupole.empty();
  }
  static leaf::result<ForceCalculator> materialize(const SpecRef &s) {
    ADPForceCalculator calc;
    calc.ntypes = ntypes(s.registry);
    BOOST_LEAF_AUTO(pt, build_pair_table(s.pair, s.registry, "pair potential"));
    calc.pair = std::move(pt);
    BOOST_LEAF_AUTO(d, build_type_array(s.density, s.registry,
                                        "density (transfer) function"));
    calc.density = std::move(d);
    BOOST_LEAF_AUTO(
        e, build_type_array(s.embedding, s.registry, "embedding function"));
    calc.embedding = std::move(e);
    BOOST_LEAF_AUTO(u,
                    build_pair_table(s.dipole, s.registry, "dipole function"));
    calc.dipole = std::move(u);
    BOOST_LEAF_AUTO(
        w, build_pair_table(s.quadrupole, s.registry, "quadrupole function"));
    calc.quadrupole = std::move(w);
    calc.set_globals(s.globals); // no-op
    calc.finalize_globals();     // no-op
    return ForceCalculator{std::move(calc)};
  }
  static leaf::result<void> decompose(const ADPForceCalculator &c,
                                      const SpeciesRegistry &reg, SpecRef &s) {
    dump_pair_table(c.pair, reg, s.pair);
    dump_type_array(c.density, reg, s.density);
    dump_type_array(c.embedding, reg, s.embedding);
    dump_pair_table(c.dipole, reg, s.dipole);
    dump_pair_table(c.quadrupole, reg, s.quadrupole);
    return {};
  }
};

template <> struct PotentialType<AngularForceCalculator> {
  static bool applies(const SpecRef &s) {
    return !s.radial.empty() || !s.angular.empty();
  }
  static leaf::result<ForceCalculator> materialize(const SpecRef &s) {
    AngularForceCalculator calc;
    calc.ntypes = ntypes(s.registry);
    BOOST_LEAF_AUTO(pt, build_pair_table(s.pair, s.registry, "pair potential"));
    calc.pair = std::move(pt);
    BOOST_LEAF_AUTO(r,
                    build_pair_table(s.radial, s.registry, "radial function"));
    calc.radial = std::move(r);
    BOOST_LEAF_AUTO(
        g, build_type_array(s.angular, s.registry, "angular function"));
    calc.angular = std::move(g);
    calc.set_globals(s.globals); // no-op
    calc.finalize_globals();     // no-op
    return ForceCalculator{std::move(calc)};
  }
  static leaf::result<void> decompose(const AngularForceCalculator &c,
                                      const SpeciesRegistry &reg, SpecRef &s) {
    dump_pair_table(c.pair, reg, s.pair);
    dump_pair_table(c.radial, reg, s.radial);
    dump_type_array(c.angular, reg, s.angular);
    return {};
  }
};

template <> struct PotentialType<TersoffForceCalculator> {
  static bool applies(const SpecRef &s) { return !s.tersoff.empty(); }
  static leaf::result<ForceCalculator> materialize(const SpecRef &s) {
    TersoffForceCalculator calc;
    calc.ntypes = ntypes(s.registry);
    BOOST_LEAF_AUTO(
        p, build_pair_table(s.tersoff, s.registry, "tersoff parameters"));
    calc.params = std::move(p);
    calc.set_globals(s.globals); // no-op
    calc.finalize_globals();     // no-op
    return ForceCalculator{std::move(calc)};
  }
  static leaf::result<void> decompose(const TersoffForceCalculator &c,
                                      const SpeciesRegistry &reg, SpecRef &s) {
    dump_pair_table(c.params, reg, s.tersoff);
    return {};
  }
};

template <> struct PotentialType<StiwebForceCalculator> {
  static bool applies(const SpecRef &s) {
    return !s.stiweb.empty() || !s.lambda.empty();
  }

  static leaf::result<ForceCalculator> materialize(const SpecRef &s) {
    StiwebForceCalculator calc;
    calc.ntypes = ntypes(s.registry);
    BOOST_LEAF_AUTO(
        p, build_pair_table(s.stiweb, s.registry, "stiweb parameters"));
    calc.params = std::move(p);
    BOOST_LEAF_AUTO(lam, build_lambda(s.lambda, s.registry));
    calc.lambda = std::move(lam);
    calc.set_globals(s.globals); // no-op
    calc.finalize_globals();     // no-op
    return ForceCalculator{std::move(calc)};
  }

  static leaf::result<void> decompose(const StiwebForceCalculator &c,
                                      const SpeciesRegistry &reg, SpecRef &s) {
    dump_pair_table(c.params, reg, s.stiweb);
    dump_lambda(c, reg, s.lambda);
    return {};
  }
};

template <> struct PotentialType<ACSF> {
  static bool applies(const SpecRef &) { return false; }
  static leaf::result<ForceCalculator> materialize(const SpecRef &) {
    return err("ML models cannot be materialized from the spec maps; load them "
               "from a model file");
  }
  static leaf::result<void> decompose(const ACSF &, const SpeciesRegistry &,
                                      SpecRef &) {
    return err("ML models cannot be decomposed for re-ranking; load from a "
               "model file with the final element ordering");
  }
};

template <> struct PotentialType<SoapModel> {
  static bool applies(const SpecRef &) { return false; }
  static leaf::result<ForceCalculator> materialize(const SpecRef &) {
    return err("ML models cannot be materialized from the spec maps; load them "
               "from a model file");
  }
  static leaf::result<void> decompose(const SoapModel &,
                                      const SpeciesRegistry &, SpecRef &) {
    return err("ML models cannot be decomposed for re-ranking; load from a "
               "model file with the final element ordering");
  }
};

template <> struct PotentialType<LMBTR> {
  static bool applies(const SpecRef &) { return false; }
  static leaf::result<ForceCalculator> materialize(const SpecRef &) {
    return err("ML models cannot be materialized from the spec maps; load them "
               "from a model file");
  }
  static leaf::result<void> decompose(const LMBTR &, const SpeciesRegistry &,
                                      SpecRef &) {
    return err("ML models cannot be decomposed for re-ranking; load from a "
               "model file with the final element ordering");
  }
};

struct Probe {
  using apply_fn = bool (*)(const SpecRef &);
  using materialize_fn = leaf::result<ForceCalculator> (*)(const SpecRef &);
  apply_fn applies;
  materialize_fn materialize;
};

template <class Calc> constexpr Probe probe_for() {
  return {&PotentialType<Calc>::applies, &PotentialType<Calc>::materialize};
}

} // namespace forcesmith::detail

#endif // FORCESMITH_API_FORCE_MODEL_STRATEGY_HPP
