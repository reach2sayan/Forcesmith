#include "potfit/api/potfit.hpp"

#include "potfit/core/neighbor_list.hpp"
#include "potfit/force/adp_force.hpp"
#include "potfit/force/angular_force.hpp"
#include "potfit/force/eam_force.hpp"
#include "potfit/force/pair_force.hpp"
#include "potfit/force/potential_table.hpp"
#include "potfit/force/stiweb_force.hpp"
#include "potfit/force/tersoff_force.hpp"
#include "potfit/io/config_reader.hpp" // ParseError
#include "potfit/io/write_model.hpp"

#include <algorithm>
#include <iterator>
#include <map>
#include <ranges>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace potfit {

namespace leaf = boost::leaf;

namespace {

[[nodiscard]] leaf::error_id err(std::string msg) {
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

[[nodiscard]] constexpr std::size_t ntypes_of(const ForceCalculator &m) {
  return std::visit([](const auto &c) { return c.ntypes; }, m);
}
[[nodiscard]] constexpr double max_cutoff_of(const ForceCalculator &m) {
  return std::visit([](const auto &c) { return c.max_cutoff(); }, m);
}
// Radial pair table (φ_ij) when the model has one; nullptr for tersoff/stiweb.
[[nodiscard]] constexpr const PotentialPair *
pair_table_of(const ForceCalculator &m) {
  return std::visit(
      [](const auto &c) -> const PotentialPair * {
        if constexpr (requires { c.pair; }) {
          return &c.pair;
        } else {
          return nullptr;
        }
      },
      m);
}

} // namespace

PotFit::PairKey PotFit::norm_key(std::string_view a,
                                         std::string_view b) {
  std::string sa(a);
  std::string sb(b);
  if (sb < sa) {
    std::swap(sa, sb);
  }
  return {std::move(sa), std::move(sb)};
}

leaf::result<Configuration *> PotFit::config_at(std::size_t cfg) {
  if (cfg >= configs_.size()) {
    return err("configuration index " + std::to_string(cfg) +
               " out of range (" + std::to_string(configs_.size()) + ")");
  }
  return &configs_[cfg];
}

leaf::result<void> PotFit::ensure_named() {
  for (auto &&[i, cfg] : std::views::enumerate(configs_)) {
    if (cfg.name.empty()) {
      cfg.name = "config-" + std::to_string(i);
    }
  }
  std::vector<std::string_view> names;
  names.reserve(configs_.size());
  std::ranges::transform(configs_, std::back_inserter(names),
                         &Configuration::name);
  std::ranges::sort(names);
  if (auto dup = std::ranges::adjacent_find(names); dup != names.end()) {
    return err("duplicate configuration name '" + std::string(*dup) + "'");
  }
  return {};
}

leaf::result<std::size_t>
PotFit::get_configuration_index(std::string_view name) {
  BOOST_LEAF_CHECK(ensure_named());
  for (auto &&[i, cfg] : configs_ | std::views::enumerate) {
    if (cfg.name == name) {
      return static_cast<std::size_t>(i);
    }
  }
  return err("no configuration named '" + std::string(name) + "'");
}

leaf::result<std::size_t>
PotFit::get_configuration_index(const Configuration &cfg) const {
  // Address equality (well-defined for any pointer), not a relational
  // pointer-range check against a possibly-foreign &cfg (unspecified behavior).
  const auto it = std::ranges::find_if(
      configs_, [&](const Configuration &c) { return &c == &cfg; });
  if (it == configs_.end()) {
    return err("configuration is not owned by this session");
  }
  return static_cast<std::size_t>(std::distance(configs_.begin(), it));
}

leaf::result<std::size_t> PotFit::get_atom_index(const Atom &atom) {
  BOOST_LEAF_CHECK(ensure_frozen()); // stamps Atom::parent for owned atoms
  if (atom.parent == nullptr) {
    return err("atom is not owned by this session");
  }
  // Legal pointer subtraction: `atom` genuinely lives in parent->atoms.
  return static_cast<std::size_t>(&atom - atom.parent->atoms.data());
}

std::size_t PotFit::add_configuration(BoundaryConditions bc) {
  Configuration cfg;
  cfg.bc = std::move(bc);
  return add_configuration(std::move(cfg));
  return configs_.size() - 1;
}

std::size_t PotFit::add_configuration(Configuration cfg) {
  configs_.push_back(std::move(cfg));
  dirty_ = true;
  return configs_.size() - 1;
}

leaf::result<void> PotFit::remove_configuration(std::size_t cfg) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  configs_.erase(configs_.begin() +
                 static_cast<std::ptrdiff_t>(c - configs_.data()));
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_cell(std::size_t cfg, const Mat3 &box) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->bc = PeriodicBC(box);
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_infinite(std::size_t cfg, double volume) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->bc = InfiniteBC(volume);
  dirty_ = true;
  return {};
}

leaf::result<std::size_t> PotFit::add_atom(std::size_t cfg,
                                               std::string_view element,
                                               const Vec3 &pos) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  BOOST_LEAF_AUTO(sp, Species::lookup(element));
  Atom a;
  a.type = sp; // slot stamped at freeze
  a.pos = pos;
  c->atoms.push_back(std::move(a));
  dirty_ = true;
  return c->atoms.size() - 1;
}

leaf::result<void> PotFit::remove_atom(std::size_t cfg, std::size_t atom) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  if (atom >= c->atoms.size()) {
    return err("atom index out of range");
  }
  c->atoms.erase(c->atoms.begin() + static_cast<std::ptrdiff_t>(atom));
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_position(std::size_t cfg, std::size_t atom,
                                            const Vec3 &pos) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  if (atom >= c->atoms.size()) {
    return err("atom index out of range");
  }
  c->atoms[atom].pos = pos;
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_element(std::size_t cfg, std::size_t atom,
                                           std::string_view element) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  if (atom >= c->atoms.size()) {
    return err("atom index out of range");
  }
  BOOST_LEAF_AUTO(sp, Species::lookup(element));
  c->atoms[atom].type = sp;
  dirty_ = true;
  return {};
}

leaf::result<std::size_t> PotFit::atom_count(std::size_t cfg) const {
  if (cfg >= configs_.size()) {
    return err("configuration index out of range");
  }
  return configs_[cfg].atoms.size();
}

// ── reference data (no dirty) ────────────────────────────────────────────────
leaf::result<void>
PotFit::write_ref_force(std::size_t cfg, std::size_t atom, const Vec3 &f) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  if (atom >= c->atoms.size()) {
    return err("atom index out of range");
  }
  c->atoms[atom].ref.force = f;
  return {};
}

leaf::result<void> PotFit::set_ref_energy(std::size_t cfg, double e) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->ref.energy = e;
  return {};
}

leaf::result<void> PotFit::set_ref_stress(std::size_t cfg,
                                              const SymTens &s) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->ref.stress = s;
  return {};
}

leaf::result<void> PotFit::set_weight(std::size_t cfg, double w) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->weight = w;
  return {};
}

leaf::result<void> PotFit::set_ref_force(const Atom &atom, const Vec3 &f) {
  BOOST_LEAF_AUTO(ai, get_atom_index(atom)); // freezes; stamps atom.parent
  BOOST_LEAF_AUTO(ci, get_configuration_index(*atom.parent));
  return write_ref_force(ci, ai, f);
}

leaf::result<void> PotFit::set_ref_force(std::string_view cfg,
                                             std::size_t atom, const Vec3 &f) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return write_ref_force(ci, atom, f);
}

leaf::result<void> PotFit::set_ref_energy(std::string_view cfg, double e) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_ref_energy(ci, e);
}

leaf::result<void> PotFit::set_ref_energy(const Configuration &cfg,
                                              double e) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_ref_energy(ci, e);
}

leaf::result<void> PotFit::set_ref_stress(std::string_view cfg,
                                              const SymTens &s) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_ref_stress(ci, s);
}

leaf::result<void> PotFit::set_ref_stress(const Configuration &cfg,
                                              const SymTens &s) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_ref_stress(ci, s);
}

leaf::result<void> PotFit::set_weight(std::string_view cfg, double w) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_weight(ci, w);
}

leaf::result<void> PotFit::set_weight(const Configuration &cfg, double w) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_weight(ci, w);
}

leaf::result<void> PotFit::declare_element(std::string_view sym) {
  BOOST_LEAF_CHECK(Species::lookup(sym)); // validate against the catalog
  if (std::ranges::find(declared_, std::string(sym)) == declared_.end()) {
    declared_.emplace_back(sym);
    dirty_ = true;
  }
  return {};
}

leaf::result<void> PotFit::set_pair_potential(std::string_view a,
                                                  std::string_view b,
                                                  Potential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  pair_.insert_or_assign(norm_key(a, b), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_density(std::string_view a, Potential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  density_.insert_or_assign(std::string(a), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_embedding(std::string_view a, Potential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  embedding_.insert_or_assign(std::string(a), std::move(p));
  dirty_ = true;
  return {};
}

void PotFit::set_global(GlobalParam g) {
  globals_.push_back(std::move(g));
  dirty_ = true;
}

leaf::result<void> PotFit::set_dipole(std::string_view a,
                                          std::string_view b, Potential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  dipole_.insert_or_assign(norm_key(a, b), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_quadrupole(std::string_view a,
                                              std::string_view b, Potential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  quadrupole_.insert_or_assign(norm_key(a, b), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_radial(std::string_view a,
                                          std::string_view b, Potential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  radial_.insert_or_assign(norm_key(a, b), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_angular(std::string_view a, Potential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  angular_.insert_or_assign(std::string(a), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_tersoff_params(std::string_view a,
                                                  std::string_view b,
                                                  TersoffParams params) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  tersoff_.insert_or_assign(norm_key(a, b), std::move(params));
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_stiweb_params(std::string_view a,
                                                 std::string_view b,
                                                 SWParams params) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  stiweb_.insert_or_assign(norm_key(a, b), std::move(params));
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_stiweb_lambda(std::string_view central,
                                                 std::string_view a,
                                                 std::string_view b,
                                                 Param value) {
  BOOST_LEAF_CHECK(Species::lookup(central));
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  const PairKey nb = norm_key(a, b);
  lambda_.insert_or_assign(
      std::tuple{std::string(central), nb.first, nb.second}, value);
  dirty_ = true;
  return {};
}

leaf::result<void> PotFit::set_pair_param(std::string_view a,
                                              std::string_view b, std::size_t i,
                                              double v) {
  BOOST_LEAF_CHECK(ensure_frozen());
  const PotentialPair *pt = pair_table_of(model_);
  if (pt == nullptr) {
    return err("this model has no pair table to edit");
  }
  BOOST_LEAF_AUTO(sa, species_of(registry_, a));
  BOOST_LEAF_AUTO(sb, species_of(registry_, b));
  // Edit the live materialized potential (value-only; does not dirty).
  std::visit(
      [&](auto &c) {
        if constexpr (requires { c.pair; }) {
          c.pair[sa.index, sb.index].set_param(i, v);
        }
      },
      model_);
  return {};
}

leaf::result<void> PotFit::seed_force_model(ForceCalculator model) {
  // Capture the element ordering this model was built against, so a later edit
  // that re-ranks slots can still recover the per-symbol potentials.
  BOOST_LEAF_AUTO(reg, build_registry());
  seeded_registry_ = std::move(reg);
  seeded_ = std::move(model);
  // Programmatic spec (if any) is superseded by the seeded model.
  pair_.clear();
  density_.clear();
  embedding_.clear();
  dipole_.clear();
  quadrupole_.clear();
  radial_.clear();
  angular_.clear();
  tersoff_.clear();
  stiweb_.clear();
  lambda_.clear();
  globals_.clear();
  dirty_ = true;
  return {};
}

// ── build helpers ────────────────────────────────────────────────────────────
leaf::result<SpeciesRegistry> PotFit::build_registry() const {
  std::vector<std::string_view> syms;
  auto add = [&](std::string_view s) {
    if (!s.empty() && std::ranges::find(syms, s) == syms.end()) {
      syms.push_back(s);
    }
  };

  auto add_pair_keys = [&](const auto &m) {
    std::ranges::for_each(m | std::views::keys, [&](const auto &k) {
      add(k.first);
      add(k.second);
    });
  };
  add_pair_keys(pair_);
  add_pair_keys(dipole_);
  add_pair_keys(quadrupole_);
  add_pair_keys(radial_);
  add_pair_keys(tersoff_);
  add_pair_keys(stiweb_);

  std::ranges::for_each(density_ | std::views::keys, add);
  std::ranges::for_each(embedding_ | std::views::keys, add);
  std::ranges::for_each(angular_ | std::views::keys, add);
  // Stiweb λ keys are (central, nb_a, nb_b) — every component is an element.
  std::ranges::for_each(lambda_ | std::views::keys, [&](const auto &k) {
    add(std::get<0>(k));
    add(std::get<1>(k));
    add(std::get<2>(k));
  });
  std::ranges::for_each(declared_, add);

  // Elements present in the loaded configurations. A model seeded from a file
  // (load_model) leaves the editable spec maps empty, so the config atoms are
  // the only source of element symbols in that path.
  for (const auto &cfg : configs_) {
    for (const auto &a : cfg.atoms) {
      add(a.type.symbol);
    }
  }

  return build_species_registry(syms);
}

namespace {

// ── model-family strategy registry ───────────────────────────────────────────
// Each force-calculator family is a strategy that knows (a) when it applies to
// the editable spec, (b) how to build itself from the spec, and (c) how to
// decompose a built model back into the spec. The forward direction is selected
// by probing `applies` in priority order; the inverse is selected by the
// variant alternative. Globals are handled uniformly through the
// WithGlobals/NoGlobals mixin interface (set_globals / finalize_globals /
// globals_empty), so no family special-cases them.

using LambdaKey = std::tuple<std::string, std::string, std::string>;

// Mutable references to the symbol-keyed spec maps + globals, bundled so the
// free strategies can read (materialize) and write (decompose) them.
struct SpecRef {
  const SpeciesRegistry &registry;
  std::map<PotFit::PairKey, Potential> &pair;
  std::map<std::string, Potential> &density;
  std::map<std::string, Potential> &embedding;
  std::map<PotFit::PairKey, Potential> &dipole;
  std::map<PotFit::PairKey, Potential> &quadrupole;
  std::map<PotFit::PairKey, Potential> &radial;
  std::map<std::string, Potential> &angular;
  std::map<PotFit::PairKey, TersoffParams> &tersoff;
  std::map<PotFit::PairKey, SWParams> &stiweb;
  std::map<LambdaKey, Param> &lambda;
  const std::vector<GlobalParam> &globals;
};

[[nodiscard]] PotFit::PairKey norm_key(std::string_view a,
                                           std::string_view b) {
  std::string sa(a), sb(b);
  if (sb < sa) {
    std::swap(sa, sb);
  }
  return {std::move(sa), std::move(sb)};
}

// ── generic builders (spec → table) ──────────────────────────────────────────
template <class V>
[[nodiscard]] leaf::result<SymmetricMatrix<V>>
build_pair_table(const std::map<PotFit::PairKey, V> &src,
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

[[nodiscard]] leaf::result<PotentialArray>
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
[[nodiscard]] leaf::result<std::vector<Param>>
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

// ── generic dumpers (table → spec) ───────────────────────────────────────────
template <class V>
void dump_pair_table(const SymmetricMatrix<V> &mat, const SpeciesRegistry &reg,
                     std::map<PotFit::PairKey, V> &dst) {
  for (auto [ti, tj] : upper_triangle(ntypes(reg))) {
    dst.insert_or_assign(
        norm_key(species_at(reg, ti).symbol, species_at(reg, tj).symbol),
        mat[ti, tj]);
  }
}

void dump_type_array(const PotentialArray &arr, const SpeciesRegistry &reg,
                     std::map<std::string, Potential> &dst) {
  for (std::size_t t = 0; t < ntypes(reg); ++t) {
    dst.insert_or_assign(std::string(species_at(reg, t).symbol), arr[t]);
  }
}

void dump_lambda(const StiwebForceCalculator &c, const SpeciesRegistry &reg,
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

constexpr char kGlobalsRerankError[] =
    "decomposing a seeded model with global parameters after a re-rank is not "
    "supported; set potentials programmatically";

// ── per-family strategies ────────────────────────────────────────────────────
template <class Calc> struct Family; // primary left undefined

template <> struct Family<PairForceCalculator> {
  static bool applies(const SpecRef &) {
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

template <> struct Family<EAMForceCalculator> {
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

template <> struct Family<ADPForceCalculator> {
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

template <> struct Family<AngularForceCalculator> {
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

template <> struct Family<TersoffForceCalculator> {
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

template <> struct Family<StiwebForceCalculator> {
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

// Forward registry: probed in priority order — most specific family first, the
// pair fallback last. ADP precedes EAM because ADP also populates
// density/embedding; angular precedes the analytic families on its own tables.
struct Probe {
  bool (*applies)(const SpecRef &);
  leaf::result<ForceCalculator> (*materialize)(const SpecRef &);
};

template <class Calc> constexpr Probe probe_for() {
  return {&Family<Calc>::applies, &Family<Calc>::materialize};
}

} // namespace

leaf::result<void> PotFit::materialize_from_spec() {
  SpecRef spec{registry_, pair_,    density_, embedding_, dipole_, quadrupole_,
               radial_,   angular_, tersoff_, stiweb_,    lambda_, globals_};

  const Probe probes[] = {
      probe_for<ADPForceCalculator>(),     probe_for<AngularForceCalculator>(),
      probe_for<TersoffForceCalculator>(), probe_for<StiwebForceCalculator>(),
      probe_for<EAMForceCalculator>(),     probe_for<PairForceCalculator>(),
  };

  const auto &chosen = *std::ranges::find_if(
      probes, [&](const Probe &p) { return p.applies(spec); });
  BOOST_LEAF_AUTO(built, chosen.materialize(spec));
  model_ = std::move(built);
  return {};
}

leaf::result<void> PotFit::detach_seeded(std::string_view action) {
  if (!seeded_) {
    return {};
  }
  if (!seeded_registry_) {
    return err("cannot " + std::string(action) +
               " a seeded model: original element ordering unknown");
  }
  BOOST_LEAF_CHECK(decompose_seeded_into_spec(*seeded_registry_));
  seeded_.reset();
  seeded_registry_.reset();
  return {};
}

leaf::result<void>
PotFit::decompose_seeded_into_spec(const SpeciesRegistry &reg) {
  if (!seeded_) {
    return {};
  }
  SpecRef spec{registry_, pair_,    density_, embedding_, dipole_, quadrupole_,
               radial_,   angular_, tersoff_, stiweb_,    lambda_, globals_};
  return std::visit(
      [&](auto &c) -> leaf::result<void> {
        return Family<std::decay_t<decltype(c)>>::decompose(c, reg, spec);
      },
      *seeded_);
}

leaf::result<void> PotFit::ensure_frozen() {
  if (!dirty_) {
    return {};
  }

  BOOST_LEAF_AUTO(reg, build_registry());
  registry_ = std::move(reg);
  const std::size_t n = ntypes(registry_);

  // Stamp the final compact slot onto every atom (in place).
  for (auto &cfg : configs_) {
    for (auto &a : cfg.atoms) {
      BOOST_LEAF_AUTO(sp, species_of(registry_, a.type.symbol));
      a.type = sp;
    }
  }

  // Build the force model.
  if (seeded_) {
    if (ntypes_of(*seeded_) == n) {
      model_ = *seeded_;
    } else {
      BOOST_LEAF_CHECK(detach_seeded("re-rank"));
      BOOST_LEAF_CHECK(materialize_from_spec());
    }
  } else {
    BOOST_LEAF_CHECK(materialize_from_spec());
  }

  // Build neighbor lists against the materialized model (must outlive them).
  const double rcut = max_cutoff_of(model_);
  const PotentialPair *pt = pair_table_of(model_);
  for (auto &cfg : configs_) {
    if (pt != nullptr) {
      build_neighbor_list(cfg, rcut, *pt);
    } else {
      build_neighbor_list(cfg, rcut);
    }
  }

  BOOST_LEAF_CHECK(ensure_named());

  // Auxiliary grouping index (configs_ must not be resized/reordered
  // hereafter).
  index_ = config_index::build_config_index(configs_);
  if (ntypes_of(model_) != n) {
    return err("model ntypes (" + std::to_string(ntypes_of(model_)) +
               ") does not match the " + std::to_string(n) +
               " element type(s) present");
  }

  dirty_ = false;
  return {};
}

// ── run / IO ─────────────────────────────────────────────────────────────────
leaf::result<force::EvalResult> PotFit::evaluate(std::size_t cfg) {
  BOOST_LEAF_CHECK(ensure_frozen());
  if (cfg >= configs_.size()) {
    return err("configuration index out of range");
  }
  return force::evaluate(model_, configs_[cfg]);
}

leaf::result<int> PotFit::optimize() {
  BOOST_LEAF_CHECK(ensure_frozen());
  if (configs_.empty()) {
    return err("no configurations to optimize against");
  }
  return run_optimizer(std::span<Configuration>(configs_), model_, opts_);
}

leaf::result<void> PotFit::write(const std::filesystem::path &path,
                                     std::string_view format) {
  BOOST_LEAF_CHECK(ensure_frozen());
  return io::write_model(model_, path, format);
}

// ── accessors ────────────────────────────────────────────────────────────────
leaf::result<const SpeciesRegistry *> PotFit::species() {
  BOOST_LEAF_CHECK(ensure_frozen());
  return &registry_;
}

leaf::result<std::span<const Configuration>> PotFit::configurations() {
  BOOST_LEAF_CHECK(ensure_frozen());
  return std::span<const Configuration>(configs_);
}

leaf::result<const config_index::ConfigIndex *> PotFit::index() {
  BOOST_LEAF_CHECK(ensure_frozen());
  return &index_.value();
}

leaf::result<Configuration *>
PotFit::config_by_name(std::string_view name) {
  BOOST_LEAF_CHECK(ensure_frozen());
  Configuration *cfg = config_index::config_by_name(index_.value(), name);
  if (cfg == nullptr) {
    return err("no configuration named '" + std::string(name) + "'");
  }
  return cfg;
}

leaf::result<const ForceCalculator *> PotFit::model() {
  BOOST_LEAF_CHECK(ensure_frozen());
  return &model_;
}

} // namespace potfit
