#include "forcesmith/api/forcesmith.hpp"

#include "forcesmith/api/force_model_strategy.hpp"
#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/force/potential_table.hpp"
#include "forcesmith/io/write_model.hpp"

#include <algorithm>
#include <execution>
#include <iterator>
#include <map>
#include <ranges>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace forcesmith {

namespace leaf = boost::leaf;

// BOOST_LEAF_CHECK expands to a GNU statement-expression ({ ... }); silence the
// pedantic complaint about that Boost idiom for this translation unit.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored                                               \
    "-Wgnu-statement-expression-from-macro-expansion"
#endif

// The model-family strategy registry (SpecRef, build_*/dump_* helpers, and the
// per-family PotentialType specializations) lives in this detail header.
using detail::err;
using detail::PotentialType;
using detail::Probe;
using detail::probe_for;
using detail::SpecRef;

namespace {

[[nodiscard]] std::size_t ntypes_of(const ForceCalculator &m) {
  return m.ntypes();
}
// ML models (ACSF/SOAP/LMBTR) re-rank by direct head remap rather than the
// analytic decompose/materialize-from-spec round-trip. The closed ML family is
// recovered through the typed escape hatch — adding an ANALYTIC calculator
// never touches this list.
[[nodiscard]] bool is_ml(const ForceCalculator &m) {
  return m.target<ACSF>() != nullptr || m.target<SoapModel>() != nullptr ||
         m.target<LMBTR>() != nullptr;
}
[[nodiscard]] double max_cutoff_of(const ForceCalculator &m) {
  return m.max_cutoff();
}
// Radial pair table (φ_ij) when the model has one; nullptr for tersoff/stiweb
// and the ML models. Pair-table access is an API/IO concern (live-edit path,
// set_pair_param), kept out of the core concept — so the four table-bearing
// families are enumerated here via the typed escape hatch.
[[nodiscard]] RadialPotentialPair *pair_table_of(ForceCalculator &m) {
  if (auto *c = m.target<PairForceCalculator>()) {
    return &c->pair;
  }
  if (auto *c = m.target<EAMForceCalculator>()) {
    return &c->pair;
  }
  if (auto *c = m.target<ADPForceCalculator>()) {
    return &c->pair;
  }
  if (auto *c = m.target<AngularForceCalculator>()) {
    return &c->pair;
  }
  return nullptr;
}

} // namespace

Forcesmith::PairKey Forcesmith::norm_key(std::string_view a,
                                         std::string_view b) {
  std::string sa(a);
  std::string sb(b);
  if (sb < sa) {
    std::swap(sa, sb);
  }
  return {std::move(sa), std::move(sb)};
}

leaf::result<Configuration *> Forcesmith::config_at(std::size_t cfg) {
  if (cfg >= configs_.size()) {
    return err("configuration index " + std::to_string(cfg) +
               " out of range (" + std::to_string(configs_.size()) + ")");
  }
  return &configs_[cfg];
}

leaf::result<void> Forcesmith::ensure_named() {
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
Forcesmith::get_configuration_index(std::string_view name) {
  BOOST_LEAF_CHECK(ensure_named());
  for (auto &&[i, cfg] : configs_ | std::views::enumerate) {
    if (cfg.name == name) {
      return static_cast<std::size_t>(i);
    }
  }
  return err("no configuration named '" + std::string(name) + "'");
}

leaf::result<std::size_t>
Forcesmith::get_configuration_index(const Configuration &cfg) const {
  const auto it = std::ranges::find_if(
      configs_, [&](const Configuration &c) { return &c == &cfg; });
  if (it == configs_.end()) {
    return err("configuration is not owned by this session");
  }
  return static_cast<std::size_t>(std::distance(configs_.begin(), it));
}

leaf::result<std::size_t> Forcesmith::get_atom_index(const Atom &atom) {
  BOOST_LEAF_CHECK(ensure_frozen()); // stamps Atom::parent for owned atoms
  if (atom.parent == nullptr) {
    return err("atom is not owned by this session");
  }
  return static_cast<std::size_t>(&atom - atom.parent->atoms.data());
}

std::size_t Forcesmith::add_configuration(BoundaryConditions bc) {
  Configuration cfg;
  cfg.bc = std::move(bc);
  return add_configuration(std::move(cfg));
  return configs_.size() - 1;
}

std::size_t Forcesmith::add_configuration(Configuration cfg) {
  configs_.push_back(std::move(cfg));
  dirty_ = true;
  return configs_.size() - 1;
}

leaf::result<void> Forcesmith::remove_configuration(std::size_t cfg) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  configs_.erase(configs_.begin() +
                 static_cast<std::ptrdiff_t>(c - configs_.data()));
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_cell(std::size_t cfg, const Mat3 &box) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->bc = PeriodicBC(box);
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_infinite(std::size_t cfg, double volume) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->bc = InfiniteBC(volume);
  dirty_ = true;
  return {};
}

leaf::result<std::size_t> Forcesmith::add_atom(std::size_t cfg,
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

leaf::result<void> Forcesmith::remove_atom(std::size_t cfg, std::size_t atom) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  if (atom >= c->atoms.size()) {
    return err("atom index out of range");
  }
  c->atoms.erase(c->atoms.begin() + static_cast<std::ptrdiff_t>(atom));
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_position(std::size_t cfg, std::size_t atom,
                                            const Vec3 &pos) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  if (atom >= c->atoms.size()) {
    return err("atom index out of range");
  }
  c->atoms[atom].pos = pos;
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_element(std::size_t cfg, std::size_t atom,
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

leaf::result<std::size_t> Forcesmith::atom_count(std::size_t cfg) const {
  if (cfg >= configs_.size()) {
    return err("configuration index out of range");
  }
  return configs_[cfg].atoms.size();
}

leaf::result<void>
Forcesmith::write_ref_force(std::size_t cfg, std::size_t atom, const Vec3 &f) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  if (atom >= c->atoms.size()) {
    return err("atom index out of range");
  }
  c->atoms[atom].ref.force = f;
  return {};
}

leaf::result<void> Forcesmith::set_ref_energy(std::size_t cfg, double e) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->ref.energy = e;
  return {};
}

leaf::result<void> Forcesmith::set_ref_stress(std::size_t cfg,
                                              const SymTens &s) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->ref.stress = s;
  return {};
}

leaf::result<void> Forcesmith::set_weight(std::size_t cfg, double w) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->weight = w;
  return {};
}

leaf::result<void> Forcesmith::set_ref_force(const Atom &atom, const Vec3 &f) {
  BOOST_LEAF_AUTO(ai, get_atom_index(atom)); // freezes; stamps atom.parent
  BOOST_LEAF_AUTO(ci, get_configuration_index(*atom.parent));
  return write_ref_force(ci, ai, f);
}

leaf::result<void> Forcesmith::set_ref_force(std::string_view cfg,
                                             std::size_t atom, const Vec3 &f) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return write_ref_force(ci, atom, f);
}

leaf::result<void> Forcesmith::set_ref_energy(std::string_view cfg, double e) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_ref_energy(ci, e);
}

leaf::result<void> Forcesmith::set_ref_energy(const Configuration &cfg,
                                              double e) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_ref_energy(ci, e);
}

leaf::result<void> Forcesmith::set_ref_stress(std::string_view cfg,
                                              const SymTens &s) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_ref_stress(ci, s);
}

leaf::result<void> Forcesmith::set_ref_stress(const Configuration &cfg,
                                              const SymTens &s) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_ref_stress(ci, s);
}

leaf::result<void> Forcesmith::set_weight(std::string_view cfg, double w) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_weight(ci, w);
}

leaf::result<void> Forcesmith::set_weight(const Configuration &cfg, double w) {
  BOOST_LEAF_AUTO(ci, get_configuration_index(cfg));
  return set_weight(ci, w);
}

leaf::result<void> Forcesmith::declare_element(std::string_view sym) {
  BOOST_LEAF_CHECK(Species::lookup(sym)); // validate against the catalog
  if (std::ranges::find(declared_, std::string(sym)) == declared_.end()) {
    declared_.emplace_back(sym);
    dirty_ = true;
  }
  return {};
}

leaf::result<void> Forcesmith::set_pair_potential(std::string_view a,
                                                  std::string_view b,
                                                  RadialPotential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  pair_.insert_or_assign(norm_key(a, b), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_density(std::string_view a,
                                           RadialPotential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  density_.insert_or_assign(std::string(a), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_embedding(std::string_view a,
                                             RadialPotential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  embedding_.insert_or_assign(std::string(a), std::move(p));
  dirty_ = true;
  return {};
}

void Forcesmith::set_global(GlobalParam g) {
  globals_.push_back(std::move(g));
  dirty_ = true;
}

leaf::result<void> Forcesmith::set_dipole(std::string_view a,
                                          std::string_view b,
                                          RadialPotential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  dipole_.insert_or_assign(norm_key(a, b), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_quadrupole(std::string_view a,
                                              std::string_view b,
                                              RadialPotential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  quadrupole_.insert_or_assign(norm_key(a, b), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_radial(std::string_view a,
                                          std::string_view b,
                                          RadialPotential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  radial_.insert_or_assign(norm_key(a, b), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_angular(std::string_view a,
                                           RadialPotential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  angular_.insert_or_assign(std::string(a), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_tersoff_params(std::string_view a,
                                                  std::string_view b,
                                                  TersoffParams params) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  tersoff_.insert_or_assign(norm_key(a, b), std::move(params));
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_stiweb_params(std::string_view a,
                                                 std::string_view b,
                                                 SWParams params) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  BOOST_LEAF_CHECK(detach_seeded("edit"));
  stiweb_.insert_or_assign(norm_key(a, b), std::move(params));
  dirty_ = true;
  return {};
}

leaf::result<void> Forcesmith::set_stiweb_lambda(std::string_view central,
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

leaf::result<void> Forcesmith::set_pair_param(std::string_view a,
                                              std::string_view b, std::size_t i,
                                              double v) {
  BOOST_LEAF_CHECK(ensure_frozen());
  // Edit the live materialized potential in place (value-only; does not dirty).
  RadialPotentialPair *pt = pair_table_of(model_);
  if (pt == nullptr) {
    return err("this model has no pair table to edit");
  }
  BOOST_LEAF_AUTO(sa, species_of(registry_, a));
  BOOST_LEAF_AUTO(sb, species_of(registry_, b));
  (*pt)[sa.index, sb.index].set_param(i, v);
  return {};
}

leaf::result<void> Forcesmith::seed_force_model(ForceCalculator model) {
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

leaf::result<SpeciesRegistry> Forcesmith::build_registry() const {
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

  for (const auto &cfg : configs_) {
    for (const auto &a : cfg.atoms) {
      add(a.type.symbol);
    }
  }

  return build_species_registry(syms);
}

leaf::result<void> Forcesmith::materialize_from_spec() {
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

leaf::result<void> Forcesmith::detach_seeded(std::string_view action) {
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
Forcesmith::decompose_seeded_into_spec(const SpeciesRegistry &reg) {
  if (!seeded_) {
    return {};
  }
  SpecRef spec{registry_, pair_,    density_, embedding_, dipole_, quadrupole_,
               radial_,   angular_, tersoff_, stiweb_,    lambda_, globals_};
  // decompose() is family-specific (the PotentialType trait); recover the
  // concrete model through the typed escape hatch and dispatch. ML families'
  // decompose returns a leaf error (they re-rank via remap, not the spec).
  ForceCalculator &m = *seeded_;
  if (auto *c = m.target<PairForceCalculator>()) {
    return PotentialType<PairForceCalculator>::decompose(*c, reg, spec);
  }
  if (auto *c = m.target<EAMForceCalculator>()) {
    return PotentialType<EAMForceCalculator>::decompose(*c, reg, spec);
  }
  if (auto *c = m.target<ADPForceCalculator>()) {
    return PotentialType<ADPForceCalculator>::decompose(*c, reg, spec);
  }
  if (auto *c = m.target<AngularForceCalculator>()) {
    return PotentialType<AngularForceCalculator>::decompose(*c, reg, spec);
  }
  if (auto *c = m.target<TersoffForceCalculator>()) {
    return PotentialType<TersoffForceCalculator>::decompose(*c, reg, spec);
  }
  if (auto *c = m.target<StiwebForceCalculator>()) {
    return PotentialType<StiwebForceCalculator>::decompose(*c, reg, spec);
  }
  if (auto *c = m.target<ACSF>()) {
    return PotentialType<ACSF>::decompose(*c, reg, spec);
  }
  if (auto *c = m.target<SoapModel>()) {
    return PotentialType<SoapModel>::decompose(*c, reg, spec);
  }
  if (auto *c = m.target<LMBTR>()) {
    return PotentialType<LMBTR>::decompose(*c, reg, spec);
  }
  return err("decompose: unknown force model");
}

leaf::result<ForceCalculator>
Forcesmith::remap_seeded_ml(const SpeciesRegistry &old_reg,
                            const SpeciesRegistry &new_reg) const {
  // ForceCalculator::remap dispatches to the held model's virtual (ML models
  // re-rank their heads; analytic models return a leaf error).
  return seeded_->remap(old_reg, new_reg);
}

leaf::result<void> Forcesmith::ensure_frozen() {
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
    } else if (is_ml(*seeded_)) {
      // ML re-rank: no symbol-keyed spec exists, so remap the model directly.
      if (!seeded_registry_) {
        return err(
            "cannot re-rank a seeded ML model: original element ordering "
            "unknown");
      }
      BOOST_LEAF_AUTO(remapped, remap_seeded_ml(*seeded_registry_, registry_));
      model_ = std::move(remapped);
      seeded_.reset();
      seeded_registry_.reset();
    } else {
      BOOST_LEAF_CHECK(detach_seeded("re-rank"));
      BOOST_LEAF_CHECK(materialize_from_spec());
    }
  } else {
    BOOST_LEAF_CHECK(materialize_from_spec());
  }

  // Build neighbor lists against the materialized model (must outlive them).
  const double rcut = max_cutoff_of(model_);
  const RadialPotentialPair *pt = pair_table_of(model_);
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

leaf::result<force::EvalResult> Forcesmith::evaluate(std::size_t cfg) {
  BOOST_LEAF_CHECK(ensure_frozen());
  if (cfg >= configs_.size()) {
    return err("configuration index out of range");
  }
  return force::evaluate(model_, configs_[cfg]);
}

leaf::result<std::vector<force::EvalResult>> Forcesmith::evaluate_all() {
  BOOST_LEAF_CHECK(ensure_frozen());
  std::vector<force::EvalResult> out(configs_.size());
  if (configs_.empty()) {
    return out;
  }

  // Warm up any lazy, model-internal descriptor state (e.g. SOAP's radial
  // basis) exactly once, serially, before the parallel fill races on it — same
  // reason MLBase::prepare() runs step_warm_up_descriptors before its parallel
  // fill.
  std::size_t warm = 0;
  while (warm < configs_.size() && configs_[warm].atoms.empty()) {
    ++warm;
  }
  if (warm == configs_.size()) {
    return out; // nothing with atoms to evaluate
  }
  out[warm] = force::evaluate(model_, configs_[warm]);

  // Every other config is independent: force::evaluate deep-copies the config
  // to a local scratch, so each task mutates only its own
  // forces/stress/neighbour list and model_ is read const-only. Same
  // std::execution::par construct the fit's step_fill_cache uses. Call
  // force::evaluate directly (plain EvalResult) so no Boost.LEAF error objects
  // are created inside the parallel region.
  std::vector<std::size_t> idx;
  idx.reserve(configs_.size());
  for (std::size_t i = 0; i < configs_.size(); ++i) {
    if (i != warm) {
      idx.push_back(i);
    }
  }
  std::for_each(
      std::execution::par, idx.begin(), idx.end(),
      [&](std::size_t i) { out[i] = force::evaluate(model_, configs_[i]); });
  return out;
}

leaf::result<int> Forcesmith::optimize() {
  BOOST_LEAF_CHECK(ensure_frozen());
  if (configs_.empty()) {
    return err("no configurations to optimize against");
  }
  if (solver_) {
    return run_optimizer(std::span<Configuration>(configs_), model_, opts_,
                         *solver_);
  }
  return run_optimizer(std::span<Configuration>(configs_), model_, opts_);
}

leaf::result<void> Forcesmith::write(const std::filesystem::path &path,
                                     std::string_view format) {
  BOOST_LEAF_CHECK(ensure_frozen());
  return io::write_model(model_, path, format);
}

leaf::result<const SpeciesRegistry *> Forcesmith::species() {
  BOOST_LEAF_CHECK(ensure_frozen());
  return &registry_;
}

leaf::result<std::span<const Configuration>> Forcesmith::configurations() {
  BOOST_LEAF_CHECK(ensure_frozen());
  return std::span<const Configuration>(configs_);
}

leaf::result<const config_index::ConfigIndex *> Forcesmith::index() {
  BOOST_LEAF_CHECK(ensure_frozen());
  if (!index_) {
    return err("internal: config index not built after freeze");
  }
  return &*index_;
}

leaf::result<Configuration *>
Forcesmith::config_by_name(std::string_view name) {
  BOOST_LEAF_CHECK(ensure_frozen());
  if (!index_) {
    return err("internal: config index not built after freeze");
  }
  Configuration *cfg = config_index::config_by_name(*index_, name);
  if (cfg == nullptr) {
    return err("no configuration named '" + std::string(name) + "'");
  }
  return cfg;
}

leaf::result<const ForceCalculator *> Forcesmith::model() {
  BOOST_LEAF_CHECK(ensure_frozen());
  return &model_;
}

} // namespace forcesmith

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
