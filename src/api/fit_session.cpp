#include "potfit/api/fit_session.hpp"

#include "potfit/core/neighbor_list.hpp"
#include "potfit/force/eam_force.hpp"
#include "potfit/force/pair_force.hpp"
#include "potfit/force/potential_table.hpp"
#include "potfit/io/config_reader.hpp" // ParseError
#include "potfit/io/write_model.hpp"

#include <algorithm>
#include <string>
#include <type_traits>
#include <variant>

namespace potfit {

namespace leaf = boost::leaf;

namespace {

[[nodiscard]] leaf::error_id err(std::string msg) {
  return leaf::new_error(io::ParseError{std::move(msg), 0});
}

// Visitor helpers over the ForceCalculator variant.
[[nodiscard]] std::size_t ntypes_of(const ForceCalculator &m) {
  return std::visit([](const auto &c) { return c.ntypes; }, m);
}
[[nodiscard]] double max_cutoff_of(const ForceCalculator &m) {
  return std::visit([](const auto &c) { return c.max_cutoff(); }, m);
}
// Radial pair table (φ_ij) when the model has one; nullptr for tersoff/stiweb.
[[nodiscard]] const PotentialPair *pair_table_of(const ForceCalculator &m) {
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

FitSession::PairKey FitSession::norm_key(std::string_view a,
                                         std::string_view b) {
  std::string sa(a);
  std::string sb(b);
  if (sb < sa) {
    std::swap(sa, sb);
  }
  return {std::move(sa), std::move(sb)};
}

leaf::result<Configuration *> FitSession::config_at(std::size_t cfg) {
  if (cfg >= configs_.size()) {
    return err("configuration index " + std::to_string(cfg) +
               " out of range (" + std::to_string(configs_.size()) + ")");
  }
  return &configs_[cfg];
}

// ── configurations ──────────────────────────────────────────────────────────
std::size_t FitSession::add_configuration(BoundaryConditions bc) {
  Configuration cfg;
  cfg.bc = std::move(bc);
  configs_.push_back(std::move(cfg));
  dirty_ = true;
  return configs_.size() - 1;
}

std::size_t FitSession::add_configuration(Configuration cfg) {
  configs_.push_back(std::move(cfg));
  dirty_ = true;
  return configs_.size() - 1;
}

leaf::result<void> FitSession::remove_configuration(std::size_t cfg) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  configs_.erase(configs_.begin() +
                 static_cast<std::ptrdiff_t>(c - configs_.data()));
  dirty_ = true;
  return {};
}

leaf::result<void> FitSession::set_cell(std::size_t cfg, const Mat3 &box) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->bc = PeriodicBC(box);
  dirty_ = true;
  return {};
}

leaf::result<void> FitSession::set_infinite(std::size_t cfg, double volume) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->bc = InfiniteBC(volume);
  dirty_ = true;
  return {};
}

// ── atoms ────────────────────────────────────────────────────────────────────
leaf::result<std::size_t> FitSession::add_atom(std::size_t cfg,
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

leaf::result<void> FitSession::remove_atom(std::size_t cfg, std::size_t atom) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  if (atom >= c->atoms.size()) {
    return err("atom index out of range");
  }
  c->atoms.erase(c->atoms.begin() + static_cast<std::ptrdiff_t>(atom));
  dirty_ = true;
  return {};
}

leaf::result<void> FitSession::set_position(std::size_t cfg, std::size_t atom,
                                            const Vec3 &pos) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  if (atom >= c->atoms.size()) {
    return err("atom index out of range");
  }
  c->atoms[atom].pos = pos;
  dirty_ = true;
  return {};
}

leaf::result<void> FitSession::set_element(std::size_t cfg, std::size_t atom,
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

leaf::result<std::size_t> FitSession::atom_count(std::size_t cfg) const {
  if (cfg >= configs_.size()) {
    return err("configuration index out of range");
  }
  return configs_[cfg].atoms.size();
}

// ── reference data (no dirty) ────────────────────────────────────────────────
leaf::result<void> FitSession::set_ref_force(std::size_t cfg, std::size_t atom,
                                             const Vec3 &f) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  if (atom >= c->atoms.size()) {
    return err("atom index out of range");
  }
  c->atoms[atom].ref.force = f;
  return {};
}

leaf::result<void> FitSession::set_ref_energy(std::size_t cfg, double e) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->ref.energy = e;
  return {};
}

leaf::result<void> FitSession::set_ref_stress(std::size_t cfg,
                                              const SymTens &s) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->ref.stress = s;
  return {};
}

leaf::result<void> FitSession::set_weight(std::size_t cfg, double w) {
  BOOST_LEAF_AUTO(c, config_at(cfg));
  c->weight = w;
  return {};
}

// ── species ──────────────────────────────────────────────────────────────────
leaf::result<void> FitSession::declare_element(std::string_view sym) {
  BOOST_LEAF_CHECK(Species::lookup(sym)); // validate against the catalog
  if (std::ranges::find(declared_, std::string(sym)) == declared_.end()) {
    declared_.emplace_back(sym);
    dirty_ = true;
  }
  return {};
}

// ── potentials ───────────────────────────────────────────────────────────────
// If a model was seeded from a file, decompose it into the symbol-keyed spec so
// programmatic edits compose with the loaded potentials instead of being lost.
leaf::result<void> FitSession::set_pair_potential(std::string_view a,
                                                  std::string_view b,
                                                  Potential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  BOOST_LEAF_CHECK(Species::lookup(b));
  if (seeded_) {
    if (!seeded_registry_) {
      return err("cannot edit a seeded model: original element ordering "
                 "unknown");
    }
    BOOST_LEAF_CHECK(decompose_seeded_into_spec(*seeded_registry_));
    seeded_.reset();
    seeded_registry_.reset();
  }
  pair_.insert_or_assign(norm_key(a, b), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> FitSession::set_density(std::string_view a, Potential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  if (seeded_) {
    if (!seeded_registry_) {
      return err("cannot edit a seeded model: original element ordering "
                 "unknown");
    }
    BOOST_LEAF_CHECK(decompose_seeded_into_spec(*seeded_registry_));
    seeded_.reset();
    seeded_registry_.reset();
  }
  density_.insert_or_assign(std::string(a), std::move(p));
  dirty_ = true;
  return {};
}

leaf::result<void> FitSession::set_embedding(std::string_view a, Potential p) {
  BOOST_LEAF_CHECK(Species::lookup(a));
  if (seeded_) {
    if (!seeded_registry_) {
      return err("cannot edit a seeded model: original element ordering "
                 "unknown");
    }
    BOOST_LEAF_CHECK(decompose_seeded_into_spec(*seeded_registry_));
    seeded_.reset();
    seeded_registry_.reset();
  }
  embedding_.insert_or_assign(std::string(a), std::move(p));
  dirty_ = true;
  return {};
}

void FitSession::set_global(GlobalParam g) {
  globals_.push_back(std::move(g));
  dirty_ = true;
}

leaf::result<void> FitSession::set_pair_param(std::string_view a,
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

// ── seeding ──────────────────────────────────────────────────────────────────
leaf::result<void> FitSession::seed_force_model(ForceCalculator model) {
  // Capture the element ordering this model was built against, so a later edit
  // that re-ranks slots can still recover the per-symbol potentials.
  BOOST_LEAF_AUTO(reg, build_registry());
  seeded_registry_ = std::move(reg);
  seeded_ = std::move(model);
  // Programmatic spec (if any) is superseded by the seeded model.
  pair_.clear();
  density_.clear();
  embedding_.clear();
  globals_.clear();
  dirty_ = true;
  return {};
}

// ── build helpers ────────────────────────────────────────────────────────────
leaf::result<SpeciesRegistry> FitSession::build_registry() const {
  std::vector<std::string_view> syms;
  auto add = [&](std::string_view s) {
    if (!s.empty() && std::ranges::find(syms, s) == syms.end()) {
      syms.push_back(s);
    }
  };
  for (const auto &cfg : configs_) {
    for (const auto &a : cfg.atoms) {
      add(a.type.symbol);
    }
  }
  for (const auto &[k, p] : pair_) {
    add(k.first);
    add(k.second);
  }
  for (const auto &[k, p] : density_) {
    add(k);
  }
  for (const auto &[k, p] : embedding_) {
    add(k);
  }
  for (const auto &s : declared_) {
    add(s);
  }
  return build_species_registry(syms);
}

leaf::result<void> FitSession::materialize_from_spec() {
  const std::size_t n = ntypes(registry_);
  if (pair_.empty()) {
    return err("no pair potentials set");
  }

  auto build_pair = [&](PotentialPair &mat) -> leaf::result<void> {
    mat.reserve(n);
    for (std::size_t ti = 0; ti < n; ++ti) {
      for (std::size_t tj = ti; tj < n; ++tj) {
        const PairKey key = norm_key(species_at(registry_, ti).symbol,
                                     species_at(registry_, tj).symbol);
        auto it = pair_.find(key);
        if (it == pair_.end()) {
          return err("missing pair potential for " + key.first + "-" +
                     key.second);
        }
        mat.emplace_back(it->second);
      }
    }
    return {};
  };

  const bool eam = !density_.empty() || !embedding_.empty();
  if (!eam) {
    PairForceCalculator calc;
    calc.ntypes = n;
    BOOST_LEAF_CHECK(build_pair(calc.pair));
    calc.globals = globals_;
    calc.finalize_globals();
    model_ = std::move(calc);
    return {};
  }

  EAMForceCalculator calc;
  calc.ntypes = n;
  BOOST_LEAF_CHECK(build_pair(calc.pair));
  calc.density.reserve(n);
  calc.embedding.reserve(n);
  for (std::size_t t = 0; t < n; ++t) {
    const std::string sym(species_at(registry_, t).symbol);
    auto di = density_.find(sym);
    if (di == density_.end()) {
      return err("missing density (transfer) function for " + sym);
    }
    calc.density.emplace_back(di->second);
    auto ei = embedding_.find(sym);
    if (ei == embedding_.end()) {
      return err("missing embedding function for " + sym);
    }
    calc.embedding.emplace_back(ei->second);
  }
  calc.globals = globals_;
  calc.finalize_globals();
  model_ = std::move(calc);
  return {};
}

leaf::result<void>
FitSession::decompose_seeded_into_spec(const SpeciesRegistry &reg) {
  if (!seeded_) {
    return {};
  }
  return std::visit(
      [&](auto &c) -> leaf::result<void> {
        using T = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<T, PairForceCalculator> ||
                      std::is_same_v<T, EAMForceCalculator>) {
          if (!c.globals.empty()) {
            return err("decomposing a seeded model with global parameters "
                       "after a re-rank is not supported; set potentials "
                       "programmatically");
          }
          const std::size_t n = c.ntypes;
          for (std::size_t ti = 0; ti < n; ++ti) {
            for (std::size_t tj = ti; tj < n; ++tj) {
              const PairKey key = norm_key(species_at(reg, ti).symbol,
                                           species_at(reg, tj).symbol);
              pair_.insert_or_assign(key, c.pair[ti, tj]);
            }
          }
          if constexpr (std::is_same_v<T, EAMForceCalculator>) {
            for (std::size_t t = 0; t < n; ++t) {
              const std::string sym(species_at(reg, t).symbol);
              density_.insert_or_assign(sym, c.density[t]);
              embedding_.insert_or_assign(sym, c.embedding[t]);
            }
          }
          return {};
        } else {
          return err("editing elements of this seeded model type is not "
                     "supported; build potentials programmatically instead");
        }
      },
      *seeded_);
}

leaf::result<void> FitSession::ensure_frozen() {
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
      model_ = *seeded_; // slot order == Z-sorted registry (potfit convention)
    } else {
      if (!seeded_registry_) {
        return err("seeded model cannot be re-ranked: original element "
                   "ordering unknown");
      }
      BOOST_LEAF_CHECK(decompose_seeded_into_spec(*seeded_registry_));
      seeded_.reset();
      seeded_registry_.reset();
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

  // Auxiliary grouping index (configs_ must not be resized/reordered hereafter).
  index_ = config_index::build_config_index(configs_);

  if (ntypes_of(model_) != n) {
    return err("model ntypes (" + std::to_string(ntypes_of(model_)) +
               ") does not match the " + std::to_string(n) +
               " element type(s) present");
  }

  dirty_ = false;
  return {};
}

leaf::result<void> FitSession::freeze() { return ensure_frozen(); }

// ── run / IO ─────────────────────────────────────────────────────────────────
leaf::result<force::EvalResult> FitSession::evaluate(std::size_t cfg) {
  BOOST_LEAF_CHECK(ensure_frozen());
  if (cfg >= configs_.size()) {
    return err("configuration index out of range");
  }
  return force::evaluate(model_, configs_[cfg]);
}

leaf::result<int> FitSession::optimize() {
  BOOST_LEAF_CHECK(ensure_frozen());
  if (configs_.empty()) {
    return err("no configurations to optimize against");
  }
  return run_optimizer(std::span<Configuration>(configs_), model_, opts_);
}

leaf::result<void> FitSession::write(const std::filesystem::path &path,
                                     std::string_view format) {
  BOOST_LEAF_CHECK(ensure_frozen());
  return io::write_model(model_, path, format);
}

// ── accessors ────────────────────────────────────────────────────────────────
leaf::result<const SpeciesRegistry *> FitSession::species() {
  BOOST_LEAF_CHECK(ensure_frozen());
  return &registry_;
}

leaf::result<std::span<const Configuration>> FitSession::configurations() {
  BOOST_LEAF_CHECK(ensure_frozen());
  return std::span<const Configuration>(configs_);
}

leaf::result<const config_index::ConfigIndex *> FitSession::index() {
  BOOST_LEAF_CHECK(ensure_frozen());
  return &index_.value();
}

leaf::result<const ForceCalculator *> FitSession::model() {
  BOOST_LEAF_CHECK(ensure_frozen());
  return &model_;
}

} // namespace potfit
