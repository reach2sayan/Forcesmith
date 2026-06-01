#pragma once

// FitSession — the programmatic Potfit API.
//
// One owning facade for building a fit in memory: create/modify potentials,
// add/remove/modify atoms, set reference forces/stresses/energy, then evaluate
// or optimize. The CLI and the file loaders (io::load_configs / io::load_model)
// are thin clients that drive exactly this API — there is no privileged bulk
// construction path.
//
// Lifecycle is automatic. Structural mutators (add/remove atoms, configs,
// elements, potentials) mark the session dirty; the expensive,
// invariant-establishing build (freeze) runs lazily the first time a run/IO
// method needs it. The user never has to call freeze() — it is exposed only as
// an optional "build now / surface errors early" hook. Reference-value writes
// (forces/energy/stress/weight) and per-parameter edits do NOT dirty the
// session: they write straight through to the live data.
//
// Auto-grow species: atoms and potentials are held by element *identity*
// (symbol); the compact dense-table slot (Species::index, Z-sorted rank) is
// assigned at freeze. Adding an atom of a new element simply re-ranks at the
// next freeze.

#include "potfit/core/atom.hpp"
#include "potfit/core/config_index.hpp"
#include "potfit/core/potential_base.hpp"
#include "potfit/core/species.hpp"
#include "potfit/core/types.hpp"
#include "potfit/force/evaluate.hpp"
#include "potfit/force/force_calculator.hpp"
#include "potfit/force/force_calculator_concept.hpp" // GlobalParam
#include "potfit/optimization/optimizer.hpp"

#include <boost/leaf/result.hpp>
#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace potfit {

class FitSession {
public:
  using PairKey = std::pair<std::string, std::string>; // sorted (min,max) symbols

  FitSession() = default;

  // ── configurations (structural; mark dirty) ───────────────────────────────
  std::size_t
  add_configuration(BoundaryConditions bc = PeriodicBC(Mat3::Identity()));
  // Push a fully-built configuration (e.g. from Configuration::from_text). Its
  // atoms' element symbols join the model's element set at freeze.
  std::size_t add_configuration(Configuration cfg);
  boost::leaf::result<void> remove_configuration(std::size_t cfg);
  boost::leaf::result<void> set_cell(std::size_t cfg, const Mat3 &box);
  boost::leaf::result<void> set_infinite(std::size_t cfg, double volume = 1.0);
  [[nodiscard]] std::size_t config_count() const { return configs_.size(); }

  // ── atoms (structural; mark dirty) ─────────────────────────────────────────
  boost::leaf::result<std::size_t>
  add_atom(std::size_t cfg, std::string_view element, const Vec3 &pos);
  boost::leaf::result<void> remove_atom(std::size_t cfg, std::size_t atom);
  boost::leaf::result<void> set_position(std::size_t cfg, std::size_t atom,
                                         const Vec3 &pos);
  boost::leaf::result<void> set_element(std::size_t cfg, std::size_t atom,
                                        std::string_view element);
  [[nodiscard]] boost::leaf::result<std::size_t>
  atom_count(std::size_t cfg) const;

  // ── reference data (value writes; do NOT dirty) ────────────────────────────
  boost::leaf::result<void> set_ref_force(std::size_t cfg, std::size_t atom,
                                          const Vec3 &f);
  boost::leaf::result<void> set_ref_energy(std::size_t cfg, double e);
  boost::leaf::result<void> set_ref_stress(std::size_t cfg, const SymTens &s);
  boost::leaf::result<void> set_weight(std::size_t cfg, double w);

  // ── species (optional; lets an element with a potential but no atoms slot) ──
  boost::leaf::result<void> declare_element(std::string_view sym);

  // ── potentials (symbol-keyed; mark dirty) ──────────────────────────────────
  // Programmatic build is currently supported for the pair and EAM models
  // (set_pair_potential alone ⇒ pair; adding density+embedding ⇒ EAM). Other
  // models load via seed_force_model and are evaluated/optimized/written as-is.
  boost::leaf::result<void> set_pair_potential(std::string_view a,
                                               std::string_view b, Potential p);
  boost::leaf::result<void> set_density(std::string_view a, Potential p);
  boost::leaf::result<void> set_embedding(std::string_view a, Potential p);
  void set_global(GlobalParam g);

  // Edit an already-placed potential in place (value writes; do NOT dirty). The
  // selectors mirror the setters. Errors if the slot is empty / not yet placed.
  boost::leaf::result<void> set_pair_param(std::string_view a,
                                           std::string_view b, std::size_t i,
                                           double v);

  // ── seed a fully-built force model (used by io::load_model / checkpoint) ────
  // Stores the model and captures the current element ordering so it can be
  // decomposed back to symbol-keyed potentials if a later edit re-ranks slots.
  boost::leaf::result<void> seed_force_model(ForceCalculator model);

  // ── optimizer options ──────────────────────────────────────────────────────
  OptimizerOptions &options() { return opts_; }
  [[nodiscard]] const OptimizerOptions &options() const { return opts_; }

  // ── lifecycle ──────────────────────────────────────────────────────────────
  // Optional: force the build now (surfaces missing-potential errors early).
  boost::leaf::result<void> freeze();

  // ── run / IO (auto-freeze on entry) ─────────────────────────────────────────
  boost::leaf::result<force::EvalResult> evaluate(std::size_t cfg);
  boost::leaf::result<int> optimize();
  boost::leaf::result<void> write(const std::filesystem::path &path,
                                  std::string_view format = "native");

  // ── accessors (auto-freeze; pointers into owned state) ──────────────────────
  boost::leaf::result<const SpeciesRegistry *> species();
  boost::leaf::result<std::span<const Configuration>> configurations();
  boost::leaf::result<const config_index::ConfigIndex *> index();
  boost::leaf::result<const ForceCalculator *> model();

private:
  boost::leaf::result<void> ensure_frozen();
  boost::leaf::result<SpeciesRegistry> build_registry() const;
  boost::leaf::result<void> materialize_from_spec(); // pair / EAM
  boost::leaf::result<void>
  decompose_seeded_into_spec(const SpeciesRegistry &model_reg); // pair / EAM
  [[nodiscard]] boost::leaf::result<Configuration *> config_at(std::size_t cfg);

  static PairKey norm_key(std::string_view a, std::string_view b);

  bool dirty_ = true;

  // The owning store, doubling as the edit buffer. Atoms carry Species by
  // identity; slots are stamped in place at freeze.
  std::vector<Configuration> configs_;

  // Symbol-keyed potentials (programmatic build path).
  std::map<PairKey, Potential> pair_;
  std::map<std::string, Potential> density_;
  std::map<std::string, Potential> embedding_;
  std::vector<GlobalParam> globals_;
  std::vector<std::string> declared_;
  OptimizerOptions opts_;

  // Seeded-from-file path: a fully-built model plus the element ordering it was
  // built against (for decomposition on re-rank).
  std::optional<ForceCalculator> seeded_;
  std::optional<SpeciesRegistry> seeded_registry_;

  // Materialized at freeze.
  SpeciesRegistry registry_;
  ForceCalculator model_ = PairForceCalculator{};
  std::optional<config_index::ConfigIndex> index_;
};

} // namespace potfit
