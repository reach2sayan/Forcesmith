#pragma once

// PotFit — the programmatic Potfit API.
//
// One owning facade for building a fit in memory: create/modify potentials,
// add/remove/modify atoms, set reference forces/stresses/energy, then evaluate
// or optimize. The CLI and the file loaders (io::load_configs / io::load_model)
// are thin clients that drive exactly this API — there is no privileged bulk
// construction path.
//
// Lifecycle is automatic. Structural mutators (add/remove atoms, configs,
// elements, potentials) mark the session dirty; the expensive,
// invariant-establishing build (freeze) runs lazily and internally the first
// time a run/IO/lookup method needs it — it is not part of the public API, so
// the user never calls it. Reference-value writes (forces/energy/stress/weight)
// and per-parameter edits do NOT dirty the session: they write straight through
// to the live data.
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
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace potfit {

class PotFit {
public:
  using PairKey = std::pair<std::string, std::string>; // sorted (min,max) symbols

  PotFit() = default;

  std::size_t
  add_configuration(BoundaryConditions bc = PeriodicBC(Mat3::Identity()));
  // Push a fully-built configuration (e.g. from Configuration::from_text). Its
  // atoms' element symbols join the model's element set at freeze.
  std::size_t add_configuration(Configuration cfg);
  // Attach a whole range of pre-built configurations at once. Accepts any
  // input_range of Configuration (vector, array, span, views...). Returns the
  // index of the FIRST appended config; the batch occupies [first, first+N).
  // Like the single-config overloads, marks the session dirty so the next
  // operation re-freezes against the enlarged config set.
  template <std::ranges::input_range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, Configuration>
  std::size_t add_configurations(R &&cfgs) {
    const std::size_t first = configs_.size();
    if constexpr (std::ranges::sized_range<R>) {
      configs_.reserve(configs_.size() + std::ranges::size(cfgs));
    }
    for (auto &&c : cfgs) {
      configs_.emplace_back(std::forward<decltype(c)>(c));
    }
    dirty_ = true;
    return first;
  }
  boost::leaf::result<void> remove_configuration(std::size_t cfg);
  boost::leaf::result<void> set_cell(std::size_t cfg, const Mat3 &box);
  boost::leaf::result<void> set_infinite(std::size_t cfg, double volume = 1.0);
  [[nodiscard]] std::size_t config_count() const { return configs_.size(); }

  [[nodiscard]] boost::leaf::result<std::size_t>
  get_configuration_index(std::string_view name);

  [[nodiscard]] boost::leaf::result<std::size_t>
  get_configuration_index(const Configuration &cfg) const;

  [[nodiscard]] boost::leaf::result<std::size_t> get_atom_index(const Atom &atom);

  boost::leaf::result<std::size_t>
  add_atom(std::size_t cfg, std::string_view element, const Vec3 &pos);
  boost::leaf::result<void> remove_atom(std::size_t cfg, std::size_t atom);

  boost::leaf::result<void> set_position(std::size_t cfg, std::size_t atom,
                                         const Vec3 &pos);
  boost::leaf::result<void> set_element(std::size_t cfg, std::size_t atom,
                                        std::string_view element);

  [[nodiscard]] boost::leaf::result<std::size_t>
  atom_count(std::size_t cfg) const;

  boost::leaf::result<void> set_ref_force(const Atom &atom, const Vec3 &f);
  boost::leaf::result<void> set_ref_force(std::string_view cfg, std::size_t atom,
                                          const Vec3 &f);

  boost::leaf::result<void> set_ref_energy(std::size_t cfg, double e);
  boost::leaf::result<void> set_ref_energy(std::string_view cfg, double e);
  boost::leaf::result<void> set_ref_energy(const Configuration &cfg, double e);

  boost::leaf::result<void> set_ref_stress(std::size_t cfg, const SymTens &s);
  boost::leaf::result<void> set_ref_stress(std::string_view cfg,
                                           const SymTens &s);
  boost::leaf::result<void> set_ref_stress(const Configuration &cfg,
                                           const SymTens &s);

  boost::leaf::result<void> set_weight(std::size_t cfg, double w);
  boost::leaf::result<void> set_weight(std::string_view cfg, double w);
  boost::leaf::result<void> set_weight(const Configuration &cfg, double w);

  // ── species (optional; lets an element with a potential but no atoms slot) ──
  boost::leaf::result<void> declare_element(std::string_view sym);

  boost::leaf::result<void> set_pair_potential(std::string_view a,
                                               std::string_view b, Potential p);
  boost::leaf::result<void> set_density(std::string_view a, Potential p);
  boost::leaf::result<void> set_embedding(std::string_view a, Potential p);
  void set_global(GlobalParam g);

  // ── extra tables for the richer model families ────────────────────────────
  // ADP: dipole u_{ij}(r) and quadrupole w_{ij}(r), per element pair.
  boost::leaf::result<void> set_dipole(std::string_view a, std::string_view b,
                                       Potential p);
  boost::leaf::result<void> set_quadrupole(std::string_view a,
                                           std::string_view b, Potential p);
  // Angular: radial modulation f_{ij}(r) per pair, angular g_i(cosθ) per type.
  boost::leaf::result<void> set_radial(std::string_view a, std::string_view b,
                                       Potential p);
  boost::leaf::result<void> set_angular(std::string_view a, Potential p);
  // Tersoff / Stiweb: analytic parameter blocks per element pair.
  boost::leaf::result<void> set_tersoff_params(std::string_view a,
                                               std::string_view b,
                                               TersoffParams params);
  boost::leaf::result<void> set_stiweb_params(std::string_view a,
                                              std::string_view b, SWParams params);
  // Stiweb 3-body strength λ for a central type and an unordered neighbour pair.
  boost::leaf::result<void> set_stiweb_lambda(std::string_view central,
                                              std::string_view a,
                                              std::string_view b, Param value);

  // Edit an already-placed potential in place (value writes; do NOT dirty). The
  // selectors mirror the setters. Errors if the slot is empty / not yet placed.
  boost::leaf::result<void> set_pair_param(std::string_view a,
                                           std::string_view b, std::size_t i,
                                           double v);

  // ── seed a fully-built force model (used by io::load_model / checkpoint) ────
  // Stores the model and captures the current element ordering so it can be
  // decomposed back to symbol-keyed potentials if a later edit re-ranks slots.
  boost::leaf::result<void> seed_force_model(ForceCalculator model);

  OptimizerOptions &options() { return opts_; }
  [[nodiscard]] const OptimizerOptions &options() const { return opts_; }

  // Inject a fully-built solver to use for optimize(). The Solver value-type
  // erases any SolverImpl together with its own tuning, so a client supplies a
  // custom algorithm simply as `set_solver(Solver{MySolver{...}})`. With none
  // set, optimize() falls back to the default Levenberg–Marquardt solver.
  void set_solver(Solver s) { solver_.emplace(std::move(s)); }

  boost::leaf::result<force::EvalResult> evaluate(std::size_t cfg);
  boost::leaf::result<int> optimize();
  boost::leaf::result<void> write(const std::filesystem::path &path,
                                  std::string_view format = "native");

  boost::leaf::result<const SpeciesRegistry *> species();
  boost::leaf::result<std::span<const Configuration>> configurations();
  boost::leaf::result<const config_index::ConfigIndex *> index();
  boost::leaf::result<Configuration *> config_by_name(std::string_view name);
  boost::leaf::result<const ForceCalculator *> model();

private:
  // The lazy build ("freeze"); runs on first run/IO/lookup that needs it.
  boost::leaf::result<void> ensure_frozen();
  // Fill empty config names with "config-<i>" and verify uniqueness. Cheap and
  // idempotent; no neighbor-list build or model materialization.
  boost::leaf::result<void> ensure_named();
  // Decompose a seeded (file-loaded) model into the editable symbol-keyed spec
  boost::leaf::result<void> detach_seeded(std::string_view action);
  boost::leaf::result<SpeciesRegistry> build_registry() const;
  boost::leaf::result<void> materialize_from_spec(); // pair / EAM
  boost::leaf::result<void>
  decompose_seeded_into_spec(const SpeciesRegistry &model_reg); // pair / EAM
  [[nodiscard]] boost::leaf::result<Configuration *> config_at(std::size_t cfg);
  // Bounds-checked per-atom force write; the public set_ref_force overloads
  // resolve their handles to (cfg, atom) indices and funnel through here.
  boost::leaf::result<void> write_ref_force(std::size_t cfg, std::size_t atom,
                                            const Vec3 &f);

  static PairKey norm_key(std::string_view a, std::string_view b);

  bool dirty_ = true;

  // The owning store, doubling as the edit buffer. Atoms carry Species by
  // identity; slots are stamped in place at freeze.
  std::vector<Configuration> configs_;

  // Symbol-keyed potentials (programmatic build path).
  std::map<PairKey, Potential> pair_;
  std::map<std::string, Potential> density_;
  std::map<std::string, Potential> embedding_;
  // Extra per-family tables (per-pair keyed by PairKey, per-type by symbol).
  std::map<PairKey, Potential> dipole_;     // ADP
  std::map<PairKey, Potential> quadrupole_; // ADP
  std::map<PairKey, Potential> radial_;     // angular
  std::map<std::string, Potential> angular_; // angular (central type)
  std::map<PairKey, TersoffParams> tersoff_;
  std::map<PairKey, SWParams> stiweb_;
  // Stiweb λ: (central symbol, sorted neighbour pair) → 3-body strength.
  std::map<std::tuple<std::string, std::string, std::string>, Param> lambda_;
  std::vector<GlobalParam> globals_;
  std::vector<std::string> declared_;
  OptimizerOptions opts_;
  // Optional client-supplied solver; empty → default LM at optimize().
  std::optional<Solver> solver_;

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
