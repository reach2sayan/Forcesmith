#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/config_index.hpp"
#include "forcesmith/core/radial_potential.hpp"
#include "forcesmith/core/species.hpp"
#include "forcesmith/core/types.hpp"
#include "forcesmith/force/evaluate.hpp"
#include "forcesmith/force/force_calculator.hpp"
#include "forcesmith/force/force_calculator_concept.hpp" // GlobalParam
#include "forcesmith/optimization/optimizer.hpp"

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

namespace forcesmith {

class Forcesmith {
public:
  using PairKey =
      std::pair<std::string, std::string>; // sorted (min,max) symbols

  Forcesmith() = default;

  std::size_t
  add_configuration(BoundaryConditions bc = PeriodicBC(Mat3::Identity()));
  std::size_t add_configuration(Configuration cfg);
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

  [[nodiscard]] boost::leaf::result<std::size_t>
  get_atom_index(const Atom &atom);

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
  boost::leaf::result<void> set_ref_force(std::string_view cfg,
                                          std::size_t atom, const Vec3 &f);

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

  boost::leaf::result<void> declare_element(std::string_view sym);

  boost::leaf::result<void>
  set_pair_potential(std::string_view a, std::string_view b, RadialPotential p);
  boost::leaf::result<void> set_density(std::string_view a, RadialPotential p);
  boost::leaf::result<void> set_embedding(std::string_view a,
                                          RadialPotential p);
  void set_global(GlobalParam g);

  boost::leaf::result<void> set_dipole(std::string_view a, std::string_view b,
                                       RadialPotential p);
  boost::leaf::result<void>
  set_quadrupole(std::string_view a, std::string_view b, RadialPotential p);
  boost::leaf::result<void> set_radial(std::string_view a, std::string_view b,
                                       RadialPotential p);
  boost::leaf::result<void> set_angular(std::string_view a, RadialPotential p);
  boost::leaf::result<void> set_tersoff_params(std::string_view a,
                                               std::string_view b,
                                               TersoffParams params);
  boost::leaf::result<void>
  set_stiweb_params(std::string_view a, std::string_view b, SWParams params);
  boost::leaf::result<void> set_stiweb_lambda(std::string_view central,
                                              std::string_view a,
                                              std::string_view b, Param value);

  boost::leaf::result<void> set_pair_param(std::string_view a,
                                           std::string_view b, std::size_t i,
                                           double v);

  boost::leaf::result<void> seed_force_model(ForceCalculator model);

  OptimizerOptions &options() { return opts_; }
  [[nodiscard]] const OptimizerOptions &options() const { return opts_; }

  void set_solver(Solver s) { solver_.emplace(std::move(s)); }

  boost::leaf::result<force::EvalResult> evaluate(std::size_t cfg);
  // Evaluate every configuration, filling descriptors in parallel across cores
  // (mirrors prepare()'s warm-up-then-parallel pattern). Results are ordered by
  // config index and bit-identical to looping evaluate(i) serially.
  boost::leaf::result<std::vector<force::EvalResult>> evaluate_all();
  boost::leaf::result<int> optimize();
  boost::leaf::result<void> write(const std::filesystem::path &path,
                                  std::string_view format = "native");

  boost::leaf::result<const SpeciesRegistry *> species();
  boost::leaf::result<std::span<const Configuration>> configurations();
  boost::leaf::result<const config_index::ConfigIndex *> index();
  boost::leaf::result<Configuration *> config_by_name(std::string_view name);
  boost::leaf::result<const ForceCalculator *> model();

private:
  boost::leaf::result<void> ensure_frozen();
  boost::leaf::result<void> ensure_named();
  boost::leaf::result<void> detach_seeded(std::string_view action);
  boost::leaf::result<SpeciesRegistry> build_registry() const;
  boost::leaf::result<void> materialize_from_spec(); // pair / EAM
  boost::leaf::result<void>
  decompose_seeded_into_spec(const SpeciesRegistry &model_reg); // pair / EAM
  boost::leaf::result<ForceCalculator>
  remap_seeded_ml(const SpeciesRegistry &old_reg,
                  const SpeciesRegistry &new_reg) const;
  [[nodiscard]] boost::leaf::result<Configuration *> config_at(std::size_t cfg);
  boost::leaf::result<void> write_ref_force(std::size_t cfg, std::size_t atom,
                                            const Vec3 &f);

  static PairKey norm_key(std::string_view a, std::string_view b);

  bool dirty_ = true;

  std::vector<Configuration> configs_;

  using PairMap = std::map<PairKey, RadialPotential>;
  using TypeMap = std::map<std::string, RadialPotential>;
  [[nodiscard]] boost::leaf::result<void> place_pair(PairMap &map,
                                                     std::string_view a,
                                                     std::string_view b,
                                                     RadialPotential p);
  [[nodiscard]] boost::leaf::result<void>
  place_type(TypeMap &map, std::string_view a, RadialPotential p);

  std::map<PairKey, RadialPotential> pair_;
  std::map<std::string, RadialPotential> density_;
  std::map<std::string, RadialPotential> embedding_;
  std::map<PairKey, RadialPotential> dipole_;      // ADP
  std::map<PairKey, RadialPotential> quadrupole_;  // ADP
  std::map<PairKey, RadialPotential> radial_;      // angular
  std::map<std::string, RadialPotential> angular_; // angular (central type)
  std::map<PairKey, TersoffParams> tersoff_;
  std::map<PairKey, SWParams> stiweb_;
  std::map<std::tuple<std::string, std::string, std::string>, Param> lambda_;
  std::vector<GlobalParam> globals_;
  std::vector<std::string> declared_;
  OptimizerOptions opts_;
  std::optional<Solver> solver_;

  std::optional<ForceCalculator> seeded_;
  std::optional<SpeciesRegistry> seeded_registry_;

  SpeciesRegistry registry_;
  ForceCalculator model_ = PairForceCalculator{};
  std::optional<config_index::ConfigIndex> index_;
};

} // namespace forcesmith
