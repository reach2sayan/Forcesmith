#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/force/descriptor_layout.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/ml_force.hpp"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace forcesmith {

enum class SymmetryFunctionFamily : std::size_t {
  G1 = 0,
  G2 = 1,
  G3 = 2,
  G4 = 3,
  G5 = 4
};

struct AcsfLayout {
  DescriptorLayout d_;
  AcsfLayout(std::size_t S, std::size_t nG1, std::size_t nG2, std::size_t nG3,
             std::size_t nG4, std::size_t nG5) {
    const std::size_t P = S * (S + 1) / 2;
    d_.add(nG1, S); // SymmetryFunctionFamily::G1
    d_.add(nG2, S); // SymmetryFunctionFamily::G2
    d_.add(nG3, S); // SymmetryFunctionFamily::G3
    d_.add(nG4, P); // SymmetryFunctionFamily::G4
    d_.add(nG5, P); // SymmetryFunctionFamily::G5
  }

  [[nodiscard]] constexpr Eigen::Index size() const { return d_.size(); }

  [[nodiscard]] Eigen::Index radial(SymmetryFunctionFamily f, std::size_t s,
                                    std::size_t t) const {
    return d_.index(std::to_underlying(f), s, t);
  }
  [[nodiscard]] Eigen::Index angular(SymmetryFunctionFamily f, std::size_t po,
                                     std::size_t t) const {
    return d_.index(std::to_underlying(f), po, t);
  }
};

struct ACSF : MLBase<ACSF> {
  struct G2 {
    double eta = 1.0; // Gaussian width
    double rs = 0.0;  // Gaussian centre (shift)
  };
  struct G3 {
    double kappa = 1.0; // cosine wavevector
  };
  struct G4 {
    double eta = 1.0;
    double zeta = 1.0;
    double lambda = 1.0; // ±1
  };
  struct G5 {
    double eta = 1.0;
    double zeta = 1.0;
    double lambda = 1.0; // ±1
  };

  std::size_t g1 = 0;     // number of parameterless G1 channels (Σ f_c)
  std::vector<G2> radial; // G2 functions
  std::vector<G3> g3;     // G3 functions
  std::vector<G4> g4;     // G4 angular functions
  std::vector<G5> g5;     // G5 angular functions
  double rcut = 6.0;
  bool use_analytic_grads = true;

  [[nodiscard]] DescriptorValue get_descriptor(const Atom &a) const;
  [[nodiscard]] constexpr double descriptor_cutoff() const { return rcut; }
  [[nodiscard]] constexpr bool analytic_grads() const {
    return use_analytic_grads;
  }
  [[nodiscard]] std::size_t descriptor_size() const {
    return static_cast<std::size_t>(
        AcsfLayout{ntypes, g1, radial.size(), g3.size(), g4.size(), g5.size()}
            .size());
  }

  [[nodiscard]] AcsfLayout layout_for(std::size_t S) const {
    return AcsfLayout{S, g1, radial.size(), g3.size(), g4.size(), g5.size()};
  }

private:
  struct Neighbor {
    std::size_t orig_index;
    Vec3 d;    // bond vector r_j − r_i
    Vec3 rhat; // d / r
    double r;
    double fc, fcp; // cosine cutoff value & derivative
    std::size_t s;  // neighbour species
  };

  [[nodiscard]] std::vector<Neighbor> collect_neighbors(const Atom &a) const;
  void accumulate_radial(DescriptorValue &out, const std::vector<Neighbor> &nb,
                         const AcsfLayout &L) const; // G1/G2/G3
  void accumulate_angular(DescriptorValue &out, const std::vector<Neighbor> &nb,
                          const AcsfLayout &L) const; // G4/G5
};

static_assert(CForceCalculator<ACSF>);

static_assert(CDescriptorModel<ACSF>);

} // namespace forcesmith
