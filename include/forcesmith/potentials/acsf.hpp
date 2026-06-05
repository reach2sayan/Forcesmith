#pragma once

// Behler-style atom-centered symmetry functions (ACSF) — a local per-atom ML
// descriptor that feeds the MLBaseImpl pipeline.
//
// Families (DScribe ACSF conventions), all using the cosine cutoff
//   f_c(r) = ½(1 + cos(π r / rcut)) for r < rcut, else 0:
//   G1  (per species)       Σ_j f_c(r_ij)
//   G2  (per species)       Σ_j exp(−η (r_ij − Rs)²) f_c(r_ij)
//   G3  (per species)       Σ_j cos(κ r_ij) f_c(r_ij)
//   G4  (per species pair)  2^{1−ζ} Σ_{j<k} (1+λ cosθ)^ζ
//                              exp(−η(r_ij²+r_ik²+r_jk²))
//                              f_c(r_ij)f_c(r_ik)f_c(r_jk)
//   G5  (per species pair)  2^{1−ζ} Σ_{j<k} (1+λ cosθ)^ζ
//                              exp(−η(r_ij²+r_ik²)) f_c(r_ij)f_c(r_ik)
// cosθ = (d_j·d_k)/(r_ij r_ik). Radial families channel over the neighbour
// species s∈[0,ntypes); angular families channel over unordered species pairs
// (the upper_triangle(ntypes) enumeration, matching SOAP). All hyperparameters
// are fixed; only the head coefficients are fitted. dD/dr is closed-form for
// every family, so the model reports analytic gradients.

#include "forcesmith/core/atom.hpp"
#include "forcesmith/force/descriptor_layout.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/ml_force.hpp"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace forcesmith {

// The five symmetry-function families, in descriptor-block order. Radial
// families (G1-G3) channel over the neighbour species s; angular families
// (G4-G5) channel over unordered species pairs. The underlying values double as
// the block ids in AcsfLayout's DescriptorLayout (pushed in this order).
enum class SymmetryFunctionFamily : std::size_t { G1, G2, G3, G4, G5 };

// Flat layout of the ACSF descriptor vector — the contiguous blocks
//   [G1 per species][G2 per species][G3 per species][G4 per pair][G5 per pair]
// — built once from the family counts + ntypes. A thin wrapper over the shared
// DescriptorLayout that keeps the radial/angular vocabulary; the single source
// of truth for both descriptor_size() and the per-component indices, so
// get_descriptor never hand-rolls offset arithmetic.
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

  // base + chan * count + t. radial() takes the per-species channel s;
  // angular() takes the per-pair channel po; the two names document which
  // channel space the caller is in.
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

  // Re-rank hook (see MLBase::remap): new-layout flat index → old-layout flat
  // index (or nullopt for a block touching a newly-added species). Delegates to
  // the generic remap_layout over this descriptor's block structure.
  [[nodiscard]] std::vector<std::optional<Eigen::Index>>
  descriptor_index_map(const SpeciesRegistry &old_reg,
                       const SpeciesRegistry &new_reg) const;

private:
  // A valid neighbour after distance/species filtering (collect_neighbours).
  // `orig` is the index into atom.neighbors so the analytic gradients scatter
  // into the full-length, neighbour-parallel grad_neigh array.
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

static_assert(ForceCalculatorModel<ACSF>);

} // namespace forcesmith
