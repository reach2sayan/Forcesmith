#include "potfit/optimization/potfit_functor.hpp"

#include "potfit/core/neighbor_list.hpp"
#include "potfit/events/signals.hpp"
#include "potfit/force/force_calculator.hpp"

#include <algorithm>
#include <cmath>
#include <span>

namespace potfit {

PotfitFunctor::PotfitFunctor(std::span<Configuration> configs,
                             std::span<Potential>     potentials,
                             double                   energy_weight)
    : configs_(configs), potentials_(potentials), energy_weight_(energy_weight)
{
    inputs_ = 0;
    for (const auto& p : potentials_)
        inputs_ += p.param_count();

    values_ = 0;
    for (const auto& cfg : configs_)
        values_ += static_cast<int>(3 * cfg.atoms.size()) + 1;
}

int PotfitFunctor::operator()(const Eigen::VectorXd& x,
                              Eigen::VectorXd& fvec) const
{
    // Scatter x into potentials.
    int off = 0;
    for (auto& p : potentials_) {
        p.scatter_params(x, off);
        off += p.param_count();
    }

    // Compute rcut = max of all span().second.
    double rcut = 0.0;
    for (const auto& p : potentials_)
        rcut = std::max(rcut, p.span().second);

    int row = 0;
    for (std::size_t c = 0; c < configs_.size(); ++c) {
        auto& cfg = configs_[c];

        // Pass potentials_ span directly — NeighborEntry::pot pointers into it
        // remain valid for the lifetime of the optimizer run.
        build_neighbor_list(cfg, rcut, std::span<const Potential>(potentials_));
        PairForceCalculator calc;
        calc.conf_index = static_cast<std::uint64_t>(c);
        calc.eval_forces(cfg);

        for (const auto& atom : cfg.atoms) {
            fvec[row++] = atom.calc_force[0] - atom.force[0];
            fvec[row++] = atom.calc_force[1] - atom.force[1];
            fvec[row++] = atom.calc_force[2] - atom.force[2];
        }
        fvec[row++] = energy_weight_ * (cfg.calc_energy - cfg.energy);
    }

    events::on_iteration(events::IterationStats{
        ++iter_,
        fvec.squaredNorm(),
        0.0
    });

    return 0;
}

int PotfitFunctor::df(const Eigen::VectorXd& x, Eigen::MatrixXd& fjac) const
{
    constexpr double delta = 1e-5;
    Eigen::VectorXd fp(values_), fm(values_);
    Eigen::VectorXd xp = x;

    for (int j = 0; j < inputs_; ++j) {
        xp[j] += delta;
        (*this)(xp, fp);
        xp[j] -= 2.0 * delta;
        (*this)(xp, fm);
        xp[j] += delta;  // restore

        fjac.col(j) = (fp - fm) / (2.0 * delta);
    }
    return 0;
}

int PotfitFunctor::inputs() const { return inputs_; }
int PotfitFunctor::values() const { return values_; }

}  // namespace potfit
