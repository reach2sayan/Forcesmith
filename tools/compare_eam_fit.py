#!/usr/bin/env python3
"""
Compare a forcesmith-fitted EAM potential against the NIST reference.

Reads the tabulated EAM JSON written by the forcesmith CLI (``{elt}_eam_fit.json``)
and the NIST reference (``{elt}_nist_true.json``), then produces a
gauge-invariant physics comparison:

  Panel A  — force parity: predicted force components (fit vs true) over a set
             of displaced FCC cells, with RMSE.
  Panel B  — energy-vs-lattice-constant E(a) curve for fit vs true, with RMSE.

Raw EAM functions phi(r)/rho(r)/F(rho) are NOT plotted: they differ by the EAM
gauge transformation even for equivalent potentials, so forces and energies are
the physically meaningful comparison.

The EAM evaluation reuses the engine in generate_eam_configs.py, so the
comparison is consistent with how the training data was produced.

Usage:
    python3 tools/compare_eam_fit.py [--element Cu] [--data-dir data]

Requirements: numpy, scipy, matplotlib
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    # Akima matches the C++ SplinePotential (boost makima) far better than a
    # natural cubic spline, which overshoots between widely spaced knots.
    from scipy.interpolate import Akima1DInterpolator
except ImportError:
    sys.exit("ERROR: scipy is required.  Install with:  pip install scipy")

import matplotlib
matplotlib.use("Agg")  # headless: write PNG without a display
import matplotlib.pyplot as plt

# Reuse the EAM force engine and FCC helpers used to generate the training data.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_eam_configs import compute_eam_forces, fcc_cell, A0  # noqa: E402


# ── Load a tabulated-EAM JSON file into evaluation splines ─────────────────────

def _section_spline(section):
    """Build a CubicSpline from one {format,potentials:[{rmin,rmax,knots}]} block.

    Returns (spline, rmin, rmax).  Single-element only (potentials[0]).
    """
    if section.get("format") != "tabulated":
        raise ValueError(f"expected tabulated section, got {section.get('format')!r}")
    pot = section["potentials"][0]
    knots = np.asarray(pot["knots"], dtype=float)
    x = np.linspace(pot["rmin"], pot["rmax"], len(knots))
    interp = Akima1DInterpolator(x, knots)
    interp.extrapolate = True  # avoid NaN at domain edges
    return interp, pot["rmin"], pot["rmax"]


def load_eam(path):
    """Load a tabulated-EAM JSON file → (F_cs, rho_at_cs, phi_cs, cutoff)."""
    j = json.loads(Path(path).read_text())
    if j.get("model") != "eam":
        raise ValueError(f"{path}: not an EAM model (model={j.get('model')!r})")
    phi_cs, _, phi_rmax = _section_spline(j["pair"])
    rho_cs, _, _        = _section_spline(j["density"])
    F_cs,   _, _        = _section_spline(j["embedding"])
    return F_cs, rho_cs, phi_cs, phi_rmax


def load_curves(path, ngrid=400):
    """Load the raw tabulated functions for plotting, evaluated on a fine grid.

    Returns {'pair':(x,y), 'density':(x,y), 'embedding':(x,y)} or None if the
    file is not tabulated (e.g. an analytic start, which has no knots).
    """
    j = json.loads(Path(path).read_text())
    if j.get("model") != "eam":
        return None
    out = {}
    for key in ("pair", "density", "embedding"):
        if j[key].get("format") != "tabulated":
            return None  # analytic start: no knots to plot
        cs, lo, hi = _section_spline(j[key])
        x = np.linspace(lo, hi, ngrid)
        out[key] = (x, cs(x))
    return out


# ── Comparison ─────────────────────────────────────────────────────────────────

def force_parity(element, true_eam, fit_eam, n_cells=10, sigma=0.05):
    """Predicted force components (fit vs true) over displaced FCC cells."""
    a0 = A0[element]
    rng = np.random.default_rng(2024)
    f_true_all, f_fit_all = [], []
    for _ in range(n_cells):
        a, pos = fcc_cell(a0)
        pos = pos + rng.normal(0, sigma, size=pos.shape)
        _, f_true = compute_eam_forces(pos, a, *true_eam)
        _, f_fit  = compute_eam_forces(pos, a, *fit_eam)
        f_true_all.append(f_true.ravel())
        f_fit_all.append(f_fit.ravel())
    return np.concatenate(f_true_all), np.concatenate(f_fit_all)


def energy_curve(element, true_eam, fit_eam, strains=None):
    """Cohesive energy/atom vs lattice constant for true and fit."""
    a0 = A0[element]
    if strains is None:
        strains = np.linspace(-0.06, 0.06, 25)
    a_vals, e_true, e_fit = [], [], []
    for s in strains:
        a, pos = fcc_cell(a0, s)
        et, _ = compute_eam_forces(pos, a, *true_eam)
        ef, _ = compute_eam_forces(pos, a, *fit_eam)
        a_vals.append(a)
        e_true.append(et / len(pos))
        e_fit.append(ef / len(pos))
    return np.array(a_vals), np.array(e_true), np.array(e_fit)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def compare_element(element, data_dir, tag=""):
    elt = element.lower()
    suffix = f"_{tag}" if tag else ""
    true_path = data_dir / f"{elt}_nist_true.json"
    fit_path  = data_dir / f"{elt}_eam{suffix}_fit.json"
    for p in (true_path, fit_path):
        if not p.exists():
            print(f"  SKIP {element}: missing {p}")
            return

    true_eam = load_eam(true_path)
    fit_eam  = load_eam(fit_path)

    f_true, f_fit = force_parity(element, true_eam, fit_eam)
    f_rmse = rmse(f_true, f_fit)

    a_vals, e_true, e_fit = energy_curve(element, true_eam, fit_eam)
    # Absolute cohesive energy is gauge/reference-dependent (the EAM gauge leaves
    # forces and energy *differences* invariant but can shift the zero), so we
    # reference both curves to their energy at the equilibrium lattice constant a0
    # before comparing.  The RMSE of the referenced curves then measures the
    # physically meaningful equation of state (curvature / bulk modulus) rather
    # than the meaningless constant offset.  We reference to a0 (not each curve's
    # own min) so a fit with an inverted/shifted EOS is shown faithfully against
    # the true equilibrium point instead of being re-zeroed at a spurious minimum.
    i0 = int(np.argmin(np.abs(a_vals - A0[element])))
    e_true_rel = e_true - e_true[i0]
    e_fit_rel  = e_fit - e_fit[i0]
    e_rmse = rmse(e_true_rel, e_fit_rel)
    e_offset = float(np.mean(e_true - e_fit))  # raw absolute offset, for reference

    label = f"{element} [{tag}]" if tag else element
    print(f"  {label}:  force RMSE = {f_rmse:.4f} eV/Ang   "
          f"energy RMSE = {e_rmse:.4f} eV/atom (ref. to a0; "
          f"abs offset = {e_offset:.4f} eV/atom)")

    fig, (axp, axe) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A — force parity
    lim = max(np.abs(f_true).max(), np.abs(f_fit).max()) * 1.05
    axp.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="ideal (y=x)")
    axp.scatter(f_true, f_fit, s=10, alpha=0.5, color="C0")
    axp.set_xlim(-lim, lim); axp.set_ylim(-lim, lim)
    axp.set_aspect("equal", "box")
    axp.set_xlabel("NIST-true force component (eV/Å)")
    axp.set_ylabel("fitted force component (eV/Å)")
    axp.set_title(f"{element} force parity\nRMSE = {f_rmse:.4f} eV/Å")
    axp.legend(loc="upper left")

    # Panel B — E(a) curve, referenced to each curve's equilibrium minimum so the
    # gauge-dependent constant offset does not swamp the comparison.
    axe.plot(a_vals, e_true_rel, "o-", color="C1", ms=4, label="NIST-true")
    axe.plot(a_vals, e_fit_rel,  "s--", color="C2", ms=4, label="fitted")
    axe.set_xlabel("lattice constant a (Å)")
    axe.set_ylabel("relative energy ΔE (eV/atom)")
    axe.set_title(f"{element} E(a) — ΔE referenced to a0 = {A0[element]:.3f} Å\n"
                  f"RMSE = {e_rmse:.4f} eV/atom")
    axe.text(0.02, 0.98, f"abs. offset (gauge) = {e_offset:.3f} eV/atom",
             transform=axe.transAxes, va="top", ha="left", fontsize=8,
             color="0.4")
    axe.legend()

    fig.suptitle(f"{label}: fitted EAM vs NIST reference", fontweight="bold")
    fig.tight_layout()
    out = data_dir / f"{elt}{suffix}_fit_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"        wrote {out}")


def plot_functions(element, data_dir, tag=""):
    """Overlay the raw EAM functions phi(r), rho(r), F(rho): NIST vs fitted.

    NOTE: raw EAM functions are gauge-dependent — F(rho)→F(rho)+λρ with a
    compensating change in phi leaves all forces/energies unchanged — so the
    fitted curves can differ from NIST even for an excellent fit.  Forces/energy
    (the fit_comparison plot) are the gauge-invariant measure of fit quality.
    """
    elt = element.lower()
    suffix = f"_{tag}" if tag else ""
    true_path  = data_dir / f"{elt}_nist_true.json"
    fit_path   = data_dir / f"{elt}_eam{suffix}_fit.json"
    start_path = data_dir / f"{elt}_eam{suffix}_start.json"
    for p in (true_path, fit_path):
        if not p.exists():
            print(f"  SKIP {element} functions: missing {p}")
            return

    true_c = load_curves(true_path)
    fit_c  = load_curves(fit_path)
    start_c = load_curves(start_path) if start_path.exists() else None

    panels = [("pair", "pair potential φ(r)", "r (Å)", "φ (eV)"),
              ("density", "electron density ρ(r)", "r (Å)", "ρ (a.u.)"),
              ("embedding", "embedding F(ρ)", "ρ (a.u.)", "F (eV)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title, xlab, ylab) in zip(axes, panels):
        ax.plot(*true_c[key], "-", color="C1", lw=2, label="NIST-true")
        ax.plot(*fit_c[key],  "--", color="C2", lw=2, label="fitted")
        if start_c is not None:
            ax.plot(*start_c[key], ":", color="C0", lw=1.5, alpha=0.7,
                    label="start")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
        ax.legend()
    # phi(r) often spans a huge repulsive range at small r; clip to the bonding
    # region so the curves are legible.
    axes[0].set_ylim(*_nice_ylim(true_c["pair"], fit_c["pair"]))

    label = f"{element} [{tag}]" if tag else element
    fig.suptitle(f"{label}: EAM functions, fitted vs NIST  "
                 f"(raw functions are gauge-dependent)", fontweight="bold")
    fig.tight_layout()
    out = data_dir / f"{elt}{suffix}_functions.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"        wrote {out}")


def _nice_ylim(*curves, pad=0.15):
    """y-limits from the function values beyond the steep small-r repulsion."""
    vals = []
    for x, y in curves:
        # ignore the inner 15% of the r-range where phi blows up
        cut = x[0] + 0.15 * (x[-1] - x[0])
        vals.append(y[x >= cut])
    v = np.concatenate(vals)
    lo, hi = float(np.min(v)), float(np.max(v))
    span = hi - lo or 1.0
    return lo - pad * span, hi + pad * span


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--element", choices=["Cu", "Al"],
                        help="element to compare (default: both)")
    parser.add_argument("--data-dir", default="data",
                        help="directory holding the JSON files (default: data/)")
    parser.add_argument("--functions", action="store_true",
                        help="also plot the raw EAM functions φ(r)/ρ(r)/F(ρ), "
                             "NIST vs fitted (gauge-dependent)")
    parser.add_argument("--tag", default="",
                        help="fit variant to compare: '' (default) reads "
                             "{elt}_eam_fit.json; 'analytic' reads "
                             "{elt}_eam_analytic_fit.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    elements = [args.element] if args.element else ["Cu", "Al"]

    print("Comparing fitted EAM vs NIST reference:")
    for element in elements:
        compare_element(element, data_dir, args.tag)
        if args.functions:
            plot_functions(element, data_dir, args.tag)


if __name__ == "__main__":
    main()
