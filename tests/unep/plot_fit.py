#!/usr/bin/env python3
"""
Plot the result of a per-element UNEP EAM fit (see fit_eam.py).

fit_eam.py produces two fits per element:
  * the **analytic fit** (stage 1: morse pair + exp_decay density + sqrt
    embedding, refined) -> fits/<el>_eam_analytic.json
  * the **tabular fit**  (stage 2: down-sampled to free knots, spline-refined)
    -> fits/<el>_eam_fit.json

This script plots both. It writes two figures under ``tests/unep/plots/``:

  <el>_functions.png  — pair / density / embedding, analytic fit (dashed) vs
                        tabular fit (solid).
  <el>_parity.png     — a 2×2 observed-vs-predicted grid: rows {analytic,
                        tabular} × cols {energy, stress}, each with its RMSE.

Energy and stress are evaluated by running ``forcesmith --evaluate`` for each fit
against the element's UNEP configs.

Notes:
  * Energy is forces-first: UNEP's per-atom zero is arbitrary, so each energy
    panel has its best constant offset removed — only the spread about the
    diagonal is meaningful.
  * Stress is now in the same unit on both axes (eV/Å³, after the converter
    units fix), so the stress panels are directly interpretable. Stress is not a
    fit target by default (weight 0); the panel is diagnostic.

Usage:
    python3 tests/unep/plot_fit.py --element Cu
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Packed stress order used by the converter, the reader, and --evaluate output.
STRESS_LABELS = ["xx", "yy", "zz", "xy", "yz", "zx"]


def load(path):
    with open(path) as fh:
        return json.load(fh)


def section_curve(section):
    """Return (x, y) for a tabulated potential section."""
    pot = section["potentials"][0]
    y = np.array(pot["knots"], dtype=float)
    x = np.linspace(pot["rmin"], pot["rmax"], y.size)
    return x, y


def evaluate(forcesmith, config, pot):
    """Run --evaluate and return the parsed report dict."""
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        report = tf.name
    cmd = [str(forcesmith), "-c", str(config), "-s", str(pot), "--evaluate", report]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"forcesmith --evaluate failed for {pot}:\n{proc.stdout}\n{proc.stderr}")
    data = load(report)
    Path(report).unlink(missing_ok=True)
    return data


# ── figure 1: the fitted functions (both fits) ────────────────────────────────

def plot_functions(element, analytic, tabular, out):
    panels = [
        ("pair", "pair potential  φ(r)", "r (Å)", "φ (eV)"),
        ("density", "density  ρ(r)", "r (Å)", "ρ (arb.)"),
        ("embedding", "embedding  F(ρ)", "ρ (arb.)", "F (eV)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title, xlab, ylab) in zip(axes, panels):
        if analytic is not None:
            xa, ya = section_curve(analytic[key])
            ax.plot(xa, ya, "--", color="C0", lw=1.8, label="analytic fit")
        xt, yt = section_curve(tabular[key])
        ax.plot(xt, yt, "-", color="C2", lw=2, label="tabular fit")
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle(f"{element}: fitted EAM functions", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ── figure 2: observed vs predicted parity, both fits × {energy, stress} ──────

def collect_parity(report):
    """Return per-atom energy arrays and per-component stress arrays."""
    e_ref, e_calc = [], []
    s_ref = [[] for _ in range(6)]
    s_calc = [[] for _ in range(6)]
    for cfg in report["configs"]:
        n = cfg["natoms"]
        e_ref.append(cfg["ref_energy"] / n)
        e_calc.append(cfg["calc_energy"] / n)
        for i in range(6):
            s_ref[i].append(cfg["ref_stress"][i])
            s_calc[i].append(cfg["calc_stress"][i])
    return (np.array(e_ref), np.array(e_calc),
            np.array(s_ref), np.array(s_calc))


def energy_panel(ax, ref, calc, row_label):
    offset = float(np.mean(calc - ref))
    calc = calc - offset
    lo = min(ref.min(), calc.min())
    hi = max(ref.max(), calc.max())
    pad = 0.05 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.scatter(ref, calc, s=12, alpha=0.5, color="C0")
    rmse = math.sqrt(float(np.mean((calc - ref) ** 2)))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("observed (DFT) energy (eV/atom)")
    ax.set_ylabel(f"{row_label}\npredicted (eV/atom)")
    ax.set_title(f"energy — RMSE {rmse:.4g} eV/atom\n(offset {offset:+.3f} removed)",
                 fontsize=9)


def stress_panel(ax, s_ref, s_calc, row_label):
    sr = s_ref.flatten(); sc = s_calc.flatten()
    lo = min(sr.min(), sc.min()); hi = max(sr.max(), sc.max())
    pad = 0.05 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
    for i, lbl in enumerate(STRESS_LABELS):
        ax.scatter(s_ref[i], s_calc[i], s=10, alpha=0.5, color=f"C{i}", label=lbl)
    rmse = math.sqrt(float(np.mean((sc - sr) ** 2)))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("observed (DFT) stress (eV/Å³)")
    ax.set_ylabel(f"{row_label}\npredicted (eV/Å³)")
    ax.set_title(f"stress — RMSE {rmse:.4g} eV/Å³", fontsize=9)
    ax.legend(fontsize=6, ncol=2)


def plot_parity(element, report_analytic, report_tabular, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for row, (label, report) in enumerate(
            [("analytic fit", report_analytic), ("tabular fit", report_tabular)]):
        e_ref, e_calc, s_ref, s_calc = collect_parity(report)
        energy_panel(axes[row][0], e_ref, e_calc, label)
        stress_panel(axes[row][1], s_ref, s_calc, label)
    fig.suptitle(f"{element}: observed vs predicted (analytic vs tabular)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--element", required=True)
    p.add_argument("--data-dir", default="data/unep")
    p.add_argument("--out-dir", default="tests/unep")
    p.add_argument("--forcesmith", default="build/release/forcesmith")
    args = p.parse_args(argv)

    el = args.element.lower()
    out_dir = Path(args.out_dir)
    tab_path = out_dir / "fits" / f"{el}_eam_fit.json"
    ana_path = out_dir / "fits" / f"{el}_eam_analytic.json"
    if not tab_path.exists():
        sys.exit(f"ERROR: no tabular fit at {tab_path}. Run fit_eam.py "
                 f"--elements {args.element} first.")
    config = Path(args.data_dir) / f"{el}_dft_unep.json"
    forcesmith = Path(args.forcesmith)
    if not forcesmith.exists():
        sys.exit(f"ERROR: forcesmith binary not found at {forcesmith}.")

    tabular = load(tab_path)
    analytic = load(ana_path) if ana_path.exists() else None

    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fn_png = plots / f"{el}_functions.png"
    par_png = plots / f"{el}_parity.png"

    plot_functions(args.element, analytic, tabular, fn_png)

    report_tabular = evaluate(forcesmith, config, tab_path)
    report_analytic = (evaluate(forcesmith, config, ana_path)
                       if analytic is not None else report_tabular)
    plot_parity(args.element, report_analytic, report_tabular, par_png)

    print(f"wrote {fn_png}")
    print(f"wrote {par_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
