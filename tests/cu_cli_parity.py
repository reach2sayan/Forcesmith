#!/usr/bin/env python3
"""Fit Cu with three models — EAM, SOAP, ACSF — using ONLY the ``forcesmith``
CLI, on the full UNEP Cu dataset, then plot force and energy parity.

Everything goes through the one built binary ``build/release/forcesmith``; there
are no separate ``unep_fit`` / ``ml_fit`` drivers involved. For each model the
pipeline is the same three CLI calls:

  1. ``forcesmith init  --model <m> ...      --out  work/<m>_start.json``
  2. ``forcesmith -c <data> -s start -e fit  --maxiter N --eweight w``
  3. ``forcesmith -c <data> -s fit  --evaluate report.json``

then we read the per-atom forces / per-config energies out of the report and
draw a 2×3 parity grid (rows {force, energy} × cols {EAM, SOAP, ACSF}).

All three fits are forces-first (small/zero energy weight): UNEP's per-atom
energy zero is arbitrary, so the energy panels have their best constant offset
removed — only the spread about the diagonal is meaningful.

Usage:
    python3 tests/cu_cli_parity.py                 # full dataset, defaults
    python3 tests/cu_cli_parity.py --maxiter 200   # more optimizer iterations
    python3 tests/cu_cli_parity.py --skip-fit      # replot from existing fits
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── model definitions ─────────────────────────────────────────────────────────
# Each model is fit entirely through `forcesmith init` + the optimizer. The init
# flags pick a well-conditioned scaffold; the fit flags set the forces-first
# weighting. Defaults mirror the UNEP/ML fit drivers:
#   * EAM uses a *morse* pair (bounded at small r) — the default `lj` scaffold
#     diverges to -nan on this data — with exp_decay density and sqrt embedding.
#   * SOAP / ACSF use a linear head and pure forces (eweight 0).
MODELS = {
    "eam": {
        "label": "EAM (morse)",
        "init": ["--model", "eam", "--functions", "morse,exp_decay,sqrt"],
        "eweight": 0.1,
        "maxiter": 1000,
    },
    "soap": {
        "label": "SOAP (linear)",
        "init": ["--model", "soap", "--n-max", "4", "--l-max", "3",
                 "--sigma", "0.5"],
        "eweight": 0.0,
        "maxiter": 150,
    },
    "acsf": {
        "label": "ACSF (linear)",
        "init": ["--model", "acsf",
                 "--g2-eta", "0.05", "0.2", "0.5", "1.0", "2.0", "4.0"],
        "eweight": 0.0,
        "maxiter": 150,
    },
}
ORDER = ["eam", "soap", "acsf"]


def run(cmd):
    """Run a forcesmith CLI command, streaming nothing, failing loudly."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"command failed ({proc.returncode}):\n  "
                 + " ".join(map(str, cmd))
                 + f"\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return proc.stdout


def fit_model(name, spec, args, work, fits):
    """init → fit → return the fit path, honouring --skip-fit."""
    start = work / f"{name}_start.json"
    fit = fits / f"cu_{name}.json"
    maxiter = args.maxiter if args.maxiter is not None else spec["maxiter"]

    if args.skip_fit and fit.exists():
        print(f"[{name}] reusing existing fit {fit}")
        return fit

    print(f"[{name}] init scaffold (cutoff {args.cutoff} Å)")
    run([args.forcesmith, "init", *spec["init"],
         "--cutoff", str(args.cutoff), "--out", str(start)])

    print(f"[{name}] fitting on full dataset "
          f"(maxiter {maxiter}, eweight {spec['eweight']}, algo {args.algorithm}) ...")
    t0 = time.time()
    run([args.forcesmith, "-c", str(args.config), "-s", str(start),
         "-e", str(fit), "--maxiter", str(maxiter),
         "--eweight", str(spec["eweight"]),
         "--stress-weight", str(args.stress_weight),
         "-a", args.algorithm])
    print(f"[{name}] fit done in {time.time() - t0:.1f}s → {fit}")
    return fit


def evaluate(name, fit, args, reports):
    """Run --evaluate and parse the per-config / per-atom report."""
    report = reports / f"cu_{name}_report.json"
    print(f"[{name}] evaluating fit against full dataset ...")
    run([args.forcesmith, "-c", str(args.config), "-s", str(fit),
         "--evaluate", str(report)])
    with open(report) as fh:
        data = json.load(fh)

    f_ref, f_calc = [], []
    e_ref, e_calc = [], []
    for cfg in data["configs"]:
        n = cfg["natoms"]
        e_ref.append(cfg["ref_energy"] / n)
        e_calc.append(cfg["calc_energy"] / n)
        for atom in cfg["atoms"]:
            f_ref.extend(atom["ref_force"])
            f_calc.extend(atom["calc_force"])
    return {
        "f_ref": np.asarray(f_ref),
        "f_calc": np.asarray(f_calc),
        "e_ref": np.asarray(e_ref),
        "e_calc": np.asarray(e_calc),
        "nconf": data["nconf"],
    }


def rmse(a, b):
    return math.sqrt(float(np.mean((a - b) ** 2)))


def force_panel(ax, res, label):
    x, y = res["f_ref"], res["f_calc"]
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad
    hb = ax.hexbin(x, y, gridsize=70, bins="log", cmap="viridis",
                   extent=(lo, hi, lo, hi), mincnt=1)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("DFT force (eV/Å)")
    ax.set_ylabel("predicted force (eV/Å)")
    ax.set_title(f"{label}\nforce RMSE {rmse(x, y):.4g} eV/Å  "
                 f"({x.size} comps)", fontsize=9)
    return hb


def energy_panel(ax, res):
    ref = res["e_ref"]
    offset = float(np.mean(res["e_calc"] - ref))
    calc = res["e_calc"] - offset
    lo = float(min(ref.min(), calc.min()))
    hi = float(max(ref.max(), calc.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.scatter(ref, calc, s=12, alpha=0.5, color="C0")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("DFT energy (eV/atom)")
    ax.set_ylabel("predicted (eV/atom)")
    ax.set_title(f"energy RMSE {rmse(ref, calc):.4g} eV/atom\n"
                 f"(offset {offset:+.3f} removed)", fontsize=9)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forcesmith", default="build/release/forcesmith",
                   help="path to the forcesmith binary")
    p.add_argument("--config", default="data/unep/cu_dft_unep.json",
                   help="UNEP Cu dataset (full dataset by default)")
    p.add_argument("--out-dir", default="tests/cli_fits",
                   help="work/fits/plots output root")
    p.add_argument("--cutoff", type=float, default=5.0, help="cutoff radius (Å)")
    p.add_argument("--maxiter", type=int, default=None,
                   help="override per-model optimizer iterations")
    p.add_argument("--stress-weight", type=float, default=0.0)
    p.add_argument("--algorithm", default="ipopt",
                   help="lm | powell | de | ls | ipopt")
    p.add_argument("--skip-fit", action="store_true",
                   help="reuse existing fits, just re-evaluate and replot")
    args = p.parse_args(argv)

    args.forcesmith = str(Path(args.forcesmith))
    if not Path(args.forcesmith).exists():
        sys.exit(f"ERROR: forcesmith binary not found at {args.forcesmith}")
    args.config = Path(args.config)
    if not args.config.exists():
        sys.exit(f"ERROR: dataset not found at {args.config}")

    root = Path(args.out_dir)
    work, fits, reports, plots = (root / "work", root / "fits",
                                  root / "reports", root / "plots")
    for d in (work, fits, reports, plots):
        d.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in ORDER:b
        spec = MODELS[name]
        fit = fit_model(name, spec, args, work, fits)
        results[name] = evaluate(name, fit, args, reports)

    # ── parity grid: rows {force, energy} × cols {eam, soap, acsf} ────────────
    fig, axes = plt.subplots(2, len(ORDER), figsize=(5.2 * len(ORDER), 10),
                             layout="constrained")
    last_hb = None
    for col, name in enumerate(ORDER):
        last_hb = force_panel(axes[0][col], results[name], MODELS[name]["label"])
        energy_panel(axes[1][col], results[name])
    fig.colorbar(last_hb, ax=axes[0].tolist(), shrink=0.85,
                 label="count (log)")
    nconf = results[ORDER[0]]["nconf"]
    fig.suptitle(f"Cu — forcesmith CLI fits on full UNEP dataset "
                 f"({nconf} configs)", fontweight="bold", fontsize=13)
    out = plots / "cu_cli_parity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print("\n=== force RMSE (eV/Å) ===")
    for name in ORDER:
        r = results[name]
        print(f"  {MODELS[name]['label']:16s} {rmse(r['f_ref'], r['f_calc']):.4f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
