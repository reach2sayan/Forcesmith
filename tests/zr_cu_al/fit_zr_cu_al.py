#!/usr/bin/env python3
"""Fit the ternary Zr–Cu–Al alloy with the ``forcesmith`` CLI and plot parity.

A true three-component (ntypes=3) fit on H.W. Sheng's ternary DFT training set
(``~/Downloads/Zr-Cu-Al.dat``). By default it runs the two machine-learning
descriptors the user asked for — **ACSF → SOAP** — both with a linear energy head
(closed-form least squares, ``-a lsq``: the global minimum in one shot, and the
engine streams the Jacobian so it stays low-memory on the full set). The analytic
EAM model is also available via ``--models`` for comparison.

The raw potfit ``.dat`` file has no ``#C`` element line, so the type-index →
element map is supplied explicitly to ``tools/dat2json.py`` (recovered from the
VASP source paths)::

    Zr-Cu-Al.dat : Zr,Cu,Al

The engine sorts species by ascending atomic number (Al=13, Cu=29, Zr=40), which
fixes the EAM section ordering used by the function plots: pairs in upper-triangle
slot order ``[Al-Al, Al-Cu, Al-Zr, Cu-Cu, Cu-Zr, Zr-Zr]``, then density Al/Cu/Zr,
then embedding Al/Cu/Zr.

Each model pipeline is:

  1. ``forcesmith init  --model <m> --ntypes 3 ...  --out work/zr_cu_al_<m>_start.json``
  2. ``forcesmith -c <data> -s start -e fit ...``   (lsq for linear heads; DE→local for EAM)
  3. ``forcesmith -c <data> -s fit  --evaluate report.json``

Outputs (under tests/zr_cu_al/plots/):
  * ``zr_cu_al_<model>_parity.png``  force+energy parity for that single fit
  * ``zr_cu_al_all_parity.png``      combined {force,energy} × {models run} grid
  * ``zr_cu_al_eam_functions.png``   6 pair φ(r) / 3 density ρ(r) / 3 embedding F(ρ)
                                     of the fitted analytic EAM (only if eam is run)

All fits are forces-first (small energy weight): the DFT per-atom energy zero is
arbitrary, so energy panels have their best constant offset removed.

Usage:
    python3 tests/zr_cu_al/fit_zr_cu_al.py                          # ACSF + SOAP, full set
    python3 tests/zr_cu_al/fit_zr_cu_al.py --max-configs 30 --maxiter 20   # smoke
    python3 tests/zr_cu_al/fit_zr_cu_al.py --models acsf,soap,eam   # add the EAM fit
    python3 tests/zr_cu_al/fit_zr_cu_al.py --skip-fit               # replot existing fits
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


# ── system ───────────────────────────────────────────────────────────────────────
# `order` is the engine's Z-sorted index space (ascending atomic number); it sets
# the EAM pair-slot and density/embedding ordering for the function plots.
SYSTEM = {
    "dat": "~/Downloads/Zr-Cu-Al.dat",
    "map": "Zr,Cu,Al",                 # type-index -> element in the raw .dat
    "order": ["Al", "Cu", "Zr"],       # engine Z-sort: Al=13 < Cu=29 < Zr=40
    "json": "data/zr_cu_al/zr_cu_al_dft.json",
    "label": "Zr–Cu–Al",
    "ntypes": 3,
}
NAME = "zr_cu_al"

# ── models (canonical run order) ──────────────────────────────────────────────────
# ntypes=3 → EAM has 6 pair φ, 3 density ρ, 3 embedding F (= 6*morse,3*exp_decay,3*sqrt).
ORDER = ["lmbtr", "acsf", "soap", "eam"]
DEFAULT_MODELS = ["lmbtr", "acsf", "soap"]   # the three linear (lsq) descriptors
MODELS = {
    "lmbtr": {
        "label": "LMBTR (linear)",
        # k2 (radial) + k3 (angular) grids.
        "init": ["--model", "lmbtr", "--ntypes", "3",
                 "--k2-n", "20", "--k3-n", "20"],
        "eweight": 0.1, "smooth_weight": 0.0, "maxiter": 150,
        # Linear head ⇒ closed-form least squares (see acsf note).
        "linear": True,
    },
    "acsf": {
        "label": "ACSF (linear)",
        "init": ["--model", "acsf", "--ntypes", "3",
                 "--g2-eta", "0.05", "0.2", "0.5", "1.0", "2.0", "4.0"],
        "eweight": 0.1, "smooth_weight": 0.0, "maxiter": 150,
        # Linear head ⇒ closed-form least squares: exact in one shot and
        # low-memory (engine streams the Jacobian, no whole-dataset cache).
        "linear": True,
    },
    "soap": {
        "label": "SOAP (linear)",
        "init": ["--model", "soap", "--ntypes", "3",
                 "--n-max", "4", "--l-max", "3", "--sigma", "0.5"],
        "eweight": 0.1, "smooth_weight": 0.0, "maxiter": 150,
        "linear": True,
    },
    "eam": {
        "label": "EAM (analytic Morse)",
        "init": ["--model", "eam", "--ntypes", "3",
                 "--functions", "6*morse,3*exp_decay,3*sqrt"],
        "eweight": 0.1, "smooth_weight": 1e-3, "maxiter": 300,
    },
}


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"command failed ({proc.returncode}):\n  "
                 + " ".join(map(str, cmd))
                 + f"\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return proc.stdout


# ── data conversion (tools/dat2json.py) ─────────────────────────────────────────

def convert(args):
    out = Path(SYSTEM["json"])
    if out.exists() and not args.convert:
        print(f"[{NAME}] reusing dataset {out}")
        return out
    dat = Path(SYSTEM["dat"]).expanduser()
    if not dat.exists():
        sys.exit(f"ERROR: raw dataset not found: {dat}")
    print(f"[{NAME}] converting {dat.name} (map {SYSTEM['map']}) → {out}")
    print(run([sys.executable, "tools/dat2json.py", "--out", str(out),
               f"{dat}:{SYSTEM['map']}"]).rstrip())
    return out


def subsample_config(src, n, dst):
    """Write a strided n-config subset of the dataset to dst (for smoke tests)."""
    data = json.load(open(src))
    if n is None or n >= len(data):
        return src
    step = max(1, len(data) // n)
    subset = data[::step][:n]
    json.dump(subset, open(dst, "w"))
    print(f"[subset] {len(subset)} of {len(data)} configs → {dst}")
    return dst


# ── fit / evaluate ──────────────────────────────────────────────────────────────

def fit_model(name, spec, config, args, work, fits):
    fit = fits / f"{NAME}_{name}.json"
    maxiter = args.maxiter if args.maxiter is not None else spec["maxiter"]

    if args.skip_fit and fit.exists():
        print(f"[{name}] reusing existing fit {fit}")
        return fit

    start = work / f"{NAME}_{name}_start.json"
    print(f"[{name}] init scaffold (ntypes 3, cutoff {args.cutoff} Å)")
    run([args.forcesmith, "init", *spec["init"],
         "--cutoff", str(args.cutoff), "--out", str(start)])

    # Same fit "config" (weights) is shared by both stages.
    weights = ["--eweight", str(spec["eweight"]),
               "--smooth-weight", str(spec["smooth_weight"]),
               "--stress-weight", str(args.stress_weight)]

    # Linear heads (SOAP/ACSF): one closed-form least-squares solve reaches the
    # exact global minimum — no DE warm-up, no iterative refinement — and the
    # engine streams the Jacobian so it stays low-memory on the full dataset.
    if spec.get("linear"):
        print(f"[{name}] closed-form least squares (-a lsq, "
              f"eweight {spec['eweight']}) ...")
        t0 = time.time()
        run([args.forcesmith, "-c", str(config), "-s", str(start),
             "-e", str(fit), *weights, "-a", "lsq"])
        print(f"[{name}] lsq done in {time.time() - t0:.1f}s → {fit}")
        return fit

    # Stage 1: global differential-evolution warm-up from the init scaffold.
    de_fit = work / f"{NAME}_{name}_de.json"
    print(f"[{name}] stage 1: DE (maxiter {args.de_steps}, "
          f"eweight {spec['eweight']}) ...")
    t0 = time.time()
    run([args.forcesmith, "-c", str(config), "-s", str(start),
         "-e", str(de_fit), "--maxiter", str(args.de_steps),
         *weights, "-a", "de"])
    print(f"[{name}] stage 1 DE done in {time.time() - t0:.1f}s → {de_fit}")

    # Stage 2: local refinement seeded from the DE result, same config as before.
    print(f"[{name}] stage 2: {args.algorithm} "
          f"(maxiter {maxiter}, eweight {spec['eweight']}) ...")
    t0 = time.time()
    run([args.forcesmith, "-c", str(config), "-s", str(de_fit),
         "-e", str(fit), "--maxiter", str(maxiter),
         *weights, "-a", args.algorithm])
    print(f"[{name}] stage 2 {args.algorithm} done in "
          f"{time.time() - t0:.1f}s → {fit}")
    return fit


def evaluate(name, fit, config, args, reports):
    report = reports / f"{NAME}_{name}_report.json"
    print(f"[{name}] evaluating ...")
    run([args.forcesmith, "-c", str(config), "-s", str(fit),
         "--evaluate", str(report)])
    data = json.load(open(report))
    f_ref, f_calc, e_ref, e_calc = [], [], [], []
    for cfg in data["configs"]:
        n = cfg["natoms"]
        e_ref.append(cfg["ref_energy"] / n)
        e_calc.append(cfg["calc_energy"] / n)
        for atom in cfg["atoms"]:
            f_ref.extend(atom["ref_force"])
            f_calc.extend(atom["calc_force"])
    return {
        "f_ref": np.asarray(f_ref), "f_calc": np.asarray(f_calc),
        "e_ref": np.asarray(e_ref), "e_calc": np.asarray(e_calc),
        "nconf": data["nconf"],
    }


def rmse(a, b):
    return math.sqrt(float(np.mean((a - b) ** 2)))


# ── parity panels ───────────────────────────────────────────────────────────────

def force_panel(ax, res, label):
    x, y = res["f_ref"], res["f_calc"]
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad
    hb = ax.hexbin(x, y, gridsize=70, bins="log", cmap="viridis",
                   extent=(lo, hi, lo, hi), mincnt=1)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("DFT force (eV/Å)")
    ax.set_ylabel("predicted force (eV/Å)")
    ax.set_title(f"{label}\nforce RMSE {rmse(x, y):.4g} eV/Å  ({x.size} comps)",
                 fontsize=9)
    return hb


def energy_panel(ax, res):
    ref = res["e_ref"]
    offset = float(np.mean(res["e_calc"] - ref))
    calc = res["e_calc"] - offset
    lo, hi = float(min(ref.min(), calc.min())), float(max(ref.max(), calc.max()))
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


def model_parity(name, res, plots):
    """A single fit's own parity plot: force (left) + energy (right). Written
    once per fit to its own file, so fits never overwrite each other."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
    hb = force_panel(axes[0], res, MODELS[name]["label"])
    fig.colorbar(hb, ax=axes[0], shrink=0.85, label="count (log)")
    energy_panel(axes[1], res)
    fig.suptitle(f"{SYSTEM['label']} — {MODELS[name]['label']} "
                 f"({res['nconf']} configs)", fontweight="bold", fontsize=12)
    out = plots / f"{NAME}_{name}_parity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def parity_grid(results, plots):
    """Combined {force,energy}×{models-done-so-far} grid, written once at the
    end to its own file. Tolerates partial `results`."""
    names = [m for m in ORDER if m in results]
    fig, axes = plt.subplots(2, len(names), figsize=(5.2 * len(names), 10),
                             squeeze=False, layout="constrained")
    last_hb = None
    for col, name in enumerate(names):
        last_hb = force_panel(axes[0][col], results[name], MODELS[name]["label"])
        energy_panel(axes[1][col], results[name])
    fig.colorbar(last_hb, ax=axes[0].tolist(), shrink=0.85, label="count (log)")
    nconf = results[names[0]]["nconf"]
    fig.suptitle(f"{SYSTEM['label']} — forcesmith CLI fits ({nconf} configs)",
                 fontweight="bold", fontsize=13)
    out = plots / f"{NAME}_all_parity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ── EAM function plot ───────────────────────────────────────────────────────────
# The analytic EAM start is fit and serialized back as a sampled (tabulated) table,
# so every fitted section carries `knots` over [rmin, rmax] regardless of the
# analytic start. Plot those directly. ntypes=3 → 6 pair / 3 density / 3 embedding.

def _curve(pot):
    y = np.asarray(pot["knots"], dtype=float)
    x = np.linspace(pot["rmin"], pot["rmax"], y.size)
    return x, y


def eam_functions_plot(fit_path, plots):
    fit = json.load(open(fit_path))
    order = SYSTEM["order"]
    nt = len(order)
    pair_labels = [f"{a}-{b}" for i, a in enumerate(order) for b in order[i:]]
    npair = len(pair_labels)                      # 6 for ntypes=3

    # 2 rows: row 0 = pair φ; row 1 = density ρ then embedding F (3 + 3 = 6 = npair).
    fig, axes = plt.subplots(2, npair, figsize=(4.0 * npair, 8.4), squeeze=False)
    for i, lbl in enumerate(pair_labels):         # 6 pair φ(r)
        ax = axes[0][i]
        x, y = _curve(fit["pair"]["potentials"][i])
        ax.plot(x, y, "-", color="C3", lw=2)
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        ax.set_title(f"pair φ  {lbl}", fontsize=10)
        ax.set_xlabel("r (Å)"); ax.set_ylabel("φ (eV)")
    for i, el in enumerate(order):                # 3 density ρ(r)
        ax = axes[1][i]
        x, y = _curve(fit["density"]["potentials"][i])
        ax.plot(x, y, "-", color="C2", lw=2)
        ax.set_title(f"density ρ  {el}", fontsize=10)
        ax.set_xlabel("r (Å)"); ax.set_ylabel("ρ (arb.)")
    for i, el in enumerate(order):                # 3 embedding F(ρ)
        ax = axes[1][nt + i]
        x, y = _curve(fit["embedding"]["potentials"][i])
        ax.plot(x, y, "-", color="C4", lw=2)
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        ax.set_title(f"embedding F  {el}", fontsize=10)
        ax.set_xlabel("ρ (arb.)"); ax.set_ylabel("F (eV)")
    fig.suptitle(f"{SYSTEM['label']} — fitted analytic EAM functions "
                 f"(Morse pair, index order {order})",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = plots / f"{NAME}_eam_functions.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ── driver ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forcesmith", default="build/release/forcesmith")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help="comma-separated subset of: " + ", ".join(ORDER)
                        + f"  (default: {','.join(DEFAULT_MODELS)})")
    p.add_argument("--out-dir", default="tests/zr_cu_al")
    p.add_argument("--cutoff", type=float, default=6.5)
    p.add_argument("--maxiter", type=int, default=None,
                   help="override per-model optimizer iterations")
    p.add_argument("--max-configs", type=int, default=None,
                   help="subsample to N configs (smoke test); default: full dataset")
    p.add_argument("--stress-weight", type=float, default=0.0)
    p.add_argument("--de-steps", type=int, default=500,
                   help="stage-1 differential-evolution iterations (EAM only)")
    p.add_argument("--algorithm", default="ipopt",
                   help="EAM stage-2 local refinement: lm | powell | ls | ipopt")
    p.add_argument("--convert", action="store_true",
                   help="force re-conversion of the raw .dat dataset")
    p.add_argument("--skip-fit", action="store_true",
                   help="reuse existing fits, just re-evaluate and replot")
    args = p.parse_args(argv)

    args.forcesmith = str(Path(args.forcesmith))
    if not Path(args.forcesmith).exists():
        sys.exit(f"ERROR: forcesmith binary not found: {args.forcesmith}")

    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    bad = [m for m in wanted if m not in ORDER]
    if bad:
        sys.exit(f"unknown model(s) {bad}; choose from {ORDER}")
    models = [m for m in ORDER if m in wanted]  # canonical acsf→soap→eam order

    root = Path(args.out_dir)
    work, fits, reports, plots = (root / "work", root / "fits",
                                  root / "reports", root / "plots")
    for d in (work, fits, reports, plots):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n===== {SYSTEM['label']} ({NAME}) — models: {', '.join(models)} =====")
    config = convert(args)
    config = subsample_config(config, args.max_configs,
                              work / f"{NAME}_subset.json")

    results, fns = {}, None
    for name in models:
        fit = fit_model(name, MODELS[name], config, args, work, fits)
        results[name] = evaluate(name, fit, config, args, reports)
        # Each fit gets its OWN parity plot file (no overwriting); the EAM
        # functions plot is emitted as soon as the EAM fit lands.
        par = model_parity(name, results[name], plots)
        print(f"[{name}] wrote {par}")
        if name == "eam":
            fns = eam_functions_plot(fits / f"{NAME}_eam.json", plots)
            print(f"[{name}] wrote {fns}")
    # One combined all-models grid (its own file).
    grid = parity_grid(results, plots)
    print(f"[{NAME}] wrote {grid}")

    print("\n=== RMSE summary ===")
    print(f"  {'model':22s} {'force (eV/Å)':>14s} {'energy (eV/atom)':>18s}")
    for name in models:
        r = results[name]
        e_off = float(np.mean(r["e_calc"] - r["e_ref"]))
        f_r = rmse(r["f_ref"], r["f_calc"])
        e_r = rmse(r["e_ref"], r["e_calc"] - e_off)
        print(f"  {MODELS[name]['label']:22s} {f_r:14.4f} {e_r:18.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
