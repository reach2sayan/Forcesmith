#!/usr/bin/env python3
"""Fit three binary alloys — Zr–Ni, Cu–Ni, Cu–Zr — each with three models, using
only the ``forcesmith`` CLI on H.W. Sheng's binary DFT datasets, then plot force
and energy parity plus the fitted (analytic) EAM functions.

Per system the models are run **in this order: ACSF → SOAP → EAM**. The EAM is the
*analytic / non-tabulated* form scaffolded straight from ``forcesmith init`` (Morse
pair + exp_decay density + sqrt embedding) — no hand-built ``.eam`` seed.

Inputs (raw potfit ``.dat`` files in ~/Downloads) are converted once via
``tools/dat2json.py``. The type-index → element map for each file was recovered
from the ``#N`` VASP source paths and compound stoichiometries:

    Zr-Ni.dat : Ni,Zr        cu-ni.dat : Cu,Ni        cu-zr.dat : Zr,Cu

The engine sorts species by ascending atomic number (Ni=28, Cu=29, Zr=40), which
fixes the EAM section ordering used by the function plots.

Each model pipeline is:

  1. ``forcesmith init  --model <m> --ntypes 2 ...  --out work/<sys>_<m>_start.json``
  2. ``forcesmith -c <data> -s start -e fit  --maxiter N --eweight w``
  3. ``forcesmith -c <data> -s fit  --evaluate report.json``

Outputs (under tests/zr_cu_ni/plots/), one set per system:
  * ``<system>_<model>_parity.png``  force+energy parity for that single fit, written
                                     at the end of each fit (one file per model)
  * ``<system>_all_parity.png``      combined 2×3 {force, energy} × {EAM, ACSF, SOAP}
  * ``<system>_eam_functions.png``   3 pair φ(r), 2 density ρ(r), 2 embedding F(ρ)
                                     of the fitted analytic EAM (no reference overlay)

All fits are forces-first (small/zero energy weight): the DFT per-atom energy zero
is arbitrary, so energy panels have their best constant offset removed.

Usage:
    python3 tests/zr_cu_ni/fit_binaries.py                       # full datasets
    python3 tests/zr_cu_ni/fit_binaries.py --max-configs 30 --maxiter 20   # smoke
    python3 tests/zr_cu_ni/fit_binaries.py --systems cu_zr       # one system
    python3 tests/zr_cu_ni/fit_binaries.py --skip-fit            # replot existing fits
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


# ── systems ────────────────────────────────────────────────────────────────────
# `order` is the engine's Z-sorted index space (ascending atomic number); it sets
# the EAM pair-slot and density/embedding ordering for the function plots.
SYSTEMS = {
    "zr_ni": {"dat": "~/Downloads/Zr-Ni.dat", "map": "Ni,Zr", "order": ["Ni", "Zr"],
              "json": "data/binary_alloys/zr_ni_dft.json", "label": "Zr–Ni"},
    "cu_ni": {"dat": "~/Downloads/cu-ni.dat", "map": "Cu,Ni", "order": ["Ni", "Cu"],
              "json": "data/binary_alloys/cu_ni_dft.json", "label": "Cu–Ni"},
    "cu_zr": {"dat": "~/Downloads/cu-zr.dat", "map": "Zr,Cu", "order": ["Cu", "Zr"],
              "json": "data/binary_alloys/cu_zr_dft.json", "label": "Cu–Zr"},
    "al_zr": {"dat": "~/Downloads/al-zr.dat", "map": "Zr,Al", "order": ["Al", "Zr"],
              "json": "data/binary_alloys/al_zr_dft.json", "label": "Al–Zr"},
}

# ── models (run in this order) ──────────────────────────────────────────────────
ORDER = ["lmbtr", "acsf", "soap", "eam"]
MODELS = {
    "lmbtr": {
        "label": "LMBTR (linear)",
        # k2 (radial) + k3 (angular) grids.
        "init": ["--model", "lmbtr", "--ntypes", "2",
                 "--k2-n", "20", "--k3-n", "20"],
        "eweight": 0.1, "smooth_weight": 0.0, "maxiter": 150,
        # Linear head ⇒ closed-form least squares (see acsf note).
        "linear": True,
    },
    "acsf": {
        "label": "ACSF (linear)",
        "init": ["--model", "acsf", "--ntypes", "2",
                 "--g2-eta", "0.05", "0.2", "0.5", "1.0", "2.0", "4.0"],
        "eweight": 0.1, "smooth_weight": 0.0, "maxiter": 150,
        # Linear head ⇒ closed-form least squares: exact in one shot and
        # low-memory (engine streams the Jacobian, no whole-dataset cache).
        "linear": True,
    },
    "eam": {
        "label": "EAM (analytic Morse)",
        # ntypes=2 → 7 analytic functions: pair×3, density×2, embedding×2.
        "init": ["--model", "eam", "--ntypes", "2",
                 "--functions", "3*morse,2*exp_decay,2*sqrt"],
        "eweight": 0.1, "smooth_weight": 1e-3, "maxiter": 300,
    },
    "soap": {
        "label": "SOAP (linear)",
        "init": ["--model", "soap", "--ntypes", "2",
                 "--n-max", "4", "--l-max", "3", "--sigma", "0.5"],
        "eweight": 0.1, "smooth_weight": 0.0, "maxiter": 150,
        # Linear head ⇒ closed-form least squares (see acsf note).
        "linear": True,
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

def convert(sys_name, spec, args):
    out = Path(spec["json"])
    if out.exists() and not args.convert:
        print(f"[{sys_name}] reusing dataset {out}")
        return out
    dat = Path(spec["dat"]).expanduser()
    if not dat.exists():
        sys.exit(f"ERROR: raw dataset not found: {dat}")
    print(f"[{sys_name}] converting {dat.name} (map {spec['map']}) → {out}")
    print(run([sys.executable, "tools/dat2json.py", "--out", str(out),
               f"{dat}:{spec['map']}"]).rstrip())
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

def fit_model(sys_name, name, spec, config, args, work, fits):
    fit = fits / f"{sys_name}_{name}.json"
    maxiter = args.maxiter if args.maxiter is not None else spec["maxiter"]

    if args.skip_fit and fit.exists():
        print(f"[{sys_name}/{name}] reusing existing fit {fit}")
        return fit

    start = work / f"{sys_name}_{name}_start.json"
    print(f"[{sys_name}/{name}] init scaffold (ntypes 2, cutoff {args.cutoff} Å)")
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
        print(f"[{sys_name}/{name}] closed-form least squares (-a lsq, "
              f"eweight {spec['eweight']}) ...")
        t0 = time.time()
        run([args.forcesmith, "-c", str(config), "-s", str(start),
             "-e", str(fit), *weights, "-a", "lsq"])
        print(f"[{sys_name}/{name}] lsq done in {time.time() - t0:.1f}s → {fit}")
        return fit

    # `no_de` models (e.g. LMBTR): skip the DE warm-up and refine straight from
    # the init scaffold with the stage-2 local algorithm (ipopt by default).
    if spec.get("no_de"):
        print(f"[{sys_name}/{name}] {args.algorithm} from scaffold "
              f"(maxiter {maxiter}, eweight {spec['eweight']}) ...")
        t0 = time.time()
        run([args.forcesmith, "-c", str(config), "-s", str(start),
             "-e", str(fit), "--maxiter", str(maxiter),
             *weights, "-a", args.algorithm])
        print(f"[{sys_name}/{name}] {args.algorithm} done in "
              f"{time.time() - t0:.1f}s → {fit}")
        return fit

    # Stage 1: global differential-evolution warm-up from the init scaffold.
    de_fit = work / f"{sys_name}_{name}_de.json"
    print(f"[{sys_name}/{name}] stage 1: DE (maxiter {args.de_steps}, "
          f"eweight {spec['eweight']}) ...")
    t0 = time.time()
    run([args.forcesmith, "-c", str(config), "-s", str(start),
         "-e", str(de_fit), "--maxiter", str(args.de_steps),
         *weights, "-a", "de"])
    print(f"[{sys_name}/{name}] stage 1 DE done in {time.time() - t0:.1f}s → {de_fit}")

    # Stage 2: local refinement seeded from the DE result, same config as before.
    print(f"[{sys_name}/{name}] stage 2: {args.algorithm} "
          f"(maxiter {maxiter}, eweight {spec['eweight']}) ...")
    t0 = time.time()
    run([args.forcesmith, "-c", str(config), "-s", str(de_fit),
         "-e", str(fit), "--maxiter", str(maxiter),
         *weights, "-a", args.algorithm])
    print(f"[{sys_name}/{name}] stage 2 {args.algorithm} done in "
          f"{time.time() - t0:.1f}s → {fit}")
    return fit


def evaluate(sys_name, name, fit, config, args, reports):
    report = reports / f"{sys_name}_{name}_report.json"
    print(f"[{sys_name}/{name}] evaluating ...")
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


def model_parity(sys_name, spec, name, res, plots):
    """A single fit's own parity plot: force (left) + energy (right). Written
    once per fit to its own file, so fits never overwrite each other."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
    hb = force_panel(axes[0], res, MODELS[name]["label"])
    fig.colorbar(hb, ax=axes[0], shrink=0.85, label="count (log)")
    energy_panel(axes[1], res)
    fig.suptitle(f"{spec['label']} — {MODELS[name]['label']} "
                 f"({res['nconf']} configs)", fontweight="bold", fontsize=12)
    out = plots / f"{sys_name}_{name}_parity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def parity_grid(sys_name, spec, results, plots):
    """Combined {force,energy}×{models-done-so-far} grid, written once at the
    end of a system to its own file. Tolerates partial `results`."""
    names = [m for m in ORDER if m in results]
    fig, axes = plt.subplots(2, len(names), figsize=(5.2 * len(names), 10),
                             squeeze=False, layout="constrained")
    last_hb = None
    for col, name in enumerate(names):
        last_hb = force_panel(axes[0][col], results[name], MODELS[name]["label"])
        energy_panel(axes[1][col], results[name])
    fig.colorbar(last_hb, ax=axes[0].tolist(), shrink=0.85, label="count (log)")
    nconf = results[names[0]]["nconf"]
    fig.suptitle(f"{spec['label']} — forcesmith CLI fits ({nconf} configs)",
                 fontweight="bold", fontsize=13)
    out = plots / f"{sys_name}_all_parity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ── EAM function plot ───────────────────────────────────────────────────────────
# The analytic EAM start is fit and serialized back as a sampled (tabulated) table,
# so every fitted section carries `knots` over [rmin, rmax] regardless of the
# analytic start. Plot those directly.

def _curve(pot):
    y = np.asarray(pot["knots"], dtype=float)
    x = np.linspace(pot["rmin"], pot["rmax"], y.size)
    return x, y


def eam_functions_plot(sys_name, spec, fit_path, plots):
    fit = json.load(open(fit_path))
    order = spec["order"]
    pair_labels = [f"{a}-{b}" for i, a in enumerate(order) for b in order[i:]]

    fig, axes = plt.subplots(1, 7, figsize=(28, 4.2))
    for i, lbl in enumerate(pair_labels):              # 3 pair φ(r)
        ax = axes[i]
        x, y = _curve(fit["pair"]["potentials"][i])
        ax.plot(x, y, "-", color="C3", lw=2)
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        ax.set_title(f"pair φ  {lbl}", fontsize=10)
        ax.set_xlabel("r (Å)"); ax.set_ylabel("φ (eV)")
    for i, el in enumerate(order):                     # 2 density ρ(r)
        ax = axes[3 + i]
        x, y = _curve(fit["density"]["potentials"][i])
        ax.plot(x, y, "-", color="C2", lw=2)
        ax.set_title(f"density ρ  {el}", fontsize=10)
        ax.set_xlabel("r (Å)"); ax.set_ylabel("ρ (arb.)")
    for i, el in enumerate(order):                     # 2 embedding F(ρ)
        ax = axes[5 + i]
        x, y = _curve(fit["embedding"]["potentials"][i])
        ax.plot(x, y, "-", color="C4", lw=2)
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        ax.set_title(f"embedding F  {el}", fontsize=10)
        ax.set_xlabel("ρ (arb.)"); ax.set_ylabel("F (eV)")
    fig.suptitle(f"{spec['label']} — fitted analytic EAM functions "
                 f"(Morse pair, index order {order})",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = plots / f"{sys_name}_eam_functions.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ── driver ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forcesmith", default="build/release/forcesmith")
    p.add_argument("--systems", default=",".join(SYSTEMS),
                   help="comma-separated subset of: " + ", ".join(SYSTEMS))
    p.add_argument("--models", default=",".join(ORDER),
                   help="comma-separated subset of: " + ", ".join(ORDER))
    p.add_argument("--out-dir", default="tests/zr_cu_ni")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--maxiter", type=int, default=None,
                   help="override per-model optimizer iterations")
    p.add_argument("--max-configs", type=int, default=None,
                   help="subsample to N configs (smoke test); default: full dataset")
    p.add_argument("--stress-weight", type=float, default=0.0)
    p.add_argument("--de-steps", type=int, default=500,
                   help="stage-1 differential-evolution iterations")
    p.add_argument("--algorithm", default="ipopt",
                   help="stage-2 local refinement: lm | powell | ls | ipopt")
    p.add_argument("--convert", action="store_true",
                   help="force re-conversion of the raw .dat datasets")
    p.add_argument("--skip-fit", action="store_true",
                   help="reuse existing fits, just re-evaluate and replot")
    args = p.parse_args(argv)

    args.forcesmith = str(Path(args.forcesmith))
    if not Path(args.forcesmith).exists():
        sys.exit(f"ERROR: forcesmith binary not found: {args.forcesmith}")

    chosen = [s.strip() for s in args.systems.split(",") if s.strip()]
    bad = [s for s in chosen if s not in SYSTEMS]
    if bad:
        sys.exit(f"unknown system(s) {bad}; choose from {list(SYSTEMS)}")

    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    bad_m = [m for m in wanted if m not in ORDER]
    if bad_m:
        sys.exit(f"unknown model(s) {bad_m}; choose from {ORDER}")
    models = [m for m in ORDER if m in wanted]  # keep canonical EAM→ACSF→SOAP order

    root = Path(args.out_dir)
    work, fits, reports, plots = (root / "work", root / "fits",
                                  root / "reports", root / "plots")
    for d in (work, fits, reports, plots):
        d.mkdir(parents=True, exist_ok=True)

    summary = {}
    for sys_name in chosen:
        spec = SYSTEMS[sys_name]
        print(f"\n===== {spec['label']} ({sys_name}) =====")
        config = convert(sys_name, spec, args)
        config = subsample_config(config, args.max_configs,
                                  work / f"{sys_name}_subset.json")

        results, fns = {}, None
        for name in models:                 # EAM → ACSF → SOAP (filtered)
            fit = fit_model(sys_name, name, MODELS[name], config, args, work, fits)
            results[name] = evaluate(sys_name, name, fit, config, args, reports)
            # Each fit gets its OWN parity plot file (no overwriting); the EAM
            # functions plot is emitted as soon as the EAM fit lands.
            par = model_parity(sys_name, spec, name, results[name], plots)
            print(f"[{sys_name}/{name}] wrote {par}")
            if name == "eam":
                fns = eam_functions_plot(
                    sys_name, spec, fits / f"{sys_name}_eam.json", plots)
                print(f"[{sys_name}/{name}] wrote {fns}")
        # One combined all-models grid per system (its own file).
        grid = parity_grid(sys_name, spec, results, plots)
        print(f"[{sys_name}] wrote {grid}")
        summary[sys_name] = {"results": results, "par": grid, "fns": fns}

    print("\n=== force RMSE (eV/Å) ===")
    print(f"  {'system':10s} " + "  ".join(f"{MODELS[m]['label']:22s}" for m in models))
    for sys_name in chosen:
        r = summary[sys_name]["results"]
        cells = "  ".join(f"{rmse(r[m]['f_ref'], r[m]['f_calc']):<22.4f}" for m in models)
        print(f"  {sys_name:10s} {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
