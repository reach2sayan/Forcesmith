#!/usr/bin/env python3
"""
Per-element EAM fitting driver for the converted UNEP DFT dataset.

For every requested element this runs a **two-stage, forces-first** fit
against ``data/unep/<el>_dft_unep.json`` (real DFT forces/energies/stresses
from UNEP-v1, Zenodo 11533864) using the existing ``build/release/forcesmith``
binary, and reports per-atom force RMSE before and after the fit.

Pipeline (per element):

  1. Derive cutoffs from the data: nearest-neighbour distance ``dmin`` (via a
     minimum-image pairwise scan, exact for the closest pair) sets
     ``re = dmin``, ``rmin = max(1.0, 0.88*dmin)``, ``rmax = min(6.5, 2.4*dmin)``.
  2. Stage-1 analytic start-pot (morse pair + exp_decay density + sqrt
     embedding, B>0).
  3. Stage-1 fit  -> dense 500-knot tabulated EAM (writer's kDefaultKnots).
  4. Stage-2 start-pot: down-sample each section of the stage-1 result to
     ``--knots`` free knots ("analytic seeds tabulated").
  5. Stage-2 fit with ``--smooth-weight`` regularizing the free splines.
  6. Force RMSE at three checkpoints (start / stage-1 / final) via
     ``forcesmith --evaluate``.

Caveats (see README.md):
  * Forces-first by design. UNEP energies carry an arbitrary per-atom zero an
    EAM cannot match, so ``--eweight`` is small and absolute energy is not fit.
  * Stress is off by default (extxyz virial->stress sign/units unverified).
  * A generic analytic EAM is a modest model: the goal is to prove the pipeline
    fits every element and *reduces* force RMSE, not to reach DFT accuracy.

Usage:
    python3 tests/unep/fit_eam.py                       # all 16 elements
    python3 tests/unep/fit_eam.py --elements Cu Al
    python3 tests/unep/fit_eam.py --elements Cu --max-configs 40 --maxiter 60

Requirements: numpy (Python 3). Reuses build/release/forcesmith; never runs cmake.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ALL_ELEMENTS = [
    "Ag", "Al", "Au", "Cr", "Cu", "Mg", "Mo", "Ni",
    "Pb", "Pd", "Pt", "Ta", "Ti", "V", "W", "Zr",
]


# ── data loading & geometry ───────────────────────────────────────────────────

def load_configs(path):
    """Load a UNEP config file (top-level JSON array of configurations)."""
    with open(path) as fh:
        configs = json.load(fh)
    if not isinstance(configs, list):
        raise ValueError(f"{path}: expected a top-level array of configurations")
    return configs


def subsample(configs, max_configs, stride):
    """Return a deterministic subset of configs for fast smoke runs."""
    out = configs[::stride] if stride > 1 else list(configs)
    if max_configs and len(out) > max_configs:
        out = out[:max_configs]
    return out


def cell_matrix(cfg):
    """Lattice as rows [a; b; c] from the X/Y/Z box vectors."""
    return np.array([cfg["X"], cfg["Y"], cfg["Z"]], dtype=float)


def nearest_neighbour_distance(configs):
    """Smallest interatomic distance across all configs (minimum-image).

    Min-image is exact for the *closest* pair, which is all we need to size
    the cutoffs.  Returns inf if no config has >=2 atoms.
    """
    dmin = math.inf
    for cfg in configs:
        pos = np.array([a["position"] for a in cfg["atoms"]], dtype=float)
        n = len(pos)
        if n < 2:
            continue
        cell = cell_matrix(cfg)                            # rows = lattice vectors a,b,c
        try:
            inv = np.linalg.inv(cell)                      # v = frac @ cell  =>  frac = v @ inv
        except np.linalg.LinAlgError:
            inv = None
        # pairwise raw differences
        diff = pos[:, None, :] - pos[None, :, :]          # (n, n, 3)
        iu = np.triu_indices(n, k=1)
        d = diff[iu]                                       # (npair, 3)
        if inv is not None:
            frac = d @ inv                                 # cartesian -> fractional
            frac -= np.round(frac)                         # wrap to [-0.5, 0.5)
            d = frac @ cell                                # back to cartesian
        dist = np.sqrt((d * d).sum(axis=1))
        dist = dist[dist > 1e-6]
        if dist.size:
            dmin = min(dmin, float(dist.min()))
    return dmin


# ── potential construction ────────────────────────────────────────────────────

def analytic_start(rmin, rmax):
    """Stage-1 analytic EAM start potential (all params free by default)."""
    return {
        "model": "eam",
        "ntypes": 1,
        "pair": {
            "format": "analytic",
            "potentials": [
                {"type": "morse", "rmin": rmin, "rmax": rmax,
                 "De": 0.5, "a": 2.0, "re": rmin},
            ],
        },
        "density": {
            "format": "analytic",
            "potentials": [
                {"type": "exp_decay", "rmin": rmin, "rmax": rmax,
                 "A": 1.0, "B": 1.0},
            ],
        },
        "embedding": {
            "format": "analytic",
            # sqrt: V = A*sqrt(rho/B), needs B>0; rho-domain span.
            "potentials": [
                {"type": "sqrt", "rmin": 1e-4, "rmax": 10.0,
                 "A": -1.0, "B": 1.0},
            ],
        },
    }


def _resample_section(sec, knots):
    """Down-sample one tabulated section's knot array to ``knots`` points."""
    y = np.array(sec["knots"], dtype=float)
    if y.size == knots:
        new = y
    else:
        xs_old = np.linspace(0.0, 1.0, y.size)
        xs_new = np.linspace(0.0, 1.0, knots)
        new = np.interp(xs_new, xs_old, y)
    return {"rmin": sec["rmin"], "rmax": sec["rmax"],
            "knots": [float(v) for v in new]}


def tabulated_start_from_dense(dense, knots):
    """Stage-2 start: down-sample each section of a dense fit to free knots."""
    out = {"model": dense.get("model", "eam"), "ntypes": dense.get("ntypes", 1)}
    for section in ("pair", "density", "embedding"):
        pots = dense[section]["potentials"]
        out[section] = {
            "format": "tabulated",
            "potentials": [_resample_section(p, knots) for p in pots],
        }
    return out


# ── forcesmith invocation ─────────────────────────────────────────────────────────

def run_forcesmith(forcesmith, config, start, *, endpot=None, evaluate=None,
               eweight=0.1, stress_weight=0.0, smooth_weight=0.0,
               maxiter=300, algorithm="lm"):
    """Invoke the forcesmith binary; returns (ok, stdout+stderr)."""
    cmd = [str(forcesmith), "-c", str(config), "-s", str(start),
           "--eweight", str(eweight), "--stress-weight", str(stress_weight),
           "--maxiter", str(maxiter), "-a", algorithm]
    if evaluate is not None:
        cmd += ["--evaluate", str(evaluate)]
    else:
        cmd += ["-e", str(endpot), "--smooth-weight", str(smooth_weight)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def eval_rmse(forcesmith, config, start):
    """Run --evaluate and return (force_rmse, stress_rmse, log).

    Force RMSE is per-atom force component (eV/Å); stress RMSE is over the six
    stress components (eV/Å³, now matching the reference units). Either RMSE is
    None if unavailable; both None on a forcesmith failure.
    """
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        report = tf.name
    ok, log = run_forcesmith(forcesmith, config, start, evaluate=report)
    if not ok:
        return None, None, log
    with open(report) as fh:
        data = json.load(fh)
    Path(report).unlink(missing_ok=True)

    fsq = 0.0
    fn = 0
    ssq = 0.0
    sn = 0
    for cfg in data["configs"]:
        for atom in cfg["atoms"]:
            for c, r in zip(atom["calc_force"], atom["ref_force"]):
                fsq += (c - r) ** 2
            fn += 3
        if "ref_stress" in cfg and "calc_stress" in cfg:
            for c, r in zip(cfg["calc_stress"], cfg["ref_stress"]):
                ssq += (c - r) ** 2
            sn += 6
    frmse = math.sqrt(fsq / fn) if fn else None
    srmse = math.sqrt(ssq / sn) if sn else None
    return frmse, srmse, log


# ── per-element driver ────────────────────────────────────────────────────────

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def fit_element(element, args):
    """Run the full two-stage fit for one element. Returns a result dict."""
    el = element.lower()
    data_path = Path(args.data_dir) / f"{el}_dft_unep.json"
    out_dir = Path(args.out_dir)
    work = out_dir / "work" / el
    fits = out_dir / "fits"

    res = {"element": element, "status": "ok"}
    try:
        configs = load_configs(data_path)
    except (OSError, ValueError) as exc:
        return {**res, "status": "error", "error": f"load: {exc}"}

    configs = subsample(configs, args.max_configs, args.stride)
    natoms = sum(len(c["atoms"]) for c in configs)
    res["nconf"] = len(configs)
    res["natoms"] = natoms

    # If we subsampled, fit against a written-out subset; else use the file.
    if len(configs) != len(load_configs(data_path)):
        cfg_path = work / "configs_subset.json"
        write_json(cfg_path, configs)
    else:
        cfg_path = data_path

    dmin = nearest_neighbour_distance(configs)
    if not math.isfinite(dmin):
        return {**res, "status": "error", "error": "no neighbour pairs found"}
    rmin = max(1.0, 0.88 * dmin)
    rmax = min(6.5, 2.4 * dmin)
    res.update({"dmin": dmin, "rmin": rmin, "rmax": rmax})

    forcesmith = args.forcesmith
    fits.mkdir(parents=True, exist_ok=True)

    # Baseline: the un-fitted analytic start potential.
    start_analytic = work / "start_analytic.json"
    write_json(start_analytic, analytic_start(rmin, rmax))
    frmse, srmse, _ = eval_rmse(forcesmith, cfg_path, start_analytic)
    res["rmse_start"] = frmse
    res["srmse_start"] = srmse

    # Stage 1 -> the *analytic fit* (analytic start refined; written dense by the
    # native writer, but it is the fitted analytic function sampled finely).
    if args.stage >= 1:
        analytic_fit = fits / f"{el}_eam_analytic.json"
        ok, log = run_forcesmith(
            forcesmith, cfg_path, start_analytic, endpot=analytic_fit,
            eweight=args.eweight, stress_weight=args.stress_weight,
            smooth_weight=0.0, maxiter=args.maxiter, algorithm=args.algorithm)
        if not ok:
            return {**res, "status": "error",
                    "error": "stage-1 (analytic) fit failed", "log": log[-1500:]}
        frmse, srmse, _ = eval_rmse(forcesmith, cfg_path, analytic_fit)
        res["rmse_analytic"] = frmse
        res["srmse_analytic"] = srmse
        res["analytic_fit"] = str(analytic_fit)

    # Stage 2 -> the *tabular fit* (analytic fit down-sampled to free knots, then
    # refined as regularized splines).
    if args.stage >= 2:
        with open(analytic_fit) as fh:
            dense = json.load(fh)
        start_tab = work / "start_tab.json"
        write_json(start_tab, tabulated_start_from_dense(dense, args.knots))

        tabular_fit = fits / f"{el}_eam_fit.json"
        ok, log = run_forcesmith(
            forcesmith, cfg_path, start_tab, endpot=tabular_fit,
            eweight=args.eweight, stress_weight=args.stress_weight,
            smooth_weight=args.smooth_weight, maxiter=args.maxiter,
            algorithm=args.algorithm)
        if not ok:
            return {**res, "status": "error",
                    "error": "stage-2 (tabular) fit failed", "log": log[-1500:]}
        frmse, srmse, _ = eval_rmse(forcesmith, cfg_path, tabular_fit)
        res["rmse_tabular"] = frmse
        res["srmse_tabular"] = srmse
        res["tabular_fit"] = str(tabular_fit)

    return res


# ── reporting ─────────────────────────────────────────────────────────────────

def fmt(x):
    return "—" if x is None else f"{x:.5f}"


def print_table(results):
    # Force RMSE (eV/Å) then stress RMSE (eV/Å³), each at start/analytic/tabular.
    cols = ["element", "nconf", "natoms", "dmin",
            "F:start", "F:analytic", "F:tabular",
            "S:start", "S:analytic", "S:tabular", "status"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in results:
        row = [
            r.get("element", "?"),
            str(r.get("nconf", "—")),
            str(r.get("natoms", "—")),
            fmt(r.get("dmin")),
            fmt(r.get("rmse_start")), fmt(r.get("rmse_analytic")),
            fmt(r.get("rmse_tabular")),
            fmt(r.get("srmse_start")), fmt(r.get("srmse_analytic")),
            fmt(r.get("srmse_tabular")),
            r.get("status", "?"),
        ]
        print("| " + " | ".join(row) + " |")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--elements", nargs="+", default=ALL_ELEMENTS,
                   help="elements to fit (default: all 16)")
    p.add_argument("--data-dir", default="data/unep")
    p.add_argument("--out-dir", default="tests/unep")
    p.add_argument("--forcesmith", default="build/release/forcesmith")
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--eweight", type=float, default=0.1)
    p.add_argument("--stress-weight", type=float, default=0.0)
    p.add_argument("--smooth-weight", type=float, default=1.0)
    p.add_argument("--knots", type=int, default=15)
    p.add_argument("--algorithm", default="lm",
                   choices=["lm", "powell", "de", "ls"])
    p.add_argument("--max-configs", type=int, default=0,
                   help="cap configs per element (0 = all)")
    p.add_argument("--stride", type=int, default=1,
                   help="use every Kth config")
    p.add_argument("--jobs", type=int, default=1,
                   help="fit this many elements in parallel")
    p.add_argument("--stage", type=int, default=2, choices=[1, 2],
                   help="highest stage to run")
    args = p.parse_args(argv)

    forcesmith = Path(args.forcesmith)
    if not forcesmith.exists():
        sys.exit(f"ERROR: forcesmith binary not found at {forcesmith}. Build it first "
                 f"(the repo convention is that you build; this script never "
                 f"runs cmake).")
    args.forcesmith = forcesmith

    elements = [e.capitalize() for e in args.elements]

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(lambda e: fit_element(e, args), elements))
    else:
        results = []
        for e in elements:
            print(f"... fitting {e}", file=sys.stderr, flush=True)
            results.append(fit_element(e, args))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print_table(results)
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{n_ok}/{len(results)} elements fit successfully.")
    print(f"Results: {out_dir / 'results.json'}")
    failed = [r for r in results if r["status"] != "ok"]
    for r in failed:
        print(f"  ! {r['element']}: {r.get('error', 'unknown error')}",
              file=sys.stderr)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
