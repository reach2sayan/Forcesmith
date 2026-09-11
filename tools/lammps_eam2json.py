#!/usr/bin/env python3
"""Convert single-element LAMMPS setfl ``.eam`` tables to the port's tabulated EAM JSON.

These are H.W. Sheng's pure-element EAM potentials (Zr/Cu/Al). The setfl layout is::

    line 1-3   comments
    line 4     "<nelements>  <El>"            (e.g. "1  Cu")
    line 5     "Nrho drho Nr dr cutoff"
    line 6     "<Z> <mass> <a0> <type>"
    then       F(rho)[Nrho], rho_at(r)[Nr], r*phi(r)[Nr]   (whitespace-separated)

The parsing here is lifted from ``tools/generate_eam_configs.py`` (parse_setfl /
to_tabulated_json) so that script stays untouched. Each input becomes a single
``ntypes:1`` tabulated EAM JSON::

    pair      φ(r)  knots over [dr, cutoff]   (setfl stores r*phi; we divide by r)
    density   ρ(r)  knots over [0,  cutoff]
    embedding F(ρ)  knots over [0,  rho_max]

Output file names are ``<element>_eam_ref.json`` (lower-case) in --out-dir.

Usage:
    python3 tools/lammps_eam2json.py \\
        ~/Downloads/Zr_lammps.eam ~/Downloads/Cu.lammps.eam ~/Downloads/Al.lammps.eam \\
        --out-dir data/zr_cu_al
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.interpolate import CubicSpline
except ImportError:
    sys.exit("ERROR: scipy is required.  Install with:  pip install scipy")


def _read_floats(f, n):
    vals = []
    while len(vals) < n:
        line = f.readline()
        if not line:
            raise EOFError(f"Unexpected end of file after {len(vals)}/{n} values")
        vals.extend(float(x) for x in line.split())
    return vals[:n]


def parse_setfl(path):
    """Parse a single-element LAMMPS setfl potential file -> dict of arrays."""
    with open(path) as f:
        for _ in range(3):          # 3 comment lines
            f.readline()
        parts = f.readline().split()  # "1  Cu"
        nelements = int(parts[0])
        if nelements != 1:
            raise ValueError(f"{path}: only single-element setfl supported, "
                             f"got {nelements}")
        element = parts[1]
        parts = f.readline().split()  # Nrho drho Nr dr cutoff
        nrho, drho = int(parts[0]), float(parts[1])
        nr, dr = int(parts[2]), float(parts[3])
        cutoff = float(parts[4])
        f.readline()                  # element block (Z mass a0 type)
        F_arr = np.array(_read_floats(f, nrho))   # F(rho)
        rho_arr = np.array(_read_floats(f, nr))   # rho_at(r)
        rphi = np.array(_read_floats(f, nr))      # r * phi(r)

    r_arr = np.arange(nr) * dr
    phi_arr = np.zeros(nr)
    phi_arr[1:] = rphi[1:] / r_arr[1:]            # r*phi -> phi, skip r=0
    return {
        "element": element, "nrho": nrho, "drho": drho,
        "nr": nr, "dr": dr, "cutoff": cutoff,
        "F": F_arr, "rho": rho_arr, "phi": phi_arr,
    }


def _subsample(x_full, y_full, n_knots, x_start, x_end):
    """n_knots uniformly spaced (x, y) over [x_start, x_end] via cubic spline."""
    x_knots = np.linspace(x_start, x_end, n_knots)
    cs = CubicSpline(x_full, y_full, extrapolate=False)
    y_knots = np.where(np.isnan(cs(x_knots)), 0.0, cs(x_knots))
    return x_knots, y_knots


def to_tabulated_json(pot, n_knots=200):
    """Parsed setfl dict -> forcesmith ntypes:1 tabulated EAM JSON."""
    nr, dr, cutoff = pot["nr"], pot["dr"], pot["cutoff"]
    nrho, drho = pot["nrho"], pot["drho"]
    r_full = np.arange(nr) * dr
    rh_full = np.arange(nrho) * drho

    x_phi, y_phi = _subsample(r_full[1:], pot["phi"][1:], n_knots, dr, cutoff)
    x_rho, y_rho = _subsample(r_full, pot["rho"], n_knots, 0.0, cutoff)
    rho_max = (nrho - 1) * drho
    x_F, y_F = _subsample(rh_full, pot["F"], n_knots, 0.0, rho_max)

    def make_pot(x, y):
        return {"rmin": float(x[0]), "rmax": float(x[-1]),
                "knots": [float(v) for v in y]}

    return {
        "model": "eam", "ntypes": 1,
        "pair": {"format": "tabulated", "potentials": [make_pot(x_phi, y_phi)]},
        "density": {"format": "tabulated", "potentials": [make_pot(x_rho, y_rho)]},
        "embedding": {"format": "tabulated", "potentials": [make_pot(x_F, y_F)]},
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="single-element .eam setfl files")
    p.add_argument("--out-dir", default="data/zr_cu_al")
    p.add_argument("--knots", type=int, default=200,
                   help="knots per function in the output table (default 200)")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            sys.exit(f"input not found: {path}")
        pot = parse_setfl(path)
        js = to_tabulated_json(pot, n_knots=args.knots)
        out = out_dir / f"{pot['element'].lower()}_eam_ref.json"
        with open(out, "w") as fh:
            json.dump(js, fh, indent=2)
        print(f"  {os.path.basename(path):18s} {pot['element']:2s}  "
              f"nr={pot['nr']} cutoff={pot['cutoff']}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
