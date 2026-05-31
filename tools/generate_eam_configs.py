#!/usr/bin/env python3
"""
Generate NIST EAM training data for potfit end-to-end validation.

Downloads Mishin Cu 2001 and Mishin Al 1999 potentials from the NIST
Interatomic Potentials Repository, converts them from LAMMPS setfl format
to potfit JSON, generates FCC training configurations, adds Gaussian noise
(mimicking DFT scatter), and writes all inputs for the potfit CLI.

After running this script, fit with:

    potfit -c data/cu_training.json -s data/cu_eam_start.json -e data/cu_eam_fit.json
    potfit -c data/al_training.json -s data/al_eam_start.json -e data/al_eam_fit.json

Usage:
    python3 tools/generate_eam_configs.py [--output-dir PATH] [--potfit PATH]

Requirements: numpy, scipy
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

try:
    from scipy.interpolate import CubicSpline
except ImportError:
    sys.exit("ERROR: scipy is required.  Install with:  pip install scipy")

# ── NIST download URLs ────────────────────────────────────────────────────────

NIST_BASE = "https://www.ctcms.nist.gov/potentials/Download"

NIST_FILES = {
    "Cu": (
        f"{NIST_BASE}/2001--Mishin-Y-Mehl-M-J-Papaconstantopoulos-D-A-et-al--Cu-1"
        f"/2/Cu01.eam.alloy",
        "Cu01.eam.alloy",
    ),
    "Al": (
        f"{NIST_BASE}/1999--Mishin-Y-Farkas-D-Mehl-M-J-Papaconstantopoulos-D-A--Al"
        f"/2/Al99.eam.alloy",
        "Al99.eam.alloy",
    ),
}

# Analytic starting guesses for potfit (physically motivated, not the NIST form)
ANALYTIC_START = {
    "Cu": {
        "model": "eam",
        "ntypes": 1,
        "pair": {
            "format": "analytic",
            "potentials": [
                {"type": "exp_decay", "rmin": 2.0, "rmax": 5.5, "A": 1.5, "B": 0.8}
            ],
        },
        "density": {
            "format": "analytic",
            "potentials": [
                {"type": "exp_decay", "rmin": 2.0, "rmax": 5.5, "A": 5.0, "B": 1.0}
            ],
        },
        "embedding": {
            "format": "analytic",
            "potentials": [
                {"type": "sqrt", "rmin": 0.01, "rmax": 100.0, "A": -2.0, "B": 0.0}
            ],
        },
    },
    "Al": {
        "model": "eam",
        "ntypes": 1,
        "pair": {
            "format": "analytic",
            "potentials": [
                {"type": "exp_decay", "rmin": 2.2, "rmax": 6.3, "A": 0.9, "B": 0.7}
            ],
        },
        "density": {
            "format": "analytic",
            "potentials": [
                {"type": "exp_decay", "rmin": 2.2, "rmax": 6.3, "A": 4.0, "B": 0.85}
            ],
        },
        "embedding": {
            "format": "analytic",
            "potentials": [
                {"type": "sqrt", "rmin": 0.01, "rmax": 100.0, "A": -1.8, "B": 0.0}
            ],
        },
    },
}

# FCC lattice constants (Å)
A0 = {"Cu": 3.615, "Al": 4.046}

# ── setfl file parsing ────────────────────────────────────────────────────────

def _read_floats(f, n):
    """Read exactly n whitespace-separated floats from file f."""
    vals = []
    while len(vals) < n:
        line = f.readline()
        if not line:
            raise EOFError(f"Unexpected end of file after {len(vals)}/{n} values")
        vals.extend(float(x) for x in line.split())
    return vals[:n]


def parse_setfl(path):
    """Parse a single-element LAMMPS setfl (eam/alloy) potential file.

    Returns a dict with keys:
        element, nrho, drho, nr, dr, cutoff,
        F   — embedding energy F(rho_i)  [eV],      shape (nrho,)
        rho — electron density rho_at(r) [au],       shape (nr,)
        phi — pair potential phi(r)      [eV],       shape (nr,)
                (note: setfl stores r*phi; we divide by r, phi[0]=0)
    """
    with open(path) as f:
        # 3 comment lines
        for _ in range(3):
            f.readline()

        # Element line:  "1  Cu"
        parts = f.readline().split()
        nelements = int(parts[0])
        if nelements != 1:
            raise ValueError(f"Only single-element setfl supported, got {nelements}")
        element = parts[1]

        # Grid parameters:  Nrho drho Nr dr cutoff
        parts = f.readline().split()
        nrho, drho = int(parts[0]), float(parts[1])
        nr,   dr   = int(parts[2]), float(parts[3])
        cutoff     = float(parts[4])

        # Element block:  atomic_num mass lattice_const crystal
        f.readline()

        F_arr   = np.array(_read_floats(f, nrho))  # F(rho)
        rho_arr = np.array(_read_floats(f, nr))    # rho_at(r)
        rphi    = np.array(_read_floats(f, nr))    # r * phi(r)

    # Convert r*phi(r) → phi(r).  Index 0 is r=0; by convention r*phi(0)=0,
    # so phi at r=0 is the repulsive limit (handled by rmin in potfit).
    r_arr = np.arange(nr) * dr
    phi_arr = np.zeros(nr)
    phi_arr[1:] = rphi[1:] / r_arr[1:]   # skip r=0

    return {
        "element": element,
        "nrho": nrho, "drho": drho,
        "nr": nr,   "dr": dr, "cutoff": cutoff,
        "F": F_arr, "rho": rho_arr, "phi": phi_arr,
    }


# ── Spline builders ───────────────────────────────────────────────────────────

def _subsample(x_full, y_full, n_knots, x_start, x_end):
    """Return (x_knots, y_knots) as n_knots uniformly spaced points in [x_start,x_end]."""
    x_knots = np.linspace(x_start, x_end, n_knots)
    cs = CubicSpline(x_full, y_full, extrapolate=False)
    y_knots = cs(x_knots)
    y_knots = np.where(np.isnan(y_knots), 0.0, y_knots)
    return x_knots, y_knots


def build_splines(pot, n_knots=200):
    """Build CubicSpline objects for F, rho_at, phi from a parsed setfl dict."""
    nr, dr, nrho, drho = pot["nr"], pot["dr"], pot["nrho"], pot["drho"]

    # r grid for rho_at and phi (skip r=0 for phi to avoid 0/0 singularity)
    r_full  = np.arange(nr) * dr
    rho_r_full = np.arange(nrho) * drho

    # Electron density rho_at(r): start from r=0
    rho_at_cs = CubicSpline(r_full, pot["rho"])

    # Pair potential phi(r): start from r=dr (skip r=0)
    phi_cs = CubicSpline(r_full[1:], pot["phi"][1:])

    # Embedding F(rho): full range
    F_cs = CubicSpline(rho_r_full, pot["F"])

    return F_cs, rho_at_cs, phi_cs


def to_tabulated_json(pot, n_knots=200):
    """Convert parsed setfl data to potfit JSON tabulated format."""
    nr, dr, cutoff = pot["nr"], pot["dr"], pot["cutoff"]
    nrho, drho     = pot["nrho"], pot["drho"]

    r_full  = np.arange(nr)   * dr
    rh_full = np.arange(nrho) * drho

    # phi(r): knots from dr to cutoff
    x_phi, y_phi = _subsample(r_full[1:], pot["phi"][1:], n_knots, dr, cutoff)

    # rho_at(r): knots from 0 to cutoff
    x_rho, y_rho = _subsample(r_full, pot["rho"], n_knots, 0.0, cutoff)

    # F(rho): knots from 0 to rho_max
    rho_max = (nrho - 1) * drho
    x_F, y_F = _subsample(rh_full, pot["F"], n_knots, 0.0, rho_max)

    def make_pot(x, y):
        return {
            "rmin": float(x[0]),
            "rmax": float(x[-1]),
            "knots": [float(v) for v in y],
        }

    return {
        "model": "eam",
        "ntypes": 1,
        "pair":    {"format": "tabulated", "potentials": [make_pot(x_phi, y_phi)]},
        "density": {"format": "tabulated", "potentials": [make_pot(x_rho, y_rho)]},
        "embedding": {"format": "tabulated", "potentials": [make_pot(x_F,  y_F)]},
    }


def make_tabulated_start(pot, n_knots, pair_scale, pair_rmin):
    """Build a *coarse* tabulated EAM start from the NIST potential.

    Fitting all three EAM functions freely is ill-posed: the EAM gauge freedoms
    (density scaling, pair/embedding linear shift) plus sparse sampling of
    intermediate r let the optimizer drive the functions wild while still
    matching forces along a flat direction.  So we set up a *well-posed*
    recovery:

      * Freeze density and embedding at their NIST values — fit the pair only,
        which removes the gauge freedom.
      * Put the pair knots only over [pair_rmin, cutoff], the range the FCC
        training data actually probes.  Knots below the nearest-neighbor
        distance (~2.5 Å) are never sampled, so leaving them free makes the
        Jacobian rank-deficient and the optimizer dumps garbage (huge spikes)
        into that null space.  The repulsive wall cannot be fit from
        equilibrium-FCC data anyway.

    The pair knots are scaled by ``pair_scale`` so the optimizer starts off the
    minimum and has a genuine target to recover.
    """
    start = to_tabulated_json(pot, n_knots=n_knots)

    # Rebuild the pair section over the probed range [pair_rmin, cutoff].
    _, _, phi_cs = build_splines(pot)
    cutoff = pot["cutoff"]
    x_phi = np.linspace(pair_rmin, cutoff, n_knots)
    y_phi = np.nan_to_num(phi_cs(x_phi)) * pair_scale
    start["pair"]["potentials"][0] = {
        "rmin": float(pair_rmin),
        "rmax": float(cutoff),
        "knots": [float(v) for v in y_phi],
    }

    # Freeze density and embedding (fit pair only).
    start["density"]["potentials"][0]["fixed"] = True
    start["embedding"]["potentials"][0]["fixed"] = True
    return start


# ── EAM force engine (NumPy + scipy) ─────────────────────────────────────────
#
# Uses a proper image-cell search so the minimum-image convention is not
# required; works correctly for small unit cells where L < 2*cutoff.

def _image_offsets(box_length, cutoff):
    """Return all (nx,ny,nz) image shift vectors needed to cover the cutoff."""
    n = int(np.ceil(cutoff / box_length)) + 1
    offsets = []
    for nx in range(-n, n + 1):
        for ny in range(-n, n + 1):
            for nz in range(-n, n + 1):
                offsets.append((nx, ny, nz))
    return offsets


def compute_eam_forces(positions, box_length, F_cs, rho_at_cs, phi_cs, cutoff):
    """Compute EAM energy and forces for atoms in a cubic periodic box.

    Args:
        positions:   (N, 3) array of Cartesian positions [Å]
        box_length:  side length of the cubic box [Å]
        F_cs:        CubicSpline for F(rho) [eV]
        rho_at_cs:   CubicSpline for rho_at(r) [au]
        phi_cs:      CubicSpline for phi(r) [eV]
        cutoff:      interaction cutoff [Å]

    Returns:
        (energy [eV], forces (N,3) [eV/Å])
    """
    natoms = len(positions)
    offsets = _image_offsets(box_length, cutoff)

    rho = np.zeros(natoms)
    forces = np.zeros((natoms, 3))
    energy = 0.0

    # Pass 1: accumulate electron density rho_i = Σ_j rho_at(r_ij)
    for i in range(natoms):
        for j in range(natoms):
            for nx, ny, nz in offsets:
                if i == j and nx == 0 and ny == 0 and nz == 0:
                    continue
                rij = (positions[j] - positions[i]
                       + np.array([nx, ny, nz]) * box_length)
                r = float(np.linalg.norm(rij))
                if 0 < r < cutoff:
                    rho[i] += float(rho_at_cs(r))

    # Embedding energy and its gradient gradF_i = dF/drho|_{rho_i}
    embed_e = np.array([float(F_cs(rh)) for rh in rho])
    gradF   = np.array([float(F_cs(rh, 1)) for rh in rho])
    energy += float(np.sum(embed_e))

    # Pass 2: pair forces + embedding-gradient coupling
    # F_i = Σ_j [phi'(r) + (gradF_i + gradF_j)*rho'(r)] * r_hat_ij
    for i in range(natoms):
        for j in range(natoms):
            for nx, ny, nz in offsets:
                if i == j and nx == 0 and ny == 0 and nz == 0:
                    continue
                rij = (positions[j] - positions[i]
                       + np.array([nx, ny, nz]) * box_length)
                r = float(np.linalg.norm(rij))
                if 0 < r < cutoff:
                    rij_hat = rij / r
                    dphi_dr  = float(phi_cs(r, 1))
                    drho_dr  = float(rho_at_cs(r, 1))
                    fac = dphi_dr + (gradF[i] + gradF[j]) * drho_dr
                    forces[i] += fac * rij_hat
                    # Pair energy: 0.5 because each ordered pair (i,j) is visited twice
                    energy += 0.5 * float(phi_cs(r))

    return energy, forces


# ── FCC configuration generation ─────────────────────────────────────────────

def fcc_cell(a0, strain=0.0):
    """4-atom FCC conventional cell at a0*(1+strain)."""
    a = a0 * (1.0 + strain)
    return a, np.array([
        [0.0,  0.0,  0.0],
        [a/2,  a/2,  0.0],
        [a/2,  0.0,  a/2],
        [0.0,  a/2,  a/2],
    ])


def generate_configs(element, F_cs, rho_at_cs, phi_cs, cutoff, rng, n_disp=5):
    """Generate 5 strained + n_disp displaced FCC configs with noisy forces."""
    a0 = A0[element]
    sigma_f = 0.02   # eV/Å per force component
    sigma_e = 0.001  # eV per config

    records = []

    strains = [-0.04, -0.02, 0.0, 0.02, 0.04]
    for s in strains:
        a, pos = fcc_cell(a0, s)
        e, f = compute_eam_forces(pos, a, F_cs, rho_at_cs, phi_cs, cutoff)
        e += rng.normal(0, sigma_e)
        f += rng.normal(0, sigma_f, size=f.shape)
        records.append((a, pos, e, f))

    for _ in range(n_disp):
        a, pos = fcc_cell(a0)
        pos = pos + rng.normal(0, 0.05, size=pos.shape)
        e, f = compute_eam_forces(pos, a, F_cs, rho_at_cs, phi_cs, cutoff)
        e += rng.normal(0, sigma_e)
        f += rng.normal(0, sigma_f, size=f.shape)
        records.append((a, pos, e, f))

    return records


def min_interatomic_distance(records):
    """Smallest interatomic distance (incl. nearest periodic images) over all
    configs.  Pair knots below this are never sampled by the data."""
    mind = float("inf")
    for a, pos, _e, _f in records:
        n = len(pos)
        for i in range(n):
            for j in range(n):
                for sx in (-1, 0, 1):
                    for sy in (-1, 0, 1):
                        for sz in (-1, 0, 1):
                            if i == j and sx == sy == sz == 0:
                                continue
                            d = np.linalg.norm(
                                pos[j] - pos[i] + np.array([sx, sy, sz]) * a)
                            mind = min(mind, d)
    return mind


def to_config_json(records, element):
    """Convert list of (a, positions, energy, forces) to potfit config JSON."""
    configs = []
    for a, pos, e, f in records:
        configs.append({
            "X": [a, 0.0, 0.0],
            "Y": [0.0, a, 0.0],
            "Z": [0.0, 0.0, a],
            "E": float(e),
            "W": 1.0,
            "atoms": [
                {
                    "element": element,
                    "position": [float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])],
                    "force":    [float(f[i, 0]),   float(f[i, 1]),   float(f[i, 2])],
                }
                for i in range(len(pos))
            ],
        })
    return configs


# ── Main ──────────────────────────────────────────────────────────────────────

def download_cached(url, cache_path):
    """Download url to cache_path if not already present."""
    if cache_path.exists():
        print(f"  Using cached {cache_path.name}")
        return
    print(f"  Downloading {url} ...")
    try:
        # NIST returns 403 to bare urllib (no User-Agent); send a browser-like one.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(cache_path, "wb") as out:
            out.write(resp.read())
    except Exception as exc:
        sys.exit(f"ERROR: Download failed: {exc}\n"
                 f"       Try manually:  curl -L -o {cache_path} '{url}'")
    print(f"  Saved to {cache_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data",
                        help="Directory for output files (default: data/)")
    parser.add_argument("--potfit", default="./build/potfit",
                        help="Path to the potfit binary for run_fitting.sh")
    parser.add_argument("--start", choices=["analytic", "tabulated"],
                        default="tabulated",
                        help="Starting potential form (default: tabulated). "
                             "'tabulated' starts from a coarse, slightly "
                             "perturbed copy of the NIST potential, which the "
                             "fit can recover closely. 'analytic' uses a rigid "
                             "exp_decay+sqrt form whose embedding F=A√(ρ+B) has "
                             "no additive constant, so it cannot match the "
                             "absolute cohesive energy (expect an E(a) offset).")
    parser.add_argument("--start-knots", type=int, default=15,
                        help="Knots per function for a tabulated start "
                             "(default: 15)")
    parser.add_argument("--start-perturb", type=float, default=1.15,
                        help="Pair-knot scale factor for a tabulated start, so "
                             "the optimizer starts off the minimum "
                             "(default: 1.15)")
    parser.add_argument("--pair-rmin", type=float, default=None,
                        help="Inner cutoff for tabulated-start pair knots (Å). "
                             "Default: auto — the minimum interatomic distance "
                             "actually present in the training configs, so no "
                             "knot sits in an unprobed (unconstrained) range.")
    parser.add_argument("--eweight", type=float, default=1.0,
                        help="Energy residual weight passed to potfit --eweight "
                             "(default: 1.0)")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    run_lines = ["#!/bin/bash", "set -e", ""]
    run_lines.append(f"POTFIT=\"{args.potfit}\"")
    run_lines.append("")

    for element, (url, filename) in NIST_FILES.items():
        print(f"\n=== {element} ===")

        # Download
        cache = outdir / filename
        download_cached(url, cache)

        # Parse setfl
        print(f"  Parsing {filename} ...")
        pot = parse_setfl(cache)
        print(f"  Nr={pot['nr']}, dr={pot['dr']:.6f}, Nrho={pot['nrho']}, "
              f"drho={pot['drho']:.6f}, cutoff={pot['cutoff']:.3f} Å")

        # Build scipy splines
        F_cs, rho_at_cs, phi_cs = build_splines(pot)

        # Convert to potfit tabulated JSON (200-knot subsampling)
        nist_json = to_tabulated_json(pot, n_knots=200)
        true_path = outdir / f"{element.lower()}_nist_true.json"
        true_path.write_text(json.dumps(nist_json, indent=2))
        print(f"  Wrote {true_path}")

        # Generate training configs (needed first so a tabulated start can set
        # its pair inner-cutoff from the distances actually present in the data)
        print(f"  Generating FCC training configs (cutoff={pot['cutoff']:.2f} Å) ...")
        records = generate_configs(element, F_cs, rho_at_cs, phi_cs,
                                   pot["cutoff"], rng, n_disp=5)
        configs = to_config_json(records, element)
        train_path = outdir / f"{element.lower()}_training.json"
        train_path.write_text(json.dumps(configs, indent=2))
        print(f"  Wrote {train_path}  ({len(configs)} configs)")

        # Write starting potential (analytic rigid form, or coarse tabulated
        # copy of NIST that the fit can recover closely)
        if args.start == "tabulated":
            # Default pair inner-cutoff = min interatomic distance in the data,
            # so every pair knot is constrained (no unprobed null space).
            pair_rmin = (args.pair_rmin if args.pair_rmin is not None
                         else min_interatomic_distance(records))
            start_json = make_tabulated_start(pot, args.start_knots,
                                              args.start_perturb, pair_rmin)
            print(f"  Start: tabulated (pair: {args.start_knots} knots over "
                  f"[{pair_rmin:.3f}, {pot['cutoff']:.2f}] ×{args.start_perturb}; "
                  f"ρ,F frozen at NIST)")
        else:
            start_json = ANALYTIC_START[element]
            print("  Start: analytic")
        start_path = outdir / f"{element.lower()}_eam_start.json"
        start_path.write_text(json.dumps(start_json, indent=2))
        print(f"  Wrote {start_path}")

        # Shell-script lines for this element
        elt = element.lower()
        run_lines += [
            f"echo '=== Fitting {element} ==='",
            f"\"$POTFIT\" \\",
            f"  -c {outdir}/{elt}_training.json \\",
            f"  -s {outdir}/{elt}_eam_start.json \\",
            f"  -e {outdir}/{elt}_eam_fit.json \\",
            f"  --algorithm lm --maxiter 500 --eweight {args.eweight}",
            f"echo '{element} done: output in {outdir}/{elt}_eam_fit.json'",
            "",
        ]

    # Write shell script
    sh_path = outdir / "run_fitting.sh"
    sh_path.write_text("\n".join(run_lines) + "\n")
    sh_path.chmod(sh_path.stat().st_mode | 0o755)
    print(f"\nWrote {sh_path}")
    print("\nNext steps:")
    print(f"  bash {sh_path}")
    print("  # or run each element separately:")
    for element in NIST_FILES:
        elt = element.lower()
        print(f"  {args.potfit} -c {outdir}/{elt}_training.json "
              f"-s {outdir}/{elt}_eam_start.json -e {outdir}/{elt}_eam_fit.json "
              f"--algorithm lm --maxiter 500 --eweight {args.eweight}")


if __name__ == "__main__":
    main()
