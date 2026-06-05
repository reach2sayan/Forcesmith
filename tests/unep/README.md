# UNEP per-element EAM fitting (`tests/unep/`)

`fit_eam.py` fits an EAM potential to **each of the 16 metals** in the converted
UNEP DFT dataset (`data/unep/<el>_dft_unep.json`, from UNEP-v1 / Zenodo
11533864 via `tools/extxyz2force.py`) and reports how much it reduces per-atom
**force RMSE**.

It reuses the existing `build/release/forcesmith` binary — it never runs cmake.
Build forcesmith yourself first; the script errors with a clear message if the
binary is missing.

## What it does (per element)

It produces **two fits per element** and plots both:
- the **analytic fit** (stage 1) → `fits/<el>_eam_analytic.json`
- the **tabular fit** (stage 2) → `fits/<el>_eam_fit.json`

A **two-stage, forces-first** fit:

1. **Derive cutoffs from the data.** Nearest-neighbour distance `dmin` (a
   minimum-image pairwise scan, exact for the closest pair) sets
   `re = dmin`, `rmin = max(1.0, 0.88·dmin)`, `rmax = min(6.5, 2.4·dmin)`.
2. **Stage-1 analytic start** — `morse` pair + `exp_decay` density + `sqrt`
   embedding (B > 0), all parameters free.
3. **Stage-1 fit** (`-a lm`) → a dense 500-knot tabulated EAM (forcesmith's native
   writer always re-samples to `kDefaultKnots = 500`).
4. **Stage-2 start** — down-sample each section (pair/density/embedding) of the
   stage-1 result to `--knots` (default 15) free knots. This is the
   "analytic seeds tabulated" step; a direct re-fit of 500×3 knots would be
   wildly over-parameterized.
5. **Stage-2 fit** with `--smooth-weight` (curvature regularization) on the free
   splines.
6. **Force RMSE** is measured via `forcesmith --evaluate` at three checkpoints:
   start (analytic), after stage 1, and after stage 2.

## Usage

```bash
# all 16 elements (slow; parallelize with --jobs)
python3 tests/unep/fit_eam.py --jobs 6

# one element
python3 tests/unep/fit_eam.py --elements Cu

# fast smoke run (subsampled configs, few iterations)
python3 tests/unep/fit_eam.py --elements Cu --max-configs 40 --maxiter 60
```

Key flags: `--elements`, `--data-dir`, `--out-dir`, `--forcesmith`, `--maxiter`,
`--eweight` (default 0.1), `--stress-weight` (default 0), `--smooth-weight`
(default 1.0), `--knots` (default 15), `--algorithm {lm,powell,de,ls}`,
`--max-configs`/`--stride` (subsample), `--jobs` (parallel elements),
`--stage {1,2}` (stop after stage 1).

## Outputs

- `fits/<el>_eam_analytic.json` — the **analytic fit** (stage 1).
- `fits/<el>_eam_fit.json` — the **tabular fit** (stage 2).
- `work/<el>/{start_analytic,start_tab,configs_subset}.json` — intermediates.
- `results.json` — one row per element with **force** RMSE (eV/Å) and **stress**
  RMSE (eV/Å³) at each checkpoint: `rmse_start/analytic/tabular` and
  `srmse_start/analytic/tabular`, plus `element, nconf, natoms, dmin, rmin,
  rmax, status`.
- A markdown summary table is printed at the end (force then stress columns).

Plot both fits with `plot_fit.py --element <El>`:
- `plots/<el>_functions.png` — pair/density/embedding, analytic (dashed) vs
  tabular (solid).
- `plots/<el>_parity.png` — a 2×2 observed-vs-predicted grid (rows
  {analytic, tabular} × cols {energy, stress}).

Re-load any fit with:

```bash
build/release/forcesmith -c data/unep/cu_dft_unep.json \
    -s tests/unep/fits/cu_eam_fit.json --evaluate /tmp/chk.json
```

## Caveats (by design)

- **Forces-first.** UNEP energies are raw VASP totals with an arbitrary
  per-atom zero an EAM cannot reproduce, so `--eweight` is small (0.1) and
  absolute energy is *not* fit. Forces (reference-free) drive the fit and are
  the reported metric.
- **Stress** units are now consistent: the converter emits `S` in **eV/Å³**, the
  same unit the C++ engine compares against (`calc_stress = virial/volume`), and
  the sign convention was verified consistent on a hydrostatic UNEP frame
  (dataset-wide corrcoef 0.96). Stress is still **off in the fit by default**
  (`--stress-weight 0`, forces-first), but it is now a usable target — set
  `--stress-weight > 0` to include it — and the parity panel is directly
  interpretable.
- **Embedding ρ-domain** span `[1e-4, 10]` for the stage-1 `sqrt` is a
  heuristic; the stage-2 free knots refine forces regardless of spacing.
- A generic analytic EAM is a **modest model**. The goal is to prove the
  pipeline fits every element and *reduces* force RMSE — not to reach DFT
  accuracy.
