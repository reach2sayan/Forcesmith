# Forcesmith

[![GCC 15](https://github.com/reach2sayan/Forcesmith/actions/workflows/gcc.yml/badge.svg)](https://github.com/reach2sayan/Forcesmith/actions/workflows/gcc.yml)
[![Clang 20](https://github.com/reach2sayan/Forcesmith/actions/workflows/clang.yml/badge.svg)](https://github.com/reach2sayan/Forcesmith/actions/workflows/clang.yml)
[![MSVC 2022](https://github.com/reach2sayan/Forcesmith/actions/workflows/msvc.yml/badge.svg)](https://github.com/reach2sayan/Forcesmith/actions/workflows/msvc.yml)

A modern **C++23** reimplementation of [potfit](https://www.potfit.net/), the
open-source force-matching tool for constructing interatomic potentials. Given a
set of reference configurations — atomic positions, forces, energies, and
optionally stresses, typically from DFT — Forcesmith optimises a potential's
parameters until the model reproduces the reference data.

This document covers the **command-line tool** and the **`forcesmith::Forcesmith`
programmatic interface**. For the internals — the value-semantic type erasure,
the CRTP bases, the customisation points, and how the optimiser is wired — see
[`docs/design.md`](docs/design.md). For how to plug in your own potentials,
solvers, heads, and force calculators, see [`docs/extension.md`](docs/extension.md).

## Model families

| Family       | Description                                                       |
|--------------|-------------------------------------------------------------------|
| **pair**     | Two-body radial potentials φ(r)                                  |
| **eam**      | Embedded Atom Method: pair + electron density + embedding F(ρ)    |
| **adp**      | Angular-Dependent Potential: EAM + dipole/quadrupole tensors      |
| **angular**  | EAM-style with a radial modulation f(r) and angular term g(cosθ)  |
| **tersoff**  | Bond-order (Tersoff / modified Tersoff)                           |
| **stiweb**   | Stillinger–Weber (2-body + 3-body)                                |
| **ml**       | Machine-learned: per-atom descriptor (ACSF, SOAP, LMBTR) → energy head |

Radial functions are either **analytic** (Lennard-Jones, Morse, EOPP, Born,
Buckingham, …; see the registry in `src/io/potential_reader.cpp`) or
**tabulated** (cubic splines over knot values). The fit is driven by one of
several optimisers — Levenberg–Marquardt, Powell dogleg, differential evolution,
Ipopt (L-BFGS), a Powell direction-set line search, or a closed-form linear
least-squares solve (`lsq`, for linear ML heads).

## Building

> Requires CMake ≥ 3.28 and a C++23 compiler (GCC 14+ / Clang 18+).

```sh
cmake --preset default          # or: cmake -S . -B build
cmake --build build
```

Targets produced:

- `forcesmith` — the CLI binary (target `forcesmith_cli`, output `forcesmith`)
- `libforcesmith.so` — the shared programmatic API library
- `forcesmith_tests` / `forcesmith_integration_tests` — GoogleTest suites (run via `ctest`)

Dependencies (fetched/built by `cmake/Dependencies.cmake`): Eigen, Boost
(parser, serialization, program_options, math, LEAF), nlohmann_json, oneTBB,
Ipopt (+MUMPS, built as an ExternalProject), and spdlog.

## Command-line usage

```
forcesmith --config configs.json --startpot start.json --endpot fitted.json [options]
```

| Flag                       | Default  | Meaning                                                        |
|----------------------------|----------|----------------------------------------------------------------|
| `--config`, `-c`           | required | atomic configuration file (reference data)                     |
| `--startpot`, `-s`         | required | initial potential / model file                                 |
| `--endpot`, `-e`           | —        | output potential file (required unless `--evaluate`)           |
| `--evaluate <file>`        | —        | evaluate the start potential, write a per-config forces/energy/stress JSON report, and exit (no optimisation) |
| `--format`, `-f`           | `native` | output format: `native` \| `lammps` \| `imd`                   |
| `--checkpoint`, `-k`       | —        | checkpoint prefix: save after each run, resume if present      |
| `--maxiter`                | `500`    | maximum optimiser iterations                                   |
| `--eweight`                | `1.0`    | energy residual weight                                         |
| `--stress-weight`          | `0.0`    | stress-tensor residual weight (0 = disabled)                   |
| `--smooth-weight`          | `0.0`    | curvature (Tikhonov) regularisation on free knots (0 = disabled) |
| `--algorithm`, `-a`        | `lm`     | optimiser: `lm` \| `powell` (dogleg) \| `de` \| `ls` (Powell direction-set) \| `ipopt` (L-BFGS) \| `lsq` (closed-form least squares; exact + low-memory for linear ML heads) |
| `--seed`                   | `0`      | RNG seed for DE (0 = `random_device`)                          |
| `--de-F`                   | `0.65`   | DE mutation factor F ∈ (0,1)                                   |
| `--de-CR`                  | `0.5`    | DE crossover probability CR ∈ (0,1)                            |
| `--de-np`                  | `15`     | DE population factor: NP = `de-np` × D                         |
| `--de-gen`                 | `1000`   | DE maximum generations                                         |

### Examples

Fit an EAM potential with Levenberg–Marquardt:

```sh
forcesmith -c cu_training.json -s cu_eam_start.json -e cu_eam_fit.json -a lm --maxiter 1000
```

Evaluate a potential against the reference configs without fitting:

```sh
forcesmith -c cu_training.json -s cu_eam_fit.json --evaluate cu_report.json
```

Global optimisation with differential evolution, then local refinement:

```sh
forcesmith -c train.json -s start.json -e de_fit.json -a de --de-gen 2000 --seed 42
forcesmith -c train.json -s de_fit.json -e final.json -a lm
```

Fit a machine-learned potential with a linear head in one closed-form solve
(`lsq` evaluates the residual and Jacobian once each and never materialises the
whole-dataset descriptor cache — the memory win for large SOAP/ACSF fits):

```sh
forcesmith init -m soap --n-max 6 --l-max 6 -o soap_start.json
forcesmith -c train.json -s soap_start.json -e soap_fit.json -a lsq
```

### Scaffolding a start potential — `forcesmith init`

`forcesmith init` is a JSON-native `makeapot`: it writes a fresh, immediately
fittable `--startpot` for any of the nine model types.

```
forcesmith init --model <type> --out <file> [--ntypes N] [--cutoff Å] [...]
```

| Flag                  | Applies to        | Meaning                                                          |
|-----------------------|-------------------|------------------------------------------------------------------|
| `--model`, `-m`       | all               | `pair`\|`eam`\|`adp`\|`angular`\|`tersoff`\|`stiweb`\|`acsf`\|`soap`\|`lmbtr` |
| `--out`, `-o`         | all               | output startpot file                                             |
| `--ntypes`, `-n`      | all               | number of atom types (default 1)                                 |
| `--cutoff`, `-c`      | all               | cutoff radius in Å (default 6.0)                                 |
| `--functions`, `-f`   | analytic families | makeapot-style list, e.g. `"3*lj"` or `"lj,exp_decay,sqrt"`      |
| `--n-max`/`--l-max`/`--sigma` | soap      | radial basis size / angular degree / atomic Gaussian width       |
| `--g1`/`--g2-eta`/`--g2-rs`   | acsf      | G1 channel count / G2 η widths / G2 rₛ centres                   |
| `--k2-n`/`--k3-n`/`--drop-k2`/`--drop-k3` | lmbtr | k2/k3 grid points; drop either term                          |
| `--bias-free`         | ml                | leave the head bias free (default fixed, for forces-first fits)  |
| `--seed`              | ml                | RNG seed for the small-Gaussian head-coefficient init            |

```sh
forcesmith init -m eam -n 1 -c 6.0 -o cu_eam_start.json     # analytic EAM scaffold
forcesmith init -m soap --n-max 6 --l-max 6 -o soap_start.json
```

## File formats

Both inputs are JSON.

### Configuration file (`--config`)

An array of configurations. Each has a cell (`X`/`Y`/`Z` lattice vectors), a
reference energy `E`, a weight `W`, and a list of atoms with `element`,
`position`, and reference `force`:

```json
[
  {
    "X": [3.47, 0.0, 0.0],
    "Y": [0.0, 3.47, 0.0],
    "Z": [0.0, 0.0, 3.47],
    "E": -13.828150,
    "W": 1.0,
    "atoms": [
      { "element": "Cu", "position": [0.0, 0.0, 0.0], "force": [-0.0208, 0.0150, 0.0188] }
    ]
  }
]
```

A reference stress tensor (`stress`) may also be supplied per configuration.

### Model / potential file (`--startpot`)

A top-level object whose `"model"` key selects the family. Each radial sub-table
declares a `"format"` (`analytic` or `tabulated`) and a list of `"potentials"`.
An analytic potential names its `"type"` and its parameters by name; a tabulated
one lists `"knots"`. Example EAM model with an analytic Morse pair term and
tabulated density/embedding:

```json
{
  "model": "eam",
  "ntypes": 1,
  "pair": {
    "format": "analytic",
    "potentials": [
      { "type": "morse", "rmin": 2.718, "rmax": 6.287,
        "De": 0.0689, "a": 1.108, "re": 3.318 }
    ]
  },
  "density":   { "format": "tabulated", "potentials": [ { "rmin": 0.0, "rmax": 6.287, "knots": [ /* … */ ] } ] },
  "embedding": { "format": "tabulated", "potentials": [ { "rmin": 0.0, "rmax": 2.0,   "knots": [ /* … */ ] } ] }
}
```

See `include/forcesmith/io/force_model_reader.hpp` for each model's required
sub-tables, and `data/` for complete worked examples.

## Programmatic API

The CLI is a thin client of `forcesmith::Forcesmith`
(`include/forcesmith/api/forcesmith.hpp`); the same fit can be built entirely in
memory. The facade owns the whole fit — configurations, atoms, reference
data, and potentials — and runs the expensive invariant-establishing build
("freeze") lazily on the first call that needs it, so there is no explicit
setup phase. Every fallible call returns `boost::leaf::result<T>`.

```cpp
#include "forcesmith/api/forcesmith.hpp"
#include "forcesmith/potentials/analytic_potential.hpp"

forcesmith::Forcesmith session;

std::size_t cfg = session.add_configuration(forcesmith::PeriodicBC(box));
BOOST_LEAF_CHECK(session.add_atom(cfg, "Cu", {0.0, 0.0, 0.0}));
session.set_ref_energy(cfg, -13.83);
// Per-atom reference forces are set by config name + atom index (or by Atom&):
session.set_ref_force("config-0", /*atom*/ 0, {fx, fy, fz});

// Place a potential — any value satisfying the potential contract.
session.set_pair_potential("Cu", "Cu",
    forcesmith::RadialPotential{forcesmith::Morse(0.34, 1.36, 2.87, 2.0, 6.0)});

session.options().energy_weight = 1.0;
BOOST_LEAF_CHECK(session.optimize());
BOOST_LEAF_CHECK(session.write("cu_fit.json", "native"));
```

### What the facade gives you

- **Configurations & atoms** — `add_configuration` / `add_configurations`
  (a whole range at once) / `add_atom` / `remove_atom` / `set_position` /
  `set_element`. Atoms are held by element *identity* (symbol); the compact
  Z-sorted table slot is assigned at freeze, so adding an atom of a new element
  just re-ranks the model at the next freeze.
- **Reference data** — `set_ref_force`, `set_ref_energy`, `set_ref_stress`,
  `set_weight`, each selectable by index, config name, or a handle. These write
  straight through to the live data and do **not** dirty the session.
- **Potentials** — `set_pair_potential` / `set_density` / `set_embedding`, plus
  the richer-family tables `set_dipole` / `set_quadrupole` (ADP),
  `set_radial` / `set_angular` (angular), and `set_tersoff_params` /
  `set_stiweb_params` / `set_stiweb_lambda` (bond-order). `set_global` links one
  optimiser slot across several potentials; `set_pair_param` edits a placed
  potential in place.
- **A pre-built model** — `seed_force_model(ForceCalculator{...})` adopts a fully
  assembled model (this is how `io::load_model` and the checkpoint reload work).
- **Run & inspect** — `evaluate(cfg)` for a single configuration, `evaluate_all()`
  for every configuration at once (descriptors filled in parallel across cores;
  results ordered by config index and bit-identical to looping `evaluate(i)`),
  `optimize()` for the fit, `write(path, format)` for output, and read-only
  accessors (`configurations()`, `species()`, `model()`, `index()`).

Custom **potentials**, **solvers**, **ML energy heads**, and whole
**force calculators** are all supplied through this facade without recompiling
the engine. The README shows *where* they plug in; the concrete contracts and
worked examples live in [`docs/extension.md`](docs/extension.md), and the design
behind them in [`docs/design.md`](docs/design.md).

## License

Distributed under the [Boost Software License, Version 1.0](LICENSE).
