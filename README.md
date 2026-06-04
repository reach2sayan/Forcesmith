# Potfit

A modern **C++23** reimplementation of [potfit](https://www.potfit.net/), the
open-source force-matching tool for constructing interatomic potentials. Given a
set of reference configurations (atomic positions, forces, energies, and
optionally stresses — typically from DFT), Potfit optimises a potential's
parameters so the model reproduces the reference data.

Supported potential families:

| Family       | Description                                            |
|--------------|--------------------------------------------------------|
| **pair**     | Two-body radial potentials φ(r)                        |
| **eam**      | Embedded Atom Method: pair + electron density + embedding F(ρ) |
| **adp**      | Angular-Dependent Potential: EAM + dipole/quadrupole tensors |
| **angular**  | EAM-style with a radial modulation f(r) and angular g(cosθ) |
| **tersoff**  | Bond-order (Tersoff / modified Tersoff)                |
| **stiweb**   | Stillinger–Weber (2-body + 3-body)                     |

Radial functions may be **analytic** (Lennard-Jones, Morse, EOPP, …) or
**tabulated** (cubic splines over knot values). The fit is driven by one of
several optimisers (Levenberg–Marquardt, Powell dogleg, differential evolution,
or a direction-set line search).

## Design

Potfit is built as a static engine library (`potfit_engine`), a first-class
shared API library (`libpotfit`, the public `potfit::PotFit` facade), and a thin
CLI (`potfit`) that is just another client of that API. Polymorphic surfaces
(potentials, solvers) use Sean-Parent-style **value-semantic type erasure**, so
new potentials and solvers can be supplied by a library client without touching
the engine — see [`docs/extension.md`](docs/extension.md).

Errors are propagated with `boost::leaf::result<T>` rather than exceptions.

### Key dependencies

Eigen, Boost (parser, serialization, program_options, math, LEAF), nlohmann_json,
and oneTBB. See `cmake/Dependencies.cmake`.

## Building

> Requires CMake ≥ 3.28 and a C++23 compiler (GCC 14+ / Clang 18+).

```sh
cmake --preset default          # or: cmake -S . -B build
cmake --build build
```

Targets produced:

- `potfit` — the CLI binary (target name `potfit_cli`, output `potfit`)
- `libpotfit.so` — the shared programmatic API library
- `potfit_tests` / `potfit_integration_tests` — GoogleTest suites (run via `ctest`)

Optional fit drivers (each a single binary; pick the element at runtime with
`--element`):

- `-DPOTFIT_BUILD_UNEP_FITS=ON` → `unep_fit` (UNEP EAM fit, e.g. `unep_fit --element Cu`)
- `-DPOTFIT_BUILD_ML_FITS=ON` → `ml_fit` (ML fit, e.g. `ml_fit --element Cu --descriptor soap|symfunc --head nn|linear`)

## Command-line usage

```
potfit --config configs.json --startpot start.json --endpot fitted.json [options]
```

| Flag                       | Default  | Meaning                                                        |
|----------------------------|----------|----------------------------------------------------------------|
| `--config`, `-c`           | required | atomic configuration file (reference data)                     |
| `--startpot`, `-s`         | required | initial potential / model file                                 |
| `--endpot`, `-e`           | —        | output potential file (required unless `--evaluate`)           |
| `--evaluate <file>`        | —        | evaluate the start potential and write a per-config forces/energy/stress JSON report, then exit (no optimisation) |
| `--format`, `-f`           | `native` | output format: `native` \| `lammps` \| `imd`                   |
| `--checkpoint`, `-k`       | —        | checkpoint prefix: save after each run, resume if present      |
| `--maxiter`                | `500`    | maximum optimiser iterations                                   |
| `--eweight`                | `1.0`    | energy residual weight                                         |
| `--stress-weight`          | `0.0`    | stress-tensor residual weight (0 = disabled)                   |
| `--smooth-weight`          | `0.0`    | curvature (Tikhonov) regularisation on free knots (0 = disabled) |
| `--algorithm`, `-a`        | `lm`     | optimiser: `lm` \| `powell` (dogleg) \| `de` \| `ls` (direction-set line search) |
| `--seed`                   | `0`      | RNG seed for DE (0 = `random_device`)                          |
| `--de-F`                   | `0.65`   | DE mutation factor F ∈ (0,1)                                   |
| `--de-CR`                  | `0.5`    | DE crossover probability CR ∈ (0,1)                            |
| `--de-np`                  | `15`     | DE population factor: NP = `de-np` × D                         |
| `--de-gen`                 | `1000`   | DE maximum generations                                         |

### Examples

Fit an EAM potential with Levenberg–Marquardt:

```sh
potfit -c cu_training.json -s cu_eam_start.json -e cu_eam_fit.json -a lm --maxiter 1000
```

Evaluate a potential against the reference configs without fitting:

```sh
potfit -c cu_training.json -s cu_eam_fit.json --evaluate cu_report.json
```

Global optimisation with differential evolution, then refine:

```sh
potfit -c train.json -s start.json -e de_fit.json -a de --de-gen 2000 --seed 42
potfit -c train.json -s de_fit.json -e final.json -a lm
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
An analytic potential names its `"type"` (see the registry in
`src/io/potential_reader.cpp`) and its parameters by name; a tabulated one lists
`"knots"`. Example EAM model with an analytic Morse pair term and tabulated
density/embedding:

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

See `include/potfit/io/force_model_reader.hpp` for the required sub-tables of
each model, and `data/` for complete worked examples.

## Programmatic API

The CLI is a thin client of `potfit::PotFit`; the same fit can be built entirely
in memory. Every fallible call returns `boost::leaf::result<T>`.

```cpp
#include "potfit/api/potfit.hpp"
#include "potfit/potentials/analytic_potential.hpp"

potfit::PotFit session;

std::size_t cfg = session.add_configuration(potfit::PeriodicBC(box));
BOOST_LEAF_CHECK(session.add_atom(cfg, "Cu", {0.0, 0.0, 0.0}));
session.set_ref_energy(cfg, -13.83);
// Per-atom reference forces are set by config name + atom index (or by Atom&):
session.set_ref_force("config-0", /*atom*/ 0, {fx, fy, fz});

// Place a potential — any value satisfying the potential contract.
session.set_pair_potential("Cu", "Cu",
    potfit::Potential{potfit::Morse(0.34, 1.36, 2.87, 2.0, 6.0)});

session.options().energy_weight = 1.0;
BOOST_LEAF_CHECK(session.optimize());
BOOST_LEAF_CHECK(session.write("cu_fit.json", "native"));
```

A custom solver is injected the same way potentials are
(`session.set_solver(potfit::Solver{MySolver{...}})`). Both extension points are
documented in [`docs/extension.md`](docs/extension.md).

## License

Distributed under the [Boost Software License, Version 1.0](LICENSE).
