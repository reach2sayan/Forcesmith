# Evaluation-parity harness: C++ port vs. original C potfit

Diffs the C++ port's force engine against the reference C `potfit` on the same
real data (the configs under `../data/potfit_upstream/`). The comparison is an
**evaluation** of a fixed start potential — no optimization — so it isolates the
force/energy/stress pipeline from optimizer-trajectory noise.

## What is compared
The **primary gate is forces-only**: both codes evaluate the start potential and
we compare the sum of squared force residuals `Σ (f_calc − f_ref)²` (potfit's
`sum of force-errors` line) plus per-atom computed forces. We deliberately do
*not* compare potfit's *total* error sum — for EAM it also folds in gauge
"dummy constraint" rows that the port models differently. Forces are also
EAM-gauge-invariant and weight-independent, so this gate is robust to those
convention differences; a clean run matches to spline-interpolation precision.

The bundled `Al_eam/Al_eam.startpot` is a physically-motivated seed (Morse pair
+ exponential density + √ embedding / Finnis-Sinclair) so the residuals are
O(1) and meaningful — not the degenerate `lj` makeapot default.

## Pieces
| File | Role |
|------|------|
| `pot2json.py`     | Convert potfit text `config`/apot `startpot` → the port's JSON |
| `gen_reference.sh`| Build upstream potfit, run it in evaluate mode (`opt 0`), capture `total error sum` + `<prefix>.force` |
| `diff_eval.py`    | Compare the port's `--evaluate` JSON against the reference within a tolerance |
| `reference/<case>/` | Generated reference outputs (created by `gen_reference.sh`) |

## Port support used
A new evaluate-only mode was added to the port:
```
forcesmith -c CONFIG.json -s STARTPOT.json --evaluate OUT.json --eweight 0 --stress-weight 0
```
`--evaluate` runs a single `forcesmith::force::evaluate()` over every config (no
optimizer, no endpot) and writes per-config computed vs. reference
forces/energy/stress and `total_sumsq` to `OUT.json`. It also exercises the
enriched `on_force_eval` callback, which now carries `const Configuration&`.

## End-to-end (Al EAM example)
```bash
cd tests/integration
H=harness ; D=data/potfit_upstream

# 0. one-time: clone the oracle (gen_reference.sh builds it)
git clone https://github.com/potfit/potfit /tmp/potfit_upstream

# 1. convert the shared inputs to the port's JSON
python3 $H/pot2json.py config   $D/Al_eam/Al.config        /tmp/al_cfg.json
python3 $H/pot2json.py startpot $D/Al_eam/Al_eam.startpot  /tmp/al_start.json --ntypes 1

# 2. generate the reference (builds upstream potfit, evaluate mode)
$H/gen_reference.sh $D/Al_eam Al_eam.param al_eam

# 3. run the port in evaluate mode (forces-only gate)
build/release/forcesmith -c /tmp/al_cfg.json -s /tmp/al_start.json \
    --evaluate /tmp/al_port_eval.json --eweight 0 --stress-weight 0

# 4. diff
python3 $H/diff_eval.py \
    --port /tmp/al_port_eval.json \
    --ref-log $H/reference/al_eam/reference.log \
    --ref-force $H/reference/al_eam/Al_eam.force
```
For the 50-config BaGe case, swap in `BaGe_eam/` with `--ntypes 2`.

## Weights (for the optional weighted comparison)
potfit accumulates the energy/stress term as `eweight · ΔE²` (weight applied
linearly to the squared residual). The port uses `(energy_weight · ΔE)²`
(weight squared). So to match a non-zero potfit `eng_weight`/`stress_weight`,
run the port with the **square roots**:
`--eweight sqrt(eng_weight)`, `--stress-weight sqrt(stress_weight)`. The
forces-only gate above sidesteps this entirely by setting all weights to 0.

## Known caveats (expected sources of small / real differences)
- **apot tabulation.** Upstream apot evaluates analytic functions via an internal
  spline table; the port evaluates them directly. Expect small (interpolation-
  level) differences. For tightest parity, drive both with *tabulated* potentials.
- **`rmin` of analytic functions.** potfit apot stores only a `cutoff` (= `rmax`);
  the port requires `rmin`. `pot2json.py --rmin` defaults to 0.0 — adjust if the
  upstream function domain differs. This is the main parity knob for analytic pots.
- **Per-config weight `#W`.** potfit multiplies residuals by `conf_weight`; the
  port currently does not. Harmless when all weights are 1 (true for the bundled
  Al/BaGe data) — otherwise a real discrepancy worth flagging.
- **lj seed potentials** (from `makeapot`) are numerical seeds, not physical EAM —
  fine for engine parity, meaningless as physics.

A large diff here is a finding, not a harness bug — that is the point. See
`reference/algorithm-verification.md` for previously confirmed C-vs-C++ bugs.
