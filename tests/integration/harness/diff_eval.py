#!/usr/bin/env python3
"""Diff the C++ port's evaluation against the original C potfit reference.

Forces-only parity gate (robust to EAM gauge / weight conventions — forces are
gauge-invariant, and we deliberately avoid potfit's *total* error sum because it
also folds in EAM gauge "dummy constraint" rows the port models differently):

  1. SCALAR  — sum of squared force residuals  Σ (f_calc − f_ref)²
               port:   computed here from configs[].atoms[] calc/ref forces
               potfit: `sum of force-errors = ...` (from reference.log / .error)
  2. PER-ATOM — computed force components (optional, if a .force file is given)
               port:   configs[].atoms[].calc_force
               potfit: the `f` column of <prefix>.force

Exit status is non-zero on any failure (suitable for CI / ctest).

Usage:
  diff_eval.py --port PORT_EVAL.json --ref-log reference.log
               [--ref-force PREFIX.force] [--rtol 1e-6] [--atol 1e-9]
"""
import argparse
import json
import re
import sys


def load_port(path):
    with open(path) as f:
        d = json.load(f)
    return d


def ref_force_sum(path):
    """Parse `sum of force-errors = <x>` from a potfit log or .error file."""
    txt = open(path).read()
    m = re.search(r"sum of force-errors\s*=\s*([0-9eE.+\-]+)", txt)
    if not m:
        sys.exit(f"error: no 'sum of force-errors' found in {path}")
    return float(m.group(1))


def port_force_sum(port):
    """Σ (f_calc − f_ref)² over all atoms / components in the port report."""
    s = 0.0
    for cfg in port["configs"]:
        for atom in cfg["atoms"]:
            for k in range(3):
                d = atom["calc_force"][k] - atom["ref_force"][k]
                s += d * d
    return s


def parse_ref_force(path):
    """Parse a potfit .force file -> {(conf, local_atom): {'calc':[3], 'ref':[3]}}.

    Row format (non-FWEIGHT build):
      <conf>:<gatom>:<c>\t<type>\t<df^2>\t<f>\t<f0>\t<df/f0>
    cols = [type, df^2, f, f0, df/f0]: <f> is the computed force, <f0> the
    reference. NB: <gatom> is the GLOBAL atom index (runs across configs), so we
    convert it to a per-config local index to align with the port's JSON, which
    numbers atoms within each config. The raw <f>/<f0> are weight-free (the
    per-config #W weight only shows up in <df^2>), so comparing them is robust to
    the port not applying #W.
    """
    comp = {"x": 0, "y": 1, "z": 2}
    out = {}
    first = {}  # conf -> first global atom index seen for that conf
    for line in open(path):
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        head, *cols = line.split("\t")
        m = re.match(r"\s*(\d+):\s*(\d+):([xyz])", head)
        if not m or len(cols) < 4:
            continue
        conf, g, c = int(m.group(1)), int(m.group(2)), m.group(3)
        first.setdefault(conf, g)
        key = (conf, g - first[conf])
        rec = out.setdefault(key, {"calc": [None] * 3, "ref": [None] * 3})
        rec["calc"][comp[c]] = float(cols[2])
        rec["ref"][comp[c]] = float(cols[3])
    return out


def ref_force_sum_raw(ref):
    """Weight-free Σ (f_calc − f_ref)² from a parsed .force dict."""
    s = 0.0
    for rec in ref.values():
        for k in range(3):
            d = rec["calc"][k] - rec["ref"][k]
            s += d * d
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="port --evaluate JSON output")
    ap.add_argument("--ref-log", required=True, help="potfit reference.log or .error")
    ap.add_argument("--ref-force", help="potfit <prefix>.force (per-atom check)")
    ap.add_argument("--rtol", type=float, default=1e-5)
    # atol floor reflects potfit's apot 500-point spline-table precision: the port
    # evaluates analytic functions directly, so direct-vs-spline agreement bottoms
    # out around 1e-6 on forces. A real engine bug is O(1e-2+) — well separated.
    ap.add_argument("--atol", type=float, default=1e-5)
    a = ap.parse_args()

    port = load_port(a.port)
    ref = parse_ref_force(a.ref_force) if a.ref_force else None
    ok = True

    # ── 1. scalar sum of squared force residuals (weight-free) ──────────────
    # Prefer the .force file: its raw f/f0 columns are unweighted, so they match
    # the port (which does not apply the per-config #W). The log's
    # `sum of force-errors` is #W-weighted — only used as a fallback.
    p_tot = port_force_sum(port)
    if ref is not None:
        r_tot = ref_force_sum_raw(ref)
        label = "Σ force resid²  "
    else:
        r_tot = ref_force_sum(a.ref_log)
        label = "Σ force resid²* "  # * = #W-weighted fallback
    denom = max(abs(r_tot), a.atol)
    rel = abs(p_tot - r_tot) / denom
    scal_ok = rel <= a.rtol or abs(p_tot - r_tot) <= a.atol
    ok &= scal_ok
    print(f"[{'PASS' if scal_ok else 'FAIL'}] {label} "
          f"port={p_tot:.10g}  potfit={r_tot:.10g}  rel={rel:.3e}")

    # ── 2. per-atom computed forces ─────────────────────────────────────────
    # Per-component pass = numpy-isclose: |Δ| <= atol + rtol·|ref|. This is the
    # right test for forces that span ~0 (perfect-lattice atoms): a tiny absolute
    # diff on a near-zero reference must not fail on a blown-up *relative* error.
    if ref is not None:
        max_abs = 0.0
        n = unmatched = n_fail = 0
        worst = None
        for cfg in port["configs"]:
            ci = cfg["index"]
            for ai, atom in enumerate(cfg["atoms"]):
                rec = ref.get((ci, ai))
                if rec is None:
                    unmatched += 1
                    continue
                for k in range(3):
                    pc = atom["calc_force"][k]
                    rc = rec["calc"][k]
                    if rc is None:
                        continue
                    d = abs(pc - rc)
                    if d > max_abs:
                        max_abs, worst = d, (ci, ai, k)
                    if d > a.atol + a.rtol * abs(rc):
                        n_fail += 1
                    n += 1
        force_ok = n_fail == 0 and unmatched == 0
        ok &= force_ok
        loc = f" worst@cfg{worst[0]}:atom{worst[1]}:c{worst[2]}" if worst else ""
        print(f"[{'PASS' if force_ok else 'FAIL'}] per-atom forces   "
              f"n={n}  unmatched={unmatched}  fail={n_fail}  max|Δ|={max_abs:.3e}{loc}")

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
