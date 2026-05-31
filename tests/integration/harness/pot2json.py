#!/usr/bin/env python3
"""Convert original (text) potfit inputs into the C++ port's JSON formats.

The original C potfit reads text `.config` / `.pot` files; the C++ port reads
JSON. This converter lets a single source-of-truth potential + config set feed
both codes so their evaluation results can be diffed (see README.md).

Subcommands
-----------
  config   <in.config>  <out.json>              potfit #N/#C/#F  -> port config JSON
  startpot <in.startpot> <out.json> --ntypes N  potfit apot (#T EAM) -> port startpot JSON

Only analytic (apot, format-0) start-potentials are converted, and only the
function types listed in TYPE_PARAMS below (extend as needed). Tabulated
potfit potentials (format 3/4) are not handled here.
"""
import argparse
import json
import sys

# potfit apot function name -> (port analytic type, ordered port param names).
# Param order mirrors src/io/potential_reader.cpp's registry, which matches the
# order potfit prints the parameters in, so values map positionally.
TYPE_PARAMS = {
    "lj":           ("lj",           ["epsilon", "sigma"]),
    "morse":        ("morse",        ["De", "a", "re"]),
    "eopp":         ("eopp",         ["A", "n", "B", "m", "k", "phi"]),
    "buck":         ("buck",         ["A", "rho", "C"]),
    "power":        ("power",        ["A", "n"]),
    "exp_decay":    ("exp_decay",    ["A", "B"]),
    "sqrt":         ("sqrt",         ["A", "B"]),
    # smooth-cutoff variants: same params as the base + a trailing `h`
    "lj_sc":        ("lj_sc",        ["epsilon", "sigma", "h"]),
    "morse_sc":     ("morse_sc",     ["De", "a", "re", "h"]),
    "eopp_sc":      ("eopp_sc",      ["A", "n", "B", "m", "k", "phi", "h"]),
    "exp_decay_sc": ("exp_decay_sc", ["A", "B", "h"]),
}


def convert_config(src: str) -> list:
    """potfit text config file -> list of port configuration objects."""
    configs = []
    lines = src.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line.startswith("#N"):
            i += 1
            continue
        natoms = int(line.split()[1])
        elements, box, energy, weight, stress = [], {}, 0.0, 1.0, None
        i += 1
        # header block: lines starting with '#', until the '#F' marker
        while i < n and not lines[i].startswith("#F"):
            h = lines[i].strip()
            if h.startswith("#C"):
                elements = h.split()[1:]
            elif h.startswith("#X"):
                box["X"] = [float(x) for x in h.split()[1:4]]
            elif h.startswith("#Y"):
                box["Y"] = [float(x) for x in h.split()[1:4]]
            elif h.startswith("#Z"):
                box["Z"] = [float(x) for x in h.split()[1:4]]
            elif h.startswith("#E"):
                energy = float(h.split()[1])
            elif h.startswith("#W"):
                weight = float(h.split()[1])
            elif h.startswith("#S"):
                stress = [float(x) for x in h.split()[1:7]]
            i += 1
        i += 1  # skip the '#F' line
        atoms = []
        for _ in range(natoms):
            parts = lines[i].split()
            tindex = int(parts[0])
            elem = elements[tindex] if tindex < len(elements) else str(tindex)
            atoms.append({
                "element": elem,
                "position": [float(parts[1]), float(parts[2]), float(parts[3])],
                "force": [float(parts[4]), float(parts[5]), float(parts[6])],
            })
            i += 1
        cfg = {"X": box["X"], "Y": box["Y"], "Z": box["Z"],
               "E": energy, "W": weight, "atoms": atoms}
        if stress is not None:
            cfg["S"] = stress
        configs.append(cfg)
    return configs


def _parse_globals(src: str) -> dict:
    """Parse a potfit `global N` block -> {name: {value, min, max}}."""
    globals_ = {}
    lines = src.splitlines()
    for i, raw in enumerate(lines):
        tok = raw.strip().split()
        if tok and tok[0] == "global":
            n = int(tok[1])
            for g in lines[i + 1:i + 1 + n]:
                gt = g.split()
                globals_[gt[0]] = {"value": float(gt[1]), "min": float(gt[2]),
                                   "max": float(gt[3])}
            break
    return globals_


def _parse_apot_functions(src: str) -> list:
    """Parse potfit apot text -> [{type, cutoff, params}] where each param is
    either a float value or a {"global": name} reference (from a `name!` line)."""
    funcs = []
    cur = None
    in_global = 0  # skip lines belonging to a `global N` block
    for raw in src.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        if tok[0] == "global":
            in_global = int(tok[1])
            continue
        if in_global > 0:
            in_global -= 1            # consume a global-definition line
            continue
        if tok[0] == "type":
            if cur is not None:
                funcs.append(cur)
            cur = {"type": tok[1], "cutoff": None, "params": []}
        elif tok[0] == "cutoff" and cur is not None:
            cur["cutoff"] = float(tok[1])
        elif cur is not None:
            # parameter line: `name value [lo hi]`, or `name!` (global ref).
            if tok[0].endswith("!"):
                cur["params"].append({"global": tok[0][:-1]})
            else:
                cur["params"].append(float(tok[1]))
    if cur is not None:
        funcs.append(cur)
    return funcs


def _to_port_pot(fn: dict, rmin: float) -> dict:
    name = fn["type"]
    if name not in TYPE_PARAMS:
        sys.exit(f"error: unsupported apot function '{fn['type']}' "
                 f"(add it to TYPE_PARAMS).")
    port_type, pnames = TYPE_PARAMS[name]
    if len(fn["params"]) != len(pnames):
        sys.exit(f"error: '{fn['type']}' expects {len(pnames)} params, "
                 f"got {len(fn['params'])}")
    obj = {"type": port_type, "rmin": rmin, "rmax": fn["cutoff"]}
    obj.update(dict(zip(pnames, fn["params"])))  # value or {"global": name}
    return obj


def convert_startpot(src: str, ntypes: int, rmin: float) -> dict:
    """potfit apot EAM start-potential -> port EAM startpot JSON."""
    funcs = _parse_apot_functions(src)
    globals_ = _parse_globals(src)
    npair = ntypes * (ntypes + 1) // 2
    expected = npair + 2 * ntypes  # pair + density + embedding
    if len(funcs) != expected:
        sys.exit(f"error: EAM with ntypes={ntypes} needs {expected} functions "
                 f"({npair} pair + {ntypes} density + {ntypes} embedding); "
                 f"found {len(funcs)}")
    pair = funcs[:npair]
    density = funcs[npair:npair + ntypes]
    embedding = funcs[npair + ntypes:]
    block = lambda group: {"format": "analytic",
                           "potentials": [_to_port_pot(f, rmin) for f in group]}
    out = {"model": "eam", "ntypes": ntypes,
           "pair": block(pair), "density": block(density),
           "embedding": block(embedding)}
    if globals_:
        out["globals"] = globals_
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("config", help="convert a potfit text config to port JSON")
    c.add_argument("infile"); c.add_argument("outfile")

    s = sub.add_parser("startpot", help="convert a potfit apot startpot to port JSON")
    s.add_argument("infile"); s.add_argument("outfile")
    s.add_argument("--ntypes", type=int, required=True)
    s.add_argument("--rmin", type=float, default=0.0,
                   help="analytic-function rmin (potfit apot has no per-fn rmin; "
                        "default 0.0 — this is a key parity knob, see README)")

    a = ap.parse_args()
    src = open(a.infile).read()
    if a.cmd == "config":
        out = convert_config(src)
    else:
        out = convert_startpot(src, a.ntypes, a.rmin)
    with open(a.outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.outfile}")


if __name__ == "__main__":
    main()
