#!/usr/bin/env python3
"""Convert potfit text ``.config`` / ``.dat`` files into the C++ port's config JSON.

These particular potfit configs (H.W. Sheng's Zr/Cu/Al VASP training sets) have
two quirks the generic ``tests/integration/harness/pot2json.py`` does not handle:

  * **No ``#C`` element line.** Atoms carry only a bare integer *type index*
    (0, 1, 2, ...). The index -> element map must be supplied explicitly, per
    input file, as ``PATH:El0,El1,...``. The mapping for this dataset (recovered
    from the per-config VASP source paths) is::

        Zr-Cu-Al.dat : Zr,Cu,Al      cu-zr.dat : Zr,Cu      al-zr.dat : Zr,Al

  * **``#E`` is energy *per atom*** (potfit convention). The port's JSON ``E``
    field is the *total* configuration energy (``--evaluate`` divides by natoms
    for its per-atom panel; the UNEP datasets store total E). So we multiply
    ``#E`` by natoms unless ``--no-energy-scale`` is given.

Other header lines (``#X/#Y/#Z`` box, ``#W`` weight, ``#S`` stress) are carried
through; the ``#M`` magnetic-moment line and the trailing ``#N`` metadata fields
are ignored. Output schema matches ``tools/extxyz2force.py``::

    [{"X":[..],"Y":[..],"Z":[..],"E":<total>,"W":<w>,"S":[6],
      "atoms":[{"element":"Cu","position":[..],"force":[..]}, ...],
      "name":"<source>#<i>"}, ...]

Usage:
    python3 tools/dat2json.py --out data/zr_cu_al/zr_cu_al_dft.json \\
        ~/Downloads/Zr-Cu-Al.dat:Zr,Cu,Al \\
        ~/Downloads/cu-zr.dat:Zr,Cu \\
        ~/Downloads/al-zr.dat:Zr,Al

The merged file concatenates all inputs into a single JSON array. Output is
streamed config-by-config so memory stays flat regardless of dataset size.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def iter_configs(path, elements, scale_energy):
    """Yield port config dicts from one potfit .dat file.

    ``elements`` maps type index -> symbol. ``scale_energy`` multiplies the
    per-atom #E by natoms to get total energy.
    """
    with open(path) as fh:
        lines = fh.readlines()
    src = os.path.basename(path)
    i, n, idx = 0, len(lines), 0
    while i < n:
        if not lines[i].startswith("#N"):
            i += 1
            continue
        natoms = int(lines[i].split()[1])
        box, energy, weight, stress = {}, 0.0, 1.0, None
        i += 1
        # header block: '#' lines until the '#F' marker
        while i < n and not lines[i].startswith("#F"):
            h = lines[i]
            if h.startswith("#X"):
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
            # '#M' and anything else: ignored
            i += 1
        i += 1  # skip the '#F' line
        atoms = []
        for _ in range(natoms):
            parts = lines[i].split()
            tindex = int(parts[0])
            if tindex >= len(elements):
                sys.exit(f"{src}: atom type index {tindex} out of range for "
                         f"element map {elements} (config #{idx})")
            atoms.append({
                "element": elements[tindex],
                "position": [float(parts[1]), float(parts[2]), float(parts[3])],
                "force": [float(parts[4]), float(parts[5]), float(parts[6])],
            })
            i += 1
        cfg = {
            "X": box["X"], "Y": box["Y"], "Z": box["Z"],
            "E": energy * natoms if scale_energy else energy,
            "W": weight,
            "atoms": atoms,
            "name": f"{src}#{idx}",
        }
        if stress is not None:
            cfg["S"] = stress
        yield cfg
        idx += 1


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", metavar="PATH:El0,El1,...",
                   help="potfit .dat file with its type-index -> element map")
    p.add_argument("--out", required=True, help="merged output JSON path")
    p.add_argument("--no-energy-scale", action="store_true",
                   help="do NOT multiply #E by natoms (use if #E is already total)")
    args = p.parse_args(argv)

    specs = []
    for spec in args.inputs:
        if ":" not in spec:
            sys.exit(f"bad input '{spec}': expected PATH:El0,El1,...")
        path, _, elems = spec.rpartition(":")
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            sys.exit(f"input not found: {path}")
        elements = [e.strip() for e in elems.split(",") if e.strip()]
        if not elements:
            sys.exit(f"no elements given for {path}")
        specs.append((path, elements))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Stream the merged JSON array so memory stays flat for large datasets.
    total, per_elem = 0, {}
    with open(out, "w") as fh:
        fh.write("[")
        first = True
        for path, elements in specs:
            count = 0
            for cfg in iter_configs(path, elements, not args.no_energy_scale):
                fh.write("\n" if first else ",\n")
                json.dump(cfg, fh, separators=(",", ":"))
                first = False
                count += 1
                for a in cfg["atoms"]:
                    per_elem[a["element"]] = per_elem.get(a["element"], 0) + 1
            print(f"  {os.path.basename(path):16s} {elements}  -> {count} configs")
            total += count
        fh.write("\n]\n")

    print(f"wrote {out}  ({total} configs)")
    print("atoms per element: " +
          ", ".join(f"{k}={v}" for k, v in sorted(per_elem.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
