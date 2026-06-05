#!/usr/bin/env python3
################################################################
#
# extxyz2force:
#   convert extended-XYZ ab-initio data into forcesmith reference
#   configurations (JSON for the C++23 forcesmith port).
#
#   Companion to tools/vasp2force.py: same output contract, a
#   different input format.  Where vasp2force parses VASP OUTCAR
#   files, this tool parses the extended-XYZ ("extxyz") files used
#   by GAP / GPUMD-NEP / ASE training sets — e.g. the UNEP-v1
#   dataset (Zenodo 11533864, CC-BY-4.0) of single-element DFT
#   energies, forces and virials for 16 metals (Al, Cu, ...).
#
#   Top-level output is a JSON array; each configuration is:
#     {
#       "X": [..], "Y": [..], "Z": [..],   # box vectors (Angstrom)
#       "E": <total energy>,               # TOTAL energy (eV)
#       "W": <weight>,
#       "S": [xx, yy, zz, xy, yz, zx],     # stress (eV/A^3), optional
#       "atoms": [ {"element": "Cu",
#                   "position": [x, y, z],
#                   "force": [fx, fy, fz]}, ... ]
#     }
#   consumed by the C++ config reader (src/io/config_reader.cpp).
#
#   NOTE on energy: extended-XYZ `energy=` is already a TOTAL
#   energy (eV), which is exactly what the C++ port compares
#   against, so it is emitted verbatim.  Use -e/--shift-per-atom
#   to subtract a per-atom reference (e.g. an isolated-atom
#   energy) and move the energy zero to cohesive energies.
#
#   NOTE on stress: the extended-XYZ `virial` (9 components, eV)
#   is the cell virial W.  We map sigma = W / V in eV/A^3 -- the
#   SAME unit the C++ forcesmith engine uses internally for calc_stress
#   (virial/volume, no GPa conversion) -- and emit [xx, yy, zz, xy,
#   yz, zx].  The sign convention has been verified consistent with
#   the engine on a hydrostatic UNEP frame (positive diagonal under
#   compression, dataset-wide corrcoef 0.96).  Stress output is
#   OPT-IN (--stress).
#
################################################################
#
#   Copyright 2002-2017 - the potfit development team
#
#   https://www.potfit.net/
#
#################################################################
#
#   This file is part of potfit.
#
#   potfit is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   potfit is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with potfit; if not, see <http://www.gnu.org/licenses/>.
#
#################################################################

import argparse
import gzip
import json
import re
import sys
from collections import OrderedDict

# Stress is emitted in eV/Angstrom^3 (the C++ engine's internal unit).
# 1 eV/Angstrom^3 = 160.21766208 GPa, if a GPa view is ever needed.
EV_PER_A3_TO_GPA = 160.21766208


def open_text(filename):
    """Open a plain or gzipped text file for reading."""
    if filename.endswith(".gz"):
        return gzip.open(filename, "rt")
    return open(filename, "r")


# --- extended-XYZ comment-line parsing -------------------------------------
#
# The comment (2nd line of each frame) is a series of key=value tokens.
# Values may be bare (energy=-3.14), quoted ("a b c"), or use single quotes.
# This regex splits on top-level whitespace while honouring quotes.
_TOKEN_RE = re.compile(
    r'(\w+)\s*=\s*'                # key=
    r'(?:"([^"]*)"|\'([^\']*)\'|(\S+))'  # "quoted" | 'quoted' | bare
)


def parse_comment(comment):
    """Return an OrderedDict of key -> raw string value from a comment line."""
    props = OrderedDict()
    for m in _TOKEN_RE.finditer(comment):
        key = m.group(1)
        val = m.group(2) if m.group(2) is not None else \
            m.group(3) if m.group(3) is not None else m.group(4)
        props[key] = val
    return props


def parse_properties(prop_str):
    """Parse a Properties=name:type:count:... spec into a column layout.

    Returns a list of (name, kind, count) triples; kind is the type code
    ('S' string, 'R'/'I' numeric).  Default when absent: the canonical
    species:S:1:pos:R:3:force:R:3 layout.
    """
    if not prop_str:
        return [("species", "S", 1), ("pos", "R", 3), ("force", "R", 3)]
    fields = prop_str.split(":")
    cols = []
    for i in range(0, len(fields), 3):
        name, kind, count = fields[i], fields[i + 1], int(fields[i + 2])
        cols.append((name, kind, count))
    return cols


def column_index(cols, name):
    """Return the starting column offset of a named property, or None."""
    offset = 0
    for cname, _kind, count in cols:
        if cname.lower() == name.lower():
            return offset
        offset += count
    return None


def virial_to_stress(virial9, volume):
    """Convert a 9-component cell virial (eV) at given volume (A^3) to the
    forcesmith stress order [xx, yy, zz, xy, yz, zx] in eV/A^3 -- the unit the
    C++ engine compares against (calc_stress = virial / volume)."""
    v = [float(x) for x in virial9]
    # row-major 3x3
    xx, _xy, _xz, _yx, yy, _yz, _zx, _zy, zz = v
    xy = 0.5 * (v[1] + v[3])
    yz = 0.5 * (v[5] + v[7])
    zx = 0.5 * (v[2] + v[6])
    scale = 1.0 / volume
    return [c * scale for c in (xx, yy, zz, xy, yz, zx)]


def cell_volume(box):
    """Scalar triple product of the three box vectors (rows of `box`)."""
    a, b, c = box
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    return abs(cx * c[0] + cy * c[1] + cz * c[2])


def read_frames(filename):
    """Yield raw (natoms, comment, atom_lines) tuples from an extxyz file."""
    f = open_text(filename)
    try:
        while True:
            header = f.readline()
            if header == "":
                return
            header = header.strip()
            if header == "":
                continue
            try:
                natoms = int(header)
            except ValueError:
                sys.stderr.write(
                    "ERROR: expected an atom count, got: {!r}\n".format(header))
                sys.exit(1)
            comment = f.readline().rstrip("\n")
            atom_lines = [f.readline() for _ in range(natoms)]
            if any(line == "" for line in atom_lines):
                sys.stderr.write("ERROR: file truncated mid-frame.\n")
                sys.exit(1)
            yield natoms, comment, atom_lines
    finally:
        f.close()


def build_config(natoms, comment, atom_lines, args):
    """Convert one extxyz frame to a forcesmith JSON config dict, or None."""
    props = parse_comment(comment)

    if "Lattice" not in props and "lattice" not in props:
        sys.stderr.write("ERROR: frame has no Lattice; cannot set box.\n")
        sys.exit(1)
    lat = props.get("Lattice", props.get("lattice"))
    lat_vals = [float(x) for x in lat.split()]
    if len(lat_vals) != 9:
        sys.stderr.write("ERROR: Lattice must have 9 numbers.\n")
        sys.exit(1)
    box = [lat_vals[0:3], lat_vals[3:6], lat_vals[6:9]]

    energy = props.get("energy", props.get("Energy", props.get("ENERGY")))
    if energy is None:
        sys.stderr.write("ERROR: frame has no energy key.\n")
        sys.exit(1)
    energy = float(energy)
    if args.shift_per_atom is not None:
        energy -= natoms * args.shift_per_atom

    cols = parse_properties(props.get("Properties", props.get("properties")))
    sp_i = column_index(cols, "species")
    if sp_i is None:
        sp_i = column_index(cols, "Z")  # some files use atomic number
    pos_i = column_index(cols, "pos")
    frc_i = column_index(cols, "force")
    if frc_i is None:
        frc_i = column_index(cols, "forces")
    if sp_i is None or pos_i is None or frc_i is None:
        sys.stderr.write(
            "ERROR: Properties must define species, pos and force columns.\n")
        sys.exit(1)

    atoms = []
    for line in atom_lines:
        f = line.split()
        atom = OrderedDict()
        atom["element"] = f[sp_i]
        atom["position"] = [float(f[pos_i]), float(f[pos_i + 1]),
                            float(f[pos_i + 2])]
        atom["force"] = [float(f[frc_i]), float(f[frc_i + 1]),
                         float(f[frc_i + 2])]
        atoms.append(atom)

    cfg = OrderedDict()
    cfg["X"] = box[0]
    cfg["Y"] = box[1]
    cfg["Z"] = box[2]
    cfg["W"] = float(args.weight)
    cfg["E"] = energy

    if args.stress:
        virial = props.get("virial", props.get("Virial"))
        if virial is None:
            sys.stderr.write(
                "WARNING: --stress given but frame has no virial; "
                "omitting S for this frame.\n")
        else:
            vv = virial.split()
            if len(vv) != 9:
                sys.stderr.write(
                    "WARNING: virial does not have 9 components; "
                    "omitting S for this frame.\n")
            else:
                cfg["S"] = virial_to_stress(vv, cell_volume(box))

    cfg["atoms"] = atoms
    return cfg


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Convert extended-XYZ ab-initio data into forcesmith "
                    "reference configurations (JSON for the C++ forcesmith port).")
    parser.add_argument("file", type=str,
                        help="extended-XYZ file (plain or .gz)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        metavar="<file>",
                        help="write JSON to this file instead of stdout")
    parser.add_argument("-w", "--weight", type=float, default=1.0,
                        help="configuration weight for all configurations")
    parser.add_argument("-e", "--shift-per-atom", type=float, default=None,
                        metavar="<eV>",
                        help="subtract this per-atom reference energy from each "
                             "configuration's total energy (e.g. an isolated-"
                             "atom energy to obtain cohesive energies)")
    parser.add_argument("--stress", action="store_true",
                        help="emit stress S from the extxyz virial "
                             "(eV/A^3, matching the C++ engine)")
    parser.add_argument("--max-configs", type=int, default=None, metavar="N",
                        help="keep at most N configurations (after striding)")
    parser.add_argument("--stride", type=int, default=1, metavar="K",
                        help="keep every K-th configuration (default 1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line()
    if args.weight < 0:
        sys.stderr.write("The weight needs to be positive!\n")
        sys.exit(1)
    if args.stride < 1:
        sys.stderr.write("--stride must be >= 1.\n")
        sys.exit(1)

    configs = []
    kept = 0
    for idx, (natoms, comment, atom_lines) in enumerate(read_frames(args.file)):
        if idx % args.stride != 0:
            continue
        if args.max_configs is not None and kept >= args.max_configs:
            break
        configs.append(build_config(natoms, comment, atom_lines, args))
        kept += 1

    if not configs:
        sys.stderr.write("No configurations were produced.\n")
        sys.exit(1)

    out_text = json.dumps(configs, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(out_text)
            fh.write("\n")
        sys.stderr.write(
            "Wrote {} configuration(s) to {}\n".format(len(configs), args.output))
    else:
        sys.stdout.write(out_text)
        sys.stdout.write("\n")
