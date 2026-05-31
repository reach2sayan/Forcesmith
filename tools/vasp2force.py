#!/usr/bin/env python3
################################################################
#
# vasp2force:
#   convert VASP OUTCAR data into potfit reference configurations
#
#   This is a port of the original potfit C-version tool
#   (reference/potfit-c/util/vasp2force) for the C++23 port.
#   The OUTCAR-parsing logic is essentially identical; only the
#   output stage differs: instead of the legacy text format
#   (#N/#C/#X/#F ...), it emits the JSON array consumed by the
#   C++ config reader (src/io/config_reader.cpp).
#
#   Top-level output is a JSON array; each configuration is:
#     {
#       "X": [..], "Y": [..], "Z": [..],   # box vectors (Angstrom)
#       "E": <total energy>,               # TOTAL energy (eV), see below
#       "W": <weight>,
#       "S": [xx, yy, zz, xy, yz, zx],     # stress (GPa), optional
#       "atoms": [ {"element": "Cu",
#                   "position": [x, y, z],
#                   "force": [fx, fy, fz]}, ... ]
#     }
#
#   NOTE on energy convention: the original C tool wrote #E as
#   energy-per-atom (total / natoms). The C++ port compares "E"
#   directly against the TOTAL computed energy, so this tool emits
#   the *total* energy (after optional single-atom-energy
#   subtraction via -e). This is the key intentional difference.
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
import os
import sys
from collections import OrderedDict


class ConfigData:
    """Mutable per-configuration scratch state during parsing."""
    MajorIteration = 0
    MinorIteration = 0
    Energy = 0
    Box_x = []
    Box_y = []
    Box_z = []
    Stress = []
    AtomData = []

    def Reset():
        ConfigData.Energy = 0
        ConfigData.Box_x = []
        ConfigData.Box_y = []
        ConfigData.Box_z = []
        ConfigData.Stress = []
        ConfigData.AtomData = []


class vasp_config(object):
    """VASP configuration holder class"""

    max_types = 0
    types = OrderedDict()
    numbers = OrderedDict()
    elements_provided = False
    single_atom_energy = []

    def __init__(self, filename):
        self.filename = filename
        self.__scan_outcar_file(filename)
        vasp_config.max_types = max(vasp_config.max_types, len(self.elements))

    def __scan_outcar_file(self, filename):
        """read metadata from outcar file"""
        if filename.endswith('.gz'):
            f = gzip.open(filename, 'rt')
        else:
            f = open(filename, 'r')
        configs = 0
        atom_types = []
        titel = []
        potcar = []
        ipt = []
        for line in f:
            if line.startswith('|'):
                continue
            if 'TOTAL-FORCE' in line:
                configs += 1
            if 'VRHFIN' in line:
                atom_types.append(line.split()[1].replace('=', '').replace(':', ''))
            if 'TITEL' in line:
                titel.append(line.split()[3][0:2])
            if 'POTCAR' in line:
                potcar.append(line.split()[2][0:2])
            if 'ions per type' in line:
                ipt = [int(s) for s in line.split()[4:]]
        f.close()
        self.nconfig = configs
        self.atoms_per_type = ipt
        self.natoms = sum(self.atoms_per_type)
        if atom_types:
            self.elements = atom_types
        elif titel:
            self.elements = titel
        elif potcar:
            potcar = uniq(potcar)
            self.elements = potcar
        else:
            sys.stderr.write('Could not determine atom types in file %s.\n' % filename)
            sys.exit()

    def read_single_atom_energies(sae_file):
        try:
            lines = 0
            for line in open(sae_file, "r"):
                lines += 1
            if lines != 1:
                sys.stderr.write("Single atom energy file contains more than one line!\n")
                sys.exit()
            with open(sae_file, "r") as f:
                vasp_config.single_atom_energy = [float(x) for x in f.read().split()]
        except SystemExit:
            raise
        except Exception:
            sys.stderr.write("Error reading single atom energy file!\n")
            sys.exit()

    def elements_check(self):
        vasp_config.elements_provided = True
        for element in self.elements:
            if element not in vasp_config.types:
                sys.stderr.write("ERROR: Undefined element in {} detected: {}\n".format(self.filename, element))
                sys.stderr.write("ERROR: Provided element configuration:")
                for key, value in vasp_config.types.items():
                    sys.stderr.write(" {}={}".format(key, value))
                sys.stderr.write("\n")
                sys.exit()

    def print_list(self):
        # diagnostics go to stderr so stdout stays pure JSON
        sys.stderr.write("File {}\n".format(self.filename))
        sys.stderr.write(" number of configurations: {}\n".format(self.nconfig))
        sys.stderr.write(" number of atom types: {}\n".format(len(self.elements)))
        sys.stderr.write(" order of elements: ")
        for i in range(len(self.elements)):
            sys.stderr.write("{}={} ".format(self.elements[i], i))
        sys.stderr.write("\n")

    def set_active_configs(self, configs, final, all):
        """check the config string for sound values"""
        self.configs = []
        if configs is None and final is False and all is False:
            sys.stderr.write("ERROR: No configuration(s) given!\n")
            sys.stderr.write("ERROR: Use the -s, -a or -f arguments to specify configurations\n")
            sys.exit()
        if all is True:
            self.configs = [int(s) + 1 for s in range(self.nconfig)]
            return
        if configs:
            for item in configs.split(','):
                if not item:
                    sys.stderr.write("ERROR: Could not read the -s string.\n")
                    sys.exit()
                if len(item.split('-')) == 1:
                    try:
                        a = int(item)
                    except ValueError:
                        sys.stderr.write("ERROR: Could not read the -s string ({}).\n".format(item))
                        sys.exit()
                    if a < 1:
                        sys.stderr.write("ERROR: Configurations start with index 1!\n")
                        sys.exit()
                    if a <= self.nconfig:
                        self.configs.append(a)
                else:
                    items = item.split('-')
                    if len(items) != 2:
                        sys.stderr.write("ERROR: Could not read the -s string ({}).\n".format(item))
                        sys.exit()
                    else:
                        try:
                            val_min = int(items[0])
                            val_max = int(items[1])
                        except ValueError:
                            sys.stderr.write("ERROR: Could not read the -s string ({}).\n".format(item))
                            sys.exit()
                        if val_min > val_max:
                            val_min = val_max
                            val_max = int(items[0])
                        if val_min < 1 or val_max > self.nconfig:
                            sys.stderr.write("ERROR: Range of the -s string invalid ({}).\n".format(item))
                            sys.stderr.write("ERROR: {} contains {} configurations.\n".format(self.filename, self.nconfig))
                            sys.exit()
                        for i in range(val_min, val_max + 1):
                            # original tool appended to the local arg by mistake;
                            # append to self.configs so ranges actually work.
                            self.configs.append(i)
        if final is True:
            self.configs.append(self.nconfig)

    def __build_config_dict(self):
        """Build the JSON dict for the current configuration, or None."""
        if ConfigData.MajorIteration not in self.configs:
            return None
        cfg = OrderedDict()
        cfg["X"] = [ConfigData.Box_x[0], ConfigData.Box_x[1], ConfigData.Box_x[2]]
        cfg["Y"] = [ConfigData.Box_y[0], ConfigData.Box_y[1], ConfigData.Box_y[2]]
        cfg["Z"] = [ConfigData.Box_z[0], ConfigData.Box_z[1], ConfigData.Box_z[2]]
        cfg["W"] = float(args.weight)
        cfg["E"] = ConfigData.Energy
        if ConfigData.Stress:
            cfg["S"] = [ConfigData.Stress[i] for i in range(6)]
        atoms = []
        for adata in ConfigData.AtomData:
            atom = OrderedDict()
            # adata[0] holds the element symbol (string), set during parsing.
            atom["element"] = adata[0]
            atom["position"] = [adata[1], adata[2], adata[3]]
            atom["force"] = [adata[4], adata[5], adata[6]]
            atoms.append(atom)
        cfg["atoms"] = atoms
        return cfg

    def collect_configs(self, out_list):
        """Parse the OUTCAR and append active configurations to out_list."""
        if self.filename.endswith('.gz'):
            f = gzip.open(self.filename, 'rt')
        else:
            f = open(self.filename, 'r')

        line = f.readline()
        ConfigData.Reset()
        ConfigData.MajorIteration = 1
        ConfigData.MinorIteration = 0
        line = f.readline()
        while line != '':
            line = f.readline()
            if 'Iteration' in line:
                this_iteration = line.split()
                this_iteration.pop()
                del(this_iteration[0])
                del(this_iteration[0])
                if len(this_iteration) == 1:
                    this_iteration = this_iteration.split('(')
                temp = int(this_iteration[0].replace('(', ''))
                ConfigData.MinorIteration = int(this_iteration[1].replace(')', ''))
                if (temp != ConfigData.MajorIteration and ConfigData.AtomData != []):
                    cfg = self.__build_config_dict()
                    if cfg is not None:
                        out_list.append(cfg)
                    ConfigData.MajorIteration = temp
                ConfigData.Reset()
            if 'energy  without' in line:
                # TOTAL energy (eV). Optionally subtract single-atom energies.
                # NOT divided by natoms (C++ port compares against total energy).
                ConfigData.Energy = float(line.split()[6])
                if len(vasp_config.single_atom_energy) > 0:
                    if len(vasp_config.single_atom_energy) < len(self.atoms_per_type):
                        sys.stderr.write("Invalid number of items in single atom energy file.\n")
                        sys.stderr.write("Expected {} but found {}.\n".format(len(self.atoms_per_type), len(vasp_config.single_atom_energy)))
                        sys.exit()
                    for i in range(len(self.atoms_per_type)):
                        ConfigData.Energy -= self.atoms_per_type[i] * vasp_config.single_atom_energy[i]
            if 'VOLUME and BASIS' in line:
                for do in range(5):
                    line = f.readline()
                ConfigData.Box_x = [float(s) for s in line.replace('-', ' -').split()[0:3]]
                line = f.readline()
                ConfigData.Box_y = [float(s) for s in line.replace('-', ' -').split()[0:3]]
                line = f.readline()
                ConfigData.Box_z = [float(s) for s in line.replace('-', ' -').split()[0:3]]
            if 'in kB' in line:
                # stress in kB -> GPa (order: xx yy zz xy yz zx, matches C++ "S")
                ConfigData.Stress = [float(s) / 1602 for s in line.split()[2:8]]
            if 'TOTAL-FORCE' in line:
                line = str(f.readline())
                for element_index in range(len(self.atoms_per_type)):
                    for atom_index in range(self.atoms_per_type[element_index]):
                        vals = [float(s) for s in str(f.readline()).split()[0:6]]
                        # store the element SYMBOL; reader maps symbol -> type idx
                        symbol = self.elements[element_index]
                        adata = [symbol, vals[0], vals[1], vals[2],
                                 vals[3], vals[4], vals[5]]
                        ConfigData.AtomData.append(adata)
        if ConfigData.MinorIteration > 0:
            cfg = self.__build_config_dict()
            if cfg is not None:
                out_list.append(cfg)
        f.close()


# return unique list without changing the order
# from http://stackoverflow.com/questions/480214
def uniq(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]


# walk directory (recursively) and return all OUTCAR* files
def get_outcar_files(directory, recursive):
    sys.stderr.write('Searching directory %s for OUTCAR* files ...\n' % directory)
    outcars = []
    if not recursive:
        for item in os.listdir(directory):
            if item.startswith('OUTCAR'):
                outcars.append(os.path.join(directory, item))
    else:
        for root, SubFolders, files in os.walk(directory):
            for item in files:
                if item.startswith('OUTCAR'):
                    outcars.append(os.path.join(root, item))
    if len(outcars) == 0:
        sys.stderr.write('Could not find any OUTCAR files in this directory.\n')
        sys.exit()
    else:
        sys.stderr.write('Found the following files:\n')
        sys.stderr.write('  {}\n'.format(('\n  ').join(outcars)))
    return outcars


# check for vasp tag
def check_for_vasp_tag(filename):
    if filename.endswith('.gz'):
        try:
            f = gzip.open(filename, 'rt')
        except Exception:
            return False
    else:
        try:
            f = open(filename, 'r')
        except Exception:
            return False
    found_vasp_tag = False
    line = f.readline()
    if 'vasp' in line or 'VASP' in line:
        found_vasp_tag = True
    f.close()
    return found_vasp_tag


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('R|'):
            return text[2:].splitlines()
        return argparse.HelpFormatter._split_lines(self, text, width)


# parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(
        description='Converts VASP OUTCAR data into potfit reference configurations (JSON for the C++ potfit port).',
        formatter_class=SmartFormatter)
    parser.add_argument('-c', type=str, required=False,
                        help='R|list of indices for chemical elements to validate\n'
                             'against (output uses element symbols, so this only\n'
                             'restricts/validates which elements are allowed)\n'
                             'e.g. -c Mg=0,Zn=1', metavar='<elem_list>')
    parser.add_argument('-e', type=str, required=False, help='file with single atom energies', metavar='<sae_file>')
    parser.add_argument('-a', '--all', action='store_true', help='use all configurations from OUTCAR')
    parser.add_argument('-f', '--final', action='store_true', help='use only the final configuration from OUTCAR')
    parser.add_argument('-l', '--list', action='store_true', help='list OUTCAR properties and exit')
    parser.add_argument('-r', '--recursive', action='store_true', help='scan recursively for OUTCAR files')
    parser.add_argument('-s', '--configs', type=str, help="R|comma separated list of configurations to use\n"
                        "supported schemes:\n"
                        " -s 1,4,12         use configs 1, 4 and 12\n"
                        " -s 1,6-9          use configs 1, 6, 7, 8 and 9\n"
                        " -s 1,4,6-9,12     combination of the above")
    parser.add_argument('-w', '--weight', type=float, default=1.0, help='configuration weight for all configurations')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='write JSON to this file instead of stdout', metavar='<file>')
    parser.add_argument('files', type=str, nargs='*', help='R|list of OUTCAR files (plain or gzipped)\n(directory in case of -r option)')
    return parser.parse_args()


# Check for sane arguments
def check_command_line_args(args):
    if args.weight < 0:
        sys.stderr.write("The weight needs to be positive!\n")
        sys.exit()


# determine all OUTCAR files
def generate_outcar_list(file_list):
    outcars = []
    if not file_list:
        outcars = get_outcar_files('.', args.recursive)
    else:
        for item in file_list:
            if os.path.isdir(item):
                outcars += get_outcar_files(item, args.recursive)
            elif os.path.isfile(item):
                outcars.append(os.path.abspath(item))
            else:
                sys.stderr.write("File {} does not exists, removing from file list!\n".format(item))
    # remove duplicate entries from the outcars table
    return uniq(outcars)


# python main function
if __name__ == "__main__":
    args = parse_command_line()
    check_command_line_args(args)
    if args.e:
        vasp_config.read_single_atom_energies(args.e)
    outcar_list = generate_outcar_list(args.files)

    config_list = []
    for outcar_file in outcar_list:
        if not check_for_vasp_tag(outcar_file):
            sys.stderr.write("No VASP data found in file {}, removing from list\n".format(outcar_file))
            continue
        try:
            config_list.append(vasp_config(outcar_file))
        except SystemExit:
            raise
        except Exception:
            # ignore any exceptions from the constructor
            pass
    if len(config_list) == 0:
        sys.stderr.write("No valid OUTCAR files!\n")
        sys.exit()

    if args.list:
        sys.stderr.write(" --- Listing configuration data for OUTCAR files --- \n")
        for config in config_list:
            config.print_list()
        sys.exit()

    if args.c:
        for item in args.c.split(','):
            if len(item.split('=')) != 2:
                sys.stderr.write("ERROR: Could not read the -c string.\n")
                sys.stderr.write("ERROR: Maybe a missing or extra '=' sign?\n")
                sys.exit()
            else:
                try:
                    name = str(item.split('=')[0])
                    number = int(item.split('=')[1])
                except ValueError:
                    sys.stderr.write("ERROR: Could not read the -c string (error in: {})\n".format(item))
                    sys.exit()
                if name in vasp_config.types:
                    sys.stderr.write("ERROR: Duplicate atom type found in -c string\n")
                    sys.stderr.write("ERROR: {} is already assigned to atom type {}\n".format(name, vasp_config.types[name]))
                    sys.exit()
                if number in vasp_config.numbers:
                    sys.stderr.write("ERROR: Duplicate atom number found in -c string\n")
                    sys.stderr.write("ERROR: {} is already assigned to element {}\n".format(number, vasp_config.numbers[number]))
                    sys.exit()
                vasp_config.types[name] = number
                vasp_config.numbers[number] = name
        for config in config_list:
            config.elements_check()

    for config in config_list:
        config.set_active_configs(args.configs, args.final, args.all)

    all_configs = []
    for config in config_list:
        config.collect_configs(all_configs)

    out_text = json.dumps(all_configs, indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(out_text)
            f.write('\n')
        sys.stderr.write("Wrote {} configuration(s) to {}\n".format(len(all_configs), args.output))
    else:
        sys.stdout.write(out_text)
        sys.stdout.write('\n')
