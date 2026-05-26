# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import re
import warnings
from collections import OrderedDict
from typing import Optional

import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixCartesian

__author__ = "Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


def read_atoms(
    filename: str="CONTCAR",
    return_velocities: bool=False,
    species_list: Optional[list[str]] = None,
    species_from_potcar: bool=False,
):
    """
    Routine to read structural static from a POSCAR type file

    Args:
        filename (str): Input filename
        return_velocities (bool): True if the predictor corrector velocities are read (only from MD output)
        species_list (list/numpy.ndarray): A list of the species (if not present in the POSCAR file or a POTCAR in the
        same directory)
        species_from_potcar (bool): True if the species list should be read from the POTCAR file in the same directory

    Returns:
        pyiron.atomistics.structure.atoms.Atoms: The generated structure object

    """
    potcar_file = os.path.join(os.path.dirname(filename), "POTCAR")
    if (species_list is None) and species_from_potcar:
        species_list = get_species_list_from_potcar(potcar_file)
        if len(species_list) == 0:
            warnings.warn(
                "Warning! Unable to read species information from POTCAR", stacklevel=2
            )
    file_string = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            file_string.append(line)
    return atoms_from_string(
        file_string, read_velocities=return_velocities, species_list=species_list
    )


def get_species_list_from_potcar(filename: str="POTCAR"):
    """
    Generates the species list from a POTCAR type file

    Args:
        filename (str): Input filename

    Returns:
        list: A list of species symbols

    """
    trigger = "VRHFIN ="
    species_list = []
    with open(filename, errors="ignore") as potcar_file:
        lines = potcar_file.readlines()
        for line in lines:
            line = line.strip()
            if trigger in line:
                str_1 = line.split(trigger)
                str_2 = str_1[-1].split(":")
                species_list.append(str_2[0].replace(" ", ""))
    return species_list


def get_number_species_atoms(structure: Atoms):
    """
    Returns a dictionary with the species in the structure and the corresponding count in the structure

    Args:
        structure

    Returns:
        collections.OrderedDict: An ordered dictionary with the species and the corresponding count

    """
    count = OrderedDict()
    # print "sorted: ", sorted(set(self.elements))
    for el in sorted(set(structure.get_chemical_symbols())):
        count[el] = 0

    for el in structure.get_chemical_symbols():
        count[el] += 1
    return count


def write_poscar(structure: Atoms, filename: str="POSCAR", write_species: bool=True, cartesian: bool=True):
    """
    Writes a POSCAR type file from a structure object

    Args:
        structure (pyiron.atomistics.structure.atoms.Atoms): The structure instance to be written to the POSCAR format
        filename (str): Output filename
        write_species (bool): True if the species should be written to the file
        cartesian (bool): True if the positions are written in Cartesian coordinates

    """
    with open(filename, "w") as f:
        f.writelines(
            "".join(
                get_poscar_content(
                    structure=structure,
                    write_species=write_species,
                    cartesian=cartesian,
                )
            )
        )


def get_poscar_content(structure: Atoms, write_species: bool=True, cartesian: bool=True):
    endline = "\n"
    selec_dyn = False
    line_lst = [
        "Poscar file generated with pyiron" + endline,
        "1.0" + endline,
    ]
    for a_i in structure.get_cell():
        x, y, z = a_i
        line_lst.append(f"{x:.15f} {y:.15f} {z:.15f}" + endline)
    atom_numbers = get_number_species_atoms(structure=structure)
    if write_species:
        line_lst.append(" ".join(atom_numbers.keys()) + endline)
    num_str = [str(val) for val in atom_numbers.values()]
    line_lst.append(" ".join(num_str))
    line_lst.append(endline)
    if len(structure.constraints) > 0:
        selective_dynamics = np.array([[True, True, True]] * len(structure))
        for con in structure.constraints:
            c = con.todict()
            if c["name"] != "FixCartesian":
                raise ValueError(
                    "Currently only FixCartesian is supported in write_poscar()."
                )
            for ind in c["kwargs"]["a"]:
                selective_dynamics[ind] = np.invert(c["kwargs"]["mask"])
        selec_dyn = True
        cartesian = False
        line_lst.append("Selective dynamics" + endline)
    sorted_coords = []
    selec_dyn_lst = []
    for species in atom_numbers:
        indices = structure.symbols.indices()[species]
        for i in indices:
            if cartesian:
                sorted_coords.append(structure.positions[i])
            else:
                sorted_coords.append(structure.get_scaled_positions()[i])
            if selec_dyn:
                selec_dyn_lst.append(selective_dynamics[i])
    if cartesian:
        line_lst.append("Cartesian" + endline)
    else:
        line_lst.append("Direct" + endline)
    if selec_dyn:
        for i, vec in enumerate(sorted_coords):
            x, y, z = vec
            sd_string = " ".join(["T" if sd else "F" for sd in selec_dyn_lst[i]])
            line_lst.append(f"{x:.15f} {y:.15f} {z:.15f}" + " " + sd_string + endline)
    else:
        for i, vec in enumerate(sorted_coords):
            x, y, z = vec
            line_lst.append(f"{x:.15f} {y:.15f} {z:.15f}" + endline)
    return line_lst


def atoms_from_string(string: list[str], read_velocities: bool=False, species_list: Optional[list[str]] = None):
    """
    Routine to convert a string list read from a input/output structure file and convert into Atoms instance

    Args:
        string (list): A list of strings (lines) read from the POSCAR/CONTCAR/CHGCAR/LOCPOT file
        read_velocities (bool): True if the velocities from a CONTCAR file should be read (predictor corrector)
        species_list (list/numpy.ndarray): A list of species of the atoms

    Returns:
        pyiron.atomistics.structure.atoms.Atoms: The required structure object

    """
    string = [s.strip() for s in string]
    string_lower = [s.lower() for s in string]
    atoms_dict = {}
    atoms_dict["first_line"] = string[0]
    # del string[0]
    atoms_dict["selective_dynamics"] = False
    atoms_dict["relative"] = False
    if "direct" in string_lower or "d" in string_lower:
        atoms_dict["relative"] = True
    atoms_dict["scaling_factor"] = float(string[1])
    unscaled_cell = []
    for i in [2, 3, 4]:
        vec = []
        for j in range(3):
            vec.append(float(string[i].split()[j]))
        unscaled_cell.append(vec)
    if atoms_dict["scaling_factor"] > 0.0:
        atoms_dict["cell"] = np.array(unscaled_cell) * atoms_dict["scaling_factor"]
    else:
        atoms_dict["cell"] = np.array(unscaled_cell) * (
            (-atoms_dict["scaling_factor"]) ** (1.0 / 3.0)
        )
    if "selective dynamics" in string_lower:
        atoms_dict["selective_dynamics"] = True
    no_of_species = len(string[5].split())
    species_dict = OrderedDict()
    position_index = 7
    if atoms_dict["selective_dynamics"]:
        position_index += 1
    for i in range(no_of_species):
        species_dict["species_" + str(i)] = {}
        try:
            species_dict["species_" + str(i)]["count"] = int(string[5].split()[i])
        except ValueError:
            species_dict["species_" + str(i)]["species"] = string[5].split()[i]
            species_dict["species_" + str(i)]["count"] = int(string[6].split()[i])
    atoms_dict["species_dict"] = species_dict
    if "species" in atoms_dict["species_dict"]["species_0"]:
        position_index += 1
    positions = []
    selective_dynamics = []
    n_atoms = sum(
        [atoms_dict["species_dict"][key]["count"] for key in atoms_dict["species_dict"]]
    )
    try:
        for i in range(position_index, position_index + n_atoms):
            string_list = np.array(string[i].split())
            positions.append([float(val) for val in string_list[0:3]])
            if atoms_dict["selective_dynamics"]:
                selective_dynamics.append(["T" in val for val in string_list[3:6]])
    except (ValueError, IndexError):
        raise AssertionError(
            "The number of positions given does not match the number of atoms"
        )
    atoms_dict["positions"] = np.array(positions)
    if not atoms_dict["relative"]:
        if atoms_dict["scaling_factor"] > 0.0:
            atoms_dict["positions"] *= atoms_dict["scaling_factor"]
        else:
            atoms_dict["positions"] *= (-atoms_dict["scaling_factor"]) ** (1.0 / 3.0)
    velocities = []
    try:
        atoms = _dict_to_atoms(atoms_dict, species_list=species_list)
    except ValueError:
        atoms = _dict_to_atoms(atoms_dict, read_from_first_line=True)
    if atoms_dict["selective_dynamics"]:
        constraints_dict = {
            label: []
            for label in ["TTT", "TTF", "FTT", "TFT", "TFF", "FFT", "FTF", "FFF"]
        }
        for i, val in enumerate(selective_dynamics):
            if val[0] and val[1] and val[2]:
                constraints_dict["TTT"].append(i)
            elif val[0] and val[1] and not val[2]:
                constraints_dict["TTF"].append(i)
            elif not val[0] and val[1] and val[2]:
                constraints_dict["FTT"].append(i)
            elif val[0] and not val[1] and val[2]:
                constraints_dict["TFT"].append(i)
            elif val[0] and not val[1] and not val[2]:
                constraints_dict["TFF"].append(i)
            elif not val[0] and val[1] and not val[2]:
                constraints_dict["FTF"].append(i)
            elif not val[0] and not val[1] and val[2]:
                constraints_dict["FFT"].append(i)
            elif not val[0] and not val[1] and not val[2]:
                constraints_dict["FFF"].append(i)
            else:
                raise ValueError("Selective Dynamics Error: " + str(val))

        constraints_lst = []
        for k, v in constraints_dict.items():
            if len(v) > 0:
                if k == "TTT":
                    constraints_lst.append(
                        FixCartesian(a=v, mask=(False, False, False))
                    )
                elif k == "TTF":
                    constraints_lst.append(FixCartesian(a=v, mask=(False, False, True)))
                elif k == "FTT":
                    constraints_lst.append(FixCartesian(a=v, mask=(True, False, False)))
                elif k == "TFT":
                    constraints_lst.append(FixCartesian(a=v, mask=(False, True, False)))
                elif k == "TFF":
                    constraints_lst.append(FixCartesian(a=v, mask=(False, True, True)))
                elif k == "FTF":
                    constraints_lst.append(FixCartesian(a=v, mask=(True, False, True)))
                elif k == "FFT":
                    constraints_lst.append(FixCartesian(a=v, mask=(True, True, False)))
                elif k == "FFF":
                    constraints_lst.append(FixCartesian(a=v, mask=(True, True, True)))
                else:
                    raise ValueError(
                        "Selective Dynamics Error: " + str(k) + ": " + str(v)
                    )

        atoms.set_constraint(constraints_lst)

    if read_velocities:
        velocity_index = position_index + n_atoms + 1
        for i in range(velocity_index, velocity_index + n_atoms):
            try:
                velocities.append([float(val) for val in string[i].split()[0:3]])
            except IndexError:
                break
        if len(velocities) != n_atoms:
            warnings.warn(
                "The velocities are either not available or they are incomplete/corrupted. Returning empty "
                "list instead",
                UserWarning,
                stacklevel=2,
            )
            return atoms, []
        return atoms, velocities
    else:
        return atoms


def _dict_to_atoms(atoms_dict: dict, species_list: Optional[list[str]] = None, read_from_first_line: bool = False):
    """
    Function to convert a generated dict into an structure object

    Args:
        atoms_dict (dict): Dictionary with the details (from string_to_atom)
        species_list (list/numpy.ndarray): List of species
        read_from_first_line (bool): True if we are to read the species information from the first line in the file

    Returns:
        pyiron.atomistics.structure.atoms.Atoms: The required structure object
    """
    is_absolute = not (atoms_dict["relative"])
    positions = atoms_dict["positions"]
    cell = atoms_dict["cell"]
    symbol = ""
    elements = []
    el_list = []
    for i, sp_key in enumerate(atoms_dict["species_dict"].keys()):
        if species_list is not None:
            try:
                el_list = np.array([species_list[i]])
                el_list = np.tile(el_list, atoms_dict["species_dict"][sp_key]["count"])
                if isinstance(species_list[i], str):
                    symbol += species_list[i] + str(
                        atoms_dict["species_dict"][sp_key]["count"]
                    )
                else:
                    symbol += species_list[i].Abbreviation + str(
                        atoms_dict["species_dict"][sp_key]["count"]
                    )
            except IndexError:
                raise ValueError(
                    "Number of species in the specified species list does not match that in the file"
                )
        elif "species" in atoms_dict["species_dict"][sp_key]:
            el_list = np.array([atoms_dict["species_dict"][sp_key]["species"]])
            el_list = np.tile(el_list, atoms_dict["species_dict"][sp_key]["count"])
            symbol += atoms_dict["species_dict"][sp_key]["species"]
            symbol += str(atoms_dict["species_dict"][sp_key]["count"])
        elif read_from_first_line:
            if not (
                len(atoms_dict["first_line"].split())
                == len(atoms_dict["species_dict"].keys())
            ):
                raise AssertionError()
            el_list = np.array(atoms_dict["first_line"].split()[i])
            el_list = np.tile(el_list, atoms_dict["species_dict"][sp_key]["count"])
            symbol += atoms_dict["first_line"].split()[i]
            symbol += str(atoms_dict["species_dict"][sp_key]["count"])
        elif species_list is None:
            raise ValueError(
                "Species list should be provided since pyiron can't detect species information"
            )
        elements.append(el_list)
    elements_new = []
    for ele in elements:
        for e in ele:
            elements_new.append(re.split("[^a-zA-Z]", e)[0])
    elements = elements_new
    if is_absolute:
        atoms = Atoms(elements, positions=positions, cell=cell, pbc=True)
    else:
        atoms = Atoms(elements, scaled_positions=positions, cell=cell, pbc=True)
    return atoms


def vasp_sorter(structure: Atoms):
    """
    Routine to sort the indices of a structure as it would be when written to a POSCAR file

    Args:
        structure (pyiron.atomistics.structure.atoms.Atoms): The structure whose indices need to be sorted

    Returns:
        list: A list of indices which is sorted by the corresponding species for writing to POSCAR

    """
    atom_numbers = get_number_species_atoms(structure)
    sorted_indices = []
    for species in atom_numbers:
        indices = structure.symbols.indices()[species]
        for i in indices:
            sorted_indices.append(i)
    return np.array(sorted_indices)


def manip_contcar(filename: str, new_filename: str, add_pos: list | np.ndarray):
    """
    Manipulate a CONTCAR/POSCAR file by adding something to the positions

    Args:
        filename (str):  Filename/path of the input file
        new_filename (str): Filename/path of the output file
        add_pos (list/numpy.ndarray): Array of values to be added to the positions of the input

    """
    actual_struct = read_atoms(filename)
    n = 0
    direct = True
    with open(filename, errors="ignore") as f:
        lines = f.readlines()
        for line in lines:
            if "Direct" in line or "Cartesian" in line:
                direct = "Direct" in line
                break
            n += 1
    pos_list = []
    sd_list = []
    if len(lines[n + 1].split()) == 6:
        for line in lines[n + 1 : n + 1 + len(actual_struct)]:
            pos_list.append([float(val) for val in line.split()[0:3]])
            sd_list.append(["T" in val for val in line.split()[3:]])
    else:
        for line in lines[n + 1 : n + 1 + len(actual_struct)]:
            pos_list.append([float(val) for val in line.split()[0:3]])
    old_pos = np.array(pos_list)
    if direct:
        add_pos_rel = np.dot(add_pos, np.linalg.inv(actual_struct.cell))
        new_pos = old_pos + add_pos_rel
    else:
        new_pos = old_pos + add_pos
    new_lines = lines
    new_pos_str = np.array(new_pos, dtype=str)
    if len(sd_list) > 0:
        bool_list = np.zeros_like(old_pos, dtype=str)
        bool_list[:] = "F"
        bool_list[np.array(sd_list)] = "T"
        for i, pos in enumerate(new_pos_str):
            linestr = np.append(pos, bool_list[i])
            new_lines[n + 1 + i] = " ".join([str(val) for val in linestr]) + "\n"
    else:
        for i, pos in enumerate(new_pos_str):
            linestr = pos
            new_lines[n + 1 + i] = " ".join([str(val) for val in linestr]) + "\n"

    # Exclude predictor corrector positions
    if len(new_lines[n + len(new_pos_str) :]) >= 2 * len(new_pos_str):
        new_lines = new_lines[: n + 2 * (len(new_pos_str)) + 2]

    with open(new_filename, "w") as f:
        for new_line in new_lines:
            f.write(new_line)
