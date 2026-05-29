# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import Dict, List

import numpy as np

__author__ = "Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2020"


class Oszicar:
    """
    This module is used to parse VASP OSZICAR files.

    Attributes:

        parse_dict (dict): A dictionary with all the useful quantities parsed from an OSZICAR file after from_file() is
                           executed

    """

    def __init__(self) -> None:
        self.parse_dict: Dict[str, np.ndarray] = dict()

    def from_file(self, filename: str = "OSZICAR") -> None:
        """
        Parse and store relevant quantities from the OSZICAR file into parse_dict.

        The OSZICAR file written by VASP contains the convergence information for each
        electronic and ionic step. This method extracts the free energy (F) printed at
        the end of each ionic step.

        Args:
            filename (str): Path to the OSZICAR file to parse
        """
        with open(filename, "r", errors="ignore") as f:
            lines = f.readlines()
        self.parse_dict["energy_pot"] = self.get_energy_pot(lines)

    @staticmethod
    def get_energy_pot(lines: List[str]) -> np.ndarray:
        """
        Extract the free energy F at the end of each ionic step from OSZICAR lines.

        The OSZICAR file lists the free energy (F=) printed after each ionic step's SCF
        convergence. This value corresponds to the free energy printed in the OUTCAR as
        ``FREE ENERGIE OF THE ION-ELECTRON SYSTEM``.

        Args:
            lines (list): Lines read from the OSZICAR file

        Returns:
            numpy.ndarray: Array of free energies (eV) for each ionic step
        """
        trigger = "F="
        energy_list = []
        for i, line in enumerate(lines):
            line = line.strip()
            if trigger in line:
                energy_list.append(float(lines[i - 1].strip().split()[2]))
        return np.array(energy_list)
