import contextlib
import os
import posixpath
import warnings
from typing import Callable, Optional, Sequence, Union

import numpy as np
from ase.atoms import Atoms

from vaspparser.dft.bader import Bader
from vaspparser.dft.waves.electronic import ElectronicStructure
from vaspparser.vasp.parser.oszicar import Oszicar
from vaspparser.vasp.parser.outcar import Outcar, OutcarCollectError
from vaspparser.vasp.procar import Procar
from vaspparser.vasp.structure import read_atoms, vasp_sorter
from vaspparser.vasp.vasprun import Vasprun as Vr
from vaspparser.vasp.vasprun import VasprunError, VasprunWarning
from vaspparser.vasp.volumetric_data import VaspVolumetricData


class Output:
    """
    Handles the output from a VASP simulation.

    Attributes:
        electronic_structure: Gives the electronic structure of the system
        electrostatic_potential: Gives the electrostatic/local potential of the system
        charge_density: Gives the charge density of the system
    """

    def __init__(self):
        self._structure = None
        self.outcar = Outcar()
        self.oszicar = Oszicar()
        self.generic_output = GenericOutput()
        self.description = (
            "This contains all the output static from this particular vasp run"
        )
        self.charge_density = VaspVolumetricData()
        self.electrostatic_potential = VaspVolumetricData()
        self.procar = Procar()
        self.electronic_structure = ElectronicStructure()
        self.vp_new = Vr()

    @property
    def structure(self):
        """
        Getter for the output structure
        """
        return self._structure

    @structure.setter
    def structure(self, atoms: Atoms):
        """
        Setter for the output structure
        """
        self._structure = atoms

    def collect(
        self,
        directory: str = os.getcwd(),
        sorted_indices: Optional[np.ndarray] = None,
        es_class=ElectronicStructure,
    ):
        """
        Collects output from the working directory

        Args:
            directory (str): Path to the directory
            sorted_indices (np.array/None):
        """
        if sorted_indices is None:
            sorted_indices = vasp_sorter(self.structure)
        files_present = os.listdir(directory)
        log_dict = {}
        vasprun_working, outcar_working = False, False
        if not ("OUTCAR" in files_present or "vasprun.xml" in files_present):
            raise OSError("Either the OUTCAR or vasprun.xml files need to be present")
        if "OSZICAR" in files_present:
            self.oszicar.from_file(filename=posixpath.join(directory, "OSZICAR"))
        if "OUTCAR" in files_present:
            try:
                self.outcar.from_file(filename=posixpath.join(directory, "OUTCAR"))
                outcar_working = True
            except OutcarCollectError as e:
                warnings.warn(
                    f"OUTCAR present, but could not be parsed: {e}!", stacklevel=2
                )
                outcar_working = False
        if "vasprun.xml" in files_present:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    self.vp_new.from_file(
                        filename=posixpath.join(directory, "vasprun.xml")
                    )
                    if any(isinstance(warn.category, VasprunWarning) for warn in w):
                        warnings.warn(
                            "vasprun.xml parsed but with some inconsistencies. "
                            "Check vasp output to be sure",
                            VasprunWarning,
                            stacklevel=2,
                        )
            except VasprunError:
                warnings.warn(
                    "Unable to parse the vasprun.xml file. Will attempt to get data from OUTCAR",
                    stacklevel=2,
                )
            else:
                # If parsing the vasprun file does not throw an error, then set to True
                vasprun_working = True
        if outcar_working:
            log_dict["temperature"] = self.outcar.parse_dict["temperatures"]
            log_dict["stresses"] = self.outcar.parse_dict["stresses"]
            log_dict["pressures"] = self.outcar.parse_dict["pressures"]
            log_dict["elastic_constants"] = self.outcar.parse_dict["elastic_constants"]
            self.generic_output.dft_log_dict["n_elect"] = self.outcar.parse_dict[
                "n_elect"
            ]
            if len(self.outcar.parse_dict["magnetization"]) > 0:
                magnetization = np.array(
                    self.outcar.parse_dict["magnetization"], dtype=object
                )
                final_magmoms = np.array(
                    self.outcar.parse_dict["final_magmoms"], dtype=object
                )
                # magnetization[sorted_indices] = magnetization.copy()
                if len(final_magmoms) != 0:
                    if len(final_magmoms.shape) == 3:
                        final_magmoms[:, sorted_indices, :] = final_magmoms.copy()
                    else:
                        final_magmoms[:, sorted_indices] = final_magmoms.copy()
                self.generic_output.dft_log_dict["magnetization"] = (
                    magnetization.tolist()
                )
                self.generic_output.dft_log_dict["final_magmoms"] = (
                    final_magmoms.tolist()
                )
            self.generic_output.dft_log_dict["e_fermi_list"] = self.outcar.parse_dict[
                "e_fermi_list"
            ]
            self.generic_output.dft_log_dict["vbm_list"] = self.outcar.parse_dict[
                "vbm_list"
            ]
            self.generic_output.dft_log_dict["cbm_list"] = self.outcar.parse_dict[
                "cbm_list"
            ]

        if vasprun_working:
            log_dict["forces"] = self.vp_new.vasprun_dict["forces"]
            log_dict["cells"] = self.vp_new.vasprun_dict["cells"]
            log_dict["volume"] = np.linalg.det(self.vp_new.vasprun_dict["cells"])
            # The vasprun parser also returns the energies printed again after the final SCF cycle under the key
            # "total_energies", but due to a bug in the VASP output, the energies reported there are wrong in Vasp 5.*;
            # instead use the last energy from the scf cycle energies
            # BUG link: https://ww.vasp.at/forum/viewtopic.php?p=19242
            try:
                # bug report is not specific to which Vasp5 versions are affected; be safe and workaround for all of
                # them
                is_vasp5 = self.vp_new.vasprun_dict["generator"]["version"].startswith(
                    "5."
                )
            except KeyError:  # in case the parser didn't read the version info
                is_vasp5 = True
            if is_vasp5:
                log_dict["energy_pot"] = np.array(
                    [e[-1] for e in self.vp_new.vasprun_dict["scf_fr_energies"]]
                )
            else:
                # total energies refers here to the total energy of the electronic system, not the total system of
                # electrons plus (potentially) moving ions; hence this is the energy_pot
                log_dict["energy_pot"] = self.vp_new.vasprun_dict["total_fr_energies"]
            if "kinetic_energies" in self.vp_new.vasprun_dict:
                log_dict["energy_tot"] = (
                    log_dict["energy_pot"]
                    + self.vp_new.vasprun_dict["kinetic_energies"]
                )
            else:
                log_dict["energy_tot"] = log_dict["energy_pot"]
            log_dict["steps"] = np.arange(len(log_dict["energy_tot"]))
            log_dict["positions"] = self.vp_new.vasprun_dict["positions"]
            log_dict["forces"][:, sorted_indices] = log_dict["forces"].copy()
            log_dict["positions"][:, sorted_indices] = log_dict["positions"].copy()
            log_dict["positions"] = np.einsum(
                "nij,njk->nik", log_dict["positions"], log_dict["cells"]
            )
            # log_dict["scf_energies"] = self.vp_new.vasprun_dict["scf_energies"]
            # log_dict["scf_dipole_moments"] = self.vp_new.vasprun_dict["scf_dipole_moments"]
            self.electronic_structure = self.vp_new.get_electronic_structure(
                es_class=es_class,
            )
            if self.electronic_structure.grand_dos_matrix is not None:
                self.electronic_structure.grand_dos_matrix[
                    :, :, :, sorted_indices, :
                ] = self.electronic_structure.grand_dos_matrix[:, :, :, :, :].copy()
            if self.electronic_structure.resolved_densities is not None:
                self.electronic_structure.resolved_densities[
                    :, sorted_indices, :, :
                ] = self.electronic_structure.resolved_densities[:, :, :, :].copy()
            self.structure.positions = log_dict["positions"][-1]
            self.structure.set_cell(log_dict["cells"][-1])
            self.generic_output.dft_log_dict["potentiostat_output"] = (
                self.vp_new.get_potentiostat_output()
            )
            valence_charges_orig = self.vp_new.get_valence_electrons_per_atom()
            valence_charges = valence_charges_orig.copy()
            valence_charges[sorted_indices] = valence_charges_orig
            self.generic_output.dft_log_dict["valence_charges"] = valence_charges

        elif outcar_working:
            # log_dict = self.outcar.parse_dict.copy()
            if len(self.outcar.parse_dict["energies"]) == 0:
                raise VaspCollectError("Error in parsing OUTCAR")
            log_dict["energy_tot"] = self.outcar.parse_dict["energies"]
            log_dict["temperature"] = self.outcar.parse_dict["temperatures"]
            log_dict["stresses"] = self.outcar.parse_dict["stresses"]
            log_dict["pressures"] = self.outcar.parse_dict["pressures"]
            log_dict["forces"] = self.outcar.parse_dict["forces"]
            log_dict["positions"] = self.outcar.parse_dict["positions"]
            log_dict["forces"][:, sorted_indices] = log_dict["forces"].copy()
            log_dict["positions"][:, sorted_indices] = log_dict["positions"].copy()
            if len(log_dict["positions"].shape) != 3 or log_dict["positions"].shape[
                1
            ] != len(sorted_indices):
                raise VaspCollectError("Improper OUTCAR parsing")
            if len(log_dict["forces"].shape) != 3 or log_dict["forces"].shape[1] != len(
                sorted_indices
            ):
                raise VaspCollectError("Improper OUTCAR parsing")
            log_dict["time"] = self.outcar.parse_dict["time"]
            log_dict["steps"] = self.outcar.parse_dict["steps"]
            log_dict["cells"] = self.outcar.parse_dict["cells"]
            log_dict["volume"] = np.array(
                [np.linalg.det(cell) for cell in self.outcar.parse_dict["cells"]]
            )
            self.generic_output.dft_log_dict["scf_energy_free"] = (
                self.outcar.parse_dict["scf_energies"]
            )
            self.generic_output.dft_log_dict["scf_dipole_mom"] = self.outcar.parse_dict[
                "scf_dipole_moments"
            ]
            self.generic_output.dft_log_dict["n_elect"] = self.outcar.parse_dict[
                "n_elect"
            ]
            self.generic_output.dft_log_dict["energy_int"] = self.outcar.parse_dict[
                "energies_int"
            ]
            self.generic_output.dft_log_dict["energy_free"] = self.outcar.parse_dict[
                "energies"
            ]
            self.generic_output.dft_log_dict["energy_zero"] = self.outcar.parse_dict[
                "energies_zero"
            ]
            self.generic_output.dft_log_dict["energy_int"] = self.outcar.parse_dict[
                "energies_int"
            ]
            if "PROCAR" in files_present:
                try:
                    self.electronic_structure = self.procar.from_file(
                        filename=posixpath.join(directory, "PROCAR")
                    )
                    #  Even the atom resolved values have to be sorted from the vasp atoms order to the Atoms order
                    self.electronic_structure.grand_dos_matrix[
                        :, :, :, sorted_indices, :
                    ] = self.electronic_structure.grand_dos_matrix[:, :, :, :, :].copy()
                    try:
                        self.electronic_structure.efermi = self.outcar.parse_dict[
                            "fermi_level"
                        ]
                    except KeyError:
                        self.electronic_structure.efermi = self.vp_new.vasprun_dict[
                            "efermi"
                        ]
                except ValueError:
                    pass
        # important that we "reverse sort" the atoms in the vasp format into the atoms in the atoms class
        self.generic_output.log_dict = log_dict
        if vasprun_working:
            # self.dft_output.log_dict["parameters"] = self.vp_new.vasprun_dict["parameters"]
            self.generic_output.dft_log_dict["scf_dipole_mom"] = (
                self.vp_new.vasprun_dict["scf_dipole_moments"]
            )
            if len(self.generic_output.dft_log_dict["scf_dipole_mom"][0]) > 0:
                total_dipole_moments = np.array(
                    [
                        dip[-1]
                        for dip in self.generic_output.dft_log_dict["scf_dipole_mom"]
                    ]
                )
                self.generic_output.dft_log_dict["dipole_mom"] = total_dipole_moments
            self.generic_output.dft_log_dict["scf_energy_int"] = (
                self.vp_new.vasprun_dict["scf_energies"]
            )
            self.generic_output.dft_log_dict["scf_energy_free"] = (
                self.vp_new.vasprun_dict["scf_fr_energies"]
            )
            self.generic_output.dft_log_dict["scf_energy_zero"] = (
                self.vp_new.vasprun_dict["scf_0_energies"]
            )
            self.generic_output.dft_log_dict["energy_int"] = np.array(
                [
                    e_int[-1]
                    for e_int in self.generic_output.dft_log_dict["scf_energy_int"]
                ]
            )
            self.generic_output.dft_log_dict["energy_free"] = np.array(
                [
                    e_free[-1]
                    for e_free in self.generic_output.dft_log_dict["scf_energy_free"]
                ]
            )
            # Overwrite energy_free with much better precision from the OSZICAR file
            if "energy_pot" in self.oszicar.parse_dict and np.array_equal(
                self.generic_output.dft_log_dict["energy_free"],
                np.round(self.oszicar.parse_dict["energy_pot"], 8),
            ):
                self.generic_output.dft_log_dict["energy_free"] = (
                    self.oszicar.parse_dict["energy_pot"]
                )
            self.generic_output.dft_log_dict["energy_zero"] = np.array(
                [
                    e_zero[-1]
                    for e_zero in self.generic_output.dft_log_dict["scf_energy_zero"]
                ]
            )
            self.generic_output.dft_log_dict["n_elect"] = float(
                self.vp_new.vasprun_dict["parameters"]["electronic"]["NELECT"]
            )
            if "kinetic_energies" in self.vp_new.vasprun_dict:
                # scf_energy_kin is for backwards compatibility
                self.generic_output.dft_log_dict["scf_energy_kin"] = (
                    self.vp_new.vasprun_dict["kinetic_energies"]
                )
                self.generic_output.dft_log_dict["energy_kin"] = (
                    self.vp_new.vasprun_dict["kinetic_energies"]
                )

        if (
            "LOCPOT" in files_present
            and os.stat(posixpath.join(directory, "LOCPOT")).st_size != 0
        ):
            self.electrostatic_potential.from_file(
                filename=posixpath.join(directory, "LOCPOT"), normalize=False
            )
        if (
            "CHGCAR" in files_present
            and os.stat(posixpath.join(directory, "CHGCAR")).st_size != 0
        ):
            self.charge_density.from_file(
                filename=posixpath.join(directory, "CHGCAR"), normalize=True
            )
        self.generic_output.bands = self.electronic_structure

    def to_dict(self) -> dict:
        """
        Serialize the parsed VASP output into a nested dictionary.

        The returned dictionary contains the following top-level keys (when available):

        - ``"description"``: human-readable label for this output object
        - ``"generic"``: positions, forces, cells, energies, and DFT-specific quantities
        - ``"structure"``: final atomistic structure as a dictionary
        - ``"electrostatic_potential"``: LOCPOT data (total and optional diff)
        - ``"charge_density"``: CHGCAR data (total and optional spin diff)
        - ``"electronic_structure"``: k-points, eigenvalues, occupancies, and DOS data
        - ``"outcar"``: OUTCAR-specific quantities not available from vasprun.xml

        Returns:
            dict: Hierarchical dictionary of parsed VASP output
        """
        output_dict = {
            "description": self.description,
            "generic": self.generic_output.to_dict(),
        }

        if self.structure is not None and type(self.structure) == Atoms:
            output_dict["structure"] = self.structure.todict()
        elif self.structure is not None:
            output_dict["structure"] = self.structure.to_dict()

        if self.electrostatic_potential.total_data is not None:
            output_dict["electrostatic_potential"] = (
                self.electrostatic_potential.to_dict()
            )

        if self.charge_density.total_data is not None:
            output_dict["charge_density"] = self.charge_density.to_dict()

        if len(self.electronic_structure.kpoint_list) > 0:
            output_dict["electronic_structure"] = self.electronic_structure.to_dict()

        if len(self.outcar.parse_dict.keys()) > 0:
            output_dict["outcar"] = self.outcar.to_dict_minimal()
        return output_dict


class GenericOutput:
    """

    This class stores the generic output like different structures, energies and forces from a simulation in a highly
    generic format. Usually the user does not have to access this class.

    Attributes:
        log_dict (dict): A dictionary of all tags and values of generic data (positions, forces, etc)
    """

    def __init__(self):
        self.log_dict = {}
        self.dft_log_dict = {}
        self.description = "generic_output contains generic output static"
        self._bands = ElectronicStructure()

    @property
    def bands(self) -> ElectronicStructure:
        """ElectronicStructure: The electronic band structure object."""
        return self._bands

    @bands.setter
    def bands(self, val: ElectronicStructure):
        self._bands = val

    def to_dict(self) -> dict:
        """
        Serialize the generic and DFT-specific output into a dictionary.

        The returned dictionary merges ``log_dict`` (generic quantities such as forces,
        positions, energies, cells) with a ``"dft"`` sub-dictionary containing
        DFT-specific quantities (magnetization, Fermi level lists, SCF energies, etc.)
        and optionally a ``"bands"`` key with the electronic structure.

        Returns:
            dict: Dictionary of all generic and DFT output quantities
        """
        go_dict, dft_dict = {}, {}
        for key, val in self.log_dict.items():
            go_dict[key] = val
        for key, val in self.dft_log_dict.items():
            dft_dict[key] = val
        go_dict["dft"] = dft_dict
        if self.bands.eigenvalue_matrix is not None:
            go_dict["dft"]["bands"] = self.bands.to_dict()
        return go_dict


class VaspCollectError(ValueError):
    """Raised when VASP output files are present but cannot be parsed correctly."""

    pass


def parse_vasp_output(
    working_directory: str,
    structure: Optional[Atoms] = None,
    sorted_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
    read_atoms_funct: Callable = read_atoms,
    es_class=ElectronicStructure,
    bader_class=Bader,
    output_parser_class=Output,
) -> dict:
    """
    Parse the VASP output in the working_directory and return it as a hierarchical dictionary.

    This is the primary entry point for parsing a finished VASP calculation. The function
    reads vasprun.xml (preferred) and/or OUTCAR, OSZICAR, LOCPOT, CHGCAR, PROCAR, and
    CONTCAR/POSCAR files as available. If both vasprun.xml and OUTCAR are present, vasprun.xml
    is used for energies, forces, positions, and electronic structure, while unique OUTCAR
    quantities (stresses, elastic constants, Broyden mixing mesh, etc.) supplement the result.
    Bader charge analysis is performed automatically when AECCAR0 and AECCAR2 are present.

    Args:
        working_directory (str): Path to the directory containing the VASP output files
        structure (Atoms): Input atomistic structure. If None, it is read from CONTCAR/POSCAR.
        sorted_indices (list/numpy.ndarray): Permutation indices mapping VASP atom order
            (grouped by species) to the order in ``structure``. Computed automatically if None.
        read_atoms_funct (callable): Function used to read structure files; defaults to
            ``read_atoms`` which parses POSCAR/CONTCAR format.
        es_class: Class used to store electronic structure data; defaults to
            ``ElectronicStructure``.
        bader_class: Class used for Bader charge analysis; defaults to ``Bader``.
        output_parser_class: Class used to aggregate all output; defaults to ``Output``.

    Returns:
        dict: Hierarchical output dictionary as returned by ``Output.to_dict()``.

    Raises:
        VaspCollectError: If required output files are found but cannot be parsed.
        OSError: If no OUTCAR or vasprun.xml is present in working_directory.
    """
    output_parser = output_parser_class()
    if structure is None or len(structure) == 0:
        try:
            structure = get_final_structure_from_file(
                working_directory=working_directory,
                filename="CONTCAR",
                read_atoms_funct=read_atoms_funct,
            )
        except OSError:
            structure = get_final_structure_from_file(
                working_directory=working_directory,
                filename="POSCAR",
                read_atoms_funct=read_atoms_funct,
            )
    if sorted_indices is None:
        sorted_indices_array = np.arange(len(structure))
    else:
        sorted_indices_array = np.asarray(sorted_indices)
    output_parser.structure = structure.copy()
    try:
        output_parser.collect(
            directory=working_directory,
            sorted_indices=sorted_indices_array,
            es_class=es_class,
        )
    except VaspCollectError:
        raise
    # Try getting high precision positions from CONTCAR
    with contextlib.suppress(IOError, ValueError, FileNotFoundError):
        output_parser.structure = get_final_structure_from_file(
            working_directory=working_directory,
            filename="CONTCAR",
            structure=structure,
            sorted_indices=sorted_indices_array,
            read_atoms_funct=read_atoms_funct,
        )

    # Bader analysis
    if os.path.isfile(os.path.join(working_directory, "AECCAR0")) and os.path.isfile(
        os.path.join(working_directory, "AECCAR2")
    ):
        bader = bader_class(working_directory=working_directory, structure=structure)
        try:
            charges_orig, volumes_orig = bader.compute_bader_charges()
        except ValueError:
            warnings.warn("Invoking Bader charge analysis failed", stacklevel=2)
        else:
            charges, volumes = charges_orig.copy(), volumes_orig.copy()
            charges[sorted_indices_array] = charges_orig
            volumes[sorted_indices_array] = volumes_orig
            if "valence_charges" in output_parser.generic_output.dft_log_dict:
                valence_charges = output_parser.generic_output.dft_log_dict[
                    "valence_charges"
                ]
                # Positive values indicate electron depletion
                output_parser.generic_output.dft_log_dict["bader_charges"] = (
                    valence_charges - charges
                )
            output_parser.generic_output.dft_log_dict["bader_volumes"] = volumes
    return output_parser.to_dict()


def get_final_structure_from_file(
    working_directory: str,
    filename: str = "CONTCAR",
    structure: Optional[Atoms] = None,
    sorted_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
    read_atoms_funct: Callable = read_atoms,
) -> Atoms:
    """
    Read the final structure from a POSCAR-format file (typically CONTCAR).

    The CONTCAR file written by VASP at the end of a relaxation or MD run contains the
    final ionic positions, cell vectors, and optionally predictor-corrector velocities.
    If ``structure`` is provided, its atom ordering is preserved and only the cell and
    positions are updated; otherwise the structure is read entirely from the file.

    Args:
        working_directory (str): Path to the directory containing the structure file
        filename (str): Name of the structure file to read (default: ``"CONTCAR"``)
        structure (Atoms): Reference input structure used to set the species list and
            atom ordering. If None, species are read directly from the file.
        sorted_indices (list/numpy.ndarray): Permutation indices mapping the VASP atom
            order to the order in ``structure``. Computed from ``structure`` if None.
        read_atoms_funct (callable): Function for parsing POSCAR-format files

    Returns:
        ase.atoms.Atoms: The final structure with updated positions and cell

    Raises:
        OSError: If the file cannot be read or the positions are inconsistent
    """
    filename = posixpath.join(working_directory, filename)
    if structure is not None and sorted_indices is None:
        sorted_indices_array = vasp_sorter(structure)
    elif sorted_indices is not None:
        sorted_indices_array = np.asarray(sorted_indices)
    else:
        sorted_indices_array = None
    if structure is None:
        try:
            output_structure = read_atoms_funct(filename=filename)
            input_structure = output_structure.copy()
        except (OSError, IndexError, ValueError):
            raise OSError("Unable to read output structure")
    else:
        input_structure = structure.copy()
        if type(input_structure) == Atoms:
            species_list = input_structure.get_chemical_symbols()
        else:
            species_list = input_structure.get_parent_symbols()
        try:
            output_structure = read_atoms_funct(
                filename=filename,
                species_list=species_list,
            )
            input_structure.cell = output_structure.cell.copy()
            input_structure.positions[sorted_indices_array] = output_structure.positions
        except (OSError, IndexError, ValueError):
            raise OSError("Unable to read output structure")
    return input_structure
