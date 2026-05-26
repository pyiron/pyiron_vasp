# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import Union, List, Tuple, Optional
import numpy as np
from ase.atoms import Atoms

from vaspparser.vasp.structure import write_poscar

__author__ = "Sudarsan Surendralal, Su-Hyun Yoo"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"


class VolumetricData:
    """
    A new class to handle 3-dimensional volumetric data elegantly (charge densities, electrostatic potentials etc) based
    on the numpy.ndarray instance. This module is adapted from the pymatgen vasp VolumtricData class

    http://pymatgen.org/_modules/pymatgen/io/vasp/outputs.html#VolumetricData

    Attributes:

        total_data (numpy.ndarray): A 3D array containing the data

    """

    def __init__(self) -> None:
        self._total_data: Optional[np.ndarray] = None
        self._atoms: Optional[Atoms] = None

    @property
    def atoms(self) -> Optional[Atoms]:
        """
        The structure related to the volumeric data

        Returns:
            pyiron_atomistics.atomistics.structure.Atoms: The structure associated with the data

        """
        return self._atoms

    @atoms.setter
    def atoms(self, val: Optional[Atoms]) -> None:
        self._atoms = val

    @property
    def total_data(self) -> Optional[np.ndarray]:
        """
        numpy.ndarray: The Nx x Ny x Nz sized array for the total data
        """
        return self._total_data

    @total_data.setter
    def total_data(self, val: Optional[Union[np.ndarray, list]]) -> None:
        if not (isinstance(val, (np.ndarray, list))):
            raise TypeError(
                "Attribute total_data should be a numpy.ndarray instance or a list and "
                f"not {type(val)}"
            )
        val = np.array(val)
        shape = np.array(np.shape(val))
        if not (len(shape) == 3):
            raise ValueError("Attribute total_data should be a 3D array")
        self._total_data = val

    @staticmethod
    def gauss_f(d: float, fwhm: float = 0.529177) -> float:
        """
        Generates a Gaussian distribution for a given distance and full width half maximum value

        Args:
            d (float): distance between target point and reference point
            fwhm (float): Full width half maximum in angstrom

        Returns:
            float: Gaussian reduction constant

        """
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        d2 = d * d
        return np.exp(-1 / (2 * sigma**2) * d2)

    @staticmethod
    def dist_between_two_grid_points(
        target_grid_point: Union[np.ndarray, list],
        n_grid_at_center: Union[np.ndarray, list],
        lattice: Union[np.ndarray, list],
        grid_shape: Union[tuple, list, np.ndarray],
    ) -> float:
        """
        Calculates the distance between a target grid point and another grid point

        Args:
            target_grid_point (numpy.ndarray/list): Target grid point
            n_grid_at_center (numpy.ndarray/list): coordinate of center of sphere
            lattice (numpy.ndarray/list): lattice vector
            grid_shape (tuple/list/numpy.ndarray): size of grid

        Returns:

            float: Distance between target grid and center of sphere in angstrom

        """
        unit_dist_in_grid = [
            np.sqrt(np.dot(lattice[0], lattice[0])) / grid_shape[0],
            np.sqrt(np.dot(lattice[1], lattice[1])) / grid_shape[1],
            np.sqrt(np.dot(lattice[2], lattice[2])) / grid_shape[2],
        ]
        dn = np.multiply(
            np.subtract(target_grid_point, n_grid_at_center), unit_dist_in_grid
        )
        dist = np.linalg.norm(dn)
        return float(dist)

    def spherical_average_potential(
        self,
        structure: Atoms,
        spherical_center: Union[list, np.ndarray],
        rad: float = 2,
        fwhm: float = 0.529177,
    ) -> float:
        """
        Calculates the spherical average about a given point in space

        Args:
            structure (pyiron_atomistics.atomistics.structure.Atoms): Input structure
            spherical_center (list/numpy.ndarray): position of spherical_center in direct coordinate
            rad (float): radius of sphere to be considered in Angstrom (recommended value: 2)
            fwhm (float): Full width half maximum of gaussian function in Angstrom (recommended value: 0.529177)

        Returns:
            float: Spherical average at the target center

        """
        if self._total_data is None:
            raise ValueError("total_data is not set")
        grid_shape = self._total_data.shape

        # Position of center of sphere at grid coordinates
        n_grid_at_center = [
            int(np.ceil(spherical_center[0] * grid_shape[0])),
            int(np.ceil(spherical_center[1] * grid_shape[1])),
            int(np.ceil(spherical_center[2] * grid_shape[2])),
        ]

        # Unit distance between grids
        dist_in_grid = [
            np.linalg.norm(structure.cell[0]) / grid_shape[0],
            np.linalg.norm(structure.cell[1]) / grid_shape[1],
            np.linalg.norm(structure.cell[2]) / grid_shape[2],
        ]

        # Range of grids to be considered within the provided radius w.r.t. center of sphere
        num_grid_in_sph: List[List[int]] = [[], []]
        for i, dist in enumerate(dist_in_grid):
            num_grid_in_sph[0].append(n_grid_at_center[i] - int(np.ceil(rad / dist)))
            num_grid_in_sph[1].append(n_grid_at_center[i] + int(np.ceil(rad / dist)))

        sph_avg_tmp = []
        weight = 0.0
        for k in range(num_grid_in_sph[0][0], num_grid_in_sph[1][0]):
            for l in range(num_grid_in_sph[0][1], num_grid_in_sph[1][1]):
                for m in range(num_grid_in_sph[0][2], num_grid_in_sph[1][2]):
                    target_grid_point = [k, l, m]
                    dist = self.dist_between_two_grid_points(
                        target_grid_point, n_grid_at_center, structure.cell, grid_shape
                    )
                    if dist <= rad:
                        sph_avg_tmp.append(
                            self._total_data[
                                k % grid_shape[0], l % grid_shape[1], m % grid_shape[2]
                            ]
                            * self.gauss_f(dist, fwhm)
                        )
                        weight += self.gauss_f(dist, fwhm)
                    else:
                        pass
        sum_list = np.sum(sph_avg_tmp)
        sph_avg = sum_list / weight
        return sph_avg

    @staticmethod
    def dist_between_two_grid_points_cyl(
        target_grid_point: Union[np.ndarray, list],
        n_grid_at_center: Union[np.ndarray, list],
        lattice: Union[np.ndarray, list],
        grid_shape: Union[tuple, list, np.ndarray],
        direction_of_cyl: int,
    ) -> float:
        """
        Distance between a target grid point and the center of a cylinder

        Args:
            target_grid_point (numpy.ndarray/list): Target grid point
            n_grid_at_center (numpy.ndarray/list): coordinate of center of sphere
            lattice (numpy.ndarray/list): lattice vector
            grid_shape (tuple/list/numpy.ndarray): size of grid
            direction_of_cyl (int): Axis of cylinder (0 (x) or 1 (y) or 2 (z))

        Returns:
            float: Distance between target grid and in-plane center of cylinder

        """
        unit_dist_in_grid = [
            np.sqrt(np.dot(lattice[0], lattice[0])) / grid_shape[0],
            np.sqrt(np.dot(lattice[1], lattice[1])) / grid_shape[1],
            np.sqrt(np.dot(lattice[2], lattice[2])) / grid_shape[2],
        ]
        dn = np.multiply(
            np.subtract(target_grid_point, n_grid_at_center), unit_dist_in_grid
        )
        if direction_of_cyl == 0:
            dn[0] = 0
        elif direction_of_cyl == 1:
            dn[1] = 0
        elif direction_of_cyl == 2:
            dn[2] = 0
        else:
            print("check the direction of cylindrical axis")
        dist = np.linalg.norm(dn)
        return float(dist)

    def cylindrical_average_potential(
        self,
        structure: Atoms,
        spherical_center: Union[list, np.ndarray],
        axis_of_cyl: int,
        rad: float = 2,
        fwhm: float = 0.529177,
    ) -> float:
        """
        Calculates the cylindrical average about a given point in space

        Args:
            structure (pyiron_atomistics.atomistics.structure.Atoms): Input structure
            spherical_center (list/numpy.ndarray): position of spherical_center in direct coordinate
            rad (float): radius of sphere to be considered in Angstrom (recommended value: 2)
            fwhm (float): Full width half maximum of gaussian function in Angstrom (recommended value: 0.529177)
            axis_of_cyl (int): Axis of cylinder (0 (x) or 1 (y) or 2 (z))

        Returns:
            float: Cylindrical average at the target center

        """
        if self._total_data is None:
            raise ValueError("total_data is not set")
        grid_shape = self._total_data.shape

        # Position of center of sphere at grid coordinates
        n_grid_at_center = [
            int(np.ceil(spherical_center[0] * grid_shape[0])),
            int(np.ceil(spherical_center[1] * grid_shape[1])),
            int(np.ceil(spherical_center[2] * grid_shape[2])),
        ]

        # Unit distance between grids
        dist_in_grid = [
            np.linalg.norm(structure.cell[0]) / grid_shape[0],
            np.linalg.norm(structure.cell[1]) / grid_shape[1],
            np.linalg.norm(structure.cell[2]) / grid_shape[2],
        ]

        # Range of grids to be considered within the provided radius w.r.t. center of sphere
        num_grid_in_cyl: List[List[int]] = [[], []]

        for i, dist in enumerate(dist_in_grid):
            if i == axis_of_cyl:
                num_grid_in_cyl[0].append(0)
                num_grid_in_cyl[1].append(grid_shape[i])
            else:
                num_grid_in_cyl[0].append(
                    n_grid_at_center[i] - int(np.ceil(rad / dist))
                )
                num_grid_in_cyl[1].append(
                    n_grid_at_center[i] + int(np.ceil(rad / dist))
                )

        cyl_avg_tmp = []
        weight = 0.0
        for k in range(num_grid_in_cyl[0][0], num_grid_in_cyl[1][0]):
            for l in range(num_grid_in_cyl[0][1], num_grid_in_cyl[1][1]):
                for m in range(num_grid_in_cyl[0][2], num_grid_in_cyl[1][2]):
                    target_grid_point = [k, l, m]
                    dist = self.dist_between_two_grid_points_cyl(
                        target_grid_point,
                        n_grid_at_center,
                        structure.cell,
                        grid_shape,
                        axis_of_cyl,
                    )
                    if dist <= rad:
                        cyl_avg_tmp.append(
                            self._total_data[
                                k % grid_shape[0], l % grid_shape[1], m % grid_shape[2]
                            ]
                            * self.gauss_f(dist, fwhm)
                        )
                        weight += self.gauss_f(dist, fwhm)
                    else:
                        pass
        sum_list = np.sum(cyl_avg_tmp)
        cyl_avg = sum_list / weight

        return cyl_avg

    def get_average_along_axis(self, ind: int = 2) -> np.ndarray:
        """
        Get the lateral average along a certain axis direction. This function is adapted from the pymatgen vasp
        VolumetricData class

        http://pymatgen.org/_modules/pymatgen/io/vasp/outputs.html#VolumetricData.get_average_along_axis

        Args:
            ind (int): Index of axis (0, 1 and 2 for the x, y, and z axis respectively)

        Returns:
            numpy.ndarray: A 1D vector with the laterally averaged values of the volumetric data
        """
        if self._total_data is None:
            raise ValueError("total_data is not set")
        if ind == 0:
            return np.average(np.average(self._total_data, axis=1), 1)
        elif ind == 1:
            return np.average(np.average(self._total_data, axis=0), 1)
        else:
            return np.average(np.average(self._total_data, axis=0), 0)

    def write_cube_file(
        self, filename: str = "cube_file.cube", cell_scaling: float = 1.0
    ) -> None:
        """
        Write the volumetric data into the CUBE file format

        Args:
            filename (str): Filename
            cell_scaling (float): Scale the cell by this fraction

        """
        atoms = self.atoms
        if atoms is None:
            raise ValueError(
                "The volumetric data object must have a valid structure assigned to it before writing "
                "to the cube format"
            )
        total_data = self.total_data
        if total_data is None:
            raise ValueError("total_data is not set")
        data = total_data
        n_x, n_y, _ = data.shape
        origin = np.zeros(3)
        flattened_data = np.hstack(
            [data[i, j, :] for i in range(n_x) for j in range(n_y)]
        )
        n_atoms = len(atoms)
        total_lines = int(len(flattened_data) / 6) * 6
        reshaped_data = np.reshape(flattened_data[0:total_lines], (-1, 6))
        last_line = [flattened_data[total_lines:]]
        head_array = np.zeros((4, 4))
        head_array[0] = np.append([n_atoms], origin)
        head_array[1:, 0] = data.shape
        head_array[1:, 1:] = atoms.cell / data.shape * cell_scaling
        position_array = np.zeros((len(atoms.positions), 5))
        position_array[:, 0] = atoms.get_atomic_numbers()
        position_array[:, 2:] = atoms.positions
        with open(filename, "w") as f:
            f.write("Cube file generated by pyiron (http://pyiron.org) \n")
            f.write("z is the fastest index \n")
            np.savetxt(f, head_array, fmt="%4d %.6f %.6f %.6f")  # type: ignore
            np.savetxt(f, position_array, fmt="%4d %.6f %.6f %.6f %.6f")  # type: ignore
            np.savetxt(f, reshaped_data, fmt="%.5e")  # type: ignore
            np.savetxt(f, last_line, fmt="%.5e")  # type: ignore

    def read_cube_file(self, filename: str = "cube_file.cube") -> None:
        """
        Generate data from a CUBE file

        Args:
            filename (str): Filename to parse

        """
        with open(filename, errors="ignore") as f:
            lines = f.readlines()
            n_atoms = int(lines[2].strip().split()[0])
            cell_data = np.genfromtxt(lines[3:6])
            cell_grid = cell_data[:, 1:]
            grid_shape = np.array(cell_data[:, 0], dtype=int)
            # total_data = np.zeros(grid_shape)
            cell = np.array([val * grid_shape[i] for i, val in enumerate(cell_grid)])
            if n_atoms > 0:
                pos_data = np.genfromtxt(lines[6 : n_atoms + 6])
                if n_atoms == 1:
                    pos_data = np.array([pos_data])
                atomic_numbers = np.array(pos_data[:, 0], dtype=int)
                positions = pos_data[:, 2:]
                self._atoms = Atoms(
                    numbers=atomic_numbers, positions=positions, cell=cell
                )
            end_int = n_atoms + 6 + int(np.prod(grid_shape) / 6)
            data = np.genfromtxt(lines[n_atoms + 6 : end_int])
            data_flatten = np.hstack(data)  # type: ignore
            if np.prod(grid_shape) % 6 > 0:
                data_flatten = np.append(
                    data_flatten, [float(val) for val in lines[end_int].split()]
                )
            n_x, n_y, n_z = grid_shape
            self._total_data = data_flatten.reshape((n_x, n_y, n_z))

    def write_vasp_volumetric(
        self, filename: str = "CHGCAR", normalize: bool = False
    ) -> None:
        """
        Writes volumetric data into a VASP CHGCAR format

        Args:
            filename (str): Filename of the new file
            normalize (bool): True if the data is to be normalized by the volume

        """
        atoms = self.atoms
        if atoms is None:
            raise ValueError("atoms is not set")
        total_data = self.total_data
        if total_data is None:
            raise ValueError("total_data is not set")
        write_poscar(structure=atoms, filename=filename)
        with open(filename, "a") as f:
            f.write("\n")
            f.write(" ".join(list(np.array(total_data.shape, dtype=str))))
            f.write("\n")
            _, n_y, n_z = total_data.shape
            flattened_data = np.hstack(
                [total_data[:, i, j] for j in range(n_z) for i in range(n_y)]
            )
            if normalize:
                flattened_data /= atoms.get_volume()
            num_lines = int(len(flattened_data) / 5) * 5
            reshaped_data = np.reshape(flattened_data[0:num_lines], (-1, 5))
            np.savetxt(f, reshaped_data, fmt="%.12f")  # type: ignore
            if len(flattened_data) % 5 > 0:
                np.savetxt(f, [flattened_data[num_lines:]], fmt="%.12f")  # type: ignore
