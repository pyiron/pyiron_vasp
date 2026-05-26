# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
import os
from ase.atoms import Atoms
from vaspparser.dft.volumetric import VolumetricData


class TestVolumetricData(unittest.TestCase):
    def setUp(self):
        self.vol_data = VolumetricData()
        self.atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=np.eye(3))
        self.data = np.random.rand(10, 10, 10)

    def test_init(self):
        self.assertIsNone(self.vol_data.atoms)
        self.assertIsNone(self.vol_data.total_data)

    def test_atoms_property(self):
        self.vol_data.atoms = self.atoms
        self.assertEqual(self.vol_data.atoms, self.atoms)

    def test_total_data_property(self):
        self.vol_data.total_data = self.data
        self.assertTrue(np.array_equal(self.vol_data.total_data, self.data))
        with self.assertRaises(TypeError):
            self.vol_data.total_data = "not_an_array"
        with self.assertRaises(ValueError):
            self.vol_data.total_data = np.random.rand(10, 10)

    def test_gauss_f(self):
        self.assertAlmostEqual(self.vol_data.gauss_f(0), 1.0)
        self.assertLess(self.vol_data.gauss_f(1), 1.0)

    def test_dist_between_two_grid_points(self):
        dist = self.vol_data.dist_between_two_grid_points(
            [0, 0, 0], [1, 1, 1], np.eye(3), (10, 10, 10)
        )
        self.assertAlmostEqual(dist, np.sqrt(0.03))

    def test_get_average_along_axis(self):
        self.vol_data.total_data = np.ones((10, 10, 10))
        avg = self.vol_data.get_average_along_axis(ind=0)
        self.assertTrue(np.allclose(avg, np.ones(10)))

    def test_read_and_write_cube_file(self):
        self.vol_data.atoms = self.atoms
        self.vol_data.total_data = self.data
        filename = "test_cube_file.cube"
        self.vol_data.write_cube_file(filename=filename)
        new_vol_data = VolumetricData()
        new_vol_data.read_cube_file(filename=filename)
        self.assertTrue(np.allclose(self.vol_data.total_data, new_vol_data.total_data))
        self.assertEqual(self.vol_data.atoms, new_vol_data.atoms)
        os.remove(filename)

    def test_write_vasp_volumetric(self):
        self.vol_data.atoms = self.atoms
        self.vol_data.total_data = self.data
        filename = "test_chgcar"
        self.vol_data.write_vasp_volumetric(filename=filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_spherical_average(self):
        self.vol_data.atoms = self.atoms
        self.vol_data.total_data = np.ones((10, 10, 10))
        avg = self.vol_data.spherical_average_potential(
            structure=self.atoms, spherical_center=[0.5, 0.5, 0.5]
        )
        self.assertAlmostEqual(avg, 1.0)

    def test_cylindrical_average(self):
        self.vol_data.atoms = self.atoms
        self.vol_data.total_data = np.ones((10, 10, 10))
        avg = self.vol_data.cylindrical_average_potential(
            structure=self.atoms, spherical_center=[0.5, 0.5, 0.5], axis_of_cyl=2
        )
        self.assertAlmostEqual(avg, 1.0)
        dist = self.vol_data.dist_between_two_grid_points_cyl(
            [0, 0, 0], [1, 1, 1], np.eye(3), (10, 10, 10), 3
        )
        self.assertAlmostEqual(dist, np.sqrt(0.03))

    def test_write_cube_file_no_atoms(self):
        self.vol_data.total_data = self.data
        with self.assertRaises(ValueError):
            self.vol_data.write_cube_file("test.cube")

    def test_write_vasp_volumetric_rem(self):
        self.vol_data.atoms = self.atoms
        self.vol_data.total_data = np.ones((6, 6, 6))
        filename = "test_chgcar"
        self.vol_data.write_vasp_volumetric(filename=filename, normalize=True)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_read_cube_file_one_atom(self):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3))
        self.vol_data.atoms = atoms
        self.vol_data.total_data = np.ones((5, 5, 5))
        filename = "test_cube_one_atom.cube"
        self.vol_data.write_cube_file(filename)
        new_vol_data = VolumetricData()
        new_vol_data.read_cube_file(filename)
        self.assertEqual(len(new_vol_data.atoms), 1)
        os.remove(filename)


if __name__ == "__main__":
    unittest.main()
