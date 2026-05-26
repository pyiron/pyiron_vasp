# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import os
import posixpath
import numpy as np
from ase.atoms import Atoms
import ase.atoms

from vaspparser.dft.waves.electronic import ElectronicStructure, Kpoint, Band
from vaspparser.vasp.vasprun import Vasprun


class TestMoreElectronicStructure(unittest.TestCase):
    def setUp(self):
        self.es_obj = ElectronicStructure()

    def test_initialization(self):
        self.assertIsInstance(self.es_obj, ElectronicStructure)
        self.assertEqual(self.es_obj.kpoints, [])
        self.assertIsNone(self.es_obj._eg)
        self.assertIsNone(self.es_obj._vbm)
        self.assertIsNone(self.es_obj._cbm)
        self.assertIsNone(self.es_obj._efermi)
        self.assertIsNone(self.es_obj._eigenvalue_matrix)
        self.assertIsNone(self.es_obj._occupancy_matrix)
        self.assertIsNone(self.es_obj._grand_dos_matrix)
        self.assertEqual(self.es_obj._kpoint_list, [])
        self.assertEqual(self.es_obj._kpoint_weights, [])
        self.assertEqual(self.es_obj.n_spins, 1)
        self.assertIsNone(self.es_obj._structure)
        self.assertIsNone(self.es_obj._orbital_dict)
        self.assertEqual(self.es_obj._output_dict, {})

    def test_properties_setters(self):
        self.es_obj.dos_energies = [1, 2, 3]
        self.assertEqual(self.es_obj.dos_energies, [1, 2, 3])
        self.es_obj.dos_densities = [4, 5, 6]
        self.assertEqual(self.es_obj.dos_densities, [4, 5, 6])
        self.es_obj.dos_idensities = [7, 8, 9]
        self.assertEqual(self.es_obj.dos_idensities, [7, 8, 9])
        self.es_obj.resolved_densities = [10, 11, 12]
        self.assertEqual(self.es_obj.resolved_densities, [10, 11, 12])
        self.es_obj.orbital_dict = {"s": 0, "p": 1}
        self.assertEqual(self.es_obj.orbital_dict, {"s": 0, "p": 1})
        self.es_obj.eigenvalue_matrix = np.array([[[1, 2], [3, 4]]])
        self.assertTrue(
            np.array_equal(self.es_obj.eigenvalue_matrix, np.array([[[1, 2], [3, 4]]]))
        )
        self.es_obj.occupancy_matrix = np.array([[[5, 6], [7, 8]]])
        self.assertTrue(
            np.array_equal(self.es_obj.occupancy_matrix, np.array([[[5, 6], [7, 8]]]))
        )
        self.es_obj.kpoint_list = [1, 2, 3]
        self.assertEqual(self.es_obj.kpoint_list, [1, 2, 3])
        self.es_obj.kpoint_weights = [4, 5, 6]
        self.assertEqual(self.es_obj.kpoint_weights, [4, 5, 6])
        self.es_obj.structure = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        self.assertIsInstance(self.es_obj.structure, Atoms)
        self.es_obj.efermi = 1.0
        self.assertEqual(self.es_obj.efermi, 1.0)
        self.es_obj.grand_dos_matrix = np.zeros((1, 1, 1, 1, 1))
        self.assertTrue(
            np.array_equal(self.es_obj.grand_dos_matrix, np.zeros((1, 1, 1, 1, 1)))
        )

    def test_get_vbm_cbm_bandgap(self):
        # Setting up a simple electronic structure
        self.es_obj.add_kpoint(value=[0, 0, 0], weight=1)
        self.es_obj.kpoints[0].add_band(eigenvalue=-1.0, occupancy=1.0, spin=0)
        self.es_obj.kpoints[0].add_band(eigenvalue=1.0, occupancy=0.0, spin=0)
        self.es_obj._eigenvalue_matrix = np.array([[[-1.0, 1.0]]])

        vbm = self.es_obj.get_vbm()
        cbm = self.es_obj.get_cbm()
        band_gap = self.es_obj.get_band_gap()

        self.assertAlmostEqual(vbm[0]["value"], -1.0)
        self.assertAlmostEqual(cbm[0]["value"], 1.0)
        self.assertAlmostEqual(band_gap[0]["band_gap"], 2.0)
        self.assertAlmostEqual(self.es_obj.vbm[0], -1.0)
        self.assertAlmostEqual(self.es_obj.cbm[0], 1.0)
        self.assertAlmostEqual(self.es_obj.eg[0], 2.0)

    def test_is_metal_exception(self):
        with self.assertRaises(ValueError):
            _ = self.es_obj.is_metal

    def test_grand_dos_matrix_value_error(self):
        # This test is to cover the ValueError in grand_dos_matrix property
        self.es_obj.add_kpoint(value=[0, 0, 0], weight=1)
        self.es_obj.kpoints[0].add_band(eigenvalue=-1.0, occupancy=1.0, spin=0)
        # Without resolved_dos_matrix, it should raise ValueError which is caught internally and returns None
        self.es_obj.grand_dos_matrix
        self.assertIsNone(self.es_obj._grand_dos_matrix)

    def test_to_dict(self):
        # Monkeypatch Atoms class for the test
        ase.atoms.Atoms.to_dict = ase.atoms.Atoms.todict

        self.es_obj.add_kpoint(value=[0, 0, 0], weight=1)
        self.es_obj.kpoints[0].add_band(eigenvalue=-1.0, occupancy=1.0, spin=0)
        self.es_obj._eigenvalue_matrix = np.array([[[-1.0]]])
        self.es_obj._occupancy_matrix = np.array([[[1.0]]])
        self.es_obj.efermi = 0.0
        self.es_obj.structure = Atoms("H", positions=[[0, 0, 0]])
        self.es_obj.dos_energies = np.array([1, 2, 3])
        self.es_obj.dos_densities = np.array([[0.1, 0.2, 0.3]])
        self.es_obj.dos_idensities = np.array([[0.1, 0.3, 0.6]])

        es_dict = self.es_obj.to_dict()
        self.assertIn("TYPE", es_dict)
        self.assertIn("k_points", es_dict)
        self.assertIn("k_weights", es_dict)
        self.assertIn("eig_matrix", es_dict)
        self.assertIn("occ_matrix", es_dict)
        self.assertIn("structure", es_dict)
        self.assertIn("efermi", es_dict)
        self.assertIn("dos", es_dict)

        # Clean up monkeypatch
        del ase.atoms.Atoms.to_dict

    def test_generate_from_matrices(self):
        self.es_obj.kpoint_list = [[0, 0, 0]]
        self.es_obj.kpoint_weights = [1]
        self.es_obj._eigenvalue_matrix = np.array([[[-1.0]]])
        self.es_obj.n_spins = 1
        self.es_obj._occupancy_matrix = np.array([[[1.0]]])
        self.es_obj.generate_from_matrices()
        self.assertEqual(len(self.es_obj.kpoints), 1)
        self.assertEqual(len(self.es_obj.kpoints[0].bands[0]), 1)
        self.assertEqual(self.es_obj.kpoints[0].bands[0][0].eigenvalue, -1.0)

    def test_get_spin_resolved_dos_exceptions(self):
        with self.assertRaises(ValueError):
            self.es_obj.get_spin_resolved_dos()

    def test_get_resolved_dos_exceptions(self):
        with self.assertRaises(ValueError):
            self.es_obj.get_resolved_dos()
        self.es_obj.dos_energies = [1, 2, 3]
        with self.assertRaises(ValueError):
            self.es_obj.get_resolved_dos()

    def test_get_resolved_dos(self):
        self.es_obj.dos_energies = np.array([1, 2, 3])
        self.es_obj.resolved_densities = np.ones((2, 2, 3, 3))  # s, a, o, n

        # Test with integer indices
        rdos = self.es_obj.get_resolved_dos(
            spin_indices=0, atom_indices=0, orbital_indices=0
        )
        self.assertTrue(np.array_equal(rdos, np.ones(3)))

        # Test with list indices
        rdos = self.es_obj.get_resolved_dos(
            spin_indices=[0, 1], atom_indices=[0, 1], orbital_indices=[0, 1, 2]
        )
        self.assertTrue(np.allclose(rdos, 2 * 2 * 3 * np.ones(3)))

        # Test with mixed indices and summation
        rdos = self.es_obj.get_resolved_dos(spin_indices=0, atom_indices=[0, 1])
        self.assertTrue(np.allclose(rdos, 2 * 3 * np.ones(3)))

        rdos = self.es_obj.get_resolved_dos(spin_indices=0, orbital_indices=[0, 1, 2])
        self.assertTrue(np.allclose(rdos, 2 * 3 * np.ones(3)))

        rdos = self.es_obj.get_resolved_dos(
            spin_indices=0, atom_indices=0, orbital_indices=[0, 1]
        )
        self.assertTrue(np.allclose(rdos, 2 * np.ones(3)))

    def test_plot_fermi_dirac(self):
        # This is a plot function, so we just check if it runs without error
        # A more thorough test would require a library to check plot outputs
        self.es_obj.add_kpoint(value=[0, 0, 0], weight=1)
        self.es_obj.kpoints[0].add_band(eigenvalue=-1.0, occupancy=1.0, spin=0)
        self.es_obj.efermi = 0
        self.es_obj._eigenvalue_matrix = np.array([[[-1.0]]])
        self.es_obj._occupancy_matrix = np.array([[[1.0]]])
        try:
            self.es_obj.plot_fermi_dirac()
        except Exception as e:
            self.fail(f"plot_fermi_dirac raised an exception: {e}")

    def test_del(self):
        # Difficult to test __del__ directly, but we can call it and check if attributes are gone
        # This is not a standard practice, but for coverage...
        es = ElectronicStructure()
        es.__del__()
        with self.assertRaises(AttributeError):
            _ = es.kpoints

    def test_repr(self):
        self.es_obj._eigenvalue_matrix = np.array([[[-1.0]]])
        self.es_obj.add_kpoint([0, 0, 0], 1)
        self.es_obj.kpoints[0].add_band(-1.0, 1.0, spin=0)
        self.assertIn("ElectronicStructure Instance", repr(self.es_obj))

    def test_get_vbm_cbm_bandgap_more_bands(self):
        self.es_obj.add_kpoint(value=[0, 0, 0], weight=1)
        self.es_obj.kpoints[0].add_band(eigenvalue=-2.0, occupancy=1.0, spin=0)
        self.es_obj.kpoints[0].add_band(eigenvalue=-1.0, occupancy=1.0, spin=0)
        self.es_obj.kpoints[0].add_band(eigenvalue=1.0, occupancy=0.0, spin=0)
        self.es_obj.kpoints[0].add_band(eigenvalue=2.0, occupancy=0.0, spin=0)
        self.es_obj._eigenvalue_matrix = np.array([[[-2.0, -1.0, 1.0, 2.0]]])
        vbm = self.es_obj.get_vbm()
        cbm = self.es_obj.get_cbm()
        self.assertAlmostEqual(vbm[0]["value"], -1.0)
        self.assertAlmostEqual(cbm[0]["value"], 1.0)

    def test_getitem(self):
        self.es_obj._output_dict["test_key"] = "test_value"
        self.assertEqual(self.es_obj["test_key"], "test_value")

    def test_to_dict_with_more_data(self):
        ase.atoms.Atoms.to_dict = ase.atoms.Atoms.todict
        self.es_obj.grand_dos_matrix = np.ones((1, 1, 1, 1, 1))
        self.es_obj.resolved_densities = np.ones((1, 1, 1, 1))
        es_dict = self.es_obj.to_dict()
        self.assertIn("grand_dos_matrix", es_dict["dos"])
        self.assertIn("resolved_densities", es_dict["dos"])
        del ase.atoms.Atoms.to_dict

    def test_generate_from_matrices_with_grand_dos(self):
        self.es_obj.kpoint_list = [[0, 0, 0]]
        self.es_obj.kpoint_weights = [1]
        self.es_obj._eigenvalue_matrix = np.array([[[-1.0]]])
        self.es_obj.n_spins = 1
        self.es_obj._occupancy_matrix = np.array([[[1.0]]])
        self.es_obj._grand_dos_matrix = np.ones((1, 1, 1, 2, 3))
        self.es_obj.generate_from_matrices()
        self.assertIsNotNone(self.es_obj.kpoints[0].bands[0][0].resolved_dos_matrix)

    def test_str_non_metal(self):
        self.es_obj.add_kpoint(value=[0, 0, 0], weight=1)
        self.es_obj.kpoints[0].add_band(eigenvalue=-1.0, occupancy=1.0, spin=0)
        self.es_obj.kpoints[0].add_band(eigenvalue=1.0, occupancy=0.0, spin=0)
        self.es_obj._eigenvalue_matrix = np.array([[[-1.0, 1.0]]])
        self.es_obj.efermi = 0.0
        self.assertIn("Is a metal: False", str(self.es_obj))

    def test_kpoint_eig_occ_matrix(self):
        kpt = Kpoint()
        kpt.add_band(eigenvalue=-1.0, occupancy=1.0, spin=0)
        kpt.add_band(eigenvalue=1.0, occupancy=0.0, spin=0)
        eig_occ = kpt.eig_occ_matrix
        self.assertEqual(eig_occ.shape, (1, 2, 2))
        self.assertTrue(np.array_equal(eig_occ[0], [[-1.0, 1.0], [1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
