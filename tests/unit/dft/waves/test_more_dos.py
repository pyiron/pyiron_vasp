# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from unittest.mock import Mock

from vaspparser.dft.waves.dos import Dos, NoResolvedDosError


class TestMoreDos(unittest.TestCase):
    def setUp(self):
        self.mock_es = Mock()
        self.mock_es.eigenvalues = [np.array([-1, 0, 1])]
        self.mock_es.grand_dos_matrix = np.ones((1, 1, 3, 2, 4))

    def test_init_with_eigenvalues(self):
        dos = Dos(eigenvalues=[np.array([-1, 0, 1])], n_bins=10)
        self.assertEqual(len(dos.energies), 1)
        self.assertEqual(len(dos.t_dos[0]), 10)

    def test_init_with_bin_density(self):
        dos = Dos(eigenvalues=[np.array([-1, 0, 1])], bin_density=5)
        self.assertEqual(len(dos.energies), 1)
        self.assertEqual(len(dos.t_dos[0]), 10)

    def test_plot_total_dos(self):
        dos = Dos(es_obj=self.mock_es)
        try:
            dos.plot_total_dos()
        except Exception as e:
            self.fail(f"plot_total_dos raised an exception: {e}")

    def test_plot_orbital_resolved_dos_no_resolved(self):
        self.mock_es.grand_dos_matrix = None
        dos = Dos(es_obj=self.mock_es)
        with self.assertRaises(NoResolvedDosError):
            dos.plot_orbital_resolved_dos()

    def test_get_spin_resolved_dos(self):
        dos = Dos(es_obj=self.mock_es)
        rdos = dos.get_spin_resolved_dos(spin_indices=0)
        self.assertEqual(rdos.shape, dos.t_dos[0].shape)

    def test_get_resolved_dos_no_grand_dos_matrix(self):
        self.mock_es.grand_dos_matrix = None
        dos = Dos(es_obj=self.mock_es)
        with self.assertRaises(NoResolvedDosError):
            dos.get_spin_resolved_dos(spin_indices=0)

    def test_get_spatially_resolved_dos_index_edge_case(self):
        # To cover the 'else' block for 'if index >= 0'
        self.mock_es.eigenvalues = [np.array([-10, 0, 1])]
        dos = Dos(es_obj=self.mock_es)
        rdos = dos.get_spatially_resolved_dos(atom_indices=[0], spin_indices=0)
        self.assertEqual(rdos.flatten().shape, dos.t_dos[0].shape)

    def test_get_orbital_resolved_dos_index_edge_case(self):
        # To cover the 'else' block for 'if index >= 0'
        self.mock_es.eigenvalues = [np.array([-10, 0, 1])]
        dos = Dos(es_obj=self.mock_es)
        rdos = dos.get_orbital_resolved_dos(orbital_indices=[0], spin_indices=0)
        self.assertEqual(rdos.flatten().shape, dos.t_dos[0].shape)

    def test_get_spatial_orbital_resolved_dos_index_edge_case(self):
        # To cover the 'else' block for 'if index >= 0'
        self.mock_es.eigenvalues = [np.array([-10, 0, 1])]
        dos = Dos(es_obj=self.mock_es)
        rdos = dos.get_spatial_orbital_resolved_dos(
            atom_indices=[0], orbital_indices=[0], spin_indices=0
        )
        self.assertEqual(rdos.flatten().shape, dos.t_dos[0].shape)


if __name__ == "__main__":
    unittest.main()
