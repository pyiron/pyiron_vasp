# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under theterms of "New BSD License", see the LICENSE file.

import os
import unittest
import numpy as np
from vaspparser.vasp.procar import Procar


class TestProcarParser(unittest.TestCase):
    def setUp(self):
        self.parser = Procar()
        self.file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../static/vasp_test_files/PROCAR_for_test",
        )

    def test_from_file(self):
        self.assertIsNone(self.parser._check_if_spin_polarized("dummy"))
        es_obj = self.parser.from_file(self.file_path)
        self.assertEqual(len(es_obj.kpoints), 1)
        self.assertEqual(len(es_obj.kpoints[0].bands[0]), 1)
        self.assertEqual(es_obj.kpoints[0].bands[0][0].eigenvalue, -17.37867948)
        self.assertEqual(es_obj.kpoints[0].bands[0][0].occupancy, 1.0)
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][0].resolved_dos_matrix,
                np.array(
                    [
                        [0.144, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.291, 0.0, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.291, 0.0, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][0].atom_resolved_dos,
                np.array([0.145, 0.298, 0.298]),
            )
        )
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][0].orbital_resolved_dos,
                np.array([0.727, 0.0, 0.013, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
        )
        self.assertEqual(es_obj.kpoints[0].bands[0][0].eigenvalue, -17.37867948)
        self.assertEqual(es_obj.kpoints[0].bands[0][0].occupancy, 1.0)
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][0].resolved_dos_matrix,
                np.array(
                    [
                        [0.144, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.291, 0.0, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.291, 0.0, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][0].atom_resolved_dos,
                np.array([0.145, 0.298, 0.298]),
            )
        )
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][0].orbital_resolved_dos,
                np.array([0.727, 0.0, 0.013, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
        )


if __name__ == "__main__":
    unittest.main()
