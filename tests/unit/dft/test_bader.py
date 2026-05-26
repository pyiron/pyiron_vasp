# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
import os
from vaspparser.dft.bader import (
    parse_charge_vol_file,
    get_valence_and_total_charge_density,
    Bader,
    call_bader,
)
from vaspparser.vasp.structure import read_atoms
from unittest.mock import patch, MagicMock


class TestBader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.bader_test_path = os.path.join(
            cls.file_location, "../../static/vasp_test_files/bader_test"
        )

    def test_parse_charge_vol(self):
        filename = os.path.join(
            self.file_location, "../../static/dft/bader_files/ACF.dat"
        )
        struct = read_atoms(os.path.join(self.bader_test_path, "POSCAR"))
        charges, volumes = parse_charge_vol_file(structure=struct, filename=filename)
        self.assertTrue(np.allclose(charges, [0.438202, 0.438197, 7.143794]))
        self.assertTrue(np.allclose(volumes, [287.284690, 297.577878, 415.155432]))

    def test_get_valence_and_total_charge_density(self):
        cd_val, cd_total = get_valence_and_total_charge_density(
            working_directory=self.bader_test_path
        )
        self.assertIsNotNone(cd_val)
        self.assertIsNotNone(cd_total)
        self.assertEqual(cd_val.total_data.shape, (20, 20, 20))
        self.assertEqual(cd_total.total_data.shape, (20, 20, 20))
        cd_val, cd_total = get_valence_and_total_charge_density(
            working_directory=os.path.dirname(self.bader_test_path)
        )
        self.assertIsNone(cd_val.total_data)
        self.assertIsNone(cd_total.total_data)

    @patch("vaspparser.dft.bader.call_bader")
    @patch("vaspparser.dft.bader.os.remove")
    def test_bader_class(self, mock_remove, mock_call_bader):
        mock_call_bader.return_value = 0
        struct = read_atoms(os.path.join(self.bader_test_path, "POSCAR"))
        bader = Bader(structure=struct, working_directory=self.bader_test_path)
        with patch(
            "vaspparser.dft.bader.parse_charge_vol_file"
        ) as mock_parse_charge_vol_file:
            mock_parse_charge_vol_file.return_value = (
                np.array([1.0]),
                np.array([1.0]),
            )
            charges, volumes = bader.compute_bader_charges()
            self.assertTrue(mock_call_bader.called)
            self.assertEqual(mock_remove.call_count, 2)
            self.assertTrue(np.allclose(charges, [1.0]))
            self.assertTrue(np.allclose(volumes, [1.0]))

    @patch("vaspparser.dft.bader.subprocess.call")
    def test_call_bader(self, mock_subprocess_call):
        mock_subprocess_call.return_value = 0
        error_code = call_bader(foldername=self.bader_test_path)
        self.assertEqual(error_code, 0)
        self.assertTrue(mock_subprocess_call.called)

    @patch("vaspparser.dft.bader.call_bader")
    @patch("vaspparser.dft.bader.os.remove")
    def test_bader_class_error(self, mock_remove, mock_call_bader):
        mock_call_bader.return_value = 1
        struct = read_atoms(os.path.join(self.bader_test_path, "POSCAR"))
        bader = Bader(structure=struct, working_directory=self.bader_test_path)
        with self.assertRaises(ValueError):
            bader.compute_bader_charges()
        self.assertTrue(mock_call_bader.called)
        self.assertEqual(mock_remove.call_count, 2)
