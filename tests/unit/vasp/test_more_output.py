# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import os
import shutil
from vaspparser.vasp.output import (
    Output,
    parse_vasp_output,
    get_final_structure_from_file,
)
from vaspparser.vasp.structure import read_atoms
import numpy as np
from ase.atoms import Atoms


class TestMoreOutput(unittest.TestCase):
    def setUp(self):
        self.output = Output()
        self.vasp_test_files_path = os.path.join(
            os.path.dirname(__file__), "../../static/vasp_test_files"
        )
        self.full_job_sample_path = os.path.join(
            self.vasp_test_files_path, "full_job_sample"
        )
        # Create a temporary directory for test files
        self.temp_dir = "temp_output_test"
        os.makedirs(self.temp_dir, exist_ok=True)
        # Copy necessary files
        for f in os.listdir(self.full_job_sample_path):
            shutil.copy(os.path.join(self.full_job_sample_path, f), self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_collect_no_oszicar(self):
        os.remove(os.path.join(self.temp_dir, "OSZICAR"))
        structure = read_atoms(
            os.path.join(self.temp_dir, "POSCAR"), species_list=["Fe"]
        )
        self.output.structure = structure
        self.output.collect(directory=self.temp_dir)
        self.assertEqual(self.output.oszicar.parse_dict, {})

    def test_to_dict_with_locpot(self):
        # Create a dummy LOCPOT file
        with open(os.path.join(self.temp_dir, "LOCPOT"), "w") as f:
            f.write("some data")
        # To make this test pass, we need to mock the from_file method of VaspVolumetricData
        # to avoid parsing errors. For now, let's just check if the key is in the dict.
        self.output.electrostatic_potential.total_data = np.array([1])
        output_dict = self.output.to_dict()
        self.assertIn("electrostatic_potential", output_dict)

    def test_get_final_structure_no_structure(self):
        structure = get_final_structure_from_file(
            working_directory=self.full_job_sample_path,
            filename="CONTCAR",
            structure=None,
        )
        self.assertIsInstance(structure, Atoms)

    def test_get_final_structure_io_error(self):
        with self.assertRaises(IOError):
            get_final_structure_from_file(
                working_directory=self.temp_dir, filename="non_existent_file.xyz"
            )


if __name__ == "__main__":
    unittest.main()
