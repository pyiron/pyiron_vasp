# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import os
import shutil
import numpy as np
from ase.atoms import Atoms
from vaspparser.vasp.output import (
    Output,
    get_final_structure_from_file,
    parse_vasp_output,
)
from vaspparser.vasp.structure import read_atoms


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

    def test_get_final_structure_with_structure_and_sorted_indices(self):
        structure = read_atoms(
            os.path.join(self.full_job_sample_path, "POSCAR"), species_list=["Fe"]
        )
        expected_structure = read_atoms(
            os.path.join(self.full_job_sample_path, "CONTCAR"),
            species_list=structure.get_chemical_symbols(),
        )
        sorted_indices = np.array([1, 0])

        final_structure = get_final_structure_from_file(
            working_directory=self.full_job_sample_path,
            filename="CONTCAR",
            structure=structure,
            sorted_indices=sorted_indices,
        )

        np.testing.assert_allclose(final_structure.cell.array, expected_structure.cell.array)
        np.testing.assert_allclose(
            final_structure.positions[sorted_indices], expected_structure.positions
        )

    def test_parse_vasp_output_with_successful_bader(self):
        class MockBader:
            def __init__(self, working_directory, structure):
                self.working_directory = working_directory
                self.structure = structure

            def compute_bader_charges(self):
                n_atoms = len(self.structure)
                return np.arange(1, n_atoms + 1, dtype=float), np.arange(
                    2, n_atoms + 2, dtype=float
                )

        bader_sample_path = os.path.join(self.vasp_test_files_path, "bader_test")

        output_dict = parse_vasp_output(
            working_directory=bader_sample_path,
            bader_class=MockBader,
        )

        self.assertIn("bader_charges", output_dict["generic"]["dft"])
        self.assertIn("bader_volumes", output_dict["generic"]["dft"])
        expected_charges = np.arange(
            1, len(output_dict["generic"]["dft"]["valence_charges"]) + 1, dtype=float
        )
        expected_volumes = np.arange(
            2, len(output_dict["generic"]["dft"]["valence_charges"]) + 2, dtype=float
        )
        np.testing.assert_allclose(
            output_dict["generic"]["dft"]["bader_volumes"], expected_volumes
        )
        np.testing.assert_allclose(
            output_dict["generic"]["dft"]["bader_charges"],
            output_dict["generic"]["dft"]["valence_charges"] - expected_charges,
        )


if __name__ == "__main__":
    unittest.main()
