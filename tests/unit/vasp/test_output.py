import unittest
import os
from vaspparser.vasp.output import Output, parse_vasp_output
from vaspparser.vasp.structure import read_atoms
import numpy as np
from ase.atoms import Atoms


class TestOutput(unittest.TestCase):
    def setUp(self):
        self.output = Output()
        self.vasp_test_files_path = os.path.join(
            os.path.dirname(__file__), "../../static/vasp_test_files"
        )
        self.full_job_sample_path = os.path.join(
            self.vasp_test_files_path, "full_job_sample"
        )

    def test_init(self):
        self.assertIsNotNone(self.output)

    def test_structure_setter(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        self.output.structure = atoms
        self.assertEqual(self.output.structure, atoms)

    def test_collect_with_vasprun(self):
        structure = read_atoms(
            os.path.join(self.full_job_sample_path, "POSCAR"), species_list=["Fe"]
        )
        self.output.structure = structure
        self.output.collect(directory=self.full_job_sample_path)
        self.assertIsNotNone(self.output.outcar)
        self.assertIsNotNone(self.output.oszicar)
        self.assertIsNotNone(self.output.generic_output)
        self.assertIsNotNone(self.output.electronic_structure)
        self.assertIsNotNone(self.output.charge_density)
        self.assertIsNone(self.output.electrostatic_potential.total_data)
        self.assertIsNotNone(self.output.procar)

    def test_to_dict(self):
        structure = read_atoms(
            os.path.join(self.full_job_sample_path, "POSCAR"), species_list=["Fe"]
        )
        self.output.structure = structure
        self.output.collect(directory=self.full_job_sample_path)
        output_dict = self.output.to_dict()
        self.assertIn("description", output_dict)
        self.assertIn("generic", output_dict)
        self.assertIn("structure", output_dict)
        self.assertNotIn("electrostatic_potential", output_dict)
        self.assertIn("charge_density", output_dict)
        self.assertIn("electronic_structure", output_dict)
        self.assertIn("outcar", output_dict)
        self.assertIn("pullay_stress", output_dict["outcar"])

    def test_parse_vasp_output(self):
        output_dict = parse_vasp_output(working_directory=self.full_job_sample_path)
        self.assertIn("description", output_dict)
        self.assertIn("generic", output_dict)
        self.assertIn("structure", output_dict)
        self.assertNotIn("electrostatic_potential", output_dict)
        self.assertIn("charge_density", output_dict)
        self.assertIn("electronic_structure", output_dict)
        self.assertIn("outcar", output_dict)

    def test_collect_with_outcar_only(self):
        outcar_sample_path = os.path.join(self.vasp_test_files_path, "outcar_samples")
        structure = read_atoms(
            os.path.join(self.full_job_sample_path, "POSCAR"), species_list=["Fe"]
        )
        self.output.structure = structure
        with self.assertRaises(IOError):
            self.output.collect(directory=outcar_sample_path)

    def test_collect_with_corrupted_vasprun(self):
        corrupted_vasprun_path = os.path.join(
            self.vasp_test_files_path, "corrupted_vasprun"
        )
        structure = read_atoms(
            os.path.join(self.full_job_sample_path, "POSCAR"), species_list=["Fe"]
        )
        self.output.structure = structure
        with self.assertWarns(UserWarning):
            self.output.collect(directory=corrupted_vasprun_path)
        self.assertIsNotNone(self.output.outcar)
        self.assertEqual(len(self.output.generic_output.log_dict), 11)

    def test_collect_with_no_output_files(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        self.output.structure = atoms
        with self.assertRaises(IOError):
            self.output.collect(directory=self.vasp_test_files_path)

    def test_get_final_structure_from_file(self):
        from vaspparser.vasp.output import get_final_structure_from_file

        structure = get_final_structure_from_file(
            working_directory=self.full_job_sample_path, filename="CONTCAR"
        )
        self.assertIsInstance(structure, Atoms)
        self.assertEqual(len(structure), 2)

    def test_parse_vasp_output_with_bader(self):
        bader_sample_path = os.path.join(self.vasp_test_files_path, "bader_test")
        with self.assertWarns(UserWarning):
            output_dict = parse_vasp_output(working_directory=bader_sample_path)
        self.assertNotIn("bader_charges", output_dict["generic"]["dft"])
        self.assertNotIn("bader_volumes", output_dict["generic"]["dft"])


if __name__ == "__main__":
    unittest.main()
