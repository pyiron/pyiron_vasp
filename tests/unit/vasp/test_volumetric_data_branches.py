import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase.atoms import Atoms

from vaspparser.vasp.volumetric_data import VaspVolumetricData


def _minimal_volumetric_content():
    return "\n".join(
        [
            "Mock CHGCAR",
            "1.0",
            "1.0 0.0 0.0",
            "0.0 1.0 0.0",
            "0.0 0.0 1.0",
            "H",
            "1",
            "Direct",
            "0.0 0.0 0.0",
            "",
            "1 1 5",
            "1.0 2.0 3.0 4.0 5.0",
            "1 1 5",
            "5.0 4.0 3.0 2.0 1.0",
        ]
    )


class TestVaspVolumetricDataBranches(unittest.TestCase):
    def test_from_file_raises_if_both_parsers_fail(self):
        vd = VaspVolumetricData()
        with (
            patch.object(vd, "_read_vol_data", side_effect=ValueError("new failed")),
            patch.object(vd, "_read_vol_data_old", side_effect=ValueError("old failed")),
        ):
            with self.assertRaises(ValueError):
                vd.from_file("dummy_file")

    def test_from_file_falls_back_to_old_parser(self):
        vd = VaspVolumetricData()
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)
        total = np.ones((1, 1, 1))
        diff = 2 * np.ones((1, 1, 1))
        with (
            patch.object(vd, "_read_vol_data", side_effect=ValueError("new failed")),
            patch.object(vd, "_read_vol_data_old", return_value=(atoms, [total, diff])),
        ):
            vd.from_file("dummy_file")
        self.assertIsNotNone(vd.atoms)
        np.testing.assert_allclose(vd.total_data, total)
        np.testing.assert_allclose(vd.diff_data, diff)

    def test_read_vol_data_old_warns_and_returns_none_for_empty_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filename = tmp.name
        try:
            with self.assertWarns(UserWarning):
                atoms, data = VaspVolumetricData._read_vol_data_old(filename)
            self.assertIsNone(atoms)
            self.assertIsNone(data)
        finally:
            os.remove(filename)

    def test_read_vol_data_warns_when_no_grid_was_parsed(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(
                "\n".join(
                    [
                        "Mock CHGCAR",
                        "1.0",
                        "1.0 0.0 0.0",
                        "0.0 1.0 0.0",
                        "0.0 0.0 1.0",
                        "H",
                        "1",
                        "Direct",
                        "0.0 0.0 0.0",
                        "",
                    ]
                )
            )
            filename = tmp.name
        try:
            with self.assertWarns(UserWarning):
                atoms, data = VaspVolumetricData()._read_vol_data(filename=filename)
            self.assertIsNone(atoms)
            self.assertIsNone(data)
        finally:
            os.remove(filename)

    def test_readers_use_potcar_species_fallback_when_structure_parse_fails(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(_minimal_volumetric_content())
            filename = tmp.name
        try:
            fallback_atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)

            def _atoms_from_string_side_effect(*args, **kwargs):
                if "species_list" not in kwargs:
                    raise ValueError("need species")
                return fallback_atoms.copy()

            with (
                patch(
                    "vaspparser.vasp.volumetric_data.atoms_from_string",
                    side_effect=_atoms_from_string_side_effect,
                ),
                patch(
                    "vaspparser.vasp.volumetric_data.get_species_list_from_potcar",
                    return_value=["H"],
                ),
            ):
                atoms_new, data_new = VaspVolumetricData()._read_vol_data(filename)
                atoms_old, data_old = VaspVolumetricData._read_vol_data_old(filename)

            self.assertIsNotNone(atoms_new)
            self.assertEqual(len(data_new), 2)
            self.assertIsNotNone(atoms_old)
            self.assertEqual(len(data_old), 2)
        finally:
            os.remove(filename)

    def test_to_dict_includes_diff_data_when_present(self):
        vd = VaspVolumetricData()
        vd.total_data = np.ones((1, 1, 1))
        vd.diff_data = 2 * np.ones((1, 1, 1))
        vd_dict = vd.to_dict()
        self.assertIn("diff", vd_dict)
        np.testing.assert_allclose(vd_dict["diff"], vd.diff_data)


if __name__ == "__main__":
    unittest.main()
