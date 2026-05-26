import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase.atoms import Atoms

from vaspparser.vasp.output import (
    VaspCollectError,
    get_final_structure_from_file,
    parse_vasp_output,
)


class _DummyOutputStructure:
    def __init__(self, positions, cell):
        self.positions = np.array(positions, dtype=float)
        self.cell = np.array(cell, dtype=float)


class _DummyStructure:
    def __init__(self, symbols, positions, cell):
        self._symbols = symbols
        self.positions = np.array(positions, dtype=float)
        self.cell = np.array(cell, dtype=float)

    def copy(self):
        return _DummyStructure(self._symbols, self.positions.copy(), self.cell.copy())

    def get_parent_symbols(self):
        return list(self._symbols)


class _ParserReraises:
    def __init__(self):
        self.structure = None

    def collect(self, directory, sorted_indices, es_class):
        raise VaspCollectError("collect failed")

    def to_dict(self):
        return {}


class _ParserCollects:
    def __init__(self):
        self.structure = None
        self.sorted_indices = None
        self.generic_output = type("G", (), {"dft_log_dict": {}})()

    def collect(self, directory, sorted_indices, es_class):
        self.sorted_indices = np.asarray(sorted_indices)

    def to_dict(self):
        return {"ok": True}


class TestOutputBranches(unittest.TestCase):
    def test_parse_vasp_output_reraises_vasp_collect_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "OUTCAR"), "w") as f:
                f.write("dummy")
            structure = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)
            with self.assertRaises(VaspCollectError):
                parse_vasp_output(
                    working_directory=tmpdir,
                    structure=structure,
                    output_parser_class=_ParserReraises,
                )

    def test_parse_vasp_output_contcar_fallback_poscar_and_sorted_indices(self):
        structure = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=np.eye(3), pbc=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "vaspparser.vasp.output.get_final_structure_from_file",
                side_effect=[OSError("contcar fail"), structure, IOError("ignore")],
            ) as mock_get_final:
                result = parse_vasp_output(
                    working_directory=tmpdir,
                    sorted_indices=[1, 0],
                    output_parser_class=_ParserCollects,
                )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_get_final.call_count, 3)
        self.assertEqual(mock_get_final.call_args_list[0].kwargs["filename"], "CONTCAR")
        self.assertEqual(mock_get_final.call_args_list[1].kwargs["filename"], "POSCAR")

    def test_get_final_structure_uses_vasp_sorter_when_indices_not_given(self):
        input_structure = Atoms(
            "H2", positions=[[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]], cell=np.eye(3), pbc=True
        )
        output_structure = Atoms(
            "H2", positions=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], cell=2 * np.eye(3), pbc=True
        )

        def _reader(filename, species_list):
            return output_structure

        with patch("vaspparser.vasp.output.vasp_sorter", return_value=np.array([1, 0])):
            final_structure = get_final_structure_from_file(
                working_directory="/tmp",
                filename="CONTCAR",
                structure=input_structure,
                read_atoms_funct=_reader,
            )
        np.testing.assert_allclose(final_structure.cell.array, output_structure.cell.array)
        np.testing.assert_allclose(
            final_structure.positions, np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )

    def test_get_final_structure_non_ase_structure_uses_parent_symbols(self):
        input_structure = _DummyStructure(
            symbols=["He", "He"],
            positions=[[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]],
            cell=np.eye(3),
        )
        output_structure = _DummyOutputStructure(
            positions=[[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            cell=3 * np.eye(3),
        )
        call_data = {}

        def _reader(filename, species_list):
            call_data["species_list"] = species_list
            return output_structure

        final_structure = get_final_structure_from_file(
            working_directory="/tmp",
            filename="CONTCAR",
            structure=input_structure,
            sorted_indices=[1, 0],
            read_atoms_funct=_reader,
        )
        self.assertEqual(call_data["species_list"], ["He", "He"])
        np.testing.assert_allclose(final_structure.cell, output_structure.cell)
        np.testing.assert_allclose(
            final_structure.positions, np.array([[4.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        )


if __name__ == "__main__":
    unittest.main()
