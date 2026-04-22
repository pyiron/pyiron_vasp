# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under theterms of "New BSD License", see the LICENSE file.

import os
import unittest
import numpy as np
from vaspparser.vasp.parser.report import Report


class TestReportParser(unittest.TestCase):
    def setUp(self):
        self.parser = Report()
        self.file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../static/vasp_test_files/REPORT_for_test",
        )

    def test_from_file(self):
        self.parser.from_file(self.file_path)
        self.assertTrue(len(self.parser.parse_dict) > 0)
        self.assertTrue(
            np.array_equal(
                self.parser.parse_dict["derivative"], np.array([1.0, 5.0, 9.0])
            )
        )
        self.assertTrue(
            np.array_equal(self.parser.parse_dict["cv_full"], np.array([0.1, 0.2, 0.3]))
        )
        self.assertTrue(
            np.array_equal(self.parser.parse_dict["cv"], np.array([0.1, 0.2]))
        )
        self.assertTrue(len(self.parser.parse_dict["free_energy"]) == 2)


if __name__ == "__main__":
    unittest.main()
