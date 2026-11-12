"""Unit tests for Mini NPU Simulator (E1-3)"""

import json
import os
import unittest
from main import (
    EPSILON,
    Matrix2D,
    PatternGenerator,
    benchmark_mac,
    benchmark_mac_1d,
    decide_mode1,
    decide_mode2,
    mac_1d,
    mac_2d,
    normalize_label,
    run_mode_2,
)


class TestMatrix2D(unittest.TestCase):
    def test_init_and_get_set(self):
        mat = Matrix2D(3, 3)
        self.assertEqual(mat.rows, 3)
        self.assertEqual(mat.cols, 3)
        self.assertEqual(mat.get(1, 1), 0.0)

        mat.set(1, 1, 5.5)
        self.assertEqual(mat.get(1, 1), 5.5)

    def test_out_of_bounds(self):
        mat = Matrix2D(2, 2)
        with self.assertRaises(IndexError):
            mat.get(2, 0)
        with self.assertRaises(IndexError):
            mat.set(-1, 0, 1.0)

    def test_to_1d_and_from_1d(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        mat = Matrix2D(2, 2, data)
        arr_1d = mat.to_1d()
        self.assertEqual(arr_1d, [1.0, 2.0, 3.0, 4.0])

        restored = Matrix2D.from_1d(arr_1d, 2, 2)
        self.assertEqual(restored.data, data)


class TestMACEngine(unittest.TestCase):
    def test_mac_2d_basic(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[2.0, 0.5], [1.0, -1.0]]
        # 1*2 + 2*0.5 + 3*1 + 4*(-1) = 2 + 1 + 3 - 4 = 2.0
        result = mac_2d(a, b)
        self.assertAlmostEqual(result, 2.0)

    def test_mac_1d(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 0.5, 1.0, -1.0]
        result = mac_1d(a, b)
        self.assertAlmostEqual(result, 2.0)


class TestLabelNormalizer(unittest.TestCase):
    def test_cross_variants(self):
        for raw in ["+", "cross", "Cross", "CROSS"]:
            self.assertEqual(normalize_label(raw), "Cross")

    def test_x_variants(self):
        for raw in ["x", "X"]:
            self.assertEqual(normalize_label(raw), "X")


class TestDecisionEngine(unittest.TestCase):
    def test_mode1_decisions(self):
        self.assertEqual(decide_mode1(5.0, 1.0), "A")
        self.assertEqual(decide_mode1(1.0, 5.0), "B")
        # Diff < 1e-9 should be undecided
        self.assertIn("판정 불가", decide_mode1(1.0, 1.0 + 1e-10))

    def test_mode2_decisions(self):
        self.assertEqual(decide_mode2(10.0, 2.0), "Cross")
        self.assertEqual(decide_mode2(2.0, 10.0), "X")
        # Diff < 1e-9 should be UNDECIDED
        self.assertEqual(decide_mode2(5.0, 5.0 + 1e-11), "UNDECIDED")


class TestPatternGenerator(unittest.TestCase):
    def test_cross_generator(self):
        mat = PatternGenerator.generate_cross(3)
        expected = [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
        self.assertEqual(mat, expected)

    def test_x_generator(self):
        mat = PatternGenerator.generate_x(3)
        expected = [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
        self.assertEqual(mat, expected)


class TestMode2Integration(unittest.TestCase):
    def test_run_mode_2_with_data_json(self):
        json_path = os.path.join(os.path.dirname(__file__), "data.json")
        if os.path.exists(json_path):
            total, passed, failed = run_mode_2(json_path)
            self.assertEqual(total, 6)
            self.assertEqual(passed, 3)
            self.assertEqual(failed, 3)


if __name__ == "__main__":
    unittest.main()
