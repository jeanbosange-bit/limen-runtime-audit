#!/usr/bin/env python3
"""Offline unit tests for extract_limen_trajectory.py."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from extract_limen_trajectory import (
    choose_device,
    choose_dtype,
    sha256_file,
    validate_arrays,
)


class TestValidation(unittest.TestCase):
    def test_valid_arrays(self):
        hidden = np.zeros((4, 6, 8), dtype=np.float32)
        logits = np.zeros((4, 32), dtype=np.float32)
        validate_arrays(hidden, logits)

    def test_hidden_rank_rejected(self):
        with self.assertRaises(ValueError):
            validate_arrays(np.zeros((4, 8)), None)

    def test_nonfinite_hidden_rejected(self):
        hidden = np.zeros((4, 6, 8))
        hidden[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_arrays(hidden, None)

    def test_token_misalignment_rejected(self):
        hidden = np.zeros((4, 6, 8))
        logits = np.zeros((3, 32))
        with self.assertRaises(ValueError):
            validate_arrays(hidden, logits)


class TestRuntimeChoices(unittest.TestCase):
    def test_cpu_auto_dtype(self):
        self.assertEqual(choose_dtype(torch.device("cpu"), "auto"), torch.float32)

    def test_explicit_dtype(self):
        self.assertEqual(
            choose_dtype(torch.device("cpu"), "bfloat16"),
            torch.bfloat16,
        )

    def test_explicit_cpu(self):
        self.assertEqual(choose_device("cpu").type, "cpu")


class TestHashing(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.bin"
            path.write_bytes(b"limen")
            self.assertEqual(
                sha256_file(path),
                "96ad72da603bfdffc6d04b3b6a22f9f90b546d1ced158ec3671d46e672457255",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
