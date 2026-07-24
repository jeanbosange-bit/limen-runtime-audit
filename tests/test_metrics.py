import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from limen_runtime_audit.audit import audit_arrays, write_report
from limen_runtime_audit.metrics import layer_trajectory_metrics, probability_baselines


class MetricsTests(unittest.TestCase):
    def test_straight_trajectory_has_unit_tortuosity_and_zero_turning(self):
        base = np.arange(5, dtype=float)[None, :, None]
        hidden = np.concatenate([base, np.zeros_like(base)], axis=2)
        metrics = layer_trajectory_metrics(hidden)
        self.assertAlmostEqual(metrics["tortuosity"][0], 1.0)
        self.assertAlmostEqual(metrics["mean_turning_angle_rad"][0], 0.0)

    def test_probability_baselines_are_finite(self):
        metrics = probability_baselines(
            np.array([[3.0, 1.0, -2.0], [0.0, 0.0, 0.0]])
        )
        self.assertTrue(np.isfinite(metrics["entropy_nats"]).all())
        self.assertAlmostEqual(metrics["top1_top2_margin"][1], 0.0)

    def test_audit_contract(self):
        rng = np.random.default_rng(7)
        audit = audit_arrays(
            rng.normal(size=(4, 5, 8)),
            rng.normal(size=(4, 11)),
            {"model_revision": "test-revision"},
        )
        self.assertEqual(audit["schema_version"], "limen.audit.v1")
        self.assertEqual(
            audit["input"]["metadata"]["model_revision"], "test-revision"
        )
        self.assertIn("baseline.entropy_nats", audit["summary"])

    def test_rejects_wrong_hidden_shape(self):
        with self.assertRaises(ValueError):
            layer_trajectory_metrics(np.zeros((4, 8)))

    def test_rejects_nonfinite_hidden_states(self):
        hidden = np.zeros((2, 4, 3))
        hidden[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            layer_trajectory_metrics(hidden)

    def test_constant_trajectory_has_undefined_tortuosity(self):
        hidden = np.ones((2, 4, 3))
        metrics = layer_trajectory_metrics(hidden)
        self.assertTrue(np.isnan(metrics["tortuosity"]).all())
        self.assertTrue((metrics["path_length"] == 0).all())

    def test_rejects_wrong_logits_shape(self):
        with self.assertRaises(ValueError):
            probability_baselines(np.zeros((3,)))

    def test_rejects_nonfinite_logits(self):
        logits = np.zeros((2, 3))
        logits[0, 0] = np.inf
        with self.assertRaises(ValueError):
            probability_baselines(logits)

    def test_write_report_records_source_hash_and_outputs(self):
        rng = np.random.default_rng(9)
        hidden = rng.normal(size=(3, 4, 5))
        audit = audit_arrays(hidden)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "trajectory.npz"
            np.savez_compressed(source, hidden_states=hidden)
            output = root / "audit"
            write_report(audit, output, source)
            stored = json.loads((output / "audit.json").read_text())
            self.assertEqual(stored["input"]["source_file"], "trajectory.npz")
            self.assertEqual(len(stored["input"]["source_sha256"]), 64)
            self.assertTrue((output / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
