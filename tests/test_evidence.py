from __future__ import annotations

import unittest

from limen_runtime_audit.evidence import EvidenceRun


class TestEvidenceChain(unittest.TestCase):
    def _build_run(self, profile: str = "generation_probability") -> EvidenceRun:
        run = EvidenceRun(capture_profile=profile)
        for i in range(3):
            step = run.new_step()
            step.set_channel("generation", "AVAILABLE", {"token_id": 100 + i})
            step.set_channel("probability", "AVAILABLE", {"entropy_nats": 0.5 + i})
            step.set_channel("latent", "NOT_CAPTURED", note="profile does not request latent")
            run.append(step)
        return run

    def test_valid_chain_verifies(self) -> None:
        run = self._build_run()
        ok, broken_at = run.verify_chain()
        self.assertTrue(ok)
        self.assertIsNone(broken_at)

    def test_tamper_detected(self) -> None:
        """Mirrors the project's own kill-test discipline: editing a sealed
        step's payload after the fact must break the chain, not pass
        silently."""
        run = self._build_run()
        tampered = run.steps[1]
        tampered.channels["generation"].value["token_id"] = 999999
        ok, broken_at = run.verify_chain()
        self.assertFalse(ok)
        self.assertEqual(broken_at, run.steps[1].event_id)

    def test_reorder_detected(self) -> None:
        """Shuffling step order must also be caught, since identity and
        chaining are explicit, never inferred from array position."""
        run = self._build_run()
        run.steps[0], run.steps[1] = run.steps[1], run.steps[0]
        ok, broken_at = run.verify_chain()
        self.assertFalse(ok)

    def test_not_captured_excluded_from_denominator(self) -> None:
        run = self._build_run(profile="generation_probability")
        cov = run.coverage()
        # latent is NOT_CAPTURED and not in this profile's expected set, so
        # it must not appear in expected_channels or reduce coverage.
        self.assertNotIn("latent", cov["expected_channels"])
        self.assertEqual(cov["available"], cov["expected_total"])
        self.assertEqual(cov["coverage_fraction"], 1.0)

    def test_failed_channel_is_flagged_not_silently_zero(self) -> None:
        run = EvidenceRun(capture_profile="generation_probability")
        step = run.new_step()
        step.set_channel("generation", "AVAILABLE", {"token_id": 1})
        step.set_channel("probability", "FAILED", note="backend returned no logits")
        run.append(step)
        cov = run.coverage()
        self.assertEqual(cov["available"], 1)
        self.assertEqual(cov["expected_total"], 2)
        self.assertIn(f"{step.event_id}:probability", cov["failed_events"])

    def test_available_channel_requires_value(self) -> None:
        run = EvidenceRun(capture_profile="generation_only")
        step = run.new_step()
        with self.assertRaises(ValueError):
            step.set_channel("generation", "AVAILABLE", value=None)

    def test_absent_channel_must_not_carry_value(self) -> None:
        run = EvidenceRun(capture_profile="generation_only")
        step = run.new_step()
        with self.assertRaises(ValueError):
            step.set_channel("generation", "NOT_CAPTURED", value={"token_id": 1})

    def test_out_of_order_step_rejected(self) -> None:
        run = EvidenceRun(capture_profile="generation_only")
        step0 = run.new_step()
        step0.set_channel("generation", "AVAILABLE", {"token_id": 1})
        run.append(step0)
        # Manually construct a step with a wrong index to simulate a bug
        # that would otherwise silently desynchronize identity.
        from limen_runtime_audit.evidence import EvidenceStep

        bad_step = EvidenceStep(run_id=run.run_id, step_index=5)
        bad_step.set_channel("generation", "AVAILABLE", {"token_id": 2})
        with self.assertRaises(ValueError):
            run.append(bad_step)

    def test_to_dict_reports_chain_integrity(self) -> None:
        run = self._build_run()
        out = run.to_dict()
        self.assertTrue(out["chain_integrity"]["valid"])
        self.assertEqual(out["coverage"]["available"], 6)


if __name__ == "__main__":
    unittest.main()
