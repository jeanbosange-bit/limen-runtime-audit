from __future__ import annotations

"""Evidence-chain primitives for multi-signal LLM runtime observability.

This module implements a *generic, adapted* pattern for linking several
observation signals captured from a single generation event (what was
generated, how the output distribution looked, and how far internal
representations moved) under one explicit identity, with a tamper-evident
hash chain and a three-way absence taxonomy.

It intentionally does not implement any decision policy, scoring model, or
threshold. It only records what was observed and makes what was *not*
observed explicit and honest, rather than silently defaulting to zero or
being inferred from array position.

Design principles (kept from the audit module's existing scientific-status
discipline):
  - a measured signal may produce a link; a missing signal produces an
    explicit absence, never an invented one.
  - identity is asserted through explicit fields, never guessed from file
    order or array position.
  - synchronization is validated, not assumed.
"""

import hashlib
import json
import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

SCHEMA_VERSION = "limen.evidence.v1"

# Three-way absence taxonomy.
#   NOT_CAPTURED  -- a legitimate, declared omission (this run's capture
#                    profile never intended to record this signal).
#   NOT_SUPPORTED -- the capture backend cannot produce this signal in this
#                    environment (e.g. no logits access, sampling backend).
#   FAILED        -- the capture profile required this signal and it is
#                    missing anyway; this is an anomaly, not a design choice.
AbsenceReason = Literal["NOT_CAPTURED", "NOT_SUPPORTED", "FAILED"]

# A capture profile declares, in advance, which signal channels this run
# intends to populate. Each channel is either "expected" or "not_expected";
# only "expected" channels that end up empty count as FAILED.
CAPTURE_PROFILES: dict[str, dict[str, bool]] = {
    "generation_only": {"generation": True, "probability": False, "latent": False},
    "generation_probability": {"generation": True, "probability": True, "latent": False},
    "full_sampled": {"generation": True, "probability": True, "latent": True},
}

CHANNELS = ("generation", "probability", "latent")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_of(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ChannelObservation:
    """One signal channel's contribution to a single generation step."""

    channel: str
    status: Literal["AVAILABLE", "NOT_CAPTURED", "NOT_SUPPORTED", "FAILED"]
    value: dict[str, Any] | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "status": self.status,
            "value": self.value,
            "note": self.note,
        }


@dataclass
class EvidenceStep:
    """A single synchronized observation event across available channels.

    Identity hierarchy (explicit, never positional):
        run_id -> event_id -> step_index
    """

    run_id: str
    step_index: int
    event_id: str = field(init=False)
    captured_at_utc: str = field(default_factory=_utc_now)
    channels: dict[str, ChannelObservation] = field(default_factory=dict)
    content_sha256: str | None = field(default=None, init=False)
    prev_sha256: str | None = None

    def __post_init__(self) -> None:
        self.event_id = f"{self.run_id}::t{self.step_index}"

    def set_channel(
        self,
        channel: str,
        status: Literal["AVAILABLE", "NOT_CAPTURED", "NOT_SUPPORTED", "FAILED"],
        value: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> None:
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel: {channel}")
        if status == "AVAILABLE" and value is None:
            raise ValueError("AVAILABLE channel requires a value")
        if status != "AVAILABLE" and value is not None:
            raise ValueError(f"{status} channel must not carry a value")
        self.channels[channel] = ChannelObservation(channel, status, value, note)

    def seal(self, prev_sha256: str | None) -> str:
        """Freeze this step and compute its content hash, chained to the
        previous step's hash. Returns the new hash. Must be called exactly
        once per step, after all channels for this step are set."""
        if self.content_sha256 is not None:
            raise RuntimeError(f"step {self.event_id} already sealed")
        self.prev_sha256 = prev_sha256
        payload = self.to_dict(include_hash=False)
        payload["prev_sha256"] = prev_sha256
        self.content_sha256 = _sha256_of(payload)
        return self.content_sha256

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        out = {
            "schema": SCHEMA_VERSION,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "step_index": self.step_index,
            "captured_at_utc": self.captured_at_utc,
            "channels": {k: v.to_dict() for k, v in self.channels.items()},
        }
        if include_hash:
            out["prev_sha256"] = self.prev_sha256
            out["content_sha256"] = self.content_sha256
        return out


@dataclass
class EvidenceRun:
    """A sequence of EvidenceStep objects sharing one run_id and one
    declared capture profile, with a genesis-anchored hash chain."""

    capture_profile: str
    run_id: str = field(default_factory=lambda: new_id("run"))
    created_at_utc: str = field(default_factory=_utc_now)
    steps: list[EvidenceStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capture_profile not in CAPTURE_PROFILES:
            raise ValueError(f"unknown capture profile: {self.capture_profile}")
        genesis_payload = {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "capture_profile": self.capture_profile,
            "created_at_utc": self.created_at_utc,
            "runtime": {"python": platform.python_version()},
        }
        self._genesis_sha256 = _sha256_of(genesis_payload)

    def new_step(self) -> EvidenceStep:
        return EvidenceStep(run_id=self.run_id, step_index=len(self.steps))

    def append(self, step: EvidenceStep) -> None:
        expected_index = len(self.steps)
        if step.step_index != expected_index:
            raise ValueError(
                f"step_index {step.step_index} out of order (expected {expected_index})"
            )
        prev_hash = self.steps[-1].content_sha256 if self.steps else self._genesis_sha256
        step.seal(prev_hash)
        self.steps.append(step)

    def verify_chain(self) -> tuple[bool, str | None]:
        """Recompute every hash from scratch and compare. Returns
        (is_valid, first_broken_event_id_or_None). This is the same kind of
        check a tamper/kill test exercises: a shuffled or edited step must
        be detectable, never silently accepted."""
        prev = self._genesis_sha256
        for step in self.steps:
            payload = step.to_dict(include_hash=False)
            payload["prev_sha256"] = prev
            recomputed = _sha256_of(payload)
            if recomputed != step.content_sha256 or step.prev_sha256 != prev:
                return False, step.event_id
            prev = step.content_sha256
        return True, None

    def coverage(self) -> dict[str, Any]:
        """Fraction of expected-per-profile channel slots that are
        AVAILABLE. NOT_CAPTURED channels (declared out of scope for this
        profile) are excluded from the denominator entirely -- a
        deliberately unobserved signal is not a penalty."""
        expected = CAPTURE_PROFILES[self.capture_profile]
        numerator = 0
        denominator = 0
        failed: list[str] = []
        for step in self.steps:
            for channel, is_expected in expected.items():
                if not is_expected:
                    continue
                denominator += 1
                obs = step.channels.get(channel)
                if obs is not None and obs.status == "AVAILABLE":
                    numerator += 1
                elif obs is not None and obs.status == "FAILED":
                    failed.append(f"{step.event_id}:{channel}")
        return {
            "capture_profile": self.capture_profile,
            "expected_channels": [c for c, v in expected.items() if v],
            "available": numerator,
            "expected_total": denominator,
            "coverage_fraction": (numerator / denominator) if denominator else None,
            "failed_events": failed,
        }

    def to_dict(self) -> dict[str, Any]:
        chain_ok, broken_at = self.verify_chain()
        return {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "capture_profile": self.capture_profile,
            "created_at_utc": self.created_at_utc,
            "genesis_sha256": self._genesis_sha256,
            "steps": [s.to_dict() for s in self.steps],
            "coverage": self.coverage(),
            "chain_integrity": {"valid": chain_ok, "first_broken_event_id": broken_at},
            "interpretation_boundary": (
                "This evidence chain records what was observed and what was "
                "declared out of scope. It does not, by itself, imply that "
                "any one channel caused another, and it carries no decision "
                "policy or trained threshold."
            ),
        }
