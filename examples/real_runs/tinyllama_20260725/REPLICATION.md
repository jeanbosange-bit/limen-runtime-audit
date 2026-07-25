# Exact replication — TinyLlama — 2026-07-25

The original extraction was repeated without changing the model, revision,
prompt, decoding, token budget, extraction point, software environment or
device.

## Verdict

`EXACT_REPLICATION`

| Check | Result |
|---|---|
| Stored array names | Identical |
| `hidden_states` | Bitwise identical; maximum absolute difference `0.0` |
| `logits` | Bitwise identical; maximum absolute difference `0.0` |
| `token_ids` | Bitwise identical; maximum absolute difference `0.0` |
| Fixed scientific metadata | All checked fields identical |
| Original NPZ SHA-256 | `22e46f57d76d8c031ad81954fbd86c8510fd75083e1eefef908cb1782985baf2` |
| Replication NPZ SHA-256 | `22e46f57d76d8c031ad81954fbd86c8510fd75083e1eefef908cb1782985baf2` |

Both runs generated:

> The sky appears blue because the sunlight reflects off the Earth's atmosphere

## Interpretation

- `[I]` The complete exported payload is exactly reproducible for these two
  executions under the locked greedy-decoding protocol.
- `[I]` The previously published metric summary is therefore reproduced
  exactly; recomputing identical deterministic arrays would add no new
  numerical estimate.
- `[III]` This supports using the first run as a stable engineering fixture for
  later matched comparisons.

This does not establish reproducibility under sampling, a different device,
software version or architecture. It also does not validate the measurements
as model-quality scores, functional localizers or causal signals.

The machine-readable evidence is in [`replication.json`](replication.json).
