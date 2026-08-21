# Phase 9 publication provenance index

## Freeze status

Phase 9 is **FULLY_RECONSTRUCTIBLE**. Reconstruction branch:
`agent/2026-08-phase9-reconstruction`. Relevant commits are:

- `e338d2cb921130b362a18d3ce71b3c2026a99412` — Phase 9 analysis replay;
- `52a84753f39535de1a61258ba0c5bc6f0f7e2e33` — `extract_core.py` recovery;
- `bcca694bf9eb0614809ca9c101f4f375a603f86f` — geometry control and packaging;
- `d5c9f914032b75347b2f53d8e1ac9e23d3a38c6c` — scalar readout extraction.

The branch was local at freeze time. This index does not assert that it was pushed.

## Recovered scripts

| Script | SHA-256 | Recovery evidence |
|---|---|---|
| `phase9_logit_lens.py` | `5d63c66b108f34fe12d5e8ded0853a6219e5982f0ee8a234333b7cd37c24a40e` | `reconstructed_phase9/PHASE9_RECONSTRUCTION_REPORT.md` |
| `extract_core.py` | `18033dbd8be2a3684f9b53e4c756a5e358de32bf2c3811b517935ad270d37a8c` | `reconstructed_extract_core/EXTRACT_CORE_RECONSTRUCTION_REPORT.md` |
| `build_geometry_control.py` | `e58a9e09016b12a80e02d1741aacf1f6c552f2bf4132f9221b3ea2ef15ca0670` | `reconstructed_geometry_control/SCRIPT_PROVENANCE.md` |
| `finalize_and_quality.py` | `754f100f6cb46d61169d680f4e7b54f694e5fe76f1c6bbc4fe9418d5866cbbf5` | same |
| `build_transfer_report_and_package.py` | `730d5a6236f22ee011a4b3b1d8ba3ea436d77b31589f33f005dd914bad54f653` | same |
| `extract_features.py` | `c8fa3842e991ba99f1ba5987b26d4c549437dc86e5549ece08d849e08e41a80a` | `reconstructed_scalar_readout/SCRIPT_RECOVERY.md` |

Recovered code provenance comes from Claude session
`5d793c3e-5e08-4a66-9200-8acfbc5066c9`. Chronological write/edit details and
uncertainties are retained in the referenced reports.

## Immutable inputs and packaged tables

| Artifact | SHA-256 |
|---|---|
| `multimodel-expansion-2026-08-19/RUN_INDEX.csv` | `8a4841e34fc060cadab2292a85687ff8d84ec4434bbd5af779ac8ca7229566d0` |
| `PROMPT_PANEL_REFERENCE.json` | `d24495ae0995d9bcc98a73ca245260bf74092b02adefcdcfd4b6194d1bf6afcc` |
| `RAW_ARTIFACT_REGISTRY.json` | `22a0cdfba29e1328faab9d2e61a631f626930c0f70ef01a2180b109b6c734f7c` |
| `PROJECTION_METADATA.json` | `c1d68c74afe7539579fa6818eac6f336fdb1fc5f323726d6578123d5ea0bf236` |
| `h_random_projection.parquet` | `0b5c204a7d1fd880d0cd8e1b3f996e4215b54587efba8be693b5d38b0e8f177b` |
| `orthogonal_subset.parquet` | `13414276f1a476d5e856f0f41cc63d0f00f49ed0eccad5ae1389450ddfd7b543` |
| `readout_features.parquet` | `e45ecf217851bb701a1f9b1929d5151334f40cdd08b707220971960e27ac773d` |

Per-raw-file hashes are retained in `RUN_INDEX.csv` and
`RAW_ARTIFACT_REGISTRY.json`; raw tensors remain outside Git. The original transfer
hash list is `deep-feature-extraction-2026-08-19/TRANSFER_SHA256SUMS.txt`.

## Readout contract and coverage

These sources are not interchangeable:

- **INTERMEDIATE_PROJECTED** — `extract_core.py` applies the recovered final norm/head
  to archived intermediate hidden states at strided layers/steps.
- **FINAL_LAYER_RECOMPUTED_READOUT** — strided final-layer rows recomputed by
  `extract_core.py`; the stored historical `readout_type` string is `FINAL_NATIVE`, but
  provenance is recomputed, not generation-native.
- **GENERATION_NATIVE_OUTPUT** — dense `FINAL_NATIVE` rows computed by
  `extract_features.py` from captured generation logits.

`FUTURE_T_PLUS_2` and `FUTURE_T_PLUS_3` also come from `extract_features.py`. Packaging
gives generation-native rows precedence on shared keys and appends only unmatched
recomputed final-layer rows.

Scalar reconstruction coverage is 18 models and 648 runs. This is historical pipeline
coverage, not the Phase 9 scientific panel. Effective Phase 9 cross-model analysis is
8 models satisfying extraction/alignment criteria. Phase 9 explicitly filters and
consumes **INTERMEDIATE_PROJECTED**; its layerwise analysis does not depend on the
generation-native scalar or future-readout rows.

## Verification evidence

- `reconstructed_phase9/REPLAY_VERIFICATION.csv`
- `reconstructed_extract_core/EXTRACT_CORE_REPLAY_VERIFICATION.csv`
- `reconstructed_geometry_control/GEOMETRY_CONTROL_REPLAY_VERIFICATION.csv`
- `reconstructed_geometry_control/FINAL_TABLE_REPLAY_VERIFICATION.csv`
- `reconstructed_scalar_readout/SCALAR_READOUT_REPLAY_VERIFICATION.csv`
- `reconstructed_scalar_readout/PACKAGING_REPLAY_VERIFICATION.csv`
- `reconstructed_scalar_readout/FINAL_SCALAR_READOUT_RECONSTRUCTION_REPORT.md`

Orthogonal projection, geometry-destroying control, intermediate projected readout,
generation-native readout, future readout, and final packaging are each covered by the
above verification chain. Final packaged readout is byte-identical.

## Phase 9 outputs

Primary outputs under `reports/decodability-v0-2026-08-19/`:

- `phase9_decodability_all8.csv` — `8eab11db184860d738c84f6c3f3113d41ddaa3f58efd747b8f5a4a8acddba5df`
- `phase9_readout_agg.csv` — `077fd495e2e8658f079dbe0b528eda35e0b74d39a790f9e6deaca03ec740362b`
- `aligned_depth_summary.csv` — `643c4e6e3bbc9e6154af1b31e7a66f6453e0236179e320005c589709e60a7690`
- `model_ordering_summary.csv` — `c45b5acd86fe6598188972b7a0b6294c73b509265f60e53d8a0d3aa40bee9a19`
- `correlation_summary.csv` — `ccfaf17c4e91c260dd95db2c6df289a19d1bf96cdc28d3dd85012fffb1880c95`
- `PHASE9_LOGIT_LENS_COMPARISON.md` — `3b7be509931b4d00ed0226965bdbe5446528db46433d3967087418a20fbb7ca4`

Threshold-free interpretation is frozen in
`PHASE9_THRESHOLD_FREE_AUDIT.md` (`b915d2a2fb0556b81ee4841d875a956bfb1f31886c96b519904b8810c0d448fe`)
and `RPTA_STATUS_AFTER_PHASE9.md`
(`d2eb513450542032026c0ef289c2f0f251fa8f80f9053a5392bdcdce3224b7ab`).

## Scientific interpretation

File provenance establishes replayability; it does not establish a mechanism.

- Phase 9 readout interpretation: **SUPPORTED_DESCRIPTIVELY**.
- Final-native/intermediate-projected distinction: **SUPPORTED_DESCRIPTIVELY**.
- RPTA: **PARTIALLY_SUPPORTED**.
- Common sequential mechanism: **NOT_ESTABLISHED**.
- Unseen-prompt generalization: **NOT_TESTED**.
- Prediction: **NOT_TESTED**.
- Control: **NOT_TESTED**.

Improved reproducibility does not upgrade the scientific claim level.

## Stop condition

Phase 9 reconstruction is closed. Do not replay it again, regenerate its artifacts,
or add models merely to increase panel size without a separately approved scientific
question.

