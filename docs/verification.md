# V0 verification — 2026-09-29

This page preserves the **initial** verification, including the missed opening head event. The subsequent user-reported regression, corrected training and current results are documented in [head-impact-fix.md](head-impact-fix.md). Original artifacts are backed up under `runs/head_fix/before`; the default checkpoint and `runs/evaluation_events.*` now refer to the corrected model.

This records the implemented foundation, not a claim of general gameplay competence.

## Completed execution

- `uv sync` and `uv lock --check` passed with isolated CPU dependencies.
- The example generator produced 288 silent H.264 clips and a separate 20-second test video. Validation accepted source-disjoint train/val/test records.
- Forty training epochs used actual ImageNet-pretrained MobileNet visual features. The trainable parameter change L2 was 13.5555; the saved best checkpoint was epoch 39.
- Independent validation decoded all 72 validation clips and reproduced the feature-cache metrics.
- Full checkpoint reload and TorchScript export numerical parity passed. Export includes the visual encoder, not just prediction heads.
- `uv run uhm evaluate-stream` ran 160 causal windows through the checkpoint and saved predictions, canonical events and a profile. Inference did not read annotations.
- `uv run uhm demo` successfully prepared/reused artifacts, built the frontend and started localhost:8000.

## Automated checks

| Check | Result |
| --- | --- |
| Python tests | 56 passed |
| Frontend tests | 11 passed, two suites |
| Ruff | Passed for `src` and `tests` |
| TypeScript + Vite production build | Passed |
| Python dependency lock | Current |

Tests cover causal frame sampling; source and file split leakage; annotation bounds; unknown/ambiguous semantics; confidence gates, duplicate suppression, cooldown and seek reset; maximum-cardinality event matching; real gradient updates, frozen encoder state, checkpoint compatibility and export; HTTP video ranges and WebSocket validation/isolation/logging; media-time pause/fade, stale replies and EOF timestamps.

Some unit tests inject a deterministic test-only predictor to isolate WebSocket behavior. Production `ModelPredictor` always loads the actual checkpoint. The separate manual WebSocket smoke used the real model and produced a player `left_torso` event at 3625ms.

The test run includes upstream Starlette test-client deprecation and PyTorch tracing warnings. Tests and numerical export parity pass for the stated input contract; this does not establish arbitrary trace-shape compatibility.

## Browser verification

- Default mode is model inference; manual mock is explicit and isolated.
- The configured video loads and plays. HTTP range delivery returns 206 for seeking.
- Full-video runs produce actual checkpoint events and update the body map. Unknown events appear in the log and do not color a region.
- A real left-torso event remained visibly red after pausing at approximately 3.7 seconds, confirming a media-time renderer clock.
- Native video controls and the playback-speed selector are available. Seek/rate change resets temporal state and invalidates stale replies.
- A service restart showed an unavailable state; reconnect restored inference. No mock fallback occurred.
- A user-facing browser preview is retained. The production service uses port8000; the temporary development server was stopped.

## Fixed-cadence held-out synthetic stream

Source: `data/raw/demo.mp4`; reference: `data/annotations/demo.jsonl`; 125ms sample step. Matching is one-to-one with maximum cardinality, then minimum onset distance, within each same-class reference interval plus/minus250ms.

| Quantity | Result |
| --- | --- |
| Reference player events | 9 |
| Emitted player events | 7 |
| Matched / false / missed | 6 / 1 / 3 |
| Event precision / recall | 85.7% / 66.7% |
| False events per minute | 3.0, from one false event in only20seconds |
| Correct regions among localized observable matches | 6/6 |
| Localization coverage across all observable references | 6/8 |
| Unobservable matched events | 0; false-localization rate therefore unavailable |
| Duplicate emissions | 0 |
| Median media-time detection delay of matches | 208ms |
| CPU decode+preprocess+model p50/p95 | Approximately39/48ms |

Raw artifacts: `runs/evaluation_events.jsonl`, `.predictions.jsonl`, `.profile.json`, `.metrics.json`; training artifacts: `runs/training_metrics.json`, `training_history.json`, `validation_metrics.json`.

The head event was missed, the left-leg event lacked consecutive eligible observations, and the occluded event was predicted early. The last case counts as both an unmatched prediction and a missed reference at the declared tolerance. We did not relax thresholds on this test video to hide these errors.

GPU utilization/VRAM in CPU mode and full source-onset-to-browser-display latency remain unmeasured and are reported as null. The future 80ms/150ms end-to-end targets have not been demonstrated. All example scores are specific to the procedural renderer; real gameplay and perceptual intensity calibration remain unvalidated.
