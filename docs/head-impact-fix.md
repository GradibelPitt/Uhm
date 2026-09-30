# Opening head-impact regression — 2026-09-29

The first demo impact was genuinely missed by the learned classifier. The previous delivery's high clip-validation scores did not establish reliable continuous playback. The original checkpoint is retained at `runs/head_fix/before/baseline.pt`.

## Diagnosis

- The original model also missed the exact isolated head segment, with impact probability about 0.000259 and player probability about 0.000309. This rules out a renderer-only failure and means temporal sampling mismatch alone was not the explanation.
- All training/validation head clips passed, but small translations or background-color changes flipped classification. A fixed block of 32 additional low-salience head examples passed the unchanged gates in only 24 cases. The model had learned a brittle dependence on the synthetic scene's layout/appearance.
- Original whole-second training labels included the approach before contact, while playback used trailing windows. This separately caused inaccurate event timing and premature occluded-event emission.
- The frontend's latest-region caption was sourced from retained event history and could remain stale after seeking. This was a distinct UI issue.

## Changes

Training now uses full source recordings and runtime-equivalent trailing 1000ms windows at a 125ms cadence, with six causal frames per window. Each target describes the current timestamp. The synthetic generator supplies explicit contact intervals starting at frame 5 (417ms into each contact second), with complete negative intervals before contact and between events.

There are 1728 base training windows from the original six sources, and 576 unaugmented validation windows from the original two validation sources. Frozen pretrained features are cached for the original training view and three fixed random augmented views. Translation, brightness, contrast and RGB offsets are shared across all frames of a view; there are no anatomical flips or future frames. No demo or fresh-evaluation video was added to training. Forty epochs selected epoch 38 by validation loss. Full reload inference and independent pixel validation passed.

Runtime thresholds, confirmation, release and cooldown are unchanged. There are no head-specific rules, demo timestamps or annotation lookups in inference. The demo's video bytes are unchanged: SHA256 `7c2076de9c51b9836bf0e101c4437eff579f2adf270c11f306f80e6a01304b26`.

Latest-response state is now separate from historical event rows and is cleared on seek/reset/reconnect and mode changes. A new per-region joint recall metric includes missed observable reference events in its denominator.

## Actual checkpoint regression

Command: `uv run python scripts/evaluate_head_regression.py --checkpoint checkpoints/baseline.pt --assert-head`.

The suite runs the actual pixel model and a fresh event aggregator for each case. All inference finishes before evaluation labels are loaded. It tests the original demo at sampling offsets 0/40/80ms and replay starts 0/1000/1250ms, plus full streams for independent seeds 100100–100102. Replay start means clearing temporal aggregation while permitting preceding frames from the prerecorded file, as normal seeking does.

| Evidence | Result |
| --- | --- |
| Opening head detected and correctly localized within contact interval | 12/12 cases |
| Original demo head confirmation time | 1580–1665ms; contact begins at 1417ms |
| Media-time head delay | 163–248ms, before transport/render delay |
| Player events per full stream | 9/9 detected |
| Visible body regions per full stream | 8/8 correctly localized |
| Strict negative, NPC/environment emissions | 0 |
| Occluded contact falsely assigned a body region | 0 |
| Duplicate emissions | 0 |
| Additional 32 low-salience head windows | 32/32 eligible, formerly 24/32 |

Raw reports: `runs/head_fix/acceptance/summary.json`, per-case JSON including predictions/events, and `low_head_32_comparison.json`. The original demo is now a regression fixture informed by the reported bug. The additional fixed seeds provide independent examples within the same renderer, not a new visual domain.

All generated source, validation and demo videos are **third-person, front-facing synthetic scenes**. Dataset annotations explicitly use `viewpoint: third_person`; the contract also supports `first_person` and `unknown`. Future first-person data should have separate annotation files and evaluation. These results do not establish first-person or real-gameplay performance, calibrated perceptual intensity, or the proposed end-to-end latency target.

## Deployed verification

- The default `baseline.pt` contains exactly the validated candidate's model tensors. Only provenance was enriched with the user's requested viewpoint field; the actual pre-enrichment training annotation SHA256 and snapshot remain recorded. `runs/head_fix/deployment.json` records both checkpoint hashes and exact tensor equality.
- The deployed checkpoint's full-stream evaluation reproduces 9/9 events and 8/8 correct visible locations. Pixel-model TorchScript export passed numerical parity.
- 92 Python tests, 17 frontend tests, Ruff and production build passed. Warnings are the existing upstream test-client/tracing notices.
- Browser playback through the restarted local service emitted a real head event at 1.64s with intensity 0.36. The page was paused with the head visibly red and `Third-person · front view` displayed. The corresponding runtime session is `runs/session-e5c2cf70f4244a59924db32d1ee06714.events.jsonl`.
