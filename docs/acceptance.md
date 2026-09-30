# V0 acceptance checks

This checklist defines evidence for a working foundation. It does not establish general gameplay understanding or validated haptic intensity calibration.

## Contract and observability

- Body regions and event classes have a single canonical ontology; `unknown` is valid. A training `no_event` label is distinct from an emitted physical event.
- Event, recipient, region, and intensity confidence remain separate. Values, timestamps, and durations are validated at the interface boundary.
- Only `recipient=player` events can illuminate the body map. Unknown regions remain in the event log and illuminate no specific region.
- Unobservable location is trained as `unknown`, or a trained observability gate suppresses localization. Masking unobservable region losses alone does not teach abstention.
- Anatomical left/right is labeled and independent of screen direction. Camera shake or screen position alone does not justify a body location.

## Trainable baseline

- The visual encoder loads identified pretrained weights. Missing weights produce an explicit error, never an undisclosed randomly initialized fallback.
- Training changes trainable parameters. Validation executes independently of training, and saved checkpoints reload for inference.
- Checkpoints identify label order, ontology, architecture, pretrained weights, preprocessing, and temporal configuration.
- Train and validation splits separate source videos. Related clips or adjacent windows from one source cannot straddle splits.
- Runtime windows use only frames available by their timestamp. Any future-frame or buffering delay is disclosed and included in latency reporting.

## Runtime and visualization

- Consecutive detections of one impact form one temporal event; release evidence allows a later impact to become a new event.
- Gating, smoothing, cooldown, and duplicate suppression are exercised with positive, negative, uncertain, and repeated-event sequences.
- Video timestamps drive event alignment and rendering. Pause, seek, restart, and playback speed changes are handled deliberately.
- Seeking or changing sources clears temporal state and invalidates stale inference results, messages, and highlights.
- Regions respond independently, intensity visibly changes red shading, and duration is followed by a smooth fade. Unknown regions never fall back to a guessed region.
- One documented command starts video playback, checkpoint inference, event transport, body visualization, and persistent event logging.
- Mock renderer events and real checkpoint inference have explicit, different mode labels.

## Metrics and denominators

Always report counts, denominators, decision thresholds, evaluated video duration, matching tolerance, and data split. Undefined metrics are unavailable, not silently zero or perfect.

| Metric | Definition |
| --- | --- |
| Event precision | Matched player event predictions / all emitted player event predictions. Use one-to-one temporal matching and state class/recipient matching rules. |
| Event recall | Matched annotated player events / all annotated player events. |
| False events per minute | Unmatched emitted player events / evaluated video minutes. Include negative footage. |
| Body-region accuracy | Correct localized regions / localized matched player events whose annotated region is observable. Report the count excluded by abstention. |
| Body-region coverage | Localized matched observable player events / all matched observable player events. Also report end-to-end coverage against all observable annotated player events to expose missed detections. |
| False localization rate | Localized matched unobservable player events / all matched unobservable player events. Report false event predictions separately. |
| Temporal stability | Duplicate emissions per matched reference event and localization changes within a continuous reference event; describe any additional stability measure. |
| Latency | Report model compute time separately from sampling/window delay, transport/render delay, and event-onset-to-display delay; include p50/p95 and measurement method. |
| Throughput | Frames or clips processed per wall-clock second, with input FPS, sampling rate, clip size, device, and resolution. |
| Resources | Peak VRAM and GPU utilization when measured; report unavailable telemetry as unavailable. |

Window classification accuracy is a useful training diagnostic, not a substitute for event-level metrics. Always-abstain behavior may have no localization mistakes but must show zero coverage. The proposed p50 <= 80 ms and p95 <= 150 ms end-to-end latencies are engineering targets until measured.

## Synthetic example limitations

Synthetic videos and labels can demonstrate import, sampling, learning, checkpoint reload, inference, event delivery, and visualization. They cannot establish robustness to real gameplay, reliable participant attribution, calibrated confidence, physically valid force, or clinically meaningful injury severity.

- Disclose how labels are generated and which visual cues identify synthetic events, recipients, and regions.
- Separate source seeds and, where feasible, scene templates across splits. State when validation still shares the same generator or visual rules as training.
- Report synthetic results as synthetic smoke-test results. Do not claim cross-game generalization or real-world localization accuracy from them.
- Do not replay annotation events or fixture timestamps while labeling the path model inference. A mock mode is useful when clearly identified.
- Interpret intensity as perceptual impact/haptic salience in [0, 1], not Newtons, damage, pain, or injury severity.

## Scope boundary

V0 uses video only, without audio, game APIs, mods, process-memory reads, title-specific hooks, wearable output, electrical stimulation, temperature, or a large VLM in the per-frame path.
