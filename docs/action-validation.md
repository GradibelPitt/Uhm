# Real-video action timeline validation — 2026-10-07

This validation uses third-person recordings of actual people. The initial validation checked download integrity, decoding, frame timing and viewer preparation. A subsequent real API smoke request for one 0.5-second high-jump window returned HTTP 429 with code credit_balance_exhausted. It does not claim successful action recognition or accuracy.

| ID | Source scene observed on four sampled positions | Source length / FPS | Local dry-run |
| --- | --- | --- | --- |
| high_jump | One athlete approaches and then jumps over a high-jump bar | 3.44 s / 25 | 7 windows, all not_run |
| hand_wave | Three people face the camera and wave; primary-person ambiguity | 5.20 s / 30 | 11 windows, all not_run |
| basketball_dunk | Distant basketball court with multiple players and rapid motion | 5.00 s / 25 | 10 windows, all not_run |
| pedestrians | Elevated fixed camera, multiple small walking pedestrians and occlusion | 79.50 s / 10 | First 10 s: 20 windows, all not_run |

All four original file SHA-256 values matched docs/action-video-sources.json. The local importer decoded the first, middle and final frame of each original and each derived MP4. It saved data/raw/action_validation/original and data/raw/action_validation/mp4, plus ignored runs/action_video_imports.json with local derivative hashes and dimensions. A four-position visual contact sheet was inspected at runs/action_video_contact_sheet.jpg. The MP4 conversions kept frame counts, frame rates and dimensions; they use H.264/yuv420p for the local viewer. This is a local transformation, not a new source or license claim.

The independently served viewer was exercised with a FastAPI client: HTML, byte-range video response, timeline JSON and explicit dry-run metadata passed. The browser automation permission for opening localhost was denied, so an in-browser screenshot and playback interaction were not completed. The project still exposes a manual launch command in README.

Focused backend tests: 10 passed. They cover ordered video frames and separate action intervals, multi-label output, refusal/unknown, action-specific observability, the middle probability band, duplicate answers, one-frame abstention, low-FPS and final-frame timestamp mapping, and the independent viewer. The preexisting haptic server tests could not load PyTorch because this computer's application-control policy blocks its native DLL; no system policy changes were made. The frontend build succeeded.

The positive/negative thresholds are uncalibrated defaults. Timestamp evidence is a CFR estimate based on decoded frame index and source FPS, with sampling error recorded. The supplied API key is stored using Windows user encryption in ignored runs/openai-key.clixml, and the runner exposes it only to its child process. Two minimal requests reached the API; the second identified credit_balance_exhausted. No model judgments or usage results were returned. A successful real-model smoke run and accuracy review remain outstanding until API billing has available credits. A key shared in chat should be rotated using scripts/save_openai_key.ps1, whose input is hidden.
