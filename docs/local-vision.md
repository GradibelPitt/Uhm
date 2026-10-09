# Local vision for body contact and haptic relevance

The local backend uses Google's Gemma 4 12B through a portable Ollama runtime on the ROG. It looks for visible support, push-off, landing, external hand loads and impacts. Positive intervals map to feet, hands/arms, legs, trunk, head or hips based on the visible contact type. It does not need a sport or activity name to select that contact region.

These are visual contact candidates. They identify the body region relevant to a possible haptic response. Physical force magnitude, perceptual intensity, calibrated probability, and anatomical left/right are not inferred by this first version. The existing checkpoint ontology represents the whole legs, so these foot-level candidates remain a separate contract from the old BodyForceEvent stream.

## Run locally

From the repository:

    .\scripts\start_local_vision.ps1
    .\.venv\Scripts\python.exe -m uhm infer-local-contacts --video data/raw/action_validation/mp4/high_jump.mp4 --output runs/high_jump.body_contacts.json
    .\.venv\Scripts\python.exe -m uhm serve-actions data/raw/action_validation/mp4/high_jump.mp4 runs/high_jump.body_contacts.json

For a short interval, add --start-seconds 1.5 --max-seconds 0.5. Timestamps stay on the original video clock. Configuration and contact definitions are in configs/local_contacts.yaml. The backend samples ordered frames at 4 fps, uses up to 2 seconds of context, and reports 0.5-second anchor intervals. Adjacent positive intervals merge independently for each contact type. Occluded or ambiguous contacts remain unknown.

The model API is bound to 127.0.0.1:11435. Frames are processed locally without an API key. The standalone viewer uses port 8001. Model files are under the current user's AppData/Local/UHM/models; the portable runtime is under AppData/Local/UHM/runtimes/ollama-v0.40.0. They are outside OneDrive. Starting the helper reuses an existing runtime on that endpoint.

## Installed model and validation

- Runtime: official Ollama v0.40.0 Windows standalone distribution; SHA-256 3623e256762ca89bd6fa99b0cc4106401919ce9df926411673e632e3ea287bb5.
- Model: gemma4:12b, 11.9B parameters, Q4_K_M, with vision capability confirmed by the runtime.
- Model manifest digest: 6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b.
- Hardware: RTX 5080 Laptop GPU, 16GB VRAM. The first request took about 80 seconds including loading and preheating; the seven-window pass with the full body-contact configuration ran at 3.8–4.4 seconds per window after that.
- Real video: the full 3.44-second source in data/raw/action_validation/mp4/high_jump.mp4 produced foot-support candidates during 0.0–2.0 seconds, foot push-off during 1.5–2.0 seconds, no positive foot contacts in the 2.0–3.0-second windows, and a trunk-impact candidate during 3.0–3.44 seconds. It also produced foot-landing candidates; those require independent contact review. These are model predictions, not independently validated force measurements or event accuracy.
- Occlusion check: the single-person crop derived from the credited hand-wave source hides the feet. All three foot-contact labels remained unknown in the checked 1.5–2.0-second interval; visible free hand movement produced no positive external-load or hand-impact candidates. Crop provenance is in ignored runs/action_video_imports.json.
- Focused Python checks: 14 tests passed; frontend checks: 17 tests passed and the build succeeded.

The legacy PyTorch DLL remains blocked by the computer's application-control policy. The local vision runtime runs independently and does not change that policy. In-browser playback inspection is still outstanding because browser access to the local viewer was declined earlier.

Sources: [Google Gemma overview](https://ai.google.dev/gemma/docs/core), [Gemma 4 model in Ollama](https://ollama.com/library/gemma4), [Ollama vision API](https://docs.ollama.com/capabilities/vision), [structured JSON responses](https://docs.ollama.com/capabilities/structured-outputs).
