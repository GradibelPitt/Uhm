# Synthetic video plumbing dataset

288 real H.264 video clips at 12 FPS, 256×256 pixels; 216 training and 72 validation clips. Six training and two validation source recordings are disjoint. The 20-second demo is a third, held-out source. The same procedural renderer is used across splits, so these are not gameplay-generalization results.

**All current generated videos are third-person, front-facing synthetic scenes, explicitly labeled `viewpoint: third_person`. There is no first-person footage.** Future first-person data uses `first_person`, a separate annotation file and independent evaluation. Undeclared external recordings retain `unknown`.

Exactly half the clips are negative for player haptics: idle/menu/camera/UI/object motion, NPC impacts, environment impacts, or an occluded unknown recipient. Positive clips include all eight anatomical regions plus player impacts whose location is deliberately hidden. Anatomical left is screen-right for this front-facing actor. Local visible body contact supplies the region labels.

Only `no_event` and `impact` are populated. `collision` and `explosion` remain reserved classes without example training support. Synthetic intensity corresponds to rendered contact salience, never Newtons.

`annotations/example.jsonl` contains train/validation labels. `annotations/demo.jsonl` is held-out evaluation truth with frame-derived contact onsets and exhaustive negatives; the runtime must never read it. `example_manifest.json` records seeds, provenance and limitations. Annotation paths are relative to the repository root. All extracted clips are silent.

Default training uses `annotations/example_stream.jsonl`: 496 precise contact/negative intervals over the original source recordings, yielding 1728 training and 576 validation causal windows (1000ms context, 125ms cadence). Labels describe each window's current timestamp, including explicit no-event labels before physical contact. Original whole-second `example.jsonl` labels remain a legacy comparison; their interval begins before actual contact.

For real data, import CFR videos, split into clips, annotate using the canonical ClipAnnotation schema, and assign entire source sessions to one split. Use `unknown` when contact or recipient cannot be observed. The loader rejects a source ID assigned to multiple splits.
