export interface HapticCue {
  cue_id: string
  source_ids: string[]
  event_type: string
  region: string
  target: 'player' | 'observed_subject' | 'camera_wearer'
  basis: 'visible' | 'inferred' | 'unspecified' | 'model_event'
  start_ms: number
  end_ms: number
  source_end_ms: number
  effect: 'pulse' | 'sustain'
  envelope: { offset_ms: number; level: number }[]
  perceptual_intensity: number | null
}
export interface HapticPreview {
  schema_version: '0.1'
  mode: 'design_preview'
  amplitude_semantics: 'unit_envelope_not_force'
  duration_ms: number
  inferred_included: boolean
  cues: HapticCue[]
}

export function envelopeAt(cue: HapticCue, timestampMs: number): number {
  if (!Number.isFinite(timestampMs) || timestampMs < cue.start_ms || timestampMs >= cue.end_ms) return 0
  const offset = timestampMs - cue.start_ms
  for (let i = 1; i < cue.envelope.length; i++) {
    const left = cue.envelope[i - 1], right = cue.envelope[i]
    if (left.offset_ms <= offset && offset <= right.offset_ms && right.offset_ms > left.offset_ms) {
      return left.level + (offset - left.offset_ms) / (right.offset_ms - left.offset_ms) * (right.level - left.level)
    }
  }
  return 0
}

export function regionLevels(preview: HapticPreview, timestampMs: number): Record<string, number> {
  const levels: Record<string, number> = {}
  for (const cue of preview.cues) {
    // Same-region overlap takes the maximum, never adds amplitudes.
    levels[cue.region] = Math.max(levels[cue.region] ?? 0, envelopeAt(cue, timestampMs))
  }
  return levels
}
