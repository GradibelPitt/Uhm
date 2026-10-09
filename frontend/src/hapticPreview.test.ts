import { describe, expect, it } from 'vitest'
import { envelopeAt, regionLevels } from './hapticPreview'
import type { HapticCue, HapticPreview } from './hapticPreview'

export const cue: HapticCue = {
  cue_id: 'cue-0', source_ids: ['contact-0'], event_type: 'foot_landing', region: 'feet',
  target: 'observed_subject', basis: 'visible', start_ms: 100, end_ms: 320, source_end_ms: 600,
  effect: 'pulse', perceptual_intensity: null,
  envelope: [{ offset_ms: 0, level: 0 }, { offset_ms: 22, level: 1 },
    { offset_ms: 88, level: .45 }, { offset_ms: 220, level: 0 }],
}
export const preview: HapticPreview = { schema_version: '0.1', mode: 'design_preview',
  amplitude_semantics: 'unit_envelope_not_force', duration_ms: 1000, inferred_included: false, cues: [cue] }

describe('media-time haptic preview', () => {
  it('has exact endpoints, a bounded peak, and clears on invalid time or seek', () => {
    expect(envelopeAt(cue, 100)).toBe(0)
    expect(envelopeAt(cue, 122)).toBe(1)
    expect(envelopeAt(cue, 111)).toBeCloseTo(.5)
    expect(envelopeAt(cue, 320)).toBe(0)
    expect(envelopeAt(cue, -1)).toBe(0)
    expect(envelopeAt(cue, NaN)).toBe(0)
    expect(envelopeAt(cue, 122)).toBe(1)
  })
  it('mixes overlap by maximum without doubling the envelope', () => {
    const combined = { ...preview, cues: [cue, { ...cue, cue_id: 'cue-1' }] }
    expect(regionLevels(combined, 122).feet).toBe(1)
    expect(regionLevels(combined, 500).feet).toBe(0)
  })
})
