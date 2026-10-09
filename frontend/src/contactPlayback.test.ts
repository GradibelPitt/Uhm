import { describe, expect, it } from 'vitest'
import { contactsAt, motionAt, regionState, windowAt } from './contactPlayback'
import type { ContactReport } from './contactPlayback'

const report: ContactReport = {
  duration_s: 3, mode: 'local_vlm', task: 'visible_contact', model: 'test', step_s: .5,
  actions: ['foot_support', 'hand_load'],
  windows: [
    { start_s: 0, end_s: .5, status: 'ok', labels: { foot_support: { state: 'positive', probability: null }, hand_load: { state: 'negative', probability: null } } },
    { start_s: .5, end_s: 1, status: 'partial', labels: { foot_support: { state: 'unknown', probability: null }, hand_load: { state: 'unknown', probability: null } } },
    { start_s: 1, end_s: 1.5, status: 'ok', labels: { foot_support: { state: 'negative', probability: null }, hand_load: { state: 'positive', probability: null } } },
  ],
  contact_events: [
    { event_type: 'foot_support', contact_region: 'feet', start_s: 0, end_s: .5, intensity: null, probability: null },
    { event_type: 'hand_load', contact_region: 'hands_and_arms', start_s: 1, end_s: 1.5, intensity: null, probability: null },
  ],
}
describe('playback uses source time and preserves model uncertainty', () => {
  it('ends each contact at the exclusive boundary and never lights future events', () => {
    expect(contactsAt(report, .49).map(event => event.contact_region)).toEqual(['feet'])
    expect(contactsAt(report, .5)).toEqual([])
    expect(contactsAt(report, 3)).toEqual([])
    expect(contactsAt(report, -1)).toEqual([])
    expect(windowAt(report, NaN)).toBeUndefined()
  })
  it('seeking backwards derives the matching event without retained pulses', () => {
    expect(regionState(report, 'hands', 1.2)).toBe('positive')
    expect(regionState(report, 'hands', .2)).toBe('negative')
    expect(regionState(report, 'feet', .2)).toBe('positive')
  })
  it('keeps occluded, unsupported and uncovered regions distinct from negative evidence', () => {
    expect(regionState(report, 'feet', .6)).toBe('unknown')
    expect(regionState(report, 'head', .2)).toBe('unanalysed')
    expect(regionState(report, 'feet', 2)).toBe('unanalysed')
  })
  it('never displays contacts from a dry run, failed window, or action-only report', () => {
    expect(contactsAt({ ...report, mode: 'dry_run' }, .2)).toEqual([])
    expect(contactsAt({ ...report, task: 'visible_actions' }, .2)).toEqual([])
    expect(contactsAt({ ...report, windows: [{ ...report.windows[0], status: 'error' }] }, .2)).toEqual([])
  })
  it('does not trust a cached segment that bridges unknown evidence', () => {
    const stale = { ...report, contact_events: [{ ...report.contact_events![0], end_s: 1 }] }
    expect(contactsAt(stale, .6)).toEqual([])
  })
  it('shows inferred POV contact and clears it on flight or unavailable frames', () => {
    const pov: ContactReport = { ...report, task: 'egocentric_motion', windows: [
      { ...report.windows[0], motion: { phase: 'landing', feet: 'both', basis: 'inferred', cues: ['deceleration'] } },
      { ...report.windows[1], labels: { foot_support: { state: 'negative', probability: null } }, motion: { phase: 'airborne', feet: 'none', basis: 'inferred', cues: ['flight'] } },
    ] }
    expect(motionAt(pov, .2)?.feet).toBe('both')
    expect(regionState(pov, 'feet', .2)).toBe('positive')
    expect(regionState(pov, 'feet', .6)).toBe('negative')
    expect(motionAt({ ...pov, mode: 'dry_run' }, .2)).toBeUndefined()
    expect(motionAt(pov, -1)).toBeUndefined()
  })
})
