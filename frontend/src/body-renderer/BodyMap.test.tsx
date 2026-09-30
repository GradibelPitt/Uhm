import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { BodyMap } from './BodyMap'
import type { BodyForceEvent } from '../event'
import { ontology } from '../ontology'

const event = (changes: Partial<BodyForceEvent> = {}): BodyForceEvent => ({
  event_id: 'impact-1', timestamp_ms: 1000, event_class: 'impact', recipient: 'player',
  body_region: 'left_torso', intensity: 0.8, duration_ms: 200, direction: 'unknown',
  confidence: { event: 0.9, recipient: 0.9, region: 0.9, intensity: 0 }, ...changes,
})
const tick = () => act(() => vi.advanceTimersByTime(32))
const active = () => screen.queryAllByRole('img', { name: /, active,/ })
const overlay = (label = 'Left torso') => screen.getByRole('img', { name: new RegExp(`^${label}`) }).nextElementSibling

describe('body event timing and observability', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => { cleanup(); vi.useRealTimers() })

  it('holds and smoothly fades in source time; paused playback preserves the current response', () => {
    let mediaTime = 1000
    const clock = () => mediaTime
    const view = render(<BodyMap events={[]} resetKey={0} mediaClock={clock} />)
    view.rerender(<BodyMap events={[event()]} resetKey={0} mediaClock={clock} />)
    tick()
    expect(overlay()?.getAttribute('opacity')).toBe('1')
    act(() => vi.advanceTimersByTime(3000))
    expect(overlay()?.getAttribute('opacity')).toBe('1')
    mediaTime = 1550
    tick()
    expect(Number(overlay()?.getAttribute('opacity'))).toBeCloseTo(0.5)
    mediaTime = 1901
    tick()
    expect(active()).toHaveLength(0)
  })

  it('never highlights a non-player recipient or an unknown region', () => {
    const view = render(<BodyMap events={[]} resetKey={0} mediaClock={() => 1000} />)
    view.rerender(<BodyMap events={[
      event({ event_id: 'other', recipient: 'other_character' }),
      event({ event_id: 'environment', recipient: 'environment' }),
      event({ event_id: 'unknown-recipient', recipient: 'unknown' }),
      event({ event_id: 'unknown-region', body_region: 'unknown' }),
    ]} resetKey={0} mediaClock={() => 1000} />)
    tick()
    expect(active()).toHaveLength(0)
  })

  it('maps a whole-body event to every region supplied by the ontology', () => {
    const view = render(<BodyMap events={[]} resetKey={0} mediaClock={() => 1000} />)
    view.rerender(<BodyMap events={[event({ body_region: 'whole_body' })]} resetKey={0} mediaClock={() => 1000} />)
    tick()
    expect(active()).toHaveLength(ontology.regions.length)
  })

  it('preserves a longer weaker event after the stronger overlapping event expires', () => {
    let mediaTime = 1100
    const clock = () => mediaTime
    const view = render(<BodyMap events={[]} resetKey={0} mediaClock={clock} />)
    view.rerender(<BodyMap events={[
      event({ event_id: 'long', duration_ms: 2000, intensity: 0.4 }),
      event({ event_id: 'strong', timestamp_ms: 1100, intensity: 0.9 }),
    ]} resetKey={0} mediaClock={clock} />)
    tick()
    expect(active()[0].getAttribute('aria-label')).toContain('90 percent')
    mediaTime = 2001
    tick()
    expect(active()[0].getAttribute('aria-label')).toContain('40 percent')
  })

  it('does not replay retained event-log rows after a seek reset and a later new event', () => {
    const retained = event()
    const view = render(<BodyMap events={[]} resetKey={0} mediaClock={() => 1000} />)
    view.rerender(<BodyMap events={[retained]} resetKey={0} mediaClock={() => 1000} />)
    tick()
    expect(active()).toHaveLength(1)
    view.rerender(<BodyMap events={[retained]} resetKey={1} mediaClock={() => 1000} />)
    tick()
    expect(active()).toHaveLength(0)
    view.rerender(<BodyMap events={[retained, event({ event_id: 'new', body_region: 'left_arm' })]} resetKey={1} mediaClock={() => 1000} />)
    tick()
    expect(active()).toHaveLength(1)
    expect(active()[0].getAttribute('aria-label')).toContain('Left arm')
  })

  it('discards an already-expired late event instead of flashing it at arrival', () => {
    const view = render(<BodyMap events={[]} resetKey={0} mediaClock={() => 4000} />)
    view.rerender(<BodyMap events={[event()]} resetKey={0} mediaClock={() => 4000} />)
    tick()
    expect(active()).toHaveLength(0)
  })

  it('uses wall time for isolated manual mock pulses', () => {
    const view = render(<BodyMap events={[]} resetKey={0} />)
    view.rerender(<BodyMap events={[event({ timestamp_ms: 50000 })]} resetKey={0} />)
    tick()
    expect(active()).toHaveLength(1)
    act(() => vi.advanceTimersByTime(1000))
    expect(active()).toHaveLength(0)
  })
})
