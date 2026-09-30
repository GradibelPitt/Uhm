import { useEffect, useRef, useState } from 'react'
import type { BodyForceEvent } from '../event'
import { ontology } from '../ontology'

const FADE_MS = 700
const BASE = '#eeede8'
const STOPS = [[249, 212, 204], [221, 91, 73], [167, 43, 34], [43, 13, 14]]

/** Intensity is perceptual salience; this scale does not represent physical force. */
export function intensityColor(intensity: number): string {
  const scaled = Math.min(1, Math.max(0, intensity)) * (STOPS.length - 1)
  const lower = Math.min(STOPS.length - 2, Math.floor(scaled))
  const fraction = scaled - lower
  const channels = STOPS[lower].map((value, index) => Math.round(value + (STOPS[lower + 1][index] - value) * fraction))
  return `rgb(${channels.join(', ')})`
}

interface ActivePulse { event: BodyForceEvent; started: number }
interface Props {
  events: BodyForceEvent[]
  resetKey: number
  selectedRegion?: string
  onSelectRegion?: (region: string) => void
  /** Source video clock in milliseconds. Omit for isolated wall-clock mock pulses. */
  mediaClock?: () => number
}

export function BodyMap({ events, resetKey, selectedRegion, onSelectRegion, mediaClock }: Props) {
  const pulses = useRef<ActivePulse[]>([])
  const seen = useRef(new Set<string>())
  const clock = useRef(mediaClock)
  clock.current = mediaClock
  const [levels, setLevels] = useState<Record<string, { intensity: number; opacity: number }>>({})

  useEffect(() => {
    pulses.current = []
    // Retained log rows must not replay after a seek, reconnect, mode switch or rate change.
    seen.current = new Set(events.map(event => event.event_id))
    setLevels({})
  }, [resetKey])

  useEffect(() => {
    const now = clock.current?.() ?? performance.now()
    // Keep deduplication memory bounded by retained rows and currently active events.
    const retained = new Set([...events.map(event => event.event_id), ...pulses.current.map(pulse => pulse.event.event_id)])
    seen.current = new Set([...seen.current].filter(id => retained.has(id)))
    for (const event of events) {
      if (seen.current.has(event.event_id)) continue
      seen.current.add(event.event_id)
      if (event.recipient !== 'player' || event.intensity <= 0 || !Number.isFinite(event.intensity)
        || !Number.isFinite(event.duration_ms) || event.duration_ms < 0) continue
      if (event.body_region !== 'whole_body' && !ontology.regions.some(region => region.id === event.body_region)) continue
      const started = clock.current ? event.timestamp_ms : now
      if (now >= started + event.duration_ms + FADE_MS) continue
      pulses.current.push({ event, started })
    }
  }, [events])

  useEffect(() => {
    let frame: number
    const animate = () => {
      const now = clock.current?.() ?? performance.now()
      pulses.current = pulses.current.filter(pulse => now < pulse.started + pulse.event.duration_ms + FADE_MS)
      const next: Record<string, { intensity: number; opacity: number }> = {}
      for (const pulse of pulses.current) {
        if (now < pulse.started) continue
        const fade = Math.max(0, now - pulse.started - pulse.event.duration_ms) / FADE_MS
        const opacity = 1 - fade * fade * (3 - 2 * fade)
        const ids = pulse.event.body_region === 'whole_body' ? ontology.regions.map(region => region.id) : [pulse.event.body_region]
        for (const id of ids) {
          if (!next[id] || pulse.event.intensity * opacity > next[id].intensity * next[id].opacity) {
            next[id] = { intensity: pulse.event.intensity, opacity }
          }
        }
      }
      setLevels(previous => Object.keys(previous).length || Object.keys(next).length ? next : previous)
      frame = requestAnimationFrame(animate)
    }
    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [])

  return <svg className="body-map" viewBox={ontology.viewBox} role="group" aria-label="Player body map, front view. Anatomical left appears on the right.">
    <defs>
      <pattern id="body-grid" width="25" height="25" patternUnits="userSpaceOnUse"><path d="M 25 0 L 0 0 0 25" fill="none" stroke="#d9d8d1" strokeWidth="0.45" /></pattern>
    </defs>
    <rect x="0" y="0" width="300" height="500" fill="url(#body-grid)" opacity="0.55" />
    <path d="M150 0V500" stroke="#c9c8bf" strokeWidth="0.8" strokeDasharray="3 5" />
    {ontology.regions.map(region => <g key={region.id}>
      <path
        d={region.path}
        fill={BASE}
        stroke={selectedRegion === region.id ? '#9f453b' : '#bfbeb4'}
        strokeWidth={selectedRegion === region.id ? 1.8 : 1}
        className={onSelectRegion ? 'body-region selectable' : 'body-region'}
        tabIndex={onSelectRegion ? 0 : undefined}
        role={onSelectRegion ? 'button' : 'img'}
        aria-label={`${region.label}${levels[region.id] ? `, active, intensity ${Math.round(levels[region.id].intensity * 100)} percent` : ''}`}
        aria-pressed={onSelectRegion ? selectedRegion === region.id : undefined}
        onClick={() => onSelectRegion?.(region.id)}
        onKeyDown={event => {
          if (onSelectRegion && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault()
            onSelectRegion(region.id)
          }
        }}
      ><title>{region.label}</title></path>
      {levels[region.id] && <path d={region.path} fill={intensityColor(levels[region.id].intensity)} opacity={levels[region.id].opacity} pointerEvents="none" />}
    </g>)}
  </svg>
}
