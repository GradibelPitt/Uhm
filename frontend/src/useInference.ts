import { useCallback, useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { isBodyForceEvent } from './event'
import type { BodyForceEvent } from './event'

export interface RuntimeMetadata {
  video_url: string
  video_name: string
  video: { fps: number; duration_ms: number; width?: number; height?: number; synthetic?: boolean; viewpoint?: 'third_person' | 'first_person' | 'unknown' }
  model: { name: string; checkpoint: string; device: string; synthetic: boolean }
  schema_version: string
  sample_step_ms?: number
}
export interface Diagnostics {
  event_class: string; recipient: string; body_region: string; intensity: number
  confidence: BodyForceEvent['confidence']
  observable_probability: number
  latency_ms: number; model_ms: number; sampling_ms: number
}
interface RuntimeStats {
  sampled_windows?: number
  pipeline_ms?: { p50: number | null; p95: number | null }
  model_ms?: { p50: number | null; p95: number | null }
  events?: number
}
type Connection = 'connecting' | 'ready' | 'disconnected' | 'error'

/** Transport only. Event attribution, thresholds and temporal aggregation live on the server. */
export function useInference(video: RefObject<HTMLVideoElement | null>, enabled: boolean) {
  const [metadata, setMetadata] = useState<RuntimeMetadata | null>(null)
  const [connection, setConnection] = useState<Connection>('connecting')
  const [error, setError] = useState<string | null>(null)
  const [events, setEvents] = useState<BodyForceEvent[]>([])
  const [latestEvent, setLatestEvent] = useState<BodyForceEvent | null>(null)
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null)
  const [stats, setStats] = useState<RuntimeStats>({})
  const [logPath, setLogPath] = useState('')
  const [resetKey, setResetKey] = useState(0)
  const [attempt, setAttempt] = useState(0)
  const socket = useRef<WebSocket | null>(null)
  const sequence = useRef(0)
  const inFlight = useRef<{ id: number; sent: number } | null>(null)
  const pendingReset = useRef<number | null>(null)
  const ready = useRef(false)
  const lastSample = useRef(-1)

  const reset = useCallback(() => {
    // Invalidate replies immediately, before the server acknowledges the reset.
    inFlight.current = null
    lastSample.current = -1
    setDiagnostics(null)
    setLatestEvent(null)
    setResetKey(value => value + 1)
    const id = ++sequence.current
    pendingReset.current = id
    if (socket.current?.readyState === WebSocket.OPEN && ready.current) {
      socket.current.send(JSON.stringify({ type: 'reset', request_id: id }))
    }
  }, [])

  const retry = useCallback(() => {
    reset()
    setAttempt(value => value + 1)
  }, [reset])

  useEffect(() => {
    const controller = new AbortController()
    setError(null)
    setConnection('connecting')
    setMetadata(null)
    fetch('/api/meta', { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error(`Runtime metadata unavailable (HTTP ${response.status}).`)
        const result = await response.json() as RuntimeMetadata
        if (!result.video_url || !result.model?.name || !result.video) throw new Error('Runtime returned incomplete metadata.')
        setMetadata(result)
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setConnection('error')
        setError(cause instanceof Error ? cause.message : 'Cannot connect to the inference runtime.')
      })
    return () => controller.abort()
  }, [attempt])

  useEffect(() => {
    if (!enabled || !metadata) return
    let disposed = false
    ready.current = false
    inFlight.current = null
    pendingReset.current = null
    lastSample.current = -1
    setConnection('connecting')
    setError(null)
    setStats({})
    setLogPath('')
    setDiagnostics(null)
    setLatestEvent(null)
    setResetKey(value => value + 1)
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/inference`)
    socket.current = ws
    const fail = (message: string) => {
      if (disposed) return
      ready.current = false
      inFlight.current = null
      setConnection('error')
      setError(message)
      ws.close()
    }
    ws.onmessage = message => {
      if (disposed) return
      let data: Record<string, unknown>
      try { data = JSON.parse(String(message.data)) as Record<string, unknown> }
      catch { fail('Runtime returned an unreadable message.'); return }
      if (data.type === 'ready') {
        ready.current = true
        setConnection('ready')
        setLogPath(typeof data.log_path === 'string' ? data.log_path : '')
        // New connection has a clean temporal state. A reset also clears previous renderer pulses.
        reset()
        return
      }
      if (data.type === 'reset') {
        if (data.request_id === pendingReset.current) pendingReset.current = null
        return
      }
      if (data.type === 'error') {
        if (data.request_id === inFlight.current?.id || data.request_id === pendingReset.current || data.request_id == null) {
          fail(typeof data.message === 'string' ? data.message : 'Inference request failed.')
        }
        return
      }
      if (data.type !== 'prediction' || data.request_id !== inFlight.current?.id || pendingReset.current !== null) return
      inFlight.current = null
      if (data.event != null) {
        if (!isBodyForceEvent(data.event)) { fail('Runtime event rejected: it does not match the canonical BodyForceEvent schema.'); return }
        const event = data.event
        setEvents(current => current.some(item => item.event_id === event.event_id) ? current : [...current.slice(-99), event])
        setLatestEvent(event)
        console.info('[BodyForceEvent / model]', event)
      }
      if (data.diagnostics && typeof data.diagnostics === 'object') setDiagnostics(data.diagnostics as unknown as Diagnostics)
      if (data.stats && typeof data.stats === 'object') setStats(data.stats as RuntimeStats)
    }
    ws.onerror = () => fail('Inference connection unavailable. Start the runtime and reconnect.')
    ws.onclose = () => {
      if (disposed) return
      ready.current = false
      inFlight.current = null
      setConnection(current => current === 'error' ? current : 'disconnected')
      setError(current => current ?? 'The inference runtime disconnected. Reconnect to continue.')
    }
    const interval = window.setInterval(() => {
      const media = video.current
      if (!ready.current || ws.readyState !== WebSocket.OPEN || pendingReset.current !== null || !media || media.paused || media.ended || media.seeking || media.readyState < 2) return
      if (inFlight.current) {
        if (performance.now() - inFlight.current.sent > 15000) fail('Inference timed out. Reconnect to retry.')
        return
      }
      const lastMillisecond = Math.max(0, Math.ceil((Number.isFinite(media.duration) ? media.duration * 1000 : metadata.video.duration_ms)) - 1)
      const timestamp = Math.max(0, Math.min(Math.floor(media.currentTime * 1000), lastMillisecond))
      if (timestamp === lastSample.current) return
      lastSample.current = timestamp
      const request = { type: 'sample', timestamp_ms: timestamp, request_id: ++sequence.current }
      inFlight.current = { id: request.request_id, sent: performance.now() }
      ws.send(JSON.stringify(request))
    }, Math.max(50, metadata.sample_step_ms ?? 125))
    return () => {
      disposed = true
      ready.current = false
      inFlight.current = null
      pendingReset.current = null
      socket.current = null
      window.clearInterval(interval)
      ws.close()
    }
  }, [enabled, metadata, reset, video])

  return { metadata, connection, error, events, latestEvent, diagnostics, stats, logPath, resetKey, reset, retry }
}
